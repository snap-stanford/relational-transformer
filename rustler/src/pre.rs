//! Preprocess a relbench-3.0.0 dataset into rustler's on-disk format.
//!
//! Input is a self-describing dataset directory (a `manifest.yaml` next to
//! `db/<table>.parquet`, and optionally `tasks/<task>/{train,val,test}.parquet`).
//! The manifest is the sole source of relational metadata (primary keys, the
//! foreign-key graph, time columns); the parquet files carry only native column
//! dtypes. Output is written to `<out_dir>/<dataset name>/`.

use crate::common::{Adj, Edge, Node, Offsets, SemType, TableInfo, TableType};
use clap::Parser;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use rkyv::rancor::Error;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::io::BufWriter;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const PBAR_TEMPLATE: &str = "{percent}% {bar} {decimal_bytes}/{decimal_total_bytes} [{elapsed_precise}<{eta_precise}, {decimal_bytes_per_sec}]";

/// Version of the preprocessed on-disk format, recorded in `meta.json`.
const PRE_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone)]
struct ColStat {
    mean: f64,
    std: f64,
}

#[derive(Debug, Clone, Default)]
struct Table {
    table_name: String,
    df: DataFrame,
    col_stats: Vec<ColStat>,
    pcol_name: Option<String>,
    fcol_name_to_ptable_name: HashMap<String, String>,
    tcol_name: Option<String>,
    node_idx_offset: i64,
}

// ---------------------------------------------------------------------------
// relbench-3.0.0 manifest schema (only the fields rustler needs). Unknown
// fields are ignored, so this stays forward-compatible with the full schema.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
struct TableSpec {
    #[serde(default)]
    pkey: Option<String>,
    #[serde(default)]
    time_col: Option<String>,
    #[serde(default)]
    fkeys: HashMap<String, String>, // fkey_col -> parent_table
}

#[derive(Debug, Deserialize)]
struct DatasetManifest {
    name: String,
    #[serde(default)]
    tables: HashMap<String, TableSpec>,
}

#[derive(Debug, Default, Deserialize)]
struct TaskManifest {
    #[serde(default)]
    entity_table: Option<String>,
    #[serde(default)]
    entity_col: Option<String>,
    #[serde(default)]
    target_col: Option<String>,
    #[serde(default)]
    task_type: Option<String>,
    #[serde(default)]
    time_col: Option<String>,
    #[serde(default)]
    src_entity_table: Option<String>,
    #[serde(default)]
    src_entity_col: Option<String>,
    #[serde(default)]
    dst_entity_table: Option<String>,
    #[serde(default)]
    dst_entity_col: Option<String>,
}

impl TaskManifest {
    /// Foreign keys of a task label table: its entity column(s) point into the
    /// database entity table(s) (src/dst for link-prediction tasks).
    fn fkeys(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        for (col, table) in [
            (&self.entity_col, &self.entity_table),
            (&self.src_entity_col, &self.src_entity_table),
            (&self.dst_entity_col, &self.dst_entity_table),
        ] {
            if let (Some(c), Some(t)) = (col, table) {
                m.insert(c.clone(), t.clone());
            }
        }
        m
    }
}

fn load_yaml<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    let s = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    serde_yaml::from_str(&s)
        .unwrap_or_else(|e| panic!("failed to parse YAML {}: {}", path.display(), e))
}

/// Read an integer value (a foreign key / row index) as i64, accepting any
/// width. relbench-3.0.0 stores reindexed keys as whatever int type fits.
fn anyvalue_to_i64(v: &AnyValue) -> Option<i64> {
    match v {
        AnyValue::Int8(x) => Some(*x as i64),
        AnyValue::Int16(x) => Some(*x as i64),
        AnyValue::Int32(x) => Some(*x as i64),
        AnyValue::Int64(x) => Some(*x),
        AnyValue::UInt8(x) => Some(*x as i64),
        AnyValue::UInt16(x) => Some(*x as i64),
        AnyValue::UInt32(x) => Some(*x as i64),
        AnyValue::UInt64(x) => Some(*x as i64),
        // Nullable-int keys upcast to float by pandas; NaN means "missing".
        AnyValue::Float32(x) => (!x.is_nan()).then_some(*x as i64),
        AnyValue::Float64(x) => (!x.is_nan()).then_some(*x as i64),
        _ => None,
    }
}

/// Read the timestamp (seconds since epoch, as i32) at row `r` of a table's
/// designated time column. Returns None for null cells, and also when the time
/// column is not a Datetime (a few datasets designate a non-temporal column,
/// e.g. an integer `year`); such edges simply carry no timestamp rather than
/// aborting the whole dataset.
fn read_timestamp(df: &DataFrame, tcol_name: &str, r: usize) -> Option<i32> {
    let col = df.column(tcol_name).ok()?;
    let dt = col.datetime().ok()?;
    dt.get(r)
        .map(|v| (v / 1_000_000_000).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
}

/// Parent row indices referenced by a single foreign-key cell. Handles scalar
/// keys and list-valued keys (a row that links to several parents, e.g. a
/// many-to-many relation); null/NaN entries are skipped.
fn fk_parent_idxs(v: &AnyValue) -> Vec<i64> {
    match v {
        AnyValue::Null => Vec::new(),
        AnyValue::List(series) => series.iter().filter_map(|x| anyvalue_to_i64(&x)).collect(),
        scalar => anyvalue_to_i64(scalar).into_iter().collect(),
    }
}

/// Normalize a freshly-read table to a uniform set of dtypes before
/// featurization: dates and non-nanosecond datetimes -> Datetime(ns), and
/// categoricals -> strings. Other dtypes are handled downstream (numbers,
/// bools, strings, binary).
fn normalize_df(df: DataFrame) -> DataFrame {
    let casts: Vec<Expr> = df
        .iter()
        .filter_map(|s| {
            let name = s.name().to_string();
            match s.dtype() {
                DataType::Date | DataType::Datetime(_, _) => Some(
                    col(name.as_str())
                        .cast(DataType::Datetime(TimeUnit::Nanoseconds, None))
                        .alias(name.as_str()),
                ),
                DataType::Categorical(_, _) => {
                    Some(col(name.as_str()).cast(DataType::String).alias(name.as_str()))
                }
                _ => None,
            }
        })
        .collect();
    if casts.is_empty() {
        return df;
    }
    df.lazy()
        .with_columns(casts)
        .collect()
        .expect("failed to normalize column dtypes")
}

#[derive(Parser)]
pub struct Cli {
    /// Local dataset directory in relbench-3.0.0 layout: a `manifest.yaml`
    /// next to `db/<table>.parquet` (and optionally `tasks/<task>/`).
    #[arg(long)]
    pub dataset_dir: String,
    /// Output root. Preprocessed data is written to `<out_dir>/<dataset name>`.
    #[arg(long)]
    pub out_dir: String,
    /// Ingest only the database tables; skip `tasks/`.
    #[arg(long, default_value_t = false)]
    pub skip_tasks: bool,
    /// Keep database tables out of the node vector (task nodes only).
    #[arg(long, default_value_t = false)]
    pub skip_db: bool,
    /// Provenance recorded in `meta.json` (e.g. the source HF dataset spec).
    #[arg(long)]
    pub source: Option<String>,
}

/// One parquet file to ingest, with relational metadata resolved from the
/// (dataset or task) manifest.
struct ReadSpec {
    path: PathBuf,
    table_name: String,
    table_type: TableType,
    pcol_name: Option<String>,
    fcol_name_to_ptable_name: HashMap<String, String>,
    tcol_name: Option<String>,
}

pub fn main(cli: Cli) {
    let dataset_dir = Path::new(&cli.dataset_dir);
    let manifest: DatasetManifest = load_yaml(&dataset_dir.join("manifest.yaml"));
    let name = manifest.name.clone();
    println!("preprocessing dataset {:?} from {:?}", name, dataset_dir);

    // Assemble the work list: database tables first, then task label tables.
    // The order fixes node_idx_offset assignment, so we sort within each group
    // for reproducible output.
    let mut specs: Vec<ReadSpec> = Vec::new();

    let mut db_specs: Vec<ReadSpec> = Vec::new();
    for entry in glob(&format!("{}/db/*.parquet", dataset_dir.display())).unwrap() {
        let path = entry.unwrap();
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let spec = manifest.tables.get(&stem).cloned().unwrap_or_else(|| {
            eprintln!("warning: table {:?} not in manifest; treating as relation-free", stem);
            TableSpec::default()
        });
        db_specs.push(ReadSpec {
            path,
            table_name: stem,
            table_type: TableType::Db,
            pcol_name: spec.pkey,
            fcol_name_to_ptable_name: spec.fkeys,
            tcol_name: spec.time_col,
        });
    }
    db_specs.sort_by(|a, b| a.path.cmp(&b.path));
    let num_db_tables = db_specs.len();
    specs.extend(db_specs);

    let mut num_task_tables = 0usize;
    // Per-task metadata (target column + type) recorded in meta.json so the
    // pretraining/eval task lists can be built straight from preprocessed data.
    let mut tasks_meta: Vec<serde_json::Value> = Vec::new();
    if !cli.skip_tasks {
        let mut task_specs: Vec<ReadSpec> = Vec::new();
        for task_entry in glob(&format!("{}/tasks/*", dataset_dir.display())).unwrap() {
            let task_dir = task_entry.unwrap();
            let tm_path = task_dir.join("manifest.yaml");
            if !task_dir.is_dir() || !tm_path.exists() {
                continue;
            }
            let tm: TaskManifest = load_yaml(&tm_path);
            let task_name = task_dir.file_name().unwrap().to_str().unwrap().to_string();
            let fkeys = tm.fkeys();
            let mut splits: Vec<&str> = Vec::new();
            for (stem, table_type) in [
                ("train", TableType::Train),
                ("val", TableType::Val),
                ("test", TableType::Test),
            ] {
                let path = task_dir.join(format!("{}.parquet", stem));
                if path.exists() {
                    splits.push(stem);
                    task_specs.push(ReadSpec {
                        path,
                        table_name: task_name.clone(),
                        table_type,
                        pcol_name: None,
                        fcol_name_to_ptable_name: fkeys.clone(),
                        tcol_name: tm.time_col.clone(),
                    });
                }
            }
            tasks_meta.push(serde_json::json!({
                "name": task_name,
                "target_col": tm.target_col,
                "task_type": tm.task_type,
                "entity_table": tm.entity_table,
                "time_col": tm.time_col,
                "splits": splits,
            }));
        }
        task_specs.sort_by(|a, b| a.path.cmp(&b.path));
        num_task_tables = task_specs.len();
        specs.extend(task_specs);
    }
    tasks_meta.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));

    println!(
        "found {} database table(s) and {} task label table(s)",
        num_db_tables, num_task_tables
    );

    println!("reading tables...");
    let tic = Instant::now();
    let mut table_map = HashMap::with_hasher(BuildHasherDefault::<DefaultHasher>::new());
    let mut num_rows_sum: i64 = 0;
    let mut num_cells_sum: i64 = 0;

    for spec in specs {
        let mut file = fs::File::open(&spec.path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();
        let df = normalize_df(df);

        let table_name = spec.table_name;
        let table_type = spec.table_type;
        let pcol_name = spec.pcol_name;
        let fcol_name_to_ptable_name = spec.fcol_name_to_ptable_name;
        let tcol_name = spec.tcol_name;

        println!(
            "read table {} of type {:?} with shape {:?}",
            table_name,
            table_type,
            df.shape()
        );
        let table_key = (table_name.clone(), table_type.clone());

        let num_rows = df.height() as i64;
        let num_cells = num_rows * df.width() as i64;
        table_map.insert(
            table_key,
            Table {
                table_name,
                df,
                col_stats: Vec::new(),
                pcol_name,
                fcol_name_to_ptable_name,
                tcol_name,
                node_idx_offset: num_rows_sum,
            },
        );
        num_rows_sum += num_rows;
        num_cells_sum += num_cells;
    }
    assert!(
        i32::try_from(num_rows_sum).is_ok(),
        "total row count {} overflows i32",
        num_rows_sum
    );
    println!("done in {:?}.", tic.elapsed());

    println!("computing column stats...");
    let tic = Instant::now();
    let mut dt_cnt: usize = 0;
    let mut dt_mean: f64 = 0.0;
    let mut dt_m2: f64 = 0.0;

    for table in table_map.values_mut() {
        for col in table.df.iter() {
            let col = col.rechunk();
            match col.dtype() {
                DataType::Boolean => {
                    let col_float = col.cast(&DataType::Float64).unwrap().drop_nulls();
                    let col_mean = col_float.mean().unwrap_or(0.0);
                    let col_std = col_float.std(1).unwrap_or(0.0);
                    table.col_stats.push(ColStat {
                        mean: col_mean,
                        std: col_std,
                    });
                }
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float64
                | DataType::Float32
                | DataType::Duration(_) => {
                    // Duration -> raw integer count of time units, then numeric.
                    let col = if matches!(col.dtype(), DataType::Duration(_)) {
                        col.cast(&DataType::Int64).unwrap()
                    } else {
                        col
                    };
                    let col = col.cast(&DataType::Float64).unwrap().drop_nulls();
                    let col = col.filter(&col.is_not_nan().unwrap()).unwrap();
                    let mean = col.mean().unwrap_or(0.0);
                    let std = col.std(1).unwrap_or(1.0);
                    let std = if std == 0.0 { 1.0 } else { std };
                    table.col_stats.push(ColStat { mean, std });
                }
                DataType::Datetime(u, _) => {
                    // Accumulate a single global mean/std over all datetime
                    // cells (Welford), so datetimes share one normalizer.
                    let col = if *u != TimeUnit::Nanoseconds {
                        col.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                            .unwrap()
                    } else {
                        col.clone()
                    };
                    assert!(*col.dtype() == DataType::Datetime(TimeUnit::Nanoseconds, None));
                    let col = col.cast(&DataType::Float64).unwrap().drop_nulls();
                    let col = col.filter(&col.is_not_nan().unwrap()).unwrap();
                    for x in col.iter() {
                        let AnyValue::Float64(f) = x else { panic!() };
                        dt_cnt += 1;
                        let delta = f - dt_mean;
                        dt_mean += delta / dt_cnt as f64;
                        let delta2 = f - dt_mean;
                        dt_m2 += delta * delta2;
                    }
                    table.col_stats.push(ColStat {
                        mean: 0.0,
                        std: 0.0,
                    });
                }
                _ => table.col_stats.push(ColStat {
                    mean: 0.0,
                    std: 0.0,
                }),
            }
        }
    }

    let dt_std = if dt_cnt > 1 {
        (dt_m2 / dt_cnt as f64).sqrt()
    } else {
        1.0
    };

    let mut col_stats_map = HashMap::new();
    for ((table_name, table_type), table) in &table_map {
        if table_type == &TableType::Train {
            col_stats_map.insert(table_name.clone(), table.col_stats.clone());
        }
    }
    for ((table_name, table_type), table) in &mut table_map {
        match table_type {
            TableType::Val | TableType::Test => {
                table.col_stats = col_stats_map.get(table_name).unwrap().clone();
            }
            _ => {}
        }
    }
    println!("done in {:?}.", tic.elapsed());

    println!("making node vector...");
    let tic = Instant::now();
    let pbar = ProgressBar::new(num_cells_sum as u64).with_style(
        ProgressStyle::default_bar()
            .template(PBAR_TEMPLATE)
            .unwrap(),
    );
    let mut text_to_idx = HashMap::new();
    let mut column_name_to_idx: Vec<(String, i32)> = Vec::new();
    let mut node_vec = (0..num_rows_sum)
        .map(|_| Node::default())
        .collect::<Vec<_>>();
    let mut p2f_adj = Adj {
        adj: vec![Vec::new(); num_rows_sum as usize],
    };

    for ((_table_name, table_type), table) in &table_map {
        if cli.skip_db && table_type == &TableType::Db {
            println!(
                "skipping table {} of type {:?}",
                table.table_name, table_type
            );
            continue;
        }

        let l = text_to_idx.len() as i32;
        let table_name_idx = *text_to_idx
            .entry(table.table_name.clone())
            .or_insert_with(|| l);

        for (col, col_stat) in table.df.iter().zip(&table.col_stats) {
            let col = col.rechunk();

            // Convert categoricals to strings
            let col = if matches!(col.dtype(), DataType::Categorical(_, _)) {
                col.cast(&DataType::String).unwrap()
            } else {
                col
            };

            // Convert datetime columns to nanoseconds if needed
            let col = if let DataType::Datetime(unit, tz) = col.dtype() {
                if *unit != TimeUnit::Nanoseconds {
                    col.cast(&DataType::Datetime(TimeUnit::Nanoseconds, tz.clone()))
                        .unwrap()
                } else {
                    col
                }
            } else {
                col
            };

            let col_name = format!("{} of {}", col.name(), table.table_name.clone());
            let l = text_to_idx.len() as i32;
            let col_name_idx = *text_to_idx.entry(col_name.clone()).or_insert_with(|| {
                column_name_to_idx.push((col_name.clone(), l));
                l
            });

            if col.name() == table.pcol_name.as_deref().unwrap_or("") {
                pbar.inc(col.len() as u64);
                continue;
            }

            if table
                .fcol_name_to_ptable_name
                .contains_key(&col.name().to_string())
            {
                if matches!(col.dtype(), polars::datatypes::DataType::Datetime(_, _)) {
                    pbar.inc(col.len() as u64);
                    continue;
                }

                let ptable_name = table
                    .fcol_name_to_ptable_name
                    .get(&col.name().to_string())
                    .unwrap();
                let ptable_offset = table_map
                    .get(&(ptable_name.to_string(), TableType::Db))
                    .unwrap_or_else(|| {
                        dbg!(ptable_name.to_string());
                        dbg!(table_map.keys());
                        panic!()
                    })
                    .node_idx_offset;
                for (r, val) in col.iter().enumerate() {
                    pbar.inc(1);
                    let parent_idxs = fk_parent_idxs(&val);
                    if parent_idxs.is_empty() {
                        continue;
                    }

                    let node_idx = i32::try_from(table.node_idx_offset + r as i64)
                        .expect("node index overflow");
                    let node = node_vec.get_mut(node_idx as usize).unwrap();
                    node.is_task_node = table_type != &TableType::Db;
                    node.node_idx = node_idx;
                    node.table_name_idx = table_name_idx;

                    let timestamp = table
                        .tcol_name
                        .as_ref()
                        .and_then(|c| read_timestamp(&table.df, c, r));
                    node.timestamp = timestamp;

                    let l = text_to_idx.len() as i32;
                    let ptable_name_idx = *text_to_idx
                        .entry(ptable_name.to_string())
                        .or_insert_with(|| l);
                    let ptable = &table_map[&(ptable_name.to_string(), TableType::Db)];

                    for pval in parent_idxs {
                        let pnode_idx = i32::try_from(ptable_offset + pval)
                            .expect("parent node index overflow");
                        node.f2p_nbr_idxs.push(pnode_idx);

                        let ptimestamp = ptable
                            .tcol_name
                            .as_ref()
                            .and_then(|tcol_name| read_timestamp(&ptable.df, tcol_name, pval as usize));

                        let f2p_edge = Edge {
                            node_idx: pnode_idx,
                            table_name_idx: ptable_name_idx,
                            table_type: TableType::Db,
                            timestamp: ptimestamp,
                        };
                        node.f2p_edges.push(f2p_edge);

                        let p2f_edge = Edge {
                            node_idx,
                            table_name_idx,
                            table_type: table_type.clone(),
                            timestamp,
                        };
                        p2f_adj.adj[pnode_idx as usize].push(p2f_edge);
                    }
                }

                continue;
            }

            for (r, val) in col.iter().enumerate() {
                pbar.inc(1);
                let node_idx =
                    i32::try_from(table.node_idx_offset + r as i64).expect("node index overflow");
                let node = &mut node_vec[node_idx as usize];
                node.is_task_node = table_type != &TableType::Db;
                node.node_idx = node_idx;
                node.table_name_idx = table_name_idx;

                let val = match val {
                    AnyValue::Boolean(val) => AnyValue::Boolean(val),
                    AnyValue::Int8(val) => AnyValue::Float64(val as f64),
                    AnyValue::Int16(val) => AnyValue::Float64(val as f64),
                    AnyValue::Int32(val) => AnyValue::Float64(val as f64),
                    AnyValue::Int64(val) => AnyValue::Float64(val as f64),
                    AnyValue::UInt8(val) => AnyValue::Float64(val as f64),
                    AnyValue::UInt16(val) => AnyValue::Float64(val as f64),
                    AnyValue::UInt32(val) => AnyValue::Float64(val as f64),
                    AnyValue::UInt64(val) => AnyValue::Float64(val as f64),
                    AnyValue::Float32(val) => AnyValue::Float64(val as f64),
                    // Duration columns (e.g. lap/pit times): treat the raw
                    // integer count of time units as a numeric value.
                    AnyValue::Duration(val, _) => AnyValue::Float64(val as f64),
                    AnyValue::Binary(val) =>
                    {
                        #[allow(clippy::unnecessary_to_owned)]
                        AnyValue::String(&String::from_utf8_lossy(val).to_string())
                    }
                    _ => val,
                };
                match val {
                    AnyValue::Null => {}
                    AnyValue::Boolean(val) => {
                        let val_float = if val { 1.0 } else { 0.0 };
                        let val_float = (val_float - col_stat.mean) / col_stat.std;
                        node.boolean_values.push(val_float as f32);
                        node.number_values.push(0.0);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Boolean);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                    }
                    AnyValue::Float64(val) => {
                        if val.is_nan() {
                            continue;
                        }
                        let val = (val - col_stat.mean) / col_stat.std;
                        if val.is_infinite() {
                            dbg!(&table.table_name);
                            dbg!(col.name());
                            dbg!(col_stat);
                            panic!();
                        }
                        node.boolean_values.push(0.0);
                        node.number_values.push(val as f32);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Number);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                    }
                    AnyValue::Datetime(val, unit, _) => {
                        assert!(unit == TimeUnit::Nanoseconds);
                        let val = (val as f64 - dt_mean) / dt_std;
                        node.boolean_values.push(0.0);
                        node.number_values.push(0.0);
                        node.text_values.push(0);
                        node.datetime_values.push(val as f32);
                        node.sem_types.push(SemType::DateTime);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                    }
                    // List/array columns (e.g. RelBench rel-amazon product.category)
                    // are flattened to their textual representation and treated as
                    // a single Text value.
                    AnyValue::String(val) => {
                        let l = text_to_idx.len() as i32;
                        let text_idx = *text_to_idx.entry(val.to_string()).or_insert_with(|| l);
                        node.boolean_values.push(0.0);
                        node.number_values.push(0.0);
                        node.text_values.push(text_idx);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Text);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(text_idx);
                    }
                    AnyValue::List(_) | AnyValue::StringOwned(_) => {
                        let s = val.to_string();
                        let l = text_to_idx.len() as i32;
                        let text_idx = *text_to_idx.entry(s).or_insert_with(|| l);
                        node.boolean_values.push(0.0);
                        node.number_values.push(0.0);
                        node.text_values.push(text_idx);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Text);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(text_idx);
                    }
                    _ => {
                        dbg!(&table.table_name);
                        dbg!(col.name());
                        dbg!(val);
                        panic!()
                    }
                }
            }
        }
    }
    pbar.finish();
    println!("done in {:?}.", tic.elapsed());

    let pre_path = format!("{}/{}", cli.out_dir, name);
    fs::create_dir_all(Path::new(&pre_path)).unwrap();

    println!("writing out text...");
    let tic = Instant::now();
    let mut text_vec = vec![String::new(); text_to_idx.len()];
    for (k, v) in text_to_idx {
        text_vec[v as usize] = k;
    }
    let num_text_strings = text_vec.len();
    let file = fs::File::create(format!("{}/text.json", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &text_vec).unwrap();

    // Column name -> index mapping collected during node processing.
    let column_index: HashMap<String, i32> = column_name_to_idx.into_iter().collect();
    let file = fs::File::create(format!("{}/column_index.json", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &column_index).unwrap();
    println!("done in {:?}.", tic.elapsed());

    println!("writing out table info...");
    let tic = Instant::now();
    let mut table_info_map = HashMap::new();
    for (table_key, table) in &table_map {
        let key = format!("{}:{:?}", table_key.0, table_key.1);
        table_info_map.insert(
            key,
            TableInfo {
                node_idx_offset: i32::try_from(table.node_idx_offset)
                    .expect("node_idx_offset overflows i32"),
                num_nodes: i32::try_from(table.df.height()).expect("num_nodes overflows i32"),
            },
        );
    }

    let file = fs::File::create(format!("{}/table_info.json", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &table_info_map).unwrap();
    println!("done in {:?}.", tic.elapsed());

    let tic = Instant::now();
    let mut offsets = vec![0];
    let pbar = ProgressBar::new(node_vec.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template(PBAR_TEMPLATE)
            .unwrap(),
    );

    println!("writing out nodes...");
    let file = fs::File::create(format!("{}/nodes.rkyv", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    for node in node_vec {
        let bytes = rkyv::to_bytes::<Error>(&node).unwrap();
        writer.write_all(&bytes).unwrap();
        offsets.push(writer.stream_position().unwrap() as i64);
        pbar.inc(1);
    }
    pbar.finish();

    println!("writing out offsets...");
    let file = fs::File::create(format!("{}/offsets.rkyv", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    let bytes = rkyv::to_bytes::<Error>(&Offsets { offsets }).unwrap();
    writer.write_all(&bytes).unwrap();

    println!("sorting p2f edges by timestamp...");
    let tic_sort = Instant::now();
    for edges in &mut p2f_adj.adj {
        edges.sort_by_key(|edge| edge.timestamp);
    }
    println!("sorted p2f edges in {:?}", tic_sort.elapsed());

    println!("writing out p2f_adj...");
    let file = fs::File::create(format!("{}/p2f_adj.rkyv", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    let bytes = rkyv::to_bytes::<Error>(&p2f_adj).unwrap();
    writer.write_all(&bytes).unwrap();
    println!("done in {:?}.", tic.elapsed());

    // Self-describing metadata for the preprocessed artifact. The embedding
    // step appends `text_embeddings` (model -> file) to this file.
    let source = cli.source.unwrap_or_else(|| dataset_dir.display().to_string());
    let meta = serde_json::json!({
        "name": name,
        "format_version": PRE_FORMAT_VERSION,
        "source": source,
        "num_db_tables": num_db_tables,
        "num_task_tables": num_task_tables,
        "num_nodes": num_rows_sum,
        "num_text_strings": num_text_strings,
        "tasks": tasks_meta,
        "files": {
            "nodes": "nodes.rkyv",
            "offsets": "offsets.rkyv",
            "p2f_adj": "p2f_adj.rkyv",
            "table_info": "table_info.json",
            "column_index": "column_index.json",
            "text": "text.json",
        },
    });
    let file = fs::File::create(format!("{}/meta.json", pre_path)).unwrap();
    serde_json::to_writer_pretty(BufWriter::new(file), &meta).unwrap();
}
