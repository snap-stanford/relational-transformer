use crate::common::{Adj, Edge, Node, Offsets, SemType, TableInfo, TableType};
use clap::Parser;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::{self, Itertools};
use parquet::file::metadata::ParquetMetaDataReader;
use polars::prelude::*;
use rkyv::rancor::Error;
use serde_json::{self, Value, from_str};
use std::collections::HashMap;
use std::env::var;
use std::fs;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::io::BufWriter;
use std::io::{Seek, Write};
use std::iter;
use std::path::Path;
use std::time::Instant;

//Function to automatically binarize a column based on its first non-null value
pub fn make_column_boolean(df: &mut DataFrame, col_name: &str) -> PolarsResult<()> {
    let s_str = df.column(col_name)?.cast(&DataType::String)?;
    let ca = s_str.str()?;
    let first = ca
        .get(0)
        .ok_or_else(|| PolarsError::NoData("column is empty".into()))?;

    let mut mask_series = ca.equal(first).into_series();
    mask_series.rename(col_name.into());
    df.replace(col_name, mask_series)?;
    Ok(())
}

const PBAR_TEMPLATE: &str = "{percent}% {bar} {decimal_bytes}/{decimal_total_bytes} [{elapsed_precise}<{eta_precise}, {decimal_bytes_per_sec}]";

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
    node_idx_offset: i32,
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    db_name: String,
    #[arg(long, default_value_t = false)]
    skip_db: bool,
}

fn cast_col_to_bool(df: DataFrame, col_name: &str) -> Result<DataFrame, PolarsError> {
    df.lazy()
        .with_column(col(col_name).cast(DataType::Boolean).alias(col_name))
        .collect()
}

pub fn main(cli: Cli) {
    let dataset_path = format!("{}/scratch/relbench/{}", var("HOME").unwrap(), cli.db_name);

    let dashes = dataset_path.matches("-").count();
    dbg!(dashes);

    println!("reading tables...");
    let tic = Instant::now();
    let mut table_map = HashMap::with_hasher(BuildHasherDefault::<DefaultHasher>::new());
    let mut num_rows_sum = 0;
    let mut num_cells_sum = 0;

    for (is_db_table, pq_path) in itertools::chain(
        iter::repeat(true).zip(
            glob(&format!("{}/db/*.parquet", dataset_path))
                .unwrap()
                .map(|p| p.unwrap())
                .sorted(),
        ),
        iter::repeat(false).zip(
            glob(&format!("{}/tasks/*/*.parquet", dataset_path))
                .unwrap()
                .map(|p| p.unwrap())
                .sorted(),
        ),
    ) {
        if pq_path.to_str().unwrap().matches("-").count() == dashes + 2 {
            continue;
        }
        dbg!(&pq_path);

        let mut file = fs::File::open(&pq_path).unwrap();
        let reader = ParquetReader::new(&mut file);
        let mut df = reader.finish().unwrap();

        let mut reader = ParquetMetaDataReader::new();
        let file = fs::File::open(&pq_path).unwrap();
        reader.try_parse(&file).unwrap();
        let metadata = reader.finish().unwrap();

        let metadata = metadata
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .iter()
            .map(|kv| {
                let key = kv.key.clone();
                let value = kv.value.as_ref().unwrap().clone();
                (key, value)
            })
            .collect::<HashMap<_, _>>();
        let tmp = metadata.get("pkey_col").unwrap();
        let pcol_name = match from_str(tmp).unwrap() {
            Value::Null => None,
            Value::String(s) => Some(s),
            _ => panic!(),
        };
        let tmp = metadata.get("fkey_col_to_pkey_table").unwrap();
        let fcol_name_to_ptable_name = match from_str(tmp).unwrap() {
            Value::Object(o) => o
                .into_iter()
                .map(|(k, v)| {
                    let mut v = match v {
                        Value::String(s) => s,
                        _ => panic!(),
                    };
                    if cli.db_name == "rel-avito" {
                        if k == "UserID" {
                            v = "UserInfo".to_string();
                        } else if k == "AdID" {
                            v = "AdsInfo".to_string();
                        }
                    }
                    (k, v)
                })
                .collect::<HashMap<_, _>>(),
            _ => panic!(),
        };
        let tmp = metadata.get("time_col").unwrap();
        let tcol_name = match from_str(tmp).unwrap() {
            Value::Null => None,
            Value::String(s) => Some(s),
            _ => panic!(),
        };

        let (table_name, table_type) = if is_db_table {
            (
                pq_path.file_stem().unwrap().to_str().unwrap().to_string(),
                TableType::Db,
            )
        } else {
            (
                pq_path
                    .parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
                match pq_path.file_stem().unwrap().to_str().unwrap() {
                    "train" => TableType::Train,
                    "val" => TableType::Val,
                    "test" => TableType::Test,
                    _ => panic!(),
                },
            )
        };
        let table_key = (table_name.clone(), table_type.clone());

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Binarize columns based on first non-null value
        if cli.db_name == "rel-stack" && table_name == "postLinks" {
            make_column_boolean(&mut df, "LinkTypeId").unwrap();
        }

        // if cli.db_name == "rel-stack" && table_name == "badges" {
        //     make_column_boolean(&mut df, "TagBased").unwrap();

        // }
        if cli.db_name == "rel-trial" && table_name == "studies" {
            make_column_boolean(&mut df, "has_dmc").unwrap();
        }
        // // if cli.db_name == "rel-trial" && table_name == "designs" {
        // //     make_column_boolean(&mut df, "allocation").unwrap();
        // // }
        // if cli.db_name == "rel-trial" && table_name == "studies" {
        //     make_column_boolean(&mut df, "is_fda_regulated_device").unwrap();
        // }

        if cli.db_name == "rel-trial" && table_name == "eligibilities" {
            for col in ["adult", "child", "older_adult"] {
                make_column_boolean(&mut df, col).unwrap();
            }
        }

        // if cli.db_name == "rel-event" && table_name == "users" {
        //     make_column_boolean(&mut df, "gender").unwrap();
        // }
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // amazon
        if cli.db_name == "rel-amazon" {
            if table_name == "user-churn" {
                df = cast_col_to_bool(df, "churn").unwrap();
            }
            if table_name == "item-churn" {
                df = cast_col_to_bool(df, "churn").unwrap();
            }
            if table_name == "product" {
                df.apply("category", |c| {
                    c.list()
                        .unwrap()
                        .into_iter()
                        // XXX: takes only the first element
                        .map(|l| l.map(|l| l.iter().next().unwrap().to_string()))
                        .collect::<StringChunked>()
                        .into_column()
                })
                .unwrap();
            }
        }

        // stack
        if cli.db_name == "rel-stack" {
            if table_name == "user-engagement" {
                df = cast_col_to_bool(df, "contribution").unwrap();
            }
            if table_name == "user-badge" {
                df = cast_col_to_bool(df, "WillGetBadge").unwrap();
            }
            if table_name == "posts" {
                df = df.drop("AcceptedAnswerId").unwrap();
            }
        }

        // trial
        if cli.db_name == "rel-trial" && table_name == "study-outcome" {
            df = cast_col_to_bool(df, "outcome").unwrap();
        }

        // f1
        if cli.db_name == "rel-f1" {
            if table_name == "driver-dnf" {
                df = cast_col_to_bool(df, "did_not_finish").unwrap();
            }
            if table_name == "driver-top3" {
                df = cast_col_to_bool(df, "qualifying").unwrap();
            }
        }

        // hm
        if cli.db_name == "rel-hm" && table_name == "user-churn" {
            df = cast_col_to_bool(df, "churn").unwrap();
        }

        // event
        if cli.db_name == "rel-event" {
            if table_name == "event_attendees" {
                df = df.drop_nulls::<String>(None).unwrap();
            }
            if table_name == "user_friends" {
                df = df.drop_nulls::<String>(None).unwrap();
                df = df.with_row_index("dummy".into(), None).unwrap();
            }
            if table_name == "user-repeat"
                || table_name == "user-ignore"
                || table_name == "user-attendance"
            {
                df = df.drop("index").unwrap();
                df = cast_col_to_bool(df, "target").unwrap();
            }
        }

        // avito
        if cli.db_name == "rel-avito" {
            if table_name == "user-visits" {
                df = cast_col_to_bool(df, "num_click").unwrap();
            }
            if table_name == "user-clicks" {
                df = cast_col_to_bool(df, "num_click").unwrap();
            }
        }

        let num_rows = df.height() as i32;
        let num_cells = num_rows * df.width() as i32;
        table_map.insert(
            table_key,
            Table {
                table_name: table_name.to_string(),
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
    println!("done in {:?}.", tic.elapsed());

    println!("computing column stats...");
    let tic = Instant::now();
    let mut dt_cnt: usize = 0;
    let mut dt_sum: f64 = 0.0;
    let mut dt_sum_sq: f64 = 0.0;

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
                DataType::UInt32
                | DataType::Int32
                | DataType::Int64
                | DataType::Float64
                | DataType::Float32 => {
                    let col = col.cast(&DataType::Float64).unwrap().drop_nulls();
                    let col = col.filter(&col.is_not_nan().unwrap()).unwrap();
                    let mean = col.mean().unwrap_or(0.0);
                    let std = col.std(1).unwrap_or(1.0);
                    let std = if std == 0.0 { 1.0 } else { std };
                    table.col_stats.push(ColStat { mean, std });
                }
                DataType::Datetime(u, _) => {
                    assert!(*u == TimeUnit::Nanoseconds);
                    let col = col.cast(&DataType::Float64).unwrap().drop_nulls();
                    let col = col.filter(&col.is_not_nan().unwrap()).unwrap();
                    dt_cnt += col.len();
                    dt_sum += col.sum::<f64>().unwrap();
                    dt_sum_sq += col
                        .iter()
                        .map(|x| {
                            if let AnyValue::Float64(f) = x {
                                f * f
                            } else {
                                panic!()
                            }
                        })
                        .sum::<f64>();
                    // let mean = col.mean().unwrap_or(0.0);
                    // let std = col.std(1).unwrap_or(1.0);
                    // let std = if std == 0.0 { 1.0 } else { std };
                    // table.col_stats.push(ColStat { mean, std });
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

    let dt_mean = dt_sum / dt_cnt as f64;
    let dt_std = (dt_sum_sq / dt_cnt as f64 - dt_mean * dt_mean).sqrt();
    dbg!(dt_cnt);
    dbg!(dt_sum);
    dbg!(dt_sum_sq);
    dbg!(dt_mean);
    dbg!(dt_std);

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
                    if val == AnyValue::Null {
                        continue;
                    }

                    let node_idx = table.node_idx_offset + r as i32;
                    let node = node_vec.get_mut(node_idx as usize).unwrap();
                    node.is_task_node = table_type != &TableType::Db;
                    node.node_idx = node_idx;
                    node.table_name_idx = table_name_idx;

                    let AnyValue::Int64(val) = val else {
                        dbg!(val);
                        dbg!(col.name());
                        panic!()
                    };
                    let pnode_idx = ptable_offset + val as i32;

                    node.f2p_nbr_idxs.push(pnode_idx);

                    let timestamp = if let Some(c) = &table.tcol_name {
                        table
                            .df
                            .column(c)
                            .unwrap()
                            .datetime()
                            .unwrap()
                            .get(r)
                            .map(|v| (v / 1_000_000_000) as i32)
                    } else {
                        None
                    };
                    node.timestamp = timestamp;

                    let l = text_to_idx.len() as i32;
                    let ptable_name_idx = *text_to_idx
                        .entry(ptable_name.to_string())
                        .or_insert_with(|| l);

                    let ptable = &table_map[&(ptable_name.to_string(), TableType::Db)];
                    let ptimestamp = if ptable.tcol_name.is_some() {
                        let tcol_name = ptable.tcol_name.as_ref().unwrap();
                        ptable
                            .df
                            .column(tcol_name)
                            .unwrap()
                            .datetime()
                            .unwrap()
                            .get(val as usize)
                            .map(|v| (v / 1_000_000_000) as i32)
                    } else {
                        None
                    };

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
                    p2f_adj.adj[pnode_idx as usize].push(p2f_edge)
                }

                continue;
            }

            for (r, val) in col.iter().enumerate() {
                pbar.inc(1);
                let node_idx = table.node_idx_offset + r as i32;
                let node = &mut node_vec[node_idx as usize];
                node.is_task_node = table_type != &TableType::Db;
                node.node_idx = node_idx;
                node.table_name_idx = table_name_idx;

                let val = match val {
                    AnyValue::Boolean(val) => AnyValue::Boolean(val),
                    AnyValue::UInt32(val) => AnyValue::Float64(val as f64),
                    AnyValue::Int32(val) => AnyValue::Float64(val as f64),
                    AnyValue::Int64(val) => AnyValue::Float64(val as f64),
                    AnyValue::Float32(val) => AnyValue::Float64(val as f64),
                    _ => val,
                };
                match val {
                    AnyValue::Null => {}
                    AnyValue::Boolean(val) => {
                        let val_float = if val { 1.0 } else { 0.0 };
                        let val_float = (val_float - col_stat.mean) / col_stat.std;
                        // println!("Column {}: boolean val: {} mean: {} std: {}", col.name(), val_float, col_stat.mean, col_stat.std);
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

    let pre_path = format!("{}/scratch/pre/{}", var("HOME").unwrap(), cli.db_name);
    fs::create_dir_all(Path::new(&pre_path)).unwrap();

    println!("writing out text...");
    let tic = Instant::now();
    let mut text_vec = vec![String::new(); text_to_idx.len()];
    for (k, v) in text_to_idx {
        text_vec[v as usize] = k;
    }
    let file = fs::File::create(format!("{}/text.json", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &text_vec).unwrap();

    // Also produce a reverse mapping for text strings
    let mut text_map: HashMap<String, i32> = HashMap::new();
    for (idx, s) in text_vec.iter().enumerate() {
        text_map.insert(s.clone(), idx as i32);
    }
    let file = fs::File::create(format!("{}/text_map.json", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &text_map).unwrap();

    // Write column name to index mapping collected during node processing
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
                node_idx_offset: table.node_idx_offset,
                num_nodes: table.df.height() as i32,
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

    println!("writing out p2f_adj...");
    let file = fs::File::create(format!("{}/p2f_adj.rkyv", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    let bytes = rkyv::to_bytes::<Error>(&p2f_adj).unwrap();
    writer.write_all(&bytes).unwrap();

    println!("done in {:?}.", tic.elapsed());
}
