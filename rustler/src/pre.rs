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
    dataset_name: String,
    #[arg(long, default_value_t = false)]
    skip_db: bool,
}

pub fn main(cli: Cli) {
    let dataset_path = format!(
        "{}/scratch/relbench/{}",
        var("HOME").unwrap(),
        cli.dataset_name
    );

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
        if pq_path.to_str().unwrap().matches("-").count() == 3 {
            continue;
        }

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
                    if cli.dataset_name == "rel-avito" {
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

        // dataset specific hacks
        if table_name == "user-repeat" || table_name == "user-ignore" {
            df = df.drop("index").unwrap();
        }

        if cli.dataset_name == "rel-event" && table_name == "user_friends" {
            df = df.drop_nulls::<String>(None).unwrap();
            df = df.with_row_index("dummy".into(), None).unwrap();
        }
        if cli.dataset_name == "rel-event" && table_name == "event_attendees" {
            df = df.drop_nulls::<String>(None).unwrap();
        }

        if cli.dataset_name == "rel-amazon" && table_name == "product" {
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

    // TODO: cleanup
    println!("computing column stats...");
    let tic = Instant::now();
    let mut dt_cnt: usize = 0;
    let mut dt_sum: f64 = 0.0;
    let mut dt_sum_sq: f64 = 0.0;
    for table in table_map.values_mut() {
        for col in table.df.iter() {
            let col = col.rechunk();
            match col.dtype() {
                DataType::Boolean
                | DataType::UInt32
                | DataType::Int32
                | DataType::Int64
                | DataType::Float64
                | DataType::Float32 => {
                    let col = col.cast(&DataType::Float64).unwrap().drop_nulls();
                    let col = col.filter(&col.is_not_nan().unwrap()).unwrap();
                    // TODO: flag and remove the column entirely
                    let mean = col.mean().unwrap_or(0.0);
                    let std = col.std(1).unwrap_or(1.0);
                    let std = if std == 0.0 { 1.0 } else { std };
                    table.col_stats.push(ColStat { mean, std });
                }
                DataType::Datetime(u, _) => {
                    assert!(*u == TimeUnit::Nanoseconds);
                    // TODO: refactor to avoid duplication with above
                    let col = col.cast(&DataType::Float64).unwrap().drop_nulls();
                    let col = col.filter(&col.is_not_nan().unwrap()).unwrap();
                    // TODO: flag and remove the column entirely
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
    let mut task_text_to_idx = HashMap::new();
    let mut node_vec = (0..num_rows_sum)
        .map(|_| Node::default())
        .collect::<Vec<_>>();
    let mut db_p2f_adj = Adj {
        adj: vec![Vec::new(); num_rows_sum as usize],
    };
    let mut task_p2f_adj = Adj {
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
        let table_name_idx = if table_type == &TableType::Db {
            let l = text_to_idx.len() as i32;
            *text_to_idx
                .entry(table.table_name.clone())
                .or_insert_with(|| l)
        } else {
            let l = task_text_to_idx.len() as i32;
            *task_text_to_idx
                .entry(table.table_name.clone())
                .or_insert_with(|| l)
        };

        for (col, col_stat) in table.df.iter().zip(&table.col_stats) {
            let col = col.rechunk();

            let col_name_idx = if table_type == &TableType::Db {
                let l = text_to_idx.len() as i32;
                *text_to_idx
                    .entry(col.name().to_string())
                    .or_insert_with(|| l)
            } else {
                let l = task_text_to_idx.len() as i32;
                *task_text_to_idx
                    .entry(col.name().to_string())
                    .or_insert_with(|| l)
            };

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
                    if table_type != &TableType::Db {
                        node.is_task_node = true;
                    } else {
                        node.is_task_node = false;
                    }
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
                    if table_type == &TableType::Db {
                        db_p2f_adj.adj[pnode_idx as usize].push(p2f_edge);
                    } else {
                        task_p2f_adj.adj[pnode_idx as usize].push(p2f_edge);
                    }
                }

                continue;
            }

            for (r, val) in col.iter().enumerate() {
                pbar.inc(1);
                let node_idx = table.node_idx_offset + r as i32;
                let node = &mut node_vec[node_idx as usize];
                if table_type != &TableType::Db {
                    node.is_task_node = true;
                } else {
                    node.is_task_node = false;
                }
                node.node_idx = node_idx;
                node.table_name_idx = table_name_idx;
                let val = match val {
                    AnyValue::Boolean(val) => AnyValue::Float64(val as i32 as f64),
                    AnyValue::UInt32(val) => AnyValue::Float64(val as f64),
                    AnyValue::Int32(val) => AnyValue::Float64(val as f64),
                    AnyValue::Int64(val) => AnyValue::Float64(val as f64),
                    AnyValue::Float32(val) => AnyValue::Float64(val as f64),
                    _ => val,
                };
                match val {
                    AnyValue::Null => {}
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
                        node.number_values.push(val as f32);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Number);
                        node.col_name_idxs.push(col_name_idx);
                    }
                    AnyValue::Datetime(val, unit, _) => {
                        assert!(unit == TimeUnit::Nanoseconds);
                        let val = (val as f64 - dt_mean) / dt_std;
                        node.number_values.push(0.0);
                        node.text_values.push(0);
                        node.datetime_values.push(val as f32);
                        node.sem_types.push(SemType::DateTime);
                        node.col_name_idxs.push(col_name_idx);
                    }
                    AnyValue::String(val) => {
                        let text_idx = if table_type == &TableType::Db {
                            let l = text_to_idx.len() as i32;
                            *text_to_idx.entry(val.to_string()).or_insert_with(|| l)
                        } else {
                            let l = task_text_to_idx.len() as i32;
                            *task_text_to_idx.entry(val.to_string()).or_insert_with(|| l)
                        };
                        node.number_values.push(0.0);
                        node.text_values.push(text_idx);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Text);
                        node.col_name_idxs.push(col_name_idx);
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

    let pre_path = format!("{}/scratch/pre/{}", var("HOME").unwrap(), cli.dataset_name);
    fs::create_dir_all(Path::new(&pre_path)).unwrap();

    println!("writing out text...");
    let tic = Instant::now();
    if !cli.skip_db {
        let mut text_vec = vec![String::new(); text_to_idx.len()];
        for (k, v) in text_to_idx {
            text_vec[v as usize] = k;
        }
        let file = fs::File::create(format!("{}/db_text.json", pre_path)).unwrap();
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &text_vec).unwrap();
    }
    let mut task_text_vec = vec![String::new(); task_text_to_idx.len()];
    for (k, v) in task_text_to_idx {
        task_text_vec[v as usize] = k;
    }
    let file = fs::File::create(format!("{}/task_text.json", pre_path)).unwrap();
    let mut task_writer = BufWriter::new(file);
    serde_json::to_writer(&mut task_writer, &task_text_vec).unwrap();
    println!("done in {:?}.", tic.elapsed());

    println!("writing out table info...");
    let tic = Instant::now();
    let mut table_info_map = HashMap::new();
    let mut task_table_info_map = HashMap::new();
    for (table_key, table) in &table_map {
        let key = format!("{}:{:?}", table_key.0, table_key.1);
        if table_key.1 == TableType::Db {
            table_info_map.insert(
                key,
                TableInfo {
                    node_idx_offset: table.node_idx_offset,
                    num_nodes: table.df.height() as i32,
                },
            );
        } else {
            task_table_info_map.insert(
                key,
                TableInfo {
                    node_idx_offset: table.node_idx_offset,
                    num_nodes: table.df.height() as i32,
                },
            );
        }
    }
    if !cli.skip_db {
        let file = fs::File::create(format!("{}/db_table_info.json", pre_path)).unwrap();
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &table_info_map).unwrap();
    }
    let file = fs::File::create(format!("{}/task_table_info.json", pre_path)).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &task_table_info_map).unwrap();
    println!("done in {:?}.", tic.elapsed());

    println!("writing out nodes...");
    let tic = Instant::now();
    let mut offsets = vec![0];
    let mut task_offsets = vec![0];
    let pbar = ProgressBar::new(node_vec.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template(PBAR_TEMPLATE)
            .unwrap(),
    );
    if !cli.skip_db {
        let file = fs::File::create(format!("{}/db_nodes.rkyv", pre_path)).unwrap();
        let mut writer = BufWriter::new(file);
        let file = fs::File::create(format!("{}/task_nodes.rkyv", pre_path)).unwrap();
        let mut task_writer = BufWriter::new(file);
        for node in node_vec {
            let bytes = rkyv::to_bytes::<Error>(&node).unwrap();
            if node.is_task_node {
                task_writer.write_all(&bytes).unwrap();
                task_offsets.push(task_writer.stream_position().unwrap() as i64);
            } else {
                writer.write_all(&bytes).unwrap();
                offsets.push(writer.stream_position().unwrap() as i64);
            }
            pbar.inc(1);
        }
    } else {
        // let file = fs::File::create(format!("{}/db_nodes.rkyv", pre_path)).unwrap();
        // let mut writer = BufWriter::new(file);
        let file = fs::File::create(format!("{}/task_nodes.rkyv", pre_path)).unwrap();
        let mut task_writer = BufWriter::new(file);
        for node in node_vec {
            let bytes = rkyv::to_bytes::<Error>(&node).unwrap();
            if node.is_task_node {
                task_writer.write_all(&bytes).unwrap();
                task_offsets.push(task_writer.stream_position().unwrap() as i64);
            } else {
                // writer.write_all(&bytes).unwrap();
                offsets.push(writer.stream_position().unwrap() as i64);
            }
            pbar.inc(1);
        }
    }
    pbar.finish();
    if !cli.skip_db {
        let file = fs::File::create(format!("{}/db_offsets.rkyv", pre_path)).unwrap();
        let mut writer = BufWriter::new(file);
        let bytes = rkyv::to_bytes::<Error>(&Offsets { offsets }).unwrap();
        writer.write_all(&bytes).unwrap();
    }
    let file = fs::File::create(format!("{}/task_offsets.rkyv", pre_path)).unwrap();
    let mut task_writer = BufWriter::new(file);
    let bytes = rkyv::to_bytes::<Error>(&Offsets {
        offsets: task_offsets,
    })
    .unwrap();
    task_writer.write_all(&bytes).unwrap();
    if !cli.skip_db {
        let file = fs::File::create(format!("{}/db_p2f_adj.rkyv", pre_path)).unwrap();
        let mut writer = BufWriter::new(file);
        let bytes = rkyv::to_bytes::<Error>(&db_p2f_adj).unwrap();
        writer.write_all(&bytes).unwrap();
    }
    let file = fs::File::create(format!("{}/task_p2f_adj.rkyv", pre_path)).unwrap();
    let mut task_writer = BufWriter::new(file);
    let bytes = rkyv::to_bytes::<Error>(&task_p2f_adj).unwrap();
    task_writer.write_all(&bytes).unwrap();
    println!("done in {:?}.", tic.elapsed());
}
