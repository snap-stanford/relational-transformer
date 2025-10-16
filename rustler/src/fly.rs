use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedTableType, Offsets,
};
use clap::Parser;
use half::bf16;
use itertools::izip;
use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::seq::index;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use std::env::var;
use std::fs;
use std::io::{BufReader, Read};
use std::str;
use std::time::Instant;

const MAX_F2P_NBRS: usize = 5;

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    col_name_idxs: Vec<i32>,
    class_value_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    boolean_values: Vec<bf16>,
    masks: Vec<bool>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    is_padding: Vec<bool>,
    true_batch_size: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    col_name_idxs: &'a mut [i32],
    class_value_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    boolean_values: &'a mut [bf16],
    masks: &'a mut [bool],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
    is_padding: &'a mut [bool],
}

impl Vecs {
    fn new(batch_size: usize, seq_len: usize, true_batch_size: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            class_value_idxs: vec![-1; l],
            col_name_values: vec![bf16::ZERO; l * d_text],
            sem_types: vec![0; l],
            number_values: vec![bf16::ZERO; l],
            text_values: vec![bf16::ZERO; l * d_text],
            datetime_values: vec![bf16::ZERO; l],
            boolean_values: vec![bf16::ZERO; l],
            masks: vec![false; l],
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            is_padding: vec![true; l],
            true_batch_size,
        }
    }

    fn chunks_exact_mut(&mut self, seq_len: usize, d_text: usize) -> impl Iterator<Item = Slices> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.class_value_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.boolean_values.chunks_exact_mut(seq_len),
            self.masks.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
            },
        )
    }
    fn into_pyobject<'a>(self, py: Python<'a>) -> PyResult<Vec<PyObject>> {
        Ok(vec![
            ("node_idxs", PyArray1::from_vec(py, self.node_idxs))
                .into_py_any(py)
                .unwrap(),
            ("f2p_nbr_idxs", PyArray1::from_vec(py, self.f2p_nbr_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "table_name_idxs",
                PyArray1::from_vec(py, self.table_name_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "class_value_idxs",
                PyArray1::from_vec(py, self.class_value_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "col_name_values",
                PyArray1::from_vec(py, self.col_name_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("sem_types", PyArray1::from_vec(py, self.sem_types))
                .into_py_any(py)
                .unwrap(),
            ("number_values", PyArray1::from_vec(py, self.number_values))
                .into_py_any(py)
                .unwrap(),
            ("text_values", PyArray1::from_vec(py, self.text_values))
                .into_py_any(py)
                .unwrap(),
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "boolean_values",
                PyArray1::from_vec(py, self.boolean_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("masks", PyArray1::from_vec(py, self.masks))
                .into_py_any(py)
                .unwrap(),
            ("is_targets", PyArray1::from_vec(py, self.is_targets))
                .into_py_any(py)
                .unwrap(),
            ("is_task_nodes", PyArray1::from_vec(py, self.is_task_nodes))
                .into_py_any(py)
                .unwrap(),
            ("is_padding", PyArray1::from_vec(py, self.is_padding))
                .into_py_any(py)
                .unwrap(),
            ("true_batch_size", self.true_batch_size)
                .into_py_any(py)
                .unwrap(),
        ])
    }
}

struct Dataset {
    mmap: Mmap,
    text_mmap: Mmap,
    p2f_adj_mmap: Mmap,
    offsets: Vec<i64>,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
}

#[pyclass]
pub struct Sampler {
    batch_size: usize,
    seq_len: usize,
    rank: usize,
    world_size: usize,
    datasets: Vec<Dataset>,
    items: Vec<Item>,
    max_bfs_width: usize,
    epoch: u64,
    d_text: usize,
    seed: u64,
    target_columns: Vec<i32>,
    columns_to_drop: Vec<Vec<i32>>,
}

#[pymethods]
impl Sampler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_tuples: Vec<(String, i32, i32)>,
        batch_size: usize,
        seq_len: usize,
        rank: usize,
        world_size: usize,
        max_bfs_width: usize,
        embedding_model: &str,
        d_text: usize,
        seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
    ) -> Self {
        let mut datasets = Vec::new();
        let mut items = Vec::new();
        for (i, (db_name, node_idx_offset, num_nodes)) in dataset_tuples.into_iter().enumerate() {
            let pre_path = format!("{}/scratch/pre/{}", var("HOME").unwrap(), db_name);
            let nodes_path = format!("{}/nodes.rkyv", pre_path);
            let file = fs::File::open(&nodes_path).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };

            let text_path = format!("{}/text_emb_{}.bin", pre_path, embedding_model);
            let text_file = fs::File::open(&text_path).unwrap();
            let text_mmap = unsafe { Mmap::map(&text_file).unwrap() };

            let offsets_path = format!("{}/offsets.rkyv", pre_path);
            let file = fs::File::open(&offsets_path).unwrap();
            let mut bytes = Vec::new();
            BufReader::new(file).read_to_end(&mut bytes).unwrap();
            let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
            let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
            let offsets = offsets.offsets;

            let p2f_adj_path = format!("{}/p2f_adj.rkyv", pre_path);
            let p2f_adj_file = fs::File::open(&p2f_adj_path).unwrap();
            let p2f_adj_mmap = unsafe { Mmap::map(&p2f_adj_file).unwrap() };
            let target = target_columns[i];

            datasets.push(Dataset {
                mmap,
                text_mmap,
                p2f_adj_mmap,
                offsets,
            });

            for j in node_idx_offset..node_idx_offset + num_nodes {
                let node = get_node(&datasets[i], j);
                // skip the node if target column was removed during preprocessing
                if node.col_name_idxs.iter().any(|&c| c == target) {
                    items.push(Item {
                        dataset_idx: i as i32,
                        node_idx: j,
                    });
                }
            }
        }

        let epoch = 0;
        Self {
            batch_size,
            seq_len,
            rank,
            world_size,
            datasets,
            items,
            max_bfs_width,
            epoch,
            d_text,
            seed,
            target_columns,
            columns_to_drop,
        }
    }

    fn len_py(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn batch_py<'a>(&self, py: Python<'a>, batch_idx: usize) -> PyResult<Vec<PyObject>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.seed));
        self.items.shuffle(&mut rng);
    }
}

impl Sampler {
    fn len(&self) -> usize {
        self.items.len().div_ceil(self.batch_size * self.world_size)
    }

    fn batch(&self, batch_idx: usize) -> Vecs {
        let true_batch_size = self.batch_size.min(
            self.items.len()
                - self.rank * self.batch_size
                - batch_idx * self.batch_size * self.world_size,
        );

        let mut vecs = Vecs::new(self.batch_size, self.seq_len, true_batch_size, self.d_text);
        vecs.chunks_exact_mut(self.seq_len, self.d_text)
            .enumerate()
            .for_each(|(i, slices)| {
                let j =
                    batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size + i;
                // when self.batch_size > true_batch_size, this will wrap around
                let j = j % self.items.len();
                let item = &self.items[j];
                self.seq(item, slices);
            });
        vecs
    }

    fn seq(&self, item: &Item, slices: Slices) {
        let dataset = &self.datasets[item.dataset_idx as usize];
        let target_column = self.target_columns[item.dataset_idx as usize];
        //define let columns to drop which is a vector of i32 and is at the same index as target_columns
        let columns_to_drop = &self.columns_to_drop[item.dataset_idx as usize];
        let seed_node_idx = item.node_idx;

        let mut visited = vec![false; dataset.offsets.len() - 1];

        let mut f2p_ftr = vec![(0, seed_node_idx)];
        let seed_node = get_node(dataset, seed_node_idx);
        let mut p2f_ftr = Vec::<Vec<_>>::new();

        let mut seq_i = 0;
        let mut rng = StdRng::seed_from_u64(
            self.epoch
                .wrapping_add(seed_node_idx as u64)
                .wrapping_add(self.seed),
        );
        loop {
            // select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                let mut depth_choices = Vec::new();
                for (i, node) in p2f_ftr.iter().enumerate() {
                    if !node.is_empty() {
                        depth_choices.push(i);
                    }
                }
                if depth_choices.is_empty() {
                    return;
                } else {
                    let depth = depth_choices[0];
                    let r = rng.random_range(0..p2f_ftr[depth].len());
                    let l = p2f_ftr[depth].len();
                    p2f_ftr[depth].swap(r, l - 1);
                    let node_idx = p2f_ftr[depth].pop().unwrap();
                    (depth, node_idx)
                }
            };

            if visited[node_idx as usize] {
                continue;
            }
            visited[node_idx as usize] = true;

            let node = get_node(dataset, node_idx);

            for edge in node.f2p_edges.iter() {
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            let p2f_edges = get_p2f_edges(dataset, node_idx);

            // temporary storage for db edges to be subsampled
            let mut db_p2f_ftr: Vec<i32> = Vec::new();

            for edge in p2f_edges.iter() {
                // include edges to task table only if seed node belongs to the task table
                if edge.table_name_idx != seed_node.table_name_idx && edge.table_type != ArchivedTableType::Db {
                    continue;
                }

                // temporal constraint
                if edge.timestamp.is_some() && seed_node.timestamp.is_some() && edge.timestamp > seed_node.timestamp {
                    continue;
                }

                if edge.table_type == ArchivedTableType::Db {
                    db_p2f_ftr.push(edge.node_idx.into());
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            let idxs = if db_p2f_ftr.len() > self.max_bfs_width {
                index::sample(&mut rng, db_p2f_ftr.len(), self.max_bfs_width).into_vec()
            } else {
                (0..db_p2f_ftr.len()).collect::<Vec<_>>()
            };

            for idx in idxs.iter() {
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(db_p2f_ftr[*idx]);
            }

            let num_cells = node.col_name_idxs.len();
            for cell_i in 0..num_cells {
                let col_idx: i32 = node.col_name_idxs[cell_i].into();
                if (node.node_idx == seed_node_idx && columns_to_drop.contains(&col_idx))
                    || (node.timestamp == seed_node.timestamp && columns_to_drop.contains(&col_idx))
                {
                    continue; // do not add this cell to the sequence
                }
                slices.node_idxs[seq_i] = node.node_idx.into();

                assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
                for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
                    slices.f2p_nbr_idxs[seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
                }

                slices.table_name_idxs[seq_i] = node.table_name_idx.into();
                slices.col_name_idxs[seq_i] = node.col_name_idxs[cell_i].into();
                slices.class_value_idxs[seq_i] = node.class_value_idx[cell_i].into();
                slices.col_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(
                        dataset,
                        slices.col_name_idxs[seq_i],
                        self.d_text,
                    ));

                let s = node.sem_types[cell_i].clone() as i32;
                slices.sem_types[seq_i] = s;

                slices.number_values[seq_i] = bf16::from_f32(node.number_values[cell_i].into());

                let text_idx: i32 = node.text_values[cell_i].into();
                slices.text_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));

                slices.datetime_values[seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());

                slices.boolean_values[seq_i] = bf16::from_f32(node.boolean_values[cell_i].into());

                let is_target = seed_node_idx == node.node_idx
                    && node.col_name_idxs[cell_i] == target_column;
                slices.is_targets[seq_i] = is_target;

                slices.masks[seq_i] = is_target;

                slices.is_task_nodes[seq_i] =
                    node.is_task_node || (node.col_name_idxs[cell_i] == target_column);
                slices.is_padding[seq_i] = false;

                seq_i += 1;
                if seq_i >= self.seq_len {
                    break;
                }
            }
            if seq_i >= self.seq_len {
                break;
            }
        }
    }
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];
    // rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
    unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
}

fn get_p2f_edges(dataset: &Dataset, idx: i32) -> &ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.p2f_adj_mmap[..];
    let p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &p2f_adj.adj[idx as usize]
}

fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {
    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };
    assert!(pref.is_empty() && suf.is_empty());
    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    db_name: String,
    #[arg(default_value = "128")]
    batch_size: usize,
    #[arg(default_value = "1024")]
    seq_len: usize,
    #[arg(default_value = "1000")]
    num_trials: usize,
}

pub fn main(cli: Cli) {
    let tic = Instant::now();
    let sampler = Sampler::new(
        vec![(cli.db_name, 0, 10)], // dataset_tuples
        cli.batch_size,             // batch_size
        cli.seq_len,                // seq_len
        0,                          // rank
        1,                          // world_size
        256,                        // max_bfs_width
        "all-MiniLM-L12-v2",        // embedding_model
        384,                        // d_text
        0,                          // seed
        vec![-1; 1],                // target_columns
        vec![Vec::<i32>::new()],    // columns_to_drop
    );
    println!("Sampler loaded in {:?}", tic.elapsed());

    let mut sum = 0;
    let mut sum_sq = 0;
    let mut rng = rand::rng();
    for _ in 0..cli.num_trials {
        let tic = Instant::now();
        let batch_idx = rng.random_range(0..sampler.len());
        let _batch = sampler.batch(batch_idx);
        let elapsed = tic.elapsed().as_millis();
        sum += elapsed;
        sum_sq += elapsed * elapsed;
    }
    let mean = sum as f64 / cli.num_trials as f64;
    let std = (sum_sq as f64 / cli.num_trials as f64 - mean * mean).sqrt();
    println!("Mean: {} ms,\tStd: {} ms", mean, std);
}
