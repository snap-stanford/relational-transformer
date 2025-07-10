use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedTableType, Offsets, SemType,
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

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    table_name_values: Vec<bf16>,
    col_name_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    masks: Vec<bool>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    true_batch_size: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    table_name_values: &'a mut [bf16],
    col_name_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    masks: &'a mut [bool],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
}

impl Vecs {
    fn new(batch_size: usize, seq_len: usize, true_batch_size: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        Self {
            node_idxs: vec![0; l],
            f2p_nbr_idxs: vec![-1; l * 4],
            table_name_idxs: vec![0; l],
            table_name_values: vec![bf16::ZERO; l * d_text],
            col_name_idxs: vec![0; l],
            col_name_values: vec![bf16::ZERO; l * d_text],
            sem_types: vec![0; l],
            number_values: vec![bf16::ZERO; l],
            text_values: vec![bf16::ZERO; l * d_text],
            datetime_values: vec![bf16::ZERO; l],
            masks: vec![false; l],
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            true_batch_size: true_batch_size,
        }
    }

    fn chunks_exact_mut(&mut self, seq_len: usize, d_text: usize) -> impl Iterator<Item = Slices> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * 4),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.table_name_values.chunks_exact_mut(seq_len * d_text),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.masks.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                table_name_values,
                col_name_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                masks,
                is_targets,
                is_task_nodes,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                table_name_values,
                col_name_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                masks,
                is_targets,
                is_task_nodes,
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
            (
                "table_name_values",
                PyArray1::from_vec(py, self.table_name_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs))
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
            ("masks", PyArray1::from_vec(py, self.masks))
                .into_py_any(py)
                .unwrap(),
            ("is_targets", PyArray1::from_vec(py, self.is_targets))
                .into_py_any(py)
                .unwrap(),
            ("is_task_nodes", PyArray1::from_vec(py, self.is_task_nodes))
                .into_py_any(py)
                .unwrap(),
            ("true_batch_size", self.true_batch_size)
                .into_py_any(py)
                .unwrap(),
        ])
    }
}

struct Dataset {
    db_mmap: Mmap,
    task_mmap: Mmap,
    db_text_mmap: Mmap,
    db_p2f_adj_mmap: Mmap,
    task_text_mmap: Mmap,
    db_offsets: Vec<i64>,
    task_offsets: Vec<i64>,
    task_p2f_adj_mmap: Mmap,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
}

#[pyclass]
pub struct Sampler {
    batch_size: usize,
    seq_len: usize,
    mask_prob: f64,
    rank: usize,
    world_size: usize,
    datasets: Vec<Dataset>,
    items: Vec<Item>,
    fake_names: bool,
    subsample_p2f_edges: usize,
    isolate_task_tables: bool,
    cos_steps: usize,
    epoch: u64,
    d_text: usize,
    mask_db_cells: bool,
    mask_task_cells: bool,
    seed: u64,
}

#[pymethods]
impl Sampler {
    #[new]
    fn new(
        dataset_tuples: Vec<(String, i32, i32)>,
        batch_size: usize,
        seq_len: usize,
        mask_prob: f64,
        rank: usize,
        world_size: usize,
        fake_names: bool,
        subsample_p2f_edges: usize,
        isolate_task_tables: bool,
        cos_steps: usize,
        embedding_model: &str,
        d_text: usize,
        mask_db_cells: bool,
        mask_task_cells: bool,
        seed: u64,
    ) -> Self {
        let mut datasets = Vec::new();
        let mut items = Vec::new();
        for (i, (dataset_name, node_idx_offset, num_nodes)) in
            dataset_tuples.into_iter().enumerate()
        {
            let pre_path = format!("{}/scratch/pre/{}", var("HOME").unwrap(), dataset_name);
            let db_nodes_path = format!("{}/db_nodes.rkyv", pre_path);
            let db_file = fs::File::open(&db_nodes_path).unwrap();
            let db_mmap = unsafe { Mmap::map(&db_file).unwrap() };
            let task_nodes_path = format!("{}/task_nodes.rkyv", pre_path);
            let task_file = fs::File::open(&task_nodes_path).unwrap();
            let task_mmap = unsafe { Mmap::map(&task_file).unwrap() };

            let db_text_path = format!("{}/db_text_emb_{}.bin", pre_path, embedding_model);
            let db_text_file = fs::File::open(&db_text_path).unwrap();
            let db_text_mmap = unsafe { Mmap::map(&db_text_file).unwrap() };
            let task_text_path = format!("{}/task_text_emb_{}.bin", pre_path, embedding_model);
            let task_text_file = fs::File::open(&task_text_path).unwrap();
            let task_text_mmap = unsafe { Mmap::map(&task_text_file).unwrap() };

            let db_offsets_path = format!("{}/db_offsets.rkyv", pre_path);
            let db_file = fs::File::open(&db_offsets_path).unwrap();
            let mut db_bytes = Vec::new();
            BufReader::new(db_file).read_to_end(&mut db_bytes).unwrap();
            // TODO: don't deserialize?
            let db_archived = rkyv::access::<ArchivedOffsets, Error>(&db_bytes).unwrap();
            let db_offsets = rkyv::deserialize::<Offsets, Error>(db_archived).unwrap();
            let db_offsets = db_offsets.offsets;
            let task_offsets_path = format!("{}/task_offsets.rkyv", pre_path);
            let task_file = fs::File::open(&task_offsets_path).unwrap();
            let mut task_bytes = Vec::new();
            BufReader::new(task_file)
                .read_to_end(&mut task_bytes)
                .unwrap();
            // TODO: don't deserialize?
            let task_archived = rkyv::access::<ArchivedOffsets, Error>(&task_bytes).unwrap();
            let task_offsets = rkyv::deserialize::<Offsets, Error>(task_archived).unwrap();
            let task_offsets = task_offsets.offsets;

            let db_p2f_adj_path = format!("{}/db_p2f_adj.rkyv", pre_path);
            let db_p2f_adj_file = fs::File::open(&db_p2f_adj_path).unwrap();
            let db_p2f_adj_mmap = unsafe { Mmap::map(&db_p2f_adj_file).unwrap() };
            let task_p2f_adj_path = format!("{}/task_p2f_adj.rkyv", pre_path);
            let task_p2f_adj_file = fs::File::open(&task_p2f_adj_path).unwrap();
            let task_p2f_adj_mmap = unsafe { Mmap::map(&task_p2f_adj_file).unwrap() };

            datasets.push(Dataset {
                db_mmap,
                task_mmap,
                db_text_mmap,
                db_p2f_adj_mmap,
                task_text_mmap,
                db_offsets,
                task_offsets,
                task_p2f_adj_mmap,
            });
            for j in node_idx_offset..node_idx_offset + num_nodes {
                items.push(Item {
                    dataset_idx: i as i32,
                    node_idx: j,
                });
            }
        }

        let epoch = 0;
        Self {
            batch_size,
            seq_len,
            mask_prob,
            rank,
            world_size,
            datasets,
            items,
            fake_names,
            subsample_p2f_edges,
            isolate_task_tables,
            cos_steps,
            epoch,
            d_text,
            mask_db_cells,
            mask_task_cells,
            seed,
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
        let steps = self.epoch * (self.len() as u64) + batch_idx as u64;
        let gain = if steps >= self.cos_steps as u64 {
            0.0
        } else {
            (steps as f64 / self.cos_steps as f64 * std::f64::consts::PI).cos() * 0.5 + 0.5
        };
        let mask_prob = self.mask_prob * gain;
        let mut vecs = Vecs::new(self.batch_size, self.seq_len, true_batch_size, self.d_text);
        vecs.chunks_exact_mut(self.seq_len, self.d_text)
            .enumerate()
            .for_each(|(i, slices)| {
                let j =
                    batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size + i;
                // when self.batch_size > true_batch_size, this will wrap around
                let j = j % self.items.len();
                let item = &self.items[j];
                self.seq(item, slices, mask_prob);
            });
        vecs
    }

    fn seq(&self, item: &Item, slices: Slices, mask_prob: f64) {
        let dataset = &self.datasets[item.dataset_idx as usize];
        let seed_node_idx = item.node_idx;

        let mut visited = vec![false; dataset.db_offsets.len() + dataset.task_offsets.len() - 2];

        let mut f2p_ftr = vec![(0, seed_node_idx)];
        let seed_node = get_node(&dataset, seed_node_idx);
        let mut p2f_ftr = Vec::<Vec<_>>::new();

        let mut seq_i = 0;
        let mut rng = StdRng::seed_from_u64(
            self.epoch.wrapping_add(seed_node_idx as u64).wrapping_add(self.seed),
        );
        loop {
            // select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                let mut depth_choices = Vec::new();
                for i in 0..p2f_ftr.len() {
                    if !p2f_ftr[i].is_empty() {
                        depth_choices.push(i);
                    }
                }
                if depth_choices.is_empty() {
                    // TODO: track stats
                    // println!("Graph exhausted after {} cells", seq_i);
                    // panic!();
                    (
                        0,
                        rng.random_range(
                            0..dataset.db_offsets.len() + dataset.task_offsets.len() - 2,
                        ) as i32,
                    )
                } else {
                    let depth = depth_choices[0];
                    let r = rng.random_range(0..p2f_ftr[depth].len());
                    let l = p2f_ftr[depth].len();
                    let tmp = p2f_ftr[depth][r];
                    p2f_ftr[depth][r] = p2f_ftr[depth][l - 1];
                    p2f_ftr[depth][l - 1] = tmp;
                    let node_idx = p2f_ftr[depth].pop().unwrap();
                    (depth, node_idx)
                }
            };

            if visited[node_idx as usize] {
                continue;
            }
            visited[node_idx as usize] = true;

            let node = get_node(&dataset, node_idx);

            for edge in node.f2p_edges.iter() {
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            let task_p2f_edges = get_task_p2f_edges(&dataset, node_idx);
            for edge in task_p2f_edges.iter() {
                if edge.timestamp.is_some()
                    && seed_node.timestamp.is_some()
                    && edge.timestamp >= seed_node.timestamp
                {
                    continue;
                }
                if self.isolate_task_tables
                    && edge.table_type != ArchivedTableType::Db
                    && edge.table_name_idx != seed_node.table_name_idx
                {
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            let db_p2f_edges = get_db_p2f_edges(&dataset, node_idx);
            let edge_idxs = if db_p2f_edges.len() > self.subsample_p2f_edges {
                index::sample(&mut rng, db_p2f_edges.len(), self.subsample_p2f_edges).into_vec()
            } else {
                (0..db_p2f_edges.len()).collect::<Vec<_>>()
            };
            for edge_idx in edge_idxs {
                let edge = &db_p2f_edges[edge_idx];
                if edge.timestamp.is_some()
                    && seed_node.timestamp.is_some()
                    && edge.timestamp >= seed_node.timestamp
                {
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            let num_cells = node.col_name_idxs.len();
            for cell_i in 0..num_cells {
                slices.node_idxs[seq_i] = node.node_idx.into();
                for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
                    slices.f2p_nbr_idxs[seq_i * 4 + j] = f2p_nbr_idx.into();
                }

                slices.table_name_idxs[seq_i] = node.table_name_idx.into();
                slices.col_name_idxs[seq_i] = node.col_name_idxs[cell_i].into();
                slices.table_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(
                        &dataset,
                        slices.table_name_idxs[seq_i],
                        node.is_task_node,
                        self.fake_names,
                        self.d_text,
                    ));
                slices.col_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(
                        &dataset,
                        slices.col_name_idxs[seq_i],
                        node.is_task_node,
                        self.fake_names,
                        self.d_text,
                    ));

                let s = node.sem_types[cell_i].clone() as i32;
                slices.sem_types[seq_i] = s;

                slices.number_values[seq_i] = bf16::from_f32(node.number_values[cell_i].into());

                let text_idx: i32 = node.text_values[cell_i].into();
                slices.text_values[seq_i * self.d_text..(seq_i + 1) * self.d_text].copy_from_slice(
                    get_text_emb(&dataset, text_idx, node.is_task_node, false, self.d_text),
                );

                slices.datetime_values[seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());

                let mask = ((s == SemType::Number as i32 && seed_node_idx == node.node_idx)
                    || (s == SemType::Number as i32 && rng.random_bool(mask_prob)))
                    && (self.mask_db_cells || node.is_task_node)
                    && (self.mask_task_cells || !node.is_task_node);

                slices.masks[seq_i] = mask;
                let is_target = s == SemType::Number as i32 && seed_node_idx == node.node_idx;
                slices.is_targets[seq_i] = is_target;
                slices.is_task_nodes[seq_i] = node.is_task_node;
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

fn get_node<'a>(dataset: &'a Dataset, idx: i32) -> &'a ArchivedNode {
    if idx < dataset.db_offsets.len() as i32 - 1 {
        let l = dataset.db_offsets[idx as usize] as usize;
        let r = dataset.db_offsets[(idx + 1) as usize] as usize;
        let bytes = &dataset.db_mmap[l..r];
        // rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
        unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
    } else {
        let idx = idx - (dataset.db_offsets.len() as i32 - 1);
        let l = dataset.task_offsets[idx as usize] as usize;
        let r = dataset.task_offsets[(idx + 1) as usize] as usize;
        let bytes = &dataset.task_mmap[l..r];
        // rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
        unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
    }
}

fn get_db_p2f_edges<'a>(dataset: &'a Dataset, idx: i32) -> &'a ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.db_p2f_adj_mmap[..];
    let db_p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &db_p2f_adj.adj[idx as usize]
}

fn get_task_p2f_edges<'a>(dataset: &'a Dataset, idx: i32) -> &'a ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.task_p2f_adj_mmap[..];
    let task_p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &task_p2f_adj.adj[idx as usize]
}

fn get_text_emb<'a>(
    dataset: &'a Dataset,
    idx: i32,
    is_task_node: bool,
    fake_names: bool,
    d_text: usize,
) -> &'a [bf16] {
    if !is_task_node {
        let (pref, db_text_emb, suf) = unsafe { dataset.db_text_mmap.align_to::<bf16>() };
        assert!(pref.is_empty() && suf.is_empty());
        let idx = if fake_names {
            (idx + 3) % (db_text_emb.len() as i32 / d_text as i32)
        } else {
            idx
        };
        &db_text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
    } else {
        let (pref, task_text_emb, suf) = unsafe { dataset.task_text_mmap.align_to::<bf16>() };
        assert!(pref.is_empty() && suf.is_empty());
        let idx = if fake_names {
            (idx + 3) % (task_text_emb.len() as i32 / d_text as i32)
        } else {
            idx
        };
        &task_text_emb[idx as usize * d_text..(idx as usize + 1) * d_text]
    }
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    dataset_name: String,
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
        vec![(cli.dataset_name, 0, 10_000)],
        cli.batch_size,
        cli.seq_len,
        0.0,
        0,
        1,
        false,
        256,
        false,
        1000,
        "stsb-distilroberta-base-v2",
        768,
        true,
        true,
        0,
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
