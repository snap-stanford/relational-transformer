use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

// node represents a row in a table
// TableInfo struct has the offset of the first node in the table and the number of nodes in the table
#[derive(SerdeSerialize, SerdeDeserialize)]
pub struct TableInfo {
    pub node_idx_offset: i32,
    pub num_nodes: i32,
}

#[derive(Archive, Deserialize, Serialize, Hash, Eq, PartialEq, Default, Clone, Debug)]
#[rkyv(derive(PartialEq, Debug))]
pub enum TableType {
    #[default]
    Db,
    Train,
    Val,
    Test,
}

#[derive(Archive, Clone, Debug, Deserialize, Serialize, Default)]
#[rkyv(derive(Debug))]
pub struct Edge {
    pub node_idx: i32,
    pub table_name_idx: i32,
    pub table_type: TableType,
    pub timestamp: Option<i32>,
}

#[derive(Archive, Serialize, Deserialize, Default)]
pub struct Adj {
    pub adj: Vec<Vec<Edge>>,
}

#[derive(Archive, Deserialize, Serialize, Default)]
#[rkyv(derive(Clone, Debug))]
pub enum SemType {
    #[default]
    Number,
    Text,
    DateTime,
    Boolean,
}

#[derive(Archive, Deserialize, Serialize, Default)]
// #[rkyv(derive(Debug))]
pub struct Node {
    pub is_task_node: bool,
    pub node_idx: i32,
    pub f2p_nbr_idxs: Vec<i32>,
    pub f2p_edges: Vec<Edge>,
    pub timestamp: Option<i32>,
    pub table_name_idx: i32,
    pub col_name_idxs: Vec<i32>,
    pub sem_types: Vec<SemType>,
    pub number_values: Vec<f32>,
    pub text_values: Vec<i32>,
    pub datetime_values: Vec<f32>,
    pub boolean_values: Vec<f32>,
    pub class_value_idx: Vec<i32>,
}

#[derive(Archive, Deserialize, Serialize, Default)]
pub struct Offsets {
    pub offsets: Vec<i64>,
}
