use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;

mod common;
use crate::common::{Adj, Node, Offsets};

fn main() {
    // Get input path from command line arguments
    let args: Vec<String> = env::args().collect();
    let input_arg = args.get(1).expect("Please provide the input path (folder or file) as the first argument");
    let input_path = Path::new(input_arg);

    if input_path.is_dir() {
        let paths = std::fs::read_dir(input_path).expect("Failed to read directory");
        for path in paths {
            let path = path.unwrap().path();
            if let Some(ext) = path.extension() {
                if ext == "rkyv" {
                    process_file(&path);
                }
            }
        }
    } else if input_path.is_file() {
        process_file(input_path);
    } else {
        eprintln!("Invalid path: {}", input_path.display());
    }
}

fn process_file(input_path: &Path) {
    let filename = input_path.file_name().unwrap().to_str().unwrap();
    let parent_dir = input_path.parent().unwrap();
    
    // Construct output filename: out_{filename}.json
    let stem = input_path.file_stem().unwrap().to_str().unwrap();
    let output_filename = format!("out_{}.json", stem);
    let output_path = parent_dir.join(output_filename);
    
    match filename {
        "offsets.rkyv" => convert_offsets(input_path, &output_path),
        "nodes.rkyv" => convert_nodes(input_path, &output_path),
        "p2f_adj.rkyv" => convert_adj(input_path, &output_path),
        _ => println!("Skipping unknown rkyv file: {}", filename),
    }
}

fn convert_offsets(input_path: &Path, output_path: &Path) {
    println!("Converting {}...", input_path.display());
    let mut file = File::open(input_path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    
    let offsets: Offsets = rkyv::from_bytes::<Offsets, rkyv::rancor::Error>(&buffer)
        .expect("Failed to deserialize offsets");
        
    let json = serde_json::to_string_pretty(&offsets).unwrap();
    std::fs::write(output_path, &json).unwrap();
    println!("Saved to {}", output_path.display());
}

fn convert_adj(input_path: &Path, output_path: &Path) {
    println!("Converting {}...", input_path.display());
    let mut file = File::open(input_path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    
    let adj: Adj = rkyv::from_bytes::<Adj, rkyv::rancor::Error>(&buffer)
        .expect("Failed to deserialize adj");
        
    let json = serde_json::to_string_pretty(&adj).unwrap();
    std::fs::write(output_path, &json).unwrap();
    println!("Saved to {}", output_path.display());
}

fn convert_nodes(input_path: &Path, output_path: &Path) {
    println!("Converting {}...", input_path.display());
    let parent_dir = input_path.parent().unwrap();
    let offsets_path = parent_dir.join("offsets.rkyv");
    
    if !offsets_path.exists() {
        eprintln!("Error: offsets.rkyv not found in {}. Cannot convert nodes.rkyv without it.", parent_dir.display());
        return;
    }

    // Read offsets
    let mut offsets_file = File::open(&offsets_path).expect("Failed to open offsets.rkyv");
    let mut offsets_buffer = Vec::new();
    offsets_file.read_to_end(&mut offsets_buffer).unwrap();
    let offsets: Offsets = rkyv::from_bytes::<Offsets, rkyv::rancor::Error>(&offsets_buffer).expect("Failed to deserialize offsets");

    // Read nodes
    let mut nodes_file = File::open(input_path).unwrap();
    let mut nodes_buffer = Vec::new();
    nodes_file.read_to_end(&mut nodes_buffer).unwrap();

    let mut nodes: Vec<Node> = Vec::new();
    for i in 0..offsets.offsets.len() - 1 {
        let start = offsets.offsets[i] as usize;
        let end = offsets.offsets[i + 1] as usize;
        let node_bytes = &nodes_buffer[start..end];
        let node: Node = rkyv::from_bytes::<Node, rkyv::rancor::Error>(node_bytes).expect("Failed to deserialize node");
        nodes.push(node);
    }

    let json = serde_json::to_string_pretty(&nodes).unwrap();
    std::fs::write(output_path, &json).unwrap();
    println!("Saved to {}", output_path.display());
}