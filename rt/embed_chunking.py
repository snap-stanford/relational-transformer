import os
import gc  # Garbage collection to free RAM
import numpy as np
import orjson
import strictfire
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TextEmbedder:
    def __init__(self, batch_size, embedding_model, device_type):
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedding_model}",
            device=device_type,
            model_kwargs={
                "dtype": torch.float16 if device_type == "cuda" else torch.float32,
            },
        )
        self.batch_size = batch_size

    # We modify this to process just one chunk at a time
    def encode_chunk(self, text_chunk, device=None):
        return self.model.encode(
            text_chunk,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=device,
        )

def main(
    dataset_name,
    device=None,
    batch_size=4096, # Internal batch size for GPU
    chunk_size=50000, # RAM SAFETY: How many rows to save to disk at a time
    embedding_model="all-MiniLM-L12-v2",
):
    # 1. Device Setup
    if device is None:
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            device = [f"cuda:{i}" for i in range(num_devices)]
            print(f"Found {num_devices} CUDA device(s): {device}")
        else:
            device = "cpu"
            print("No CUDA devices available, using CPU")

    if isinstance(device, list):
        device_type = torch.device(device[0]).type
    else:
        device_type = torch.device(device).type

    # 2. File Paths
    input_path = f"{os.environ['USERPROFILE']}/scratch/pre/{dataset_name}/text.json"
    output_path = f"{os.environ['USERPROFILE']}/scratch/pre/{dataset_name}/text_emb_{embedding_model}.bin"

    # 3. Load Input Data
    print(f"Loading data from: {input_path}")
    with open(input_path, "rb") as f:
        raw = f.read()
    text_list = orjson.loads(raw)
    total_rows = len(text_list)
    print(f"Loaded {total_rows:,} sentences.")

    # 4. Initialize Model
    embedder = TextEmbedder(
        batch_size,
        embedding_model=embedding_model,
        device_type=device_type,
    )

    # 5. CHUNKED PROCESSING LOOP (The Fix)
    print(f"Starting generation. Saving incrementally to: {output_path}")
    
    # Open file in 'wb' (Write Binary) mode. This wipes the file and starts fresh.
    with open(output_path, "wb") as f_out:
        
        # Loop through data in steps of 'chunk_size'
        for i in range(0, total_rows, chunk_size):
            # Slice the data
            end_idx = min(i + chunk_size, total_rows)
            current_batch = text_list[i : end_idx]
            
            print(f"Processing chunk {i} to {end_idx} ({((i/total_rows)*100):.1f}%)...")
            
            # Generate Embeddings for this chunk only
            batch_embs = embedder.encode_chunk(current_batch, device=device)
            
            # Cast to float16 (Standard, low memory) and Write to disk immediately
            # Note: We use tobytes() to append raw binary data, identical to .tofile()
            f_out.write(batch_embs.astype(np.float16).tobytes())
            
            # CRITICAL: Delete from RAM immediately
            del batch_embs
            del current_batch
            gc.collect() # Force Python to release memory

    print(f"✅ DONE! All embeddings saved to: {output_path}")

if __name__ == "__main__":
    strictfire.Def(main)