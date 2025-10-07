import os

import numpy as np
import orjson
import strictfire
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TextEmbedder:
    def __init__(self, device, batch_size, embedding_model):
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedding_model}",
            device=device,
        )
        self.model = torch.compile(self.model)
        self.batch_size = batch_size

    def __call__(self, text_list):
        return self.model.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )


def main(
    dataset_name,
    device="cuda:0",
    batch_size=8192,
    embedding_model="all-MiniLM-L12-v2",
):
    text_path = f"{os.environ['HOME']}/scratch/pre/{dataset_name}/text.json"
    with open(text_path) as f:
        raw = f.read()
    text_list = orjson.loads(raw)

    text_embedder = TextEmbedder(device, batch_size, embedding_model=embedding_model)
    emb_list = text_embedder(text_list)

    emb_path = f"{os.environ['HOME']}/scratch/pre/{dataset_name}/text_emb_{embedding_model}.bin"
    emb = np.stack(emb_list).astype(bfloat16)
    emb.tofile(emb_path)


if __name__ == "__main__":
    strictfire.StrictFire(main)
