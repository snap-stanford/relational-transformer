import os
import orjson

import strictfire
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from ml_dtypes import bfloat16


class RandomEmbedder:
    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.model = dict()

    def encode(
        self, text_list, convert_to_numpy=True, show_progress_bar=False, **kwargs
    ):
        if show_progress_bar:
            pbar = tqdm(text_list, desc="Encoding")
        embeddings = []
        for text in text_list:
            if text not in self.model:
                self.model[text] = np.random.randn(self.embedding_dim).astype(bfloat16)
            embedding = self.model[text]
            if convert_to_numpy:
                embedding = np.array(embedding, dtype=bfloat16)
            embeddings.append(embedding)
            if show_progress_bar:
                pbar.update(1)
        return embeddings


class TextEmbedder:
    def __init__(self, device, batch_size, embedding_model):
        if embedding_model == "random":
            self.model = RandomEmbedder()
        else:
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
    dataset_name="rel-f1",
    device="cuda:0",
    batch_size=2048,
    task=False,
    embedding_model="stsb-distilroberta-base-v2",
):
    pref = "task_" if task else "db_"
    text_path = f"{os.environ['HOME']}/scratch/pre/{dataset_name}/{pref}text.json"
    with open(text_path) as f:
        raw = f.read()
    text_list = orjson.loads(raw)

    text_embedder = TextEmbedder(device, batch_size, embedding_model=embedding_model)
    emb_list = text_embedder(text_list)

    emb_path = f"{os.environ['HOME']}/scratch/pre/{dataset_name}/{pref}text_emb_{embedding_model}.bin"
    emb = np.stack(emb_list).astype(bfloat16)
    emb.tofile(emb_path)


if __name__ == "__main__":
    strictfire.StrictFire(main)
