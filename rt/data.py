import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))

import json
import os
from functools import cache

import ml_dtypes
import numpy as np
import torch
from rustler import Sampler
from torch.utils.data import Dataset


@cache
def _load_column_index(db_name: str) -> dict:
    """
    Load the column index mapping for a dataset (cached).
    """
    home = os.environ.get("HOME", ".")
    column_index_path = os.path.join(
        home, "scratch", "pre", db_name, "column_index.json"
    )

    with open(column_index_path) as f:
        return json.load(f)


def get_column_index(column_name: str, table_name: str, db_name: str) -> int:
    """
    Get the index of a column in the text embeddings for a given dataset.
    """
    column_index = _load_column_index(db_name)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(
            f'Column "{target}" not found in column_index.json for dataset {db_name}.'
        )

    return column_index[target]


class RelationalDataset(Dataset):
    def __init__(
        self,
        tasks,
        batch_size,
        seq_len,
        rank,
        world_size,
        max_bfs_width,
        embedding_model,
        d_text,
        seed,
    ):
        dataset_tuples = []
        target_column_indices = []
        drop_column_indices = []

        for db_name, table_name, target_column, split, columns_to_drop in tasks:
            if split == "train":
                split = "Train"
            elif split == "val":
                split = "Val"
            elif split == "test":
                split = "Test"

            table_info_path = (
                f"{os.environ['HOME']}/scratch/pre/{db_name}/table_info.json"
            )
            with open(table_info_path) as f:
                table_info = json.load(f)

            table_info_key = (
                f"{table_name}:Db"
                if f"{table_name}:Db" in table_info
                else f"{table_name}:{split}"
            )
            info = table_info[table_info_key]
            node_idx_offset = info["node_idx_offset"]
            num_nodes = info["num_nodes"]

            target_idx = get_column_index(target_column, table_name, db_name)
            target_column_indices.append(target_idx)

            drop_indices = [
                get_column_index(col, table_name, db_name) for col in columns_to_drop
            ]
            drop_column_indices.append(drop_indices)

            dataset_tuples.append((db_name, node_idx_offset, num_nodes))

        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=rank,
            world_size=world_size,
            max_bfs_width=max_bfs_width,
            embedding_model=embedding_model,
            d_text=d_text,
            seed=seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
        )

        self.seq_len = seq_len
        self.d_text = d_text

    def __len__(self):
        return self.sampler.len_py()

    def __getitem__(self, batch_idx):
        tup = self.sampler.batch_py(batch_idx)
        out = dict(tup)
        for k, v in out.items():
            if k in [
                "number_values",
                "datetime_values",
                "text_values",
                "col_name_values",
                "boolean_values",
            ]:
                out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
            elif k == "true_batch_size":
                pass
            else:
                out[k] = torch.from_numpy(v)

        out["node_idxs"] = out["node_idxs"].view(-1, self.seq_len)
        out["sem_types"] = out["sem_types"].view(-1, self.seq_len)
        out["masks"] = out["masks"].view(-1, self.seq_len)
        out["is_targets"] = out["is_targets"].view(-1, self.seq_len)
        out["is_task_nodes"] = out["is_task_nodes"].view(-1, self.seq_len)
        out["is_padding"] = out["is_padding"].view(-1, self.seq_len)
        out["table_name_idxs"] = out["table_name_idxs"].view(-1, self.seq_len)
        out["col_name_idxs"] = out["col_name_idxs"].view(-1, self.seq_len)
        out["class_value_idxs"] = out["class_value_idxs"].view(-1, self.seq_len)

        out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, self.seq_len, 5)
        out["number_values"] = out["number_values"].view(-1, self.seq_len, 1)
        out["datetime_values"] = out["datetime_values"].view(-1, self.seq_len, 1)
        out["boolean_values"] = (
            out["boolean_values"].view(-1, self.seq_len, 1).bfloat16()
        )
        out["text_values"] = out["text_values"].view(-1, self.seq_len, self.d_text)
        out["col_name_values"] = out["col_name_values"].view(
            -1, self.seq_len, self.d_text
        )

        return out
