import os
import json

import torch
from rustler import Sampler
from torch.utils.data import Dataset
import ml_dtypes
import numpy as np


class FineTuneDataset(Dataset):
    def __init__(
        self,
        tasks,
        batch_size,
        seq_len,
        mask_prob,
        rank,
        world_size,
        fake_names,
        subsample_p2f_edges,
        isolate_task_tables,
        cos_steps,
        embedding_model,
        d_text,
        mask_db_cells,
        mask_task_cells,
        seed,
    ):
        dataset_tuples = []
        for dataset_name, task_name, task_split in tasks:
            table_info_path = (
                f"{os.environ['HOME']}/scratch/pre/{dataset_name}/task_table_info.json"
            )
            with open(table_info_path) as f:
                table_info = json.load(f)

            if task_split == "train":
                task_split = "Train"
            elif task_split == "val":
                task_split = "Val"
            elif task_split == "test":
                task_split = "Test"

            info = table_info[f"{task_name}:{task_split}"]
            node_idx_offset = info["node_idx_offset"]
            num_nodes = info["num_nodes"]

            dataset_tuples.append((dataset_name, node_idx_offset, num_nodes))

        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            batch_size=batch_size,
            seq_len=seq_len,
            mask_prob=mask_prob,
            rank=rank,
            world_size=world_size,
            fake_names=fake_names,
            subsample_p2f_edges=subsample_p2f_edges,
            isolate_task_tables=isolate_task_tables,
            cos_steps=cos_steps,
            embedding_model=embedding_model,
            d_text=d_text,
            mask_db_cells=mask_db_cells,
            mask_task_cells=mask_task_cells,
            seed=seed,
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
                "table_name_values",
                "col_name_values",
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
        out["table_name_idxs"] = out["table_name_idxs"].view(-1, self.seq_len)
        out["col_name_idxs"] = out["col_name_idxs"].view(-1, self.seq_len)

        out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, self.seq_len, 4)
        out["number_values"] = out["number_values"].view(-1, self.seq_len, 1)
        out["datetime_values"] = out["datetime_values"].view(-1, self.seq_len, 1)
        out["text_values"] = out["text_values"].view(-1, self.seq_len, self.d_text)
        out["table_name_values"] = out["table_name_values"].view(
            -1, self.seq_len, self.d_text
        )
        out["col_name_values"] = out["col_name_values"].view(
            -1, self.seq_len, self.d_text
        )

        return out
