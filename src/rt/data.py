import json
import math
import random
from functools import cache
from pathlib import Path

import ml_dtypes  # noqa: F401  # registers bfloat16 numpy dtype for rustler
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from rt._rustler import Sampler
from rt.pre import resolve_pre_dir

# rustler's Sampler is an unpicklable Rust object, so any DataLoader over a
# RustlerDataset must use the 'fork' start method -- Python 3.14 defaults to
# 'forkserver'/'spawn', which pickle the worker's arguments and would fail with
# "cannot pickle 'builtins.Sampler'". We also share worker tensors via node-local
# files instead of /dev/shm (which dense multi-worker eval nodes, plus segments
# leaked by preempted jobs, exhaust -> "No space left on device"). Set both once,
# here, at import of the module that introduces the Sampler, so every entry point
# that touches rt.data (eval / baseline / scaling / training) is covered without
# each needing its own copy.
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

MAX_F2P_NBRS = 5  # See fly.rs L32


@cache
def _load_column_index(db_name: str, pre_dir: str) -> dict:
    pre_dir = Path(pre_dir).expanduser()
    column_index_path = f"{pre_dir}/{db_name}/column_index.json"
    with open(column_index_path) as f:
        return json.load(f)


def get_column_index(
    column_name: str, table_name: str, db_name: str, pre_dir: str
) -> int:
    column_index = _load_column_index(db_name, pre_dir)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(
            f'Column "{target}" not found in {pre_dir}/{db_name}/column_index.json.'
        )

    return column_index[target]


SEM_TYPE_NUMBER = 0
SEM_TYPE_BOOLEAN = 3


def process_batch(tup, d_text, bool_as_num):
    out = dict(tup)
    seq_len = out.pop("seq_len")

    for k, v in out.items():
        if k in [
            "number_values",
            "datetime_values",
            "text_values",
            "col_name_values",
            "boolean_values",
        ]:
            out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
        else:
            out[k] = torch.from_numpy(v)

    out["node_idxs"] = out["node_idxs"].view(-1, seq_len)
    out["sem_types"] = out["sem_types"].view(-1, seq_len)
    out["is_targets"] = out["is_targets"].view(-1, seq_len)
    out["is_task_nodes"] = out["is_task_nodes"].view(-1, seq_len)
    out["is_padding"] = out["is_padding"].view(-1, seq_len)
    out["table_name_idxs"] = out["table_name_idxs"].view(-1, seq_len)
    out["col_name_idxs"] = out["col_name_idxs"].view(-1, seq_len)
    out["class_value_idxs"] = out["class_value_idxs"].view(-1, seq_len)
    out["timestamps"] = out["timestamps"].view(-1, seq_len)
    out["seed_node_idxs"] = out["seed_node_idxs"].view(-1, seq_len)
    out["bfs_depths"] = out["bfs_depths"].view(-1, seq_len)

    out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, seq_len, MAX_F2P_NBRS)
    out["number_values"] = out["number_values"].view(-1, seq_len, 1)
    out["datetime_values"] = out["datetime_values"].view(-1, seq_len, 1)
    out["boolean_values"] = out["boolean_values"].view(-1, seq_len, 1).bfloat16()
    out["text_values"] = out["text_values"].view(-1, seq_len, d_text)
    out["col_name_values"] = out["col_name_values"].view(-1, seq_len, d_text)

    if bool_as_num:
        bool_mask = out["sem_types"] == SEM_TYPE_BOOLEAN
        out["number_values"][bool_mask] = out["boolean_values"][bool_mask]
        out["boolean_values"][bool_mask] = 0
        out["sem_types"][bool_mask] = SEM_TYPE_NUMBER

    return out


class RustlerDataset:
    def __init__(
        self,
        tasks,
        pre_dir: str,
        global_rank,
        local_rank,
        world_size,
        local_ctx_sizes: list[int],
        bfs_widths: list[int],
        num_walks,
        walk_length,
        prefer_latest: list[bool],
        mask_prob_max,
        embedding_model,
        d_text,
        shuffle_seed,
        context_seed,
        items_per_task,
        quiet,
        bool_as_num,
        ignore_data_errors,
        skip_text_cols,
        mmap_populate,
        balance_labels: list[bool],
        timeout_per_item,
        ablate_schema_semantics,
        vector_db_path: str | None,
        train_only_fallback: bool,
    ):
        # Accept either rt.tasks.Task dataclasses or the legacy 5-tuples
        # (db_name, table_name, target_column, split, columns_to_drop).
        tasks = [
            t
            if isinstance(t, tuple)
            else (t.db_name, t.table_name, t.target_column, t.split, t.leakage_columns)
            for t in tasks
        ]

        # `pre_dir` may be a local path or a HuggingFace repo spec; resolve to a
        # local root, downloading only the files needed for these databases.
        pre_dir = resolve_pre_dir(pre_dir, [t[0] for t in tasks], embedding_model)
        if vector_db_path is not None:
            vector_db_path = str(Path(vector_db_path).expanduser())

        dataset_tuples = []
        target_column_indices = []
        drop_column_indices = []
        skipped_tasks = []

        for db_name, table_name, target_column, split, columns_to_drop in tasks:
            try:
                if split == "train":
                    split = "Train"
                elif split == "val":
                    split = "Val"
                elif split == "test":
                    split = "Test"

                table_info_path = f"{pre_dir}/{db_name}/table_info.json"
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

                target_idx = get_column_index(
                    target_column, table_name, db_name, pre_dir
                )
                target_column_indices.append(target_idx)

                drop_indices = []
                for col in columns_to_drop:
                    if col == target_column:
                        continue
                    try:
                        drop_indices.append(
                            get_column_index(col, table_name, db_name, pre_dir)
                        )
                    except ValueError:
                        pass  # skip_col not in task parquet; ignore
                drop_column_indices.append(drop_indices)

                dataset_tuples.append((db_name, table_name, node_idx_offset, num_nodes))
            except Exception as e:
                if not ignore_data_errors:
                    raise
                task_name = f"{db_name}/{table_name}/{target_column}"
                skipped_tasks.append((task_name, e))

        if skipped_tasks and local_rank == 0 and not quiet:
            print(
                f"\033[31mskipped {len(skipped_tasks)} task(s)\033[0m",
                flush=True,
            )
            for task_name, e in skipped_tasks:
                print(f"  \033[31mskipped {task_name}: {e}\033[0m", flush=True)

        self.world_size = world_size
        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            local_ctx_sizes=local_ctx_sizes,
            bfs_widths=bfs_widths,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest=prefer_latest,
            mask_prob_max=mask_prob_max,
            embedding_model=embedding_model,
            pre_dir=pre_dir,
            d_text=d_text,
            shuffle_seed=shuffle_seed,
            context_seed=context_seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
            items_per_task=items_per_task,
            quiet=quiet,
            ignore_data_errors=ignore_data_errors,
            num_prev_skipped=len(skipped_tasks),
            skip_text_cols=skip_text_cols,
            mmap_populate=mmap_populate,
            balance_labels=balance_labels,
            timeout_per_item=timeout_per_item,
            ablate_schema_semantics=ablate_schema_semantics,
            vector_db_path=vector_db_path,
            train_only_fallback=train_only_fallback,
        )
        self.num_items = self.sampler.num_items

        self.d_text = d_text
        self.bool_as_num = bool_as_num

    def _process_batch(self, tup):
        return process_batch(tup, self.d_text, self.bool_as_num)


class TrainDataset(RustlerDataset, IterableDataset):
    def __init__(
        self,
        tasks,
        pre_dir: str,
        train_ctx_sizes,
        train_tokens_per_gpu,
        total_bs,
        global_rank,
        local_rank,
        world_size,
        local_ctx_sizes: list[int],
        bfs_widths: list[int],
        num_walks,
        walk_length,
        prefer_latest: list[bool],
        mask_prob_max,
        embedding_model,
        d_text,
        seed,
        items_per_task,
        mask_prob_max_shared,
        bool_as_num,
        skip_text_cols,
        mmap_populate,
        balance_labels: list[bool],
        timeout_per_item,
        ablate_schema_semantics,
        vector_db_path: str | None,
        train_only_fallback: bool,
    ):
        # TrainDataset drives both shuffle and context construction from the
        # same seed — this matches prior single-seed behavior.
        RustlerDataset.__init__(
            self,
            tasks=tasks,
            pre_dir=pre_dir,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            local_ctx_sizes=local_ctx_sizes,
            bfs_widths=bfs_widths,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest=prefer_latest,
            mask_prob_max=mask_prob_max,
            embedding_model=embedding_model,
            d_text=d_text,
            shuffle_seed=seed,
            context_seed=seed,
            items_per_task=items_per_task,
            quiet=False,
            bool_as_num=bool_as_num,
            ignore_data_errors=True,
            skip_text_cols=skip_text_cols,
            mmap_populate=mmap_populate,
            balance_labels=balance_labels,
            timeout_per_item=timeout_per_item,
            ablate_schema_semantics=ablate_schema_semantics,
            vector_db_path=vector_db_path,
            train_only_fallback=train_only_fallback,
        )
        self.train_ctx_sizes = train_ctx_sizes
        self.seed = random.Random(seed).getrandbits(64)
        self.train_tokens_per_gpu = train_tokens_per_gpu
        self.total_bs = total_bs
        self.mask_prob_max_shared = mask_prob_max_shared
        # total_bs must split evenly into world_size * per_gpu_bs so the global
        # batch is exactly total_bs. Pick a GPU count that divides
        # total_bs / per_gpu_bs (the launcher does this); fail loudly otherwise.
        for c in train_ctx_sizes:
            train_bs = max(1, train_tokens_per_gpu // c)
            if total_bs < world_size * train_bs:
                assert total_bs % world_size == 0, (
                    f"total_bs={total_bs} not divisible by world_size={world_size}"
                    f" for ctx_size={c}"
                )
            else:
                assert total_bs % (world_size * train_bs) == 0, (
                    f"total_bs={total_bs} not divisible by world_size*bs_per_gpu="
                    f"{world_size * train_bs} for ctx_size={c}"
                )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.sampler.set_step_py(worker_info.id)
            self.sampler.set_stride_py(worker_info.num_workers)
            step = worker_info.id
            stride = worker_info.num_workers
        else:
            step = 0
            stride = 1
        # Single ctx_size: yield individual microbatches so workers prefetch
        # in parallel. List-yielding (multi-ctx case) blocks each worker for
        # grad_accum batches before yielding, which is unnecessary here since
        # all microbatches share the only ctx_size anyway.
        single_ctx = len(self.train_ctx_sizes) == 1
        while True:
            if self.mask_prob_max_shared is not None:
                self.sampler.set_mask_prob_max_py(self.mask_prob_max_shared.value)
            train_ctx_size = random.Random(self.seed + step).choice(
                self.train_ctx_sizes
            )
            train_bs = max(1, self.train_tokens_per_gpu // train_ctx_size)
            if self.total_bs < self.world_size * train_bs:
                train_bs = max(1, self.total_bs // self.world_size)
                grad_accum = 1
            else:
                grad_accum = self.total_bs // (self.world_size * train_bs)
            if single_ctx:
                tup = self.sampler.batch_py(None, train_bs, train_ctx_size)
                yield self._process_batch(tup)
            else:
                # Multi ctx_size: yield grad_accum batches atomically with
                # shared ctx_size to avoid worker-round-robin interleaving
                # ctx_sizes within an optimizer step.
                batches = []
                for _ in range(grad_accum):
                    tup = self.sampler.batch_py(None, train_bs, train_ctx_size)
                    batches.append(self._process_batch(tup))
                yield batches
            step += stride


class EvalDataset(Dataset):
    def __init__(
        self,
        rustler_dataset: RustlerDataset,
        eval_bs,
        eval_ctx_size,
    ):
        self.rustler_dataset = rustler_dataset
        self.eval_bs = eval_bs
        self.eval_ctx_size = eval_ctx_size

    def __len__(self):
        # Uniform across ranks: every rank iterates the same number of
        # batches. Higher-rank offsets on the last batch may legitimately
        # overshoot num_items; the rustler sampler fills those slots as
        # phantoms (batch_mask[i]=false) so the downstream fixed-size
        # gather is simple and correct.
        return math.ceil(
            self.rustler_dataset.num_items
            / (self.eval_bs * self.rustler_dataset.world_size)
        )

    def __getitem__(self, i):
        return self.rustler_dataset._process_batch(
            self.rustler_dataset.sampler.batch_py(i, self.eval_bs, self.eval_ctx_size)
        )
