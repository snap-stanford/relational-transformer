"""Standard eval driver shared by training (rt/main.py), offline SWA
(rt/swa.py), and grid eval (rt/eval_grid.py).

Owns: per-task DataLoaders (built once at construction), prefetch
iterators, per-batch forward + DDP gather, per-task metric computation,
per-split aggregation, stdout printing, wandb logging.

``Evaluator.evaluate_raw`` is the single per-task pipeline primitive: it
drives the loader, forward passes, and DDP gather, and yields raw
post-mask arrays (labels, preds for each net, per-row label counts) per
``(task, ctx_size)`` on rank 0. ``Evaluator.evaluate`` is a thin
wrapper that consumes ``evaluate_raw`` and adds metric computation,
per-split aggregation, stdout printing, and wandb logging — mirroring
the inline ``evaluate()`` previously in rt/main.py.

Construction is rank-aware (rustler fans out per-rank items via
``global_rank``/``world_size``). On non-zero ranks both methods drive
all collectives so NCCL stays in lockstep, but only rank 0 sees yielded
tensors / metric values.
"""

import time

import lazy_loader as lazy
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import EvalDataset, RustlerDataset

wandb = lazy.load("wandb")
sklearn = lazy.load("sklearn")


def fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


class Evaluator:
    """Standard per-task eval over a fixed task list.

    Build once with sampler/loader knobs; call ``evaluate`` (or
    ``evaluate_raw``) one or more times. Re-using an instance across
    eval points reuses prefetch state and avoids loader rebuild —
    important for the in-loop training eval.

    Synthetic-DB tasks (``"synthetic" in db_name``) are dropped at
    construction; they were never evaluated by the inline path either.
    """

    def __init__(
        self,
        *,
        tasks,
        pre_dir,
        eval_bs,
        ctx_sizes,
        items_per_task,
        num_workers,
        prefetch_factor,
        persistent_workers,
        local_ctx_size,
        bfs_width,
        num_walks,
        walk_length,
        prefer_latest,
        bool_as_num,
        skip_text_cols,
        mmap_populate,
        balance_labels,
        ablate_schema_semantics,
        embedding_model,
        d_text,
        shuffle_seed,
        context_seed,
        vector_db_path,
        train_only_fallback,
        global_rank,
        local_rank,
        world_size,
        ddp,
        device,
    ):
        self.tasks = [t for t in tasks if "synthetic" not in t.db_name]
        self.eval_splits = sorted(set(t.split for t in self.tasks if t.split))
        self.ctx_sizes = ctx_sizes
        self.eval_bs = eval_bs
        self.items_per_task = items_per_task
        self.bool_as_num = bool_as_num
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.ddp = ddp
        self.device = device

        max_eval_ctx_size = max(ctx_sizes)

        self.eval_loaders = {}
        self.eval_loader_iters = {}

        init_pbar = tqdm(
            total=len(self.tasks),
            desc="load eval data",
            disable=local_rank != 0,
            leave=False,
        )
        init_tic = time.time()
        prefetch_time = 0.0

        for eval_task in self.tasks:
            rustler_dataset = RustlerDataset(
                tasks=[
                    (
                        eval_task.db_name,
                        eval_task.table_name,
                        eval_task.target_column,
                        eval_task.split,
                        eval_task.leakage_columns,
                    )
                ],
                pre_dir=pre_dir,
                global_rank=global_rank,
                local_rank=local_rank,
                world_size=world_size,
                local_ctx_sizes=[local_ctx_size],
                bfs_widths=[bfs_width],
                num_walks=num_walks,
                walk_length=walk_length,
                prefer_latest=[prefer_latest],
                mask_prob_max=0.0,
                embedding_model=embedding_model,
                d_text=d_text,
                shuffle_seed=shuffle_seed,
                context_seed=context_seed,
                items_per_task=items_per_task,
                quiet=True,
                bool_as_num=bool_as_num,
                ignore_data_errors=False,
                skip_text_cols=skip_text_cols,
                mmap_populate=mmap_populate,
                balance_labels=[balance_labels],
                timeout_per_item=3600.0,
                ablate_schema_semantics=ablate_schema_semantics,
                vector_db_path=vector_db_path,
                train_only_fallback=train_only_fallback,
            )
            eval_dataset = EvalDataset(
                rustler_dataset=rustler_dataset,
                eval_bs=eval_bs,
                eval_ctx_size=max_eval_ctx_size,
            )
            self.eval_loaders[eval_task] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers,
                pin_memory=True,
                # in_order=True guarantees sampler-order yields; with False the
            # row order is worker-completion order — a timing race that breaks
            # cross-seed prediction averaging (context ensembling) on rows
            # written by eval_grid.
            in_order=True,
            )
            _prefetch_tic = time.time()
            self.eval_loader_iters[eval_task] = iter(self.eval_loaders[eval_task])
            prefetch_time += time.time() - _prefetch_tic
            init_pbar.update(1)
        init_pbar.close()
        if local_rank == 0:
            print(
                f"\neval data loaded in"
                f" \033[1m{fmt_duration(time.time() - init_tic)}\033[0m",
                flush=True,
            )
            print(
                f"  prefetch init took \033[1m{fmt_duration(prefetch_time)}\033[0m",
                flush=True,
            )

    def evaluate_raw(self, nets_with_prefix, eval_ctx_sizes_to_use,
                     with_node_idxs=False):
        """Per-task pipeline primitive.

        Drives per-batch forward + DDP gather + ``batch_mask`` filtering
        for every task. Yields one tuple per ``(task, ctx_size)`` on
        rank 0::

            (task, ctx_size, labels_np, preds_by_prefix_np, num_labels_np)

        - ``labels_np``: ``(n_real,)`` per-row labels.
        - ``preds_by_prefix_np``: dict ``prefix → (n_real,) preds``,
          one entry per ``(net, prefix)`` in ``nets_with_prefix``.
        - ``num_labels_np``: ``(n_real,) int64`` per-row count of
          in-context training labels for that row's target column at
          ``ctx_size`` (the ``mean_labels`` source data).

        ``n_real`` is the number of real (non-phantom) rows across all
        ranks for that task — already filtered by ``batch_mask``.

        With ``with_node_idxs=True`` a sixth element ``node_idxs_np`` is
        appended to the yielded tuple: the ``(n_real,) int64`` global
        rustler node index of each row's *seed* (target) node. Because
        rustler assigns a task row the node index ``node_idx_offset + r``
        (``r`` the 0-based row index in the relbench task-table parquet),
        ``node_idx - node_idx_offset`` recovers the exact parquet row, which
        is how :mod:`rt.eval_utils` keys predictions back to the relbench
        ``(entity_col, time_col)`` for a leaderboard submission (eval row
        order is *not* the parquet row order, so a positional join is wrong).

        Other ranks drive every collective but yield nothing.
        """
        device = self.device
        ddp = self.ddp
        world_size = self.world_size
        global_rank = self.global_rank
        local_rank = self.local_rank

        for net, _ in nets_with_prefix:
            net.eval()

        with torch.inference_mode():
            for eval_task, eval_loader_iter in self.eval_loader_iters.items():
                eval_loader = self.eval_loaders[eval_task]

                # The number of eval batches per task MUST be identical on every
                # rank, or NCCL deadlocks (ranks issue a different number of
                # collective calls). ``len(eval_loader.dataset)`` is
                # ``ceil(num_items / (eval_bs * world_size))`` -- uniform across
                # ranks (``num_items`` is the task's total item count, not a
                # per-rank count), and the rustler sampler fills any overshoot
                # slots as phantoms (batch_mask[i]=False). ``items_per_task``
                # only caps how many batches we bother running; the cap is the
                # same integer on every rank, so it never desyncs the schedule.
                n_batches = len(eval_loader.dataset)
                if self.items_per_task is not None:
                    n_batches = min(
                        n_batches,
                        max(1, self.items_per_task // self.eval_bs // world_size),
                    )

                preds_per_prefix_per_ctx = {
                    prefix: {ctx: [] for ctx in eval_ctx_sizes_to_use}
                    for _, prefix in nets_with_prefix
                }
                num_labels_per_ctx = {ctx: [] for ctx in eval_ctx_sizes_to_use}
                labels = []
                batch_masks = []
                node_idxs_acc = []
                pbar = tqdm(
                    total=n_batches,
                    desc=f"{eval_task.db_name}/{eval_task.table_name}/{eval_task.split}",
                    disable=local_rank != 0,
                    leave=False,
                )
                # Drive the loop by the fixed, cross-rank-uniform batch count.
                # Every rank processes exactly ``n_batches`` batches (each of
                # ``eval_bs`` rows, phantom-padded as needed), so every rank
                # contributes exactly ``n_batches * eval_bs`` rows to every
                # collective below -- no StopIteration / local-count breaks.
                for _ in range(n_batches):
                    batch = next(eval_loader_iter)

                    batch_mask = batch.pop("batch_mask")

                    # Per-row in-context training-label count for the
                    # target column, for each requested ctx_size. Gathered
                    # and masked alongside labels/preds so the eventual
                    # mean_labels stat is uniform over real items.
                    for eval_ctx_size in eval_ctx_sizes_to_use:
                        tb = {k: v[:, :eval_ctx_size] for k, v in batch.items()}
                        tb_is_targets = tb["is_targets"]
                        tb_target_col = torch.full(
                            (tb_is_targets.shape[0], 1),
                            -1,
                            dtype=tb["col_name_idxs"].dtype,
                        )
                        tb_target_node = tb_target_col.clone()
                        tb_bidxs, tb_sidxs = tb_is_targets.nonzero(as_tuple=True)
                        tb_target_col[tb_bidxs, 0] = tb["col_name_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        tb_target_node[tb_bidxs, 0] = tb["node_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        is_label_cell = (
                            tb["is_task_nodes"]
                            & ~tb["is_padding"]
                            & (tb["col_name_idxs"] == tb_target_col)
                            & (tb["node_idxs"] != tb_target_node)
                        )
                        num_labels_per_ctx[eval_ctx_size].append(
                            is_label_cell.sum(dim=1).to(torch.int64)
                        )

                    for net, prefix in nets_with_prefix:
                        preds_by_ctx = net.predict(
                            batch,
                            eval_ctx_sizes_to_use,
                            device,
                            eval_task,
                            bool_as_num=self.bool_as_num,
                        )
                        for ctx_size, yhat in preds_by_ctx.items():
                            assert yhat.size(0) == batch_mask.size(0)
                            preds_per_prefix_per_ctx[prefix][ctx_size].append(yhat)

                    val_key = (
                        "boolean_values"
                        if eval_task.task_type == "clf" and not self.bool_as_num
                        else "number_values"
                    )
                    y = (
                        batch[val_key].squeeze(-1)
                        * batch["is_targets"].to(batch[val_key].dtype)
                    ).sum(dim=1)
                    assert y.size(0) == batch_mask.size(0)
                    labels.append(y)
                    batch_masks.append(batch_mask)
                    if with_node_idxs:
                        # Seed (target) node's global rustler index per row. Exactly
                        # one target cell per real row, so the masked sum picks it
                        # out; phantom rows have no target → 0, dropped by batch_mask.
                        nidx = (
                            batch["node_idxs"].to(torch.int64)
                            * batch["is_targets"].to(torch.int64)
                        ).sum(dim=1)
                        assert nidx.size(0) == batch_mask.size(0)
                        node_idxs_acc.append(nidx)
                    pbar.update(1)

                pbar.close()

                # prefetch next pass while we run gather + metric compute.
                self.eval_loader_iters[eval_task] = iter(eval_loader)

                # Every rank ran exactly ``n_batches`` batches of ``eval_bs``
                # rows, so ``labels_cat`` has the same length on every rank and
                # the all_gathers are inherently in lockstep -- no cross-rank
                # MIN reduce or truncation needed. Phantom rows are filtered out
                # via ``masks_gathered`` on rank 0 after the gather.
                labels_cat = torch.cat(labels, dim=0).to(device)
                masks_cat = torch.cat(batch_masks, dim=0).to(device)
                if ddp:
                    labels_gathered = torch.empty(
                        labels_cat.size(0) * world_size,
                        dtype=labels_cat.dtype,
                        device=device,
                    )
                    masks_gathered = torch.empty(
                        masks_cat.size(0) * world_size,
                        dtype=masks_cat.dtype,
                        device=device,
                    )
                    dist.all_gather_into_tensor(
                        labels_gathered, labels_cat.contiguous()
                    )
                    dist.all_gather_into_tensor(masks_gathered, masks_cat.contiguous())
                else:
                    labels_gathered = labels_cat
                    masks_gathered = masks_cat

                if global_rank == 0:
                    labels_np = labels_gathered[masks_gathered].float().cpu().numpy()

                node_idxs_np = None
                if with_node_idxs:
                    nidx_cat = torch.cat(node_idxs_acc, dim=0).to(device)
                    if ddp:
                        nidx_gathered = torch.empty(
                            nidx_cat.size(0) * world_size,
                            dtype=nidx_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_into_tensor(
                            nidx_gathered, nidx_cat.contiguous()
                        )
                    else:
                        nidx_gathered = nidx_cat
                    if global_rank == 0:
                        node_idxs_np = nidx_gathered[masks_gathered].cpu().numpy()

                for eval_ctx_size in eval_ctx_sizes_to_use:
                    nlabels_cat = torch.cat(
                        num_labels_per_ctx[eval_ctx_size], dim=0
                    ).to(device)
                    if ddp:
                        nlabels_gathered = torch.empty(
                            nlabels_cat.size(0) * world_size,
                            dtype=nlabels_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_into_tensor(
                            nlabels_gathered, nlabels_cat.contiguous()
                        )
                    else:
                        nlabels_gathered = nlabels_cat

                    preds_by_prefix_np = {}
                    for _, prefix in nets_with_prefix:
                        preds = torch.cat(
                            preds_per_prefix_per_ctx[prefix][eval_ctx_size], dim=0
                        ).to(device)
                        if ddp:
                            preds_gathered = torch.empty(
                                preds.size(0) * world_size,
                                dtype=preds.dtype,
                                device=device,
                            )
                            dist.all_gather_into_tensor(
                                preds_gathered, preds.contiguous()
                            )
                            preds = preds_gathered
                        if global_rank == 0:
                            preds_by_prefix_np[prefix] = (
                                preds[masks_gathered].float().cpu().numpy()
                            )

                    if global_rank == 0:
                        num_labels_np = nlabels_gathered[masks_gathered].cpu().numpy()
                        out = (
                            eval_task,
                            eval_ctx_size,
                            labels_np,
                            preds_by_prefix_np,
                            num_labels_np,
                        )
                        if with_node_idxs:
                            out = out + (node_idxs_np,)
                        yield out

    def evaluate(self, nets_with_prefix, eval_ctx_sizes_to_use, steps, reg_metric):
        """Full main.py-style pass: per-task metrics, per-split avg
        aggregation, stdout printing, wandb logging.

        ``nets_with_prefix``: list of ``(net, prefix_str)``. Prefixes
        feed into wandb keys + console labels (e.g. ``""`` for the
        live net, ``"swa_"`` for the SWA snapshot).

        ``reg_metric``: ``"mae"`` (mean absolute error) or ``"r2"``
        (coefficient of determination). Selects the per-task metric
        computed for ``task_type == "reg"`` tasks; both the wandb key
        and the ``all_metrics`` aggregate key are named
        ``avg_{reg_metric}``.

        Returns the empty-prefix net's metrics dict, or the first
        net's if no empty-prefix entry exists.
        """
        assert reg_metric in ("mae", "r2"), (
            f"reg_metric must be 'mae' or 'r2', got {reg_metric!r}"
        )
        eval_tic = time.time()
        local_rank = self.local_rank
        global_rank = self.global_rank

        if local_rank == 0:
            tqdm.write(f"[step {steps}]")

        avg_reg_key = f"avg_{reg_metric}"

        all_metrics = {}
        all_reg_scores = {}
        all_auc_scores = {}
        for _, prefix in nets_with_prefix:
            all_metrics[prefix] = {
                split: {ctx: {} for ctx in eval_ctx_sizes_to_use}
                for split in self.eval_splits
            }
            all_reg_scores[prefix] = {
                (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
            }
            all_auc_scores[prefix] = {
                (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
            }
        all_mean_labels_reg = {
            (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
        }
        all_mean_labels_clf = {
            (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
        }

        outer_pbar = tqdm(
            total=len(self.eval_loaders),
            desc=f"eval@{steps}",
            disable=local_rank != 0,
            leave=False,
        )

        last_task = None
        for (
            eval_task,
            eval_ctx_size,
            labels_np,
            preds_by_prefix_np,
            num_labels_np,
        ) in self.evaluate_raw(nets_with_prefix, eval_ctx_sizes_to_use):
            if last_task is not None and eval_task is not last_task:
                outer_pbar.update(1)
            last_task = eval_task

            # Uniform per-real-item average. Length matches labels_np.
            task_mean_labels = float(num_labels_np.mean())
            if eval_task.task_type == "reg":
                all_mean_labels_reg[(eval_ctx_size, eval_task.split)].append(
                    task_mean_labels
                )
            elif eval_task.task_type == "clf":
                all_mean_labels_clf[(eval_ctx_size, eval_task.split)].append(
                    task_mean_labels
                )

            for prefix, preds_np in preds_by_prefix_np.items():
                if eval_task.task_type == "reg":
                    metric_name = reg_metric
                    if reg_metric == "mae":
                        metric = sklearn.metrics.mean_absolute_error(
                            labels_np, preds_np
                        )
                    elif reg_metric == "r2":
                        metric = sklearn.metrics.r2_score(labels_np, preds_np)
                    all_reg_scores[prefix][(eval_ctx_size, eval_task.split)].append(
                        metric
                    )
                    metric_str = f"{metric:<6.4f}"
                elif eval_task.task_type == "clf":
                    metric_name = "auc"
                    labels_int = [int(x > 0) for x in labels_np]
                    try:
                        metric = sklearn.metrics.roc_auc_score(labels_int, preds_np)
                    except Exception as e:
                        n_classes = len(set(labels_int))
                        n_nan_labels = int(np.isnan(labels_np).sum())
                        n_nan_preds = int(np.isnan(preds_np).sum())
                        tqdm.write(
                            f"\033[31mroc_auc_score failed for "
                            f"{eval_task.db_name}/{eval_task.table_name}/"
                            f"{eval_task.split} ctx={eval_ctx_size}: "
                            f"{type(e).__name__}: {e} | "
                            f"n={len(labels_int)} n_classes={n_classes} "
                            f"n_nan_labels={n_nan_labels} "
                            f"n_nan_preds={n_nan_preds} "
                            f"→ falling back to AUC=0\033[0m"
                        )
                        metric = 0.0
                    all_auc_scores[prefix][(eval_ctx_size, eval_task.split)].append(
                        metric
                    )
                    metric_str = f"{metric * 100:<6.1f}"

                short_db = eval_task.db_name.split("/")[-1].split("-")[1]
                tqdm.write(
                    f"  {f'{prefix}{short_db}/{eval_task.table_name}/{eval_task.split}':<30}"
                    f"ctx: {eval_ctx_size:<5}   "
                    f"{metric_name}: \033[1m{metric_str}\033[0m  "
                    f"mean_labels: \033[1m{task_mean_labels:<5.1f}\033[0m"
                )
                all_metrics[prefix][eval_task.split][eval_ctx_size][
                    (eval_task.db_name, eval_task.table_name)
                ] = metric
                all_metrics[prefix][eval_task.split][eval_ctx_size][
                    (eval_task.db_name, eval_task.table_name, "mean_labels")
                ] = task_mean_labels

        if last_task is not None:
            outer_pbar.update(1)
        outer_pbar.close()

        if global_rank == 0:
            for _, prefix in nets_with_prefix:
                for split in self.eval_splits:
                    for eval_ctx_size in eval_ctx_sizes_to_use:
                        def _avg(xs):
                            # Single-task-type recipes (per-task fine-tuning)
                            # have no scores for the other type.
                            return sum(xs) / len(xs) if xs else float("nan")

                        avg_reg = _avg(all_reg_scores[prefix][(eval_ctx_size, split)])
                        avg_auc = _avg(all_auc_scores[prefix][(eval_ctx_size, split)])
                        wandb.log(
                            {
                                f"{prefix}{avg_reg_key}/{split}/{eval_ctx_size}": avg_reg,
                                f"{prefix}avg_auc/{split}/{eval_ctx_size}": avg_auc,
                            },
                            step=steps,
                        )
                        avg_mean_labels_reg = _avg(
                            all_mean_labels_reg[(eval_ctx_size, split)]
                        )
                        avg_mean_labels_clf = _avg(
                            all_mean_labels_clf[(eval_ctx_size, split)]
                        )
                        all_metrics[prefix][split][eval_ctx_size][avg_reg_key] = avg_reg
                        all_metrics[prefix][split][eval_ctx_size]["avg_auc"] = avg_auc
                        all_metrics[prefix][split][eval_ctx_size][
                            "avg_mean_labels_reg"
                        ] = avg_mean_labels_reg
                        all_metrics[prefix][split][eval_ctx_size][
                            "avg_mean_labels_clf"
                        ] = avg_mean_labels_clf
                        tqdm.write(
                            f"  {f'{prefix}avg/{split}':<30}"
                            f"ctx: {eval_ctx_size:<7}"
                            f"{reg_metric}: \033[1m{avg_reg:<6.4f}\033[0m  "
                            f"auc: \033[1m{avg_auc * 100:<5.1f}\033[0m  "
                            f"mean_labels_reg: \033[1m{avg_mean_labels_reg:<5.1f}\033[0m  "
                            f"mean_labels_clf: \033[1m{avg_mean_labels_clf:<5.1f}\033[0m"
                        )

        if global_rank == 0:
            tasks_by_split: dict[str, list] = {s: [] for s in self.eval_splits}
            for t in self.tasks:
                tasks_by_split[t.split].append(t)
            for _, prefix in nets_with_prefix:
                for split in self.eval_splits:
                    for eval_ctx_size in eval_ctx_sizes_to_use:
                        payload = {
                            "ctx_size": eval_ctx_size,
                            f"{prefix}ctx_scaling/steps={steps}/{split}/{avg_reg_key}": all_metrics[
                                prefix
                            ][split][eval_ctx_size][avg_reg_key],
                            f"{prefix}ctx_scaling/steps={steps}/{split}/avg_auc": all_metrics[
                                prefix
                            ][split][eval_ctx_size]["avg_auc"],
                            f"{prefix}ctx_scaling/steps={steps}/{split}/avg_mean_labels_reg": all_metrics[
                                prefix
                            ][split][eval_ctx_size]["avg_mean_labels_reg"],
                            f"{prefix}ctx_scaling/steps={steps}/{split}/avg_mean_labels_clf": all_metrics[
                                prefix
                            ][split][eval_ctx_size]["avg_mean_labels_clf"],
                        }
                        for t in tasks_by_split[split]:
                            metric_name = reg_metric if t.task_type == "reg" else "auc"
                            base = (
                                f"per_task/{prefix}ctx_scaling/steps={steps}/"
                                f"{t.db_name}/{t.table_name}/{split}"
                            )
                            payload[f"{base}/{metric_name}"] = all_metrics[prefix][
                                split
                            ][eval_ctx_size][(t.db_name, t.table_name)]
                            payload[f"{base}/mean_labels"] = all_metrics[prefix][split][
                                eval_ctx_size
                            ][(t.db_name, t.table_name, "mean_labels")]
                        wandb.log(payload)

        if local_rank == 0:
            tqdm.write(
                f"  eval done in \033[1m{fmt_duration(time.time() - eval_tic)}\033[0m"
            )
        return all_metrics
