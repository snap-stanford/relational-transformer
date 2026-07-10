"""Shared evaluation engine for scripts/eval.py and scripts/baseline.py.

Builds an :class:`rt.evaluator.Evaluator` over a task list, runs single-process
inference for a model that implements ``.predict(batch, ctx_sizes, device, task,
bool_as_num)`` (RT or a rel2tab baseline), and reports metrics **through
RelBench's own leaderboard evaluator**.

For each task it:

1. gathers per-row predictions plus each row's *seed node index* from the
   evaluator (``evaluate_raw(..., with_node_idxs=True)``);
2. maps every prediction back to its relbench task-table row via
   ``row = node_idx - node_idx_offset`` (eval row order is **not** the parquet
   row order, so a positional join would be wrong);
3. puts predictions on the relbench submission scale -- regression is
   *denormalized* with the train-split target stats rustler normalized with
   (``y = pred * std + mean``, ``std`` ddof=1), classification logits are
   passed through a sigmoid to a probability in ``[0, 1]``;
4. writes a relbench prediction-table CSV ``<dataset>__<task>.csv`` keyed by
   ``(entity_col, time_col)`` into ``out_dir`` (a valid leaderboard submission
   directory, checkable with ``python -m relbench.leaderboard <out_dir>``); and
5. scores it with :func:`relbench.leaderboard.evaluate_task` (AUROC for clf,
   NMAE for reg) against the same relbench dataset the data was preprocessed
   from (``meta.json``'s ``source``).
"""

from __future__ import annotations

import json
import tempfile
from functools import cache
from pathlib import Path

from rt.pre import read_meta, resolve_pre_dir


def metric_for(task_type: str, labels, preds, reg_metric: str = "mae") -> tuple[str, float]:
    """rt-internal metric on the *normalized* scale (val-set tuning + debug).

    Kept for ensemble context-tuning on the validation split (a relative
    comparison, scale-invariant) and as a labeled debug number alongside the
    authoritative RelBench metric. Not the submission metric.
    """
    import sklearn.metrics as M

    if task_type == "reg":
        if reg_metric == "r2":
            return "r2", float(M.r2_score(labels, preds))
        return "mae", float(M.mean_absolute_error(labels, preds))
    return "roc_auc", float(M.roc_auc_score((labels > 0).astype(int), preds))


# --------------------------------------------------------------------------- #
# RelBench submission: denormalize / sigmoid, key by node index, score.
# --------------------------------------------------------------------------- #
def _relbench():
    try:
        import relbench  # noqa: F401
        from relbench.leaderboard import evaluate_task

        return relbench, evaluate_task
    except Exception as e:  # pragma: no cover - import-time guidance
        raise RuntimeError(
            "relbench is required for evaluation. It is a declared dependency "
            "(relbench @ relbench-hf), e.g. `pixi install`."
        ) from e


@cache
def _seed_offset(pre_dir: str, db: str, table: str, split: str, embedding_model: str) -> int:
    """Global rustler ``node_idx`` of the first row of ``table``'s ``split``
    (so ``node_idx - offset`` is the relbench parquet row index)."""
    local = resolve_pre_dir(pre_dir, [db], embedding_model)
    ti = json.loads((Path(local) / db / "table_info.json").read_text())
    split_cap = {"train": "Train", "val": "Val", "test": "Test"}.get(split, split.capitalize())
    key = f"{table}:Db" if f"{table}:Db" in ti else f"{table}:{split_cap}"
    return int(ti[key]["node_idx_offset"])


@cache
def _load_relbench_task(source: str, table: str):
    relbench, _ = _relbench()
    return relbench.load_task(source, table)


def _train_stats(rtask) -> tuple[float, float]:
    """Train-split target ``(mean, std)`` -- the exact inverse of rustler's
    ``(val - mean) / std`` normalization (``std`` ddof=1, 0 -> 1.0)."""
    df = rtask.get_table("train").df
    col = rtask.target_col
    mean = float(df[col].mean())
    std = float(df[col].std(ddof=1))
    return mean, (std if std != 0.0 else 1.0)


def _emit_and_score(out_dir: Path, task, pre_dir: str, embedding_model: str,
                    labels, preds, node_idxs, *, keep_csv: bool):
    """Denormalize/sigmoid ``preds``, write a relbench prediction-table CSV keyed
    by ``(entity_col, time_col)``, and score it with relbench's evaluator.

    Returns ``(metric_name, metric_value, n, align_str, csv_path | None)``.
    ``align_str`` is a built-in alignment guard: denormalized rt labels are
    compared row-for-row to the relbench ground truth (max abs diff for reg,
    class agreement for clf) -- both should be ~perfect when the node-index
    join is correct.
    """
    import numpy as np

    relbench, evaluate_task = _relbench()

    meta = read_meta(pre_dir, task.db_name)
    source = meta.get("source")
    if not source:
        raise RuntimeError(
            f"{task.db_name}/meta.json has no 'source'; cannot locate the relbench task"
        )
    rtask = _load_relbench_task(source, task.table_name)
    offset = _seed_offset(pre_dir, task.db_name, task.table_name, task.split, embedding_model)

    node_idxs = np.asarray(node_idxs, dtype=np.int64)
    rowidx = node_idxs - offset
    masked = rtask.get_table("test", mask_input_cols=True).df.reset_index(drop=True)
    gt = rtask.get_table("test", mask_input_cols=False).df.reset_index(drop=True)
    n_test = len(masked)
    if rowidx.size and (rowidx.min() < 0 or rowidx.max() >= n_test):
        raise RuntimeError(
            f"{task.db_name}/{task.table_name}: seed row indices out of range "
            f"[{int(rowidx.min())}, {int(rowidx.max())}] vs {n_test} relbench test rows"
        )

    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    gt_vals = gt[rtask.target_col].to_numpy(dtype=np.float64)[rowidx]
    if task.task_type == "reg":
        mean, std = _train_stats(rtask)
        out_preds = preds * std + mean
        align = f"|dy|<={float(np.max(np.abs(labels * std + mean - gt_vals))):.1e}" if rowidx.size else "n/a"
    else:  # clf -> probability in [0, 1]; AUROC is invariant to the sigmoid.
        out_preds = 1.0 / (1.0 + np.exp(-preds))
        agree = float(np.mean((labels > 0).astype(int) == (gt_vals > 0).astype(int))) if rowidx.size else float("nan")
        align = f"cls={agree:.3f}"

    sub = masked.iloc[rowidx][[rtask.entity_col, rtask.time_col]].copy()
    sub[rtask.target_col] = out_preds

    if keep_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{task.db_name}__{task.table_name}.csv"
        sub.to_csv(csv_path, index=False)
        score_path, ret_path = csv_path, csv_path
    else:
        tf = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        sub.to_csv(tf.name, index=False)
        tf.close()
        score_path, ret_path = Path(tf.name), None

    metrics = evaluate_task(f"{task.db_name}/{task.table_name}", str(score_path), dataset=source)
    if ret_path is None:
        Path(score_path).unlink(missing_ok=True)
    mname, mval = next(iter(metrics.items()))
    return mname, float(mval), int(rowidx.size), align, ret_path


def build_evaluator(tasks, pre_dir, *, embedding_model, d_text, device, ctx_size=8192,
                    local_ctx_size=256, bfs_width=32, num_walks=10_000, walk_length=20,
                    tokens_per_gpu=2**18, items_per_task=None, num_workers=2, context_seed=0,
                    prefer_latest=True, shuffle_seed=0, mmap_populate=True):
    from rt.evaluator import Evaluator

    # mmap_populate=True by default: pre-fault the eval data into RAM so the
    # context build is fed instead of cold-faulting it from shared storage per
    # item (the same starvation that hits training without it).
    #
    # prefer_latest / shuffle_seed are real knobs (not constants): prefer_latest
    # picks the same-table neighbor sort (recency vs frequency); shuffle_seed
    # fixes the val/test subset selection + item shuffle, so an --items-per-task
    # subsample stays the *same* rows across configs (context tuning, ensembling).
    return Evaluator(
        tasks=tasks, pre_dir=pre_dir, eval_bs=max(1, tokens_per_gpu // ctx_size),
        ctx_sizes=[ctx_size], items_per_task=items_per_task, num_workers=num_workers,
        prefetch_factor=2, persistent_workers=False, local_ctx_size=local_ctx_size,
        bfs_width=bfs_width, num_walks=num_walks, walk_length=walk_length,
        prefer_latest=prefer_latest, bool_as_num=True, skip_text_cols=False,
        mmap_populate=mmap_populate, balance_labels=False,
        ablate_schema_semantics=False, embedding_model=embedding_model, d_text=d_text,
        shuffle_seed=shuffle_seed, context_seed=context_seed, vector_db_path=None,
        train_only_fallback=False,
        global_rank=0, local_rank=0, world_size=1, ddp=False, device=device,
    )


def run_and_report(model, tasks, pre_dir, *, ctx_size, reg_metric, out_dir, no_csv,
                   evaluator, embedding_model):
    """Run inference, write relbench submission CSVs, score via relbench's
    evaluator, print per-task + mean metrics. Returns a results dict."""
    out_dir = Path(out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}  debug")
    for task, _ctx, labels, preds_by_prefix, _nl, node_idxs in evaluator.evaluate_raw(
        [(model, "")], [ctx_size], with_node_idxs=True
    ):
        preds = preds_by_prefix[""]
        mname, mval, n, align, _ = _emit_and_score(
            out_dir, task, pre_dir, embedding_model, labels, preds, node_idxs,
            keep_csv=not no_csv,
        )
        nm, nv = metric_for(task.task_type, labels, preds, reg_metric)  # normalized-scale debug
        by_metric.setdefault(mname, []).append(mval)
        results[f"{task.db_name}/{task.table_name}"] = {"metric": mname, "value": mval, "n": n}
        print(f"{task.db_name + '/' + task.table_name:40} {mname:8} {mval:>9.4f} {n:>7}  "
              f"{align:>11}  norm[{nm}]={nv:.4f}")
    print(f"\n{'mean':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    if not no_csv:
        print(f"\nsubmission CSVs written to {out_dir}/  "
              f"(validate: python -m relbench.leaderboard {out_dir})")
    return results


def _is_better(task_type, a, b):
    return a > b if task_type == "clf" else a < b  # higher auc / lower mae


def run_ensemble(model, pre_dir, val_tasks, test_tasks, *, grid, ensemble_size, ctx_size,
                 reg_metric, out_dir, no_csv, **eval_kwargs):
    """Context-tuned + ensembled evaluation.

    Tune: for each task, pick the (local_ctx_size, bfs_width) in ``grid`` with the
    best *validation* metric. Ensemble: on test, run that config with
    ``ensemble_size`` context seeds and average the per-item predictions, then
    score the averaged submission through relbench's evaluator.
    """
    from collections import defaultdict

    import numpy as np

    embedding_model = eval_kwargs["embedding_model"]

    # ---- tune on val: best context config per task ----
    best = {}  # (db, table) -> {"cfg", "value", "task_type"}
    for cfg in grid:
        lcs, bw = cfg
        ev = build_evaluator(val_tasks, pre_dir, ctx_size=ctx_size, local_ctx_size=lcs,
                             bfs_width=bw, **eval_kwargs)
        for task, _c, labels, preds_by_prefix, _nl in ev.evaluate_raw([(model, "")], [ctx_size]):
            _, v = metric_for(task.task_type, labels, preds_by_prefix[""], reg_metric)
            key = (task.db_name, task.table_name)
            if key not in best or _is_better(task.task_type, v, best[key]["value"]):
                best[key] = {"cfg": cfg, "value": v, "task_type": task.task_type}
            print(f"  tune {task.db_name}/{task.table_name} cfg={cfg}: {v:.4f}")

    # ---- ensemble on test: best config per task, averaged over context seeds ----
    groups = defaultdict(list)
    for t in test_tasks:
        b = best.get((t.db_name, t.table_name))
        if b is not None:
            groups[b["cfg"]].append(t)

    out_dir = Path(out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'cfg':14} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}")
    for cfg, tasks in groups.items():
        lcs, bw = cfg
        acc = {}  # key -> [labels, sum_preds, task, node_idxs]
        for seed in range(ensemble_size):
            ev = build_evaluator(tasks, pre_dir, ctx_size=ctx_size, local_ctx_size=lcs,
                                 bfs_width=bw, context_seed=seed, **eval_kwargs)
            for task, _c, labels, preds_by_prefix, _nl, node_idxs in ev.evaluate_raw(
                [(model, "")], [ctx_size], with_node_idxs=True
            ):
                key = (task.db_name, task.table_name)
                p = preds_by_prefix[""].astype(np.float64)
                if key not in acc:
                    acc[key] = [labels, np.zeros_like(p), task, node_idxs]
                acc[key][1] += p
        for key, (labels, sp, task, node_idxs) in acc.items():
            preds = sp / ensemble_size
            mname, mval, n, align, _ = _emit_and_score(
                out_dir, task, pre_dir, embedding_model, labels, preds, node_idxs,
                keep_csv=not no_csv,
            )
            by_metric.setdefault(mname, []).append(mval)
            results[f"{task.db_name}/{task.table_name}"] = {"metric": mname, "value": mval,
                                                            "cfg": cfg, "n": n}
            print(f"{task.db_name + '/' + task.table_name:40} {str(cfg):14} {mname:8} "
                  f"{mval:>9.4f} {n:>7}  {align:>11}")
    print(f"\n{'mean (ensembled)':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    if not no_csv:
        print(f"\nsubmission CSVs written to {out_dir}/  "
              f"(validate: python -m relbench.leaderboard {out_dir})")
    return results
