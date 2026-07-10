"""RT task lists, built straight from preprocessed datasets (local dir or Hub repo).

A *task* is one ``(database, task-table, target-column)`` prediction problem at a
given split. Two kinds of task are surfaced:

* **Forecast tasks** -- the explicit, time-split prediction tables shipped under a
  dataset's ``tasks/`` dir and recorded in ``meta.json``'s ``tasks`` list.
* **Autocomplete tasks** -- schema-derived "predict a masked column" tasks. For
  every database (forecast or not) we emit, per table per feature column, a clf
  task for boolean columns and a reg task for numeric columns. No data is
  materialized: the "label" is an existing preprocessed column. This recovers the
  schema-only autocomplete pretraining mixture the original pipeline used (most
  ``the-join`` databases ship *only* DB tables, no ``tasks/`` dir).

Feature columns are exactly the non-foreign-key, non-primary-key, non-time,
non-text columns: rustler ``pre`` already emits no cell for foreign (and, in
relbench-3.0.0, primary) keys, so :func:`rustler.column_sem_types` (which reads
sem-types straight off the preprocessed nodes) excludes them automatically; any
primary-key/time columns that do survive are pruned here using the source
``manifest.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from rt.pre import list_datasets, read_meta

# relbench task_type -> RT task_type. Only node-level clf/reg tasks are modeled;
# link_prediction (recommendation) tasks are skipped.
_TASK_TYPE = {"binary_classification": "clf", "regression": "reg"}

# autocomplete: which sem-type becomes which task. Text/DateTime are not targets.
_SEM_TASK_TYPE = {"Boolean": "clf", "Number": "reg"}


@dataclass(frozen=True)
class Task:
    db_name: str
    table_name: str
    target_column: str
    task_type: str  # "clf" | "reg"
    split: str = ""  # "train" | "val" | "test"
    leakage_columns: tuple[str, ...] = ()


@cache
def _pkey_time_cols(pre_dir: str, db: str) -> frozenset[str]:
    """Primary-key / time columns to exclude as autocomplete targets, read from
    a ``manifest.yaml`` sitting next to the preprocessed db if one is present.
    Keyed as ``"<col> of <table>"`` to match :func:`rustler.column_sem_types`.

    This is a redundant safety net, not a hard dependency: relbench-3.0.0
    reindexes foreign *and* primary keys to row indices, and rustler ``pre``
    emits no cell for those, so :func:`rustler.column_sem_types` already excludes
    them; time columns surface as the ``DateTime`` sem-type, which is not a clf/
    reg target. We therefore never fetch over the network (that would hang the
    pretraining-task enumeration on compute nodes) -- only a local manifest is
    consulted, and its absence is fine.
    """
    from pathlib import Path

    p = Path(pre_dir).expanduser()
    manifest_path = p / db / "manifest.yaml"
    if not manifest_path.exists():
        return frozenset()
    try:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text())
        excl: set[str] = set()
        for table, info in (manifest.get("tables") or {}).items():
            for key in ("pkey", "time_col"):
                col = info.get(key)
                if col:
                    excl.add(f"{col} of {table}")
        return frozenset(excl)
    except Exception:
        return frozenset()


def _autocomplete_tasks(pre_dir: str, db: str, splits, task_types) -> list[Task]:
    """Schema-derived autocomplete tasks for one database.

    Emitted at the ``train`` split only (autocomplete is a pretraining-only
    signal; there is no held-out forecast horizon). The target column is masked
    and predicted from the rest of its row's context, so no leakage columns are
    needed.
    """
    if "train" not in splits:
        return []
    from rt._rustler import column_sem_types

    sem = column_sem_types(pre_dir, db)
    excluded = _pkey_time_cols(pre_dir, db)
    out: list[Task] = []
    for col_of_table, sem_type in sem.items():
        tt = _SEM_TASK_TYPE.get(sem_type)
        if tt is None or tt not in task_types or col_of_table in excluded:
            continue
        col, table = col_of_table.rsplit(" of ", 1)
        out.append(Task(db, table, col, tt, "train"))
    return out


def tasks_from_preprocessed(
    pre_dir: str,
    *,
    splits,
    task_types=("clf", "reg"),
    dbs=None,
) -> list[Task]:
    """Tasks across the preprocessed datasets under ``pre_dir`` for the given splits.

    A database with explicit forecast tasks (a non-empty ``meta.json`` ``tasks``
    list) contributes those; a database without any (DB-tables-only, the common
    ``the-join`` case) contributes schema-derived autocomplete tasks instead.
    """
    out: list[Task] = []
    for db in dbs if dbs is not None else list_datasets(pre_dir):
        meta = read_meta(pre_dir, db)
        explicit = [
            t
            for t in meta.get("tasks", [])
            if _TASK_TYPE.get(t.get("task_type")) and t.get("target_col")
        ]
        if explicit:
            for t in explicit:
                tt = _TASK_TYPE[t["task_type"]]
                if tt not in task_types:
                    continue
                for split in splits:
                    if split in t.get("splits", []):
                        out.append(Task(db, t["name"], t["target_col"], tt, split))
        else:
            out.extend(_autocomplete_tasks(pre_dir, db, splits, task_types))
    return out


def pretrain_tasks(pre_dir: str, *, dbs=None) -> list[Task]:
    """Train-split tasks across every preprocessed dataset (the pretraining mixture)."""
    return tasks_from_preprocessed(pre_dir, splits=("train",), dbs=dbs)


# The curated RelBench evaluation benchmark used by the released runs: 12 clf +
# 9 reg forecasting tasks (the "relbench_eval_w_event" set from the original
# repo). We evaluate on exactly these, not every explicit relbench-pre task, so
# the eval matches the released numbers and stays cheap (~21 vs ~34 tasks). Each
# entry is (db, task_table, target_column, task_type).
RELBENCH_EVAL_TASKS: tuple[tuple[str, str, str, str], ...] = (
    # clf
    ("rel-amazon", "user-churn", "churn", "clf"),
    ("rel-hm", "user-churn", "churn", "clf"),
    ("rel-stack", "user-badge", "WillGetBadge", "clf"),
    ("rel-amazon", "item-churn", "churn", "clf"),
    ("rel-stack", "user-engagement", "contribution", "clf"),
    ("rel-avito", "user-visits", "num_click", "clf"),
    ("rel-avito", "user-clicks", "num_click", "clf"),
    ("rel-event", "user-ignore", "target", "clf"),
    ("rel-trial", "study-outcome", "outcome", "clf"),
    ("rel-f1", "driver-dnf", "did_not_finish", "clf"),
    ("rel-event", "user-repeat", "target", "clf"),
    ("rel-f1", "driver-top3", "qualifying", "clf"),
    # reg
    ("rel-hm", "item-sales", "sales", "reg"),
    ("rel-amazon", "user-ltv", "ltv", "reg"),
    ("rel-amazon", "item-ltv", "ltv", "reg"),
    ("rel-stack", "post-votes", "popularity", "reg"),
    ("rel-trial", "site-success", "success_rate", "reg"),
    ("rel-trial", "study-adverse", "num_of_adverse_events", "reg"),
    ("rel-event", "user-attendance", "target", "reg"),
    ("rel-f1", "driver-position", "position", "reg"),
    ("rel-avito", "ad-ctr", "num_click", "reg"),
)


def eval_tasks(pre_dir: str, *, splits=("val", "test"), dbs=None) -> list[Task]:
    """The curated RelBench benchmark (:data:`RELBENCH_EVAL_TASKS`) at the given
    splits -- exactly the 21 forecasting tasks the released runs evaluated on,
    not an enumerate-all over the preprocessed eval datasets."""
    return [
        Task(db, table, col, tt, split)
        for (db, table, col, tt) in RELBENCH_EVAL_TASKS
        if dbs is None or db in dbs
        for split in splits
    ]
