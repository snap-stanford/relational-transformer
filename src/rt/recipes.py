"""Named task recipes -- a thin dispatch over :mod:`rt.tasks`.

A recipe maps a name to a builder that takes a preprocessed ``pre_dir`` (local
path or Hub repo) and returns a list of tasks. Recipes are intentionally tiny:
the actual task set lives in the preprocessed data, not in code.
"""

from __future__ import annotations

from typing import Callable

from rt.tasks import Task, eval_tasks, pretrain_tasks

REGISTRY: dict[str, Callable[[str], list[Task]]] = {
    # Pretraining: train-split tasks over every preprocessed dataset (the Join).
    "pretrain": pretrain_tasks,
    # Evaluation: the benchmark tasks (point pre_dir at RelBench-preprocessed).
    "relbench_eval": lambda pre_dir: eval_tasks(pre_dir, splits=("val", "test")),
    "relbench_eval_val": lambda pre_dir: eval_tasks(pre_dir, splits=("val",)),
    "relbench_eval_test": lambda pre_dir: eval_tasks(pre_dir, splits=("test",)),
}


def get_tasks(recipe: str, pre_dir: str) -> list[Task]:
    if recipe not in REGISTRY:
        raise ValueError(f"unknown recipe {recipe!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[recipe](pre_dir)
