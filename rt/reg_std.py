import json
import os

from relbench.base import TaskType
from relbench.datasets import get_dataset_names
from relbench.tasks import get_task, get_task_names


def main():
    std_dict = {}
    for dataset_name in get_dataset_names():
        for task_name in get_task_names(dataset_name):
            task = get_task(dataset_name, task_name)
            if task.task_type != TaskType.REGRESSION:
                continue
            table = task.get_table("train")
            std = table.df[task.target_col].std().item()
            std_dict[f"{dataset_name}/{task_name}"] = std
            print(f"{dataset_name}\t {task_name}\t {std:.4f}")
    path = f"{os.environ['HOME']}/scratch/pre/reg_std.json"
    with open(path, "w") as f:
        json.dump(std_dict, f, indent=2)


if __name__ == "__main__":
    main()
