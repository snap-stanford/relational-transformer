from relbench.datasets import get_dataset_names
from relbench.tasks import get_task_names, get_task

if __name__ == "__main__":
    for dataset_name in get_dataset_names():
        for task_name in get_task_names(dataset_name):
            print(f"Downloading task: {task_name} from dataset: {dataset_name}")
            get_task(dataset_name, task_name, download=True)
