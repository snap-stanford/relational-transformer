from relbench.datasets import get_dataset_names, get_dataset

if __name__ == "__main__":
    for dataset_name in get_dataset_names():
        print(f"Downloading dataset: {dataset_name}")
        get_dataset(dataset_name, download=True)
