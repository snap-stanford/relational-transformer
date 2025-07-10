import wandb
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from roach.store import store
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm
from torch import nn

from .model import Model
from .text_embedder import GloveTextEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--date", type=str, default="2025-06-25_dev")
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-dnf")
parser.add_argument("--eval_freq", type=int, default=0)
parser.add_argument("--log_eval", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_schedule", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=2**14 + 1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/scratch/relbench_cache"),
)
args = parser.parse_args()

store.init(f"/dfs/user/ranjanr/roach/stores/{args.date}")
args.script_name = "gnn_node"
store.save(args.__dict__, "args")

run = wandb.init(project=args.date, config=args.__dict__)
print(run.name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)


stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]

    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=split == "train",
    )


def count_params(net):
    out = {"emb": 0, "non_emb": 0}
    for module in net.modules():
        if list(module.children()):
            continue
        k = "emb" if isinstance(module, nn.Embedding) else "non_emb"
        out[k] += sum(p.numel() for p in module.parameters())
    return out


steps = 0


def train() -> float:
    global steps

    model.train()

    loss_accum = count_accum = 0
    total_steps = len(loader_dict["train"])
    for batch in tqdm(loader_dict["train"], total=total_steps):
        if steps >= args.max_steps:
            break

        if args.eval_freq and steps % args.eval_freq == 0:
            evaluate()
        if args.log_eval and steps & (steps - 1) == 0:
            evaluate()

        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()
        if args.lr_schedule:
            lrs.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        wandb.log(
            {
                "epochs": steps / total_steps,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": loss.item(),
            }
        )
        steps += 1

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
).to(device)
print(f"Model params: {count_params(model)}")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.lr_schedule:
    lrs = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max_steps,
        pct_start=0.2,
        anneal_strategy="linear",
    )
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
while steps < args.max_steps:

    def evaluate():
        model.eval()
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        if task.task_type == TaskType.REGRESSION:
            metric_name = "r2"
            metric_key = "r2"
        elif task.task_type == TaskType.BINARY_CLASSIFICATION:
            metric_name = "auc"
            metric_key = "roc_auc"
        k = f"{metric_name}/{args.dataset}/{args.task}/val"
        metric = float(val_metrics[metric_key])
        log_dict = {k: metric}
        print(log_dict)
        wandb.log(log_dict, step=steps)
        store.log("steps", steps)
        store.log("epochs", steps / len(loader_dict["train"]))
        store.log(k, metric)
        model.train()

    train_loss = train()
