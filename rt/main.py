import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))

import os
import json

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.utils import get_total_norm, clip_grads_with_norm_
import wandb
from sklearn.metrics import roc_auc_score, r2_score
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import FineTuneDataset
from rt.model import Transformer

import os
import random
import numpy as np


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# TODO: add type hints and make sure that strictfire validates them
def main(
    # data
    pairs=[
        # clf
        # ("rel-amazon", "user-churn"),
        # ("rel-hm", "user-churn"),
        # ("rel-stack", "user-badge"),
        # ("rel-amazon", "item-churn"),
        # ("rel-stack", "user-engagement"),
        # ("rel-avito", "user-visits"),
        # ("rel-avito", "user-clicks"),
        # ("rel-event", "user-ignore"),
        # ("rel-trial", "study-outcome"),
        ("rel-f1", "driver-dnf"),
        # ("rel-event", "user-repeat"),
        # ("rel-f1", "driver-top3"),
        # reg
        # ("rel-hm", "item-sales"),
        # ("rel-amazon", "user-ltv"),
        # ("rel-amazon", "item-ltv"),
        # ("rel-stack", "post-votes"),
        # ("rel-trial", "site-success"),
        # ("rel-trial", "study-adverse"),
        # ("rel-event", "user-attendance"),
        # ("rel-f1", "driver-position"),
        # ("rel-avito", "ad-ctr"),
    ],
    mask_prob=0.5,
    zero_mask_prob_steps=100,
    mask_db_cells=False,
    mask_task_cells=True,
    fake_names=False,
    batch_size=32,
    num_workers=8,
    subsample_p2f_edges=256,
    isolate_task_tables=False,
    # misc
    date="2025-07-01_dev",
    profile=False,
    eval_splits=[],
    eval_freq=None,
    log_eval=False,
    ckpt_freq=None,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    # optimization
    lr=1e-3,
    lr_schedule=True,
    max_grad_norm=1.0,
    max_steps=1_001,
    # model
    embedding_model="stsb-distilroberta-base-v2",
    d_text=768,
    seq_len=1024,
    num_layers=12,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    loss="huber",
    compile_=True,
    seed=0,
):
    seed_everything(seed)

    ddp = "LOCAL_RANK" in os.environ
    device = "cuda"
    if ddp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    if ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        run = wandb.init(project=date, config=locals())
        print(run.name)

    # torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch._dynamo.config.cache_size_limit = 16
    torch._dynamo.config.compiled_autograd = compile_
    # torch._dynamo.config.optimize_ddp = "python_reducer"
    torch._dynamo.config.optimize_ddp = True
    torch.set_num_threads(1)

    dataset = FineTuneDataset(
        tasks=[(dataset_name, task_name, "train") for dataset_name, task_name in pairs],
        batch_size=batch_size,
        seq_len=seq_len,
        mask_prob=mask_prob,
        rank=rank,
        world_size=world_size,
        fake_names=fake_names,
        subsample_p2f_edges=subsample_p2f_edges,
        isolate_task_tables=isolate_task_tables,
        cos_steps=max_steps - zero_mask_prob_steps,
        embedding_model=embedding_model,
        d_text=d_text,
        mask_db_cells=mask_db_cells,
        mask_task_cells=mask_task_cells,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        in_order=False,
    )

    eval_loaders = {}
    for dataset_name, task_name in pairs:
        for split in eval_splits:
            eval_dataset = FineTuneDataset(
                tasks=[(dataset_name, task_name, split)],
                batch_size=batch_size,
                seq_len=seq_len,
                mask_prob=0.0,
                rank=rank,
                world_size=world_size,
                fake_names=fake_names,
                subsample_p2f_edges=subsample_p2f_edges,
                isolate_task_tables=isolate_task_tables,
                cos_steps=0,
                embedding_model=embedding_model,
                d_text=d_text,
                mask_db_cells=False,
                mask_task_cells=True,
                seed=0,
            )
            eval_loaders[(dataset_name, task_name, split)] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                persistent_workers=True,
                pin_memory=True,
                in_order=False,
            )

    net = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        d_text=d_text,
        num_heads=num_heads,
        d_ff=d_ff,
        loss=loss,
    )
    if load_ckpt_path is not None:
        load_ckpt_path = Path(load_ckpt_path).expanduser()
        net.load_state_dict(torch.load(load_ckpt_path, map_location="cpu"))

    if rank == 0:
        param_count = sum(p.numel() for p in net.parameters())
        print(f"{param_count=:_}")

    net = net.to(device)
    net = net.to(torch.bfloat16)
    if ddp:
        net = DDP(net)
    net = torch.compile(
        net,
        # fullgraph=True,
        # dynamic=False,
        disable=not compile_,
    )
    opt = optim.AdamW(net.parameters(), lr=lr, fused=True)
    if lr_schedule:
        lrs = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=max_steps,
            pct_start=0.2,
            anneal_strategy="linear",
        )

    steps = 0
    if rank == 0:
        wandb.log({"epochs": 0}, step=steps)

    with open(f"{os.environ['HOME']}/scratch/pre/reg_std.json") as f:
        std_dict = json.load(f)

    def evaluate():
        net.eval()
        with torch.inference_mode():
            for (
                dataset_name,
                task_name,
                split,
            ), eval_loader in eval_loaders.items():
                preds = []
                labels = []
                losses = []
                for batch in tqdm(eval_loader, desc=split, disable=rank != 0):
                    true_batch_size = batch.pop("true_batch_size")
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=True)

                    batch["masks"][true_batch_size:, :] = False
                    batch["is_targets"][true_batch_size:, :] = False

                    loss, yhat = net(batch)

                    yhat = yhat[batch["is_targets"], :]
                    y = batch["number_values"][batch["is_targets"], :]
                    assert yhat.size(0) == true_batch_size
                    assert y.size(0) == true_batch_size

                    losses.append(loss)
                    preds.append(yhat.squeeze(-1).tolist())
                    labels.append(y.squeeze(-1).tolist())

                losses = [x.item() for x in losses]
                preds = sum(preds, [])
                labels = sum(labels, [])

                if ddp:
                    labels_list = [None] * dist.get_world_size()
                    dist.all_gather_object(labels_list, labels)

                    preds_list = [None] * dist.get_world_size()
                    dist.all_gather_object(preds_list, preds)
                else:
                    labels_list = [labels]
                    preds_list = [preds]

                if rank == 0:
                    loss = sum(losses) / len(losses)
                    k = f"loss/{dataset_name}/{task_name}/{split}"
                    wandb.log({k: loss}, step=steps)

                    labels = sum(labels_list, [])
                    preds = sum(preds_list, [])
                    if task_name in [
                        "item-sales",
                        "user-ltv",
                        "item-ltv",
                        "post-votes",
                        "site-success",
                        "study-adverse",
                        "user-attendance",
                        "driver-position",
                        "ad-ctr",
                    ]:
                        # metric_name = "mae"
                        # std = std_dict[f"{dataset_name}/{task_name}"]
                        # metric = mean_absolute_error(labels, preds) * std
                        metric_name = "r2"
                        metric = r2_score(labels, preds)
                    else:
                        metric_name = "auc"
                        labels = [int(x) for x in labels]
                        metric = roc_auc_score(labels, preds)
                    k = f"{metric_name}/{dataset_name}/{task_name}/{split}"
                    wandb.log({k: metric}, step=steps)
                    print(f"step={steps}, \t{k}: {metric}")

    def checkpoint():
        if rank != 0:
            return
        save_ckpt_dir_ = Path(save_ckpt_dir).expanduser()
        save_ckpt_dir_.mkdir(parents=True, exist_ok=True)
        save_ckpt_path = f"{save_ckpt_dir_}/{steps=}.pt"
        state_dict = net.module.state_dict() if ddp else net.state_dict()
        torch.save(state_dict, save_ckpt_path)
        print(f"saved checkpoint to {save_ckpt_path}")

    pbar = tqdm(
        total=max_steps,
        desc="steps",
        disable=rank != 0,
    )
    while steps < max_steps:
        loader.dataset.sampler.shuffle_py(int(steps / len(loader)))
        loader_iter = iter(loader)
        while steps < max_steps:
            if eval_freq is not None and steps % eval_freq == 0:
                evaluate()
            if log_eval and steps & (steps - 1) == 0:
                evaluate()
            if ckpt_freq is not None and steps % ckpt_freq == 0:
                checkpoint()

            net.train()

            tic = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch.pop("true_batch_size")
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            toc = time.time()
            load_time = toc - tic
            if rank == 0:
                wandb.log({"load_time": load_time}, step=steps)

            loss, _yhat = net(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = get_total_norm(
                [p.grad for p in net.parameters() if p.grad is not None]
            )
            clip_grads_with_norm_(
                net.parameters(), max_norm=max_grad_norm, total_norm=grad_norm
            )

            opt.step()
            if lr_schedule:
                lrs.step()

            steps += 1

            if ddp:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                # TODO: calculate auc of the batch
                # (maybe only once in a while if it hurts training throughput)
                wandb.log(
                    {
                        "loss": loss,
                        "lr": opt.param_groups[0]["lr"],
                        "mask_frac": batch["masks"].float().mean(),
                        "epochs": steps / len(loader),
                        "grad_norm": grad_norm,
                    },
                    step=steps,
                )

            if profile and steps != 0:
                prof.step()  # noqa FIXME

            pbar.update(1)

            if profile and steps == 1:
                prof = torch.profiler.profile(with_modules=True)
                prof.__enter__()

    if profile:
        prof.__exit__(None, None, None)
        if rank == 0:
            prof.export_chrome_trace("trace.json")
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import strictfire

    strictfire.StrictFire(main)
