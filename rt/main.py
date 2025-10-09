import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from sklearn.metrics import r2_score, roc_auc_score
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import RelationalDataset
from rt.model import RelationalTransformer


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def all_gather_nd(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.
    Adapted from: https://stackoverflow.com/a/71433508

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        list[Tensor]: List of tensors gathered from all processes.
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[: size[0]])
    return all_tensors


def main(
    # misc
    project,
    eval_splits,
    eval_freq,
    eval_pow2,
    max_eval_steps,
    load_ckpt_path,
    save_ckpt_dir,
    compile_,
    seed,
    # data
    train_tasks,
    eval_tasks,
    batch_size,
    num_workers,
    max_bfs_width,
    # optimization
    lr,
    wd,
    lr_schedule,
    max_grad_norm,
    max_steps,
    # model
    embedding_model,
    d_text,
    seq_len,
    num_blocks,
    d_model,
    num_heads,
    d_ff,
):
    seed_everything(seed)

    ddp = "LOCAL_RANK" in os.environ
    device = "cuda"
    if ddp:
        os.environ["OMP_NUM_THREADS"] = f"{num_workers}"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    if ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        run = wandb.init(project=project, config=locals())
        print(run.name)

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch._dynamo.config.cache_size_limit = 16
    torch._dynamo.config.compiled_autograd = compile_ if ddp else False
    torch._dynamo.config.optimize_ddp = True
    torch.set_num_threads(1)

    dataset = RelationalDataset(
        tasks=[
            (db_name, table_name, target_column, "train", columns_to_drop)
            for db_name, table_name, target_column, columns_to_drop in train_tasks
        ],
        batch_size=batch_size,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        max_bfs_width=max_bfs_width,
        embedding_model=embedding_model,
        d_text=d_text,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        in_order=True,
    )

    eval_loaders = {}
    for db_name, table_name, target_column, columns_to_drop in eval_tasks:
        for split in eval_splits:
            eval_dataset = RelationalDataset(
                tasks=[(db_name, table_name, target_column, split, columns_to_drop)],
                batch_size=batch_size,
                seq_len=seq_len,
                rank=rank,
                world_size=world_size,
                max_bfs_width=max_bfs_width,
                embedding_model=embedding_model,
                d_text=d_text,
                seed=0,
            )
            eval_dataset.sampler.shuffle_py(0)
            eval_loaders[(db_name, table_name, split)] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                persistent_workers=True,
                pin_memory=True,
                in_order=True,
            )

    net = RelationalTransformer(
        num_blocks=num_blocks,
        d_model=d_model,
        d_text=d_text,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    if load_ckpt_path is not None:
        load_ckpt_path = Path(load_ckpt_path).expanduser()
        state_dict = torch.load(load_ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)

    if rank == 0:
        param_count = sum(p.numel() for p in net.parameters())
        print(f"{param_count=:_}")

    net = net.to(device)
    net = net.to(torch.bfloat16)
    opt = optim.AdamW(
        net.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )

    if lr_schedule:
        lrs = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=max_steps,
            pct_start=0.2,
            anneal_strategy="linear",
        )

    if ddp:
        net = DDP(net)

    net = torch.compile(net, dynamic=False, disable=not compile_)

    steps = 0
    if rank == 0:
        wandb.log({"epochs": 0}, step=steps)

    eval_loader_iters = {}
    for k, eval_loader in eval_loaders.items():
        eval_loader_iters[k] = iter(eval_loader)

    def evaluate(net):
        metrics = {"val": {}, "test": {}}
        net.eval()
        with torch.inference_mode():
            for (
                db_name,
                table_name,
                split,
            ), eval_loader_iter in eval_loader_iters.items():
                if table_name in [
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
                    task_type = "reg"
                else:
                    task_type = "clf"

                preds = []
                labels = []
                losses = []
                eval_load_times = []
                eval_loader = eval_loaders[(db_name, table_name, split)]
                pbar = tqdm(
                    total=(
                        min(max_eval_steps, len(eval_loader))
                        if max_eval_steps > -1
                        else len(eval_loader)
                    ),
                    desc=f"{db_name}/{table_name}/{split}",
                    disable=rank != 0,
                )

                batch_idx = 0
                while True:
                    tic = time.time()
                    try:
                        batch = next(eval_loader_iter)
                        batch_idx += 1
                    except StopIteration:
                        break
                    toc = time.time()
                    pbar.update(1)

                    eval_load_time = toc - tic
                    if rank == 0:
                        eval_load_times.append(eval_load_time)

                    true_batch_size = batch.pop("true_batch_size")
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=True)

                    batch["masks"][true_batch_size:, :] = False
                    batch["is_targets"][true_batch_size:, :] = False
                    batch["is_padding"][true_batch_size:, :] = True

                    loss, yhat_dict = net(batch)

                    if task_type == "clf":
                        yhat = yhat_dict["boolean"][batch["is_targets"]]
                        y = batch["boolean_values"][batch["is_targets"]].flatten()
                    elif task_type == "reg":
                        yhat = yhat_dict["number"][batch["is_targets"]]
                        y = batch["number_values"][batch["is_targets"]].flatten()

                    assert yhat.size(0) == true_batch_size
                    assert y.size(0) == true_batch_size

                    pred = yhat.flatten()

                    losses.append(loss.item())
                    preds.append(pred)
                    labels.append(y)

                    if max_eval_steps > -1 and batch_idx >= max_eval_steps:
                        break

                eval_loader_iters[(db_name, table_name, split)] = iter(eval_loader)

                pbar.close()
                preds = torch.cat(preds, dim=0)
                labels = torch.cat(labels, dim=0)

                if ddp:
                    # ensure the predictions and labels are gathered jointly
                    preds = all_gather_nd(preds)
                    labels = all_gather_nd(labels)
                else:
                    preds = [preds]
                    labels = [labels]

                if rank == 0:
                    loss = sum(losses) / len(losses)
                    k = f"loss/{db_name}/{table_name}/{split}"
                    avg_eval_load_time = sum(eval_load_times) / len(eval_load_times)
                    wandb.log(
                        {
                            k: loss,
                            f"avg_eval_load_time/{db_name}/{table_name}": avg_eval_load_time,
                        },
                        step=steps,
                    )

                    preds = torch.cat(preds, dim=0).float().cpu().numpy()
                    labels = torch.cat(labels, dim=0).float().cpu().numpy()

                    if task_type == "reg":
                        metric_name = "r2"
                        metric = r2_score(labels, preds)
                    elif task_type == "clf":
                        metric_name = "auc"
                        labels = [int(x > 0) for x in labels]
                        metric = roc_auc_score(labels, preds)

                    k = f"{metric_name}/{db_name}/{table_name}/{split}"
                    wandb.log({k: metric}, step=steps)
                    print(f"\nstep={steps}, \t{k}: {metric}")
                    metrics[split][(db_name, table_name)] = metric

        return metrics

    def checkpoint(best=False, db_name="", table_name=""):
        if rank != 0:
            return
        save_ckpt_dir_ = Path(save_ckpt_dir).expanduser()
        save_ckpt_dir_.mkdir(parents=True, exist_ok=True)
        if best:
            save_ckpt_path = f"{save_ckpt_dir_}/{db_name}_{table_name}_best.pt"
        else:
            save_ckpt_path = f"{save_ckpt_dir_}/{steps=}.pt"

        state_dict = net.module.state_dict() if ddp else net.state_dict()
        torch.save(state_dict, save_ckpt_path)
        print(f"saved checkpoint to {save_ckpt_path}")

    pbar = tqdm(
        total=max_steps,
        desc="steps",
        disable=rank != 0,
    )

    best_val_metrics = dict()
    best_test_metrics = dict()

    while steps < max_steps:
        loader.dataset.sampler.shuffle_py(int(steps / len(loader)))
        loader_iter = iter(loader)
        while steps < max_steps:
            if (eval_freq is not None and steps % eval_freq == 0) or (
                eval_pow2 and steps & (steps - 1) == 0
            ):
                metrics = evaluate(net)
                if save_ckpt_dir is not None:
                    for (db_name, table_name), metric in metrics["val"].items():
                        # Eval metric is always higher is better (auc, r2)
                        best_metric = best_val_metrics.get(
                            (db_name, table_name), -float("inf")
                        )
                        if metric > best_metric:
                            best_val_metrics[(db_name, table_name)] = metric
                            best_test_metrics[(db_name, table_name)] = metrics["test"][(db_name, table_name)]
                            checkpoint(
                                best=True,
                                db_name=db_name,
                                table_name=table_name
                            )
                        else:
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

            loss, _yhat_dict = net(batch)
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
                wandb.log(
                    {
                        "loss": loss,
                        "lr": opt.param_groups[0]["lr"],
                        "epochs": steps / len(loader),
                        "grad_norm": grad_norm,
                    },
                    step=steps,
                )

            pbar.update(1)

    # Print best test metrics
    if rank == 0:
        print("\n" + "="*80)
        print("Best test metrics:")
        print("="*80)
        for (db_name, table_name), metric in best_test_metrics.items():
            print(f"{db_name}/{table_name}/test: {metric:.4f}")
        print("="*80 + "\n")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import strictfire

    strictfire.StrictFire(main)
