"""SWA (stochastic weight averaging) state + offline averaging driver.

Two pieces, used independently:

- ``SwaState``: a small parameter-averaging container shared with
  ``rt/main.py`` (in-loop SWA over training params). Backed by an fp32
  dict; updates are in-place ``lerp_`` with alpha derived from
  ``momentum`` and the current update count.

- ``run_swa``: offline driver. Loads ``steps=*.pt`` ckpts from
  ``cfg.load_ckpt_dir`` in latest→earliest order, accumulates an
  equal-weight running average, and evaluates every
  ``eval_freq`` ckpts (and at the end) on a fixed eval recipe
  using a single fixed hparam config (no grid). The starting (latest)
  ckpt can be pinned via ``cfg.start_step`` — useful for reproducing
  the SWA average that was logged at a specific step during training.
  Loaders + the inference net are built once and reused across all
  eval points for speed. Metrics are logged to wandb tagged with
  ``swa_n`` so they can be plotted vs swa_n.
"""

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lazy_loader as lazy
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from rt.config import LoggerConfig, ModelConfig
from rt.evaluator import Evaluator, fmt_duration
from rt.model import RelationalTransformer
from rt.recipes import get_tasks

wandb = lazy.load("wandb")


# -----------------------------------------------------------------------------
# SwaState — shared with rt/main.py
# -----------------------------------------------------------------------------


class SwaState:
    """Equal-weight or EMA running average of a fixed set of named tensors.

    Backed by an fp32 dict of clones; updates are in-place via
    ``lerp_``. The averaging weight schedule depends on ``momentum``:

    - ``1.0``: equal-weight averaging (``alpha = 1/n``). After ``k``
      updates, ``params[name]`` equals the arithmetic mean of the
      ``k`` inputs.
    - ``< 1.0``: bias-corrected EMA (``alpha = (1-m) / (1-m^n)``).
      First update has ``alpha=1.0``, asymptotes to ``1-m`` as ``n``
      grows.

    Used both from training (``rt/main.py``: in-loop SWA over
    ``raw_net`` parameters) and from offline averaging drivers below
    (combining saved-ckpt state_dicts).
    """

    def __init__(self, named_tensors, momentum):
        """``named_tensors``: iterable of ``(name, tensor)`` pairs. The
        fp32 storage is allocated as clones on the source tensors'
        devices. Initial values are arbitrary — the first ``update``
        sets them exactly (``alpha=1.0``)."""
        self.momentum = momentum
        self.params = {name: t.detach().float().clone() for name, t in named_tensors}
        self.n = 0

    @torch.no_grad()
    def update(self, named_tensors):
        """Add one snapshot to the running average. Source key set must
        equal the stored key set."""
        self.n += 1
        if self.momentum == 1.0:
            alpha = 1.0 / self.n
        else:
            m = self.momentum
            alpha = (1.0 - m) / (1.0 - m**self.n)
        src = dict(named_tensors)
        assert src.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(src) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(src))}"
        )
        for name, target in self.params.items():
            target.lerp_(src[name].float(), alpha)

    def state_dict(self):
        """CPU-serializable snapshot for training resume."""
        return {
            "momentum": self.momentum,
            "n": self.n,
            "params": {k: v.detach().cpu().clone() for k, v in self.params.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        assert state["momentum"] == self.momentum, (
            f"momentum mismatch: ckpt={state['momentum']} cfg={self.momentum}"
        )
        assert state["params"].keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(state['params']) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(state['params']))}"
        )
        self.n = state["n"]
        for k, v in self.params.items():
            v.copy_(state["params"][k].to(v.device))

    @torch.no_grad()
    def sync_to(self, named_tensors):
        """Copy the running average into the target tensors in-place.
        Target key set must equal the stored key set."""
        dst = dict(named_tensors)
        assert dst.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(dst) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(dst))}"
        )
        for name, target in dst.items():
            target.copy_(self.params[name])


# -----------------------------------------------------------------------------
# Offline driver
# -----------------------------------------------------------------------------


@dataclass
class SwaConfig:
    """All knobs for one offline SWA averaging run.

    The driver iterates the live (non-``swa_*``) ``steps=*.pt`` ckpts
    in ``load_ckpt_dir`` from latest→earliest, accumulating an
    equal-weight running average. Every ``eval_freq`` ckpts
    (and at the end), evaluates on the eval ``recipe`` with the fixed
    hparams below, logging metrics to wandb tagged with ``swa_n``.

    ``start_step`` pins the iteration's starting (latest) ckpt: when
    not None, ckpts with ``step > start_step`` are dropped before
    iteration, so accumulation begins at ``steps={start_step}.pt``.
    Use this to reproduce the SWA average that training logged at a
    specific step. ``start_step`` must match an existing live ckpt.
    When None, accumulation starts from the latest live ckpt.
    """

    load_ckpt_dir: str
    start_step: int | None
    eval_freq: int
    save_ckpt_root_dir: str | None

    recipe: str
    pre_dir: str
    tokens_per_gpu: int
    num_workers: int
    prefetch_factor: int
    local_ctx_size: int
    bfs_width: int
    num_walks: int
    walk_length: int
    prefer_latest: bool
    items_per_task: int
    ctx_sizes: list[int]
    bool_as_num: bool
    skip_text_cols: bool
    mmap_populate: bool
    balance_labels: bool
    ablate_schema_semantics: bool
    reg_metric: str
    shuffle_seed: int
    context_seed: int
    vector_db_path: str | None

    model: ModelConfig
    logger: LoggerConfig


def _discover_live_ckpts(ckpt_dir: Path) -> list[Path]:
    """Return ``steps=N.pt`` files in ``ckpt_dir``, sorted latest→earliest.

    Strict prefix match on ``steps=`` so ``swa_steps=*.pt`` is excluded.
    """
    live = [
        p
        for p in ckpt_dir.iterdir()
        if p.is_file() and p.name.startswith("steps=") and p.suffix == ".pt"
    ]
    live.sort(key=lambda p: int(p.stem.split("=")[1]), reverse=True)
    return live


def run_swa(cfg: SwaConfig):
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    ddp = int(os.environ.get("WORLD_SIZE", "0")) > 1
    device = "cuda"
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = global_rank = 0
        world_size = 1
    print(
        f"[rank init] global_rank={global_rank} local_rank={local_rank}"
        f" world_size={world_size}",
        flush=True,
    )

    ckpt_dir = Path(cfg.load_ckpt_dir).expanduser()
    live_ckpts = _discover_live_ckpts(ckpt_dir)
    assert live_ckpts, f"no live ckpts (steps=*.pt) found in {ckpt_dir}"
    if cfg.start_step is not None:
        all_steps = [int(p.stem.split("=")[1]) for p in live_ckpts]
        assert cfg.start_step in all_steps, (
            f"start_step={cfg.start_step} has no matching steps=*.pt in"
            f" {ckpt_dir} (available: {sorted(all_steps)})"
        )
        live_ckpts = [
            p for p in live_ckpts if int(p.stem.split("=")[1]) <= cfg.start_step
        ]
    if local_rank == 0:
        latest_step = int(live_ckpts[0].stem.split("=")[1])
        earliest_step = int(live_ckpts[-1].stem.split("=")[1])
        print(
            f"discovered \033[1m{len(live_ckpts)}\033[0m live ckpts in {ckpt_dir}"
            + (
                f" (pinned to start_step={cfg.start_step:_})"
                if cfg.start_step is not None
                else ""
            ),
            flush=True,
        )
        print(
            f"  step range: \033[1m{earliest_step:_}\033[0m → "
            f"\033[1m{latest_step:_}\033[0m (will iterate latest→earliest)",
            flush=True,
        )

    save_ckpt_dir_ = None
    if global_rank == 0:
        run = wandb.init(
            entity="rtv2",
            project=cfg.logger.project,
            name=cfg.logger.wandb_run_name,
            config=asdict(cfg),
            reinit="finish_previous",
            mode="disabled" if cfg.logger.wandb_disabled else "online",
        )
        print(f"wandb run name: \033[1m{run.name}\033[0m", flush=True)
        if cfg.save_ckpt_root_dir is not None:
            save_ckpt_root_dir_ = Path(cfg.save_ckpt_root_dir).expanduser()
            save_ckpt_dir_ = save_ckpt_root_dir_ / run.entity / run.project / run.id
            save_ckpt_dir_.mkdir(parents=True, exist_ok=True)
            print(f"will save SWA ckpts to {save_ckpt_dir_}", flush=True)

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch._dynamo.config.cache_size_limit = 16
    torch._dynamo.config.optimize_ddp = True
    torch.set_num_threads(1)

    eval_tasks = get_tasks(cfg.recipe, cfg.pre_dir)
    if local_rank == 0:
        n_real = sum(1 for t in eval_tasks if "synthetic" not in t.db_name)
        print(
            f"recipe \033[1m{cfg.recipe}\033[0m → \033[1m{n_real}\033[0m tasks",
            flush=True,
        )

    max_ctx = max(cfg.ctx_sizes)
    eval_bs = max(1, cfg.tokens_per_gpu // max_ctx)

    evaluator = Evaluator(
        tasks=eval_tasks,
        pre_dir=cfg.pre_dir,
        eval_bs=eval_bs,
        ctx_sizes=cfg.ctx_sizes,
        items_per_task=cfg.items_per_task,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.num_workers > 0,
        local_ctx_size=cfg.local_ctx_size,
        bfs_width=cfg.bfs_width,
        num_walks=cfg.num_walks,
        walk_length=cfg.walk_length,
        prefer_latest=cfg.prefer_latest,
        bool_as_num=cfg.bool_as_num,
        skip_text_cols=cfg.skip_text_cols,
        mmap_populate=cfg.mmap_populate,
        balance_labels=cfg.balance_labels,
        ablate_schema_semantics=cfg.ablate_schema_semantics,
        embedding_model=cfg.model.embedding_model,
        d_text=cfg.model.d_text,
        shuffle_seed=cfg.shuffle_seed,
        context_seed=cfg.context_seed,
        vector_db_path=cfg.vector_db_path,
        train_only_fallback=False,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        ddp=ddp,
        device=device,
    )

    eval_net = (
        RelationalTransformer(
            num_blocks=cfg.model.num_blocks,
            d_model=cfg.model.d_model,
            d_text=cfg.model.d_text,
            num_heads=cfg.model.num_heads,
            d_ff=cfg.model.d_ff,
            compile=cfg.model.compile,
            materialize_attn_masks=cfg.model.materialize_attn_masks,
        )
        .to(device)
        .to(torch.bfloat16)
    )
    eval_net.eval()
    if local_rank == 0:
        param_count = sum(p.numel() for p in eval_net.parameters())
        print(f"params: \033[1m{param_count:_}\033[0m", flush=True)

    # equal-weight averaging across saved ckpts
    swa = SwaState(eval_net.named_parameters(), momentum=1.0)

    if local_rank == 0:
        print(
            f"\nstarting SWA accumulation (eval every {cfg.eval_freq} ckpts)\n",
            flush=True,
        )

    ckpt_pbar = tqdm(
        total=len(live_ckpts),
        desc="swa accumulate",
        disable=local_rank != 0,
        dynamic_ncols=True,
    )
    grand_tic = time.time()
    n_ckpts = len(live_ckpts)
    for ckpt_idx, ckpt_path in enumerate(live_ckpts):
        ckpt_step = int(ckpt_path.stem.split("=")[1])
        if local_rank == 0:
            tqdm.write(
                f"\n[ckpt {ckpt_idx + 1}/{n_ckpts}] loading {ckpt_path.name}"
                f" (step={ckpt_step:_})..."
            )
        load_tic = time.time()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Use load_state_dict so all model state (incl. any future buffers)
        # is restored, with strict-key checking. Then update swa from
        # eval_net's now-loaded parameters.
        eval_net.load_state_dict(ckpt["model"])
        swa.update(eval_net.named_parameters())
        del ckpt
        if local_rank == 0:
            tqdm.write(
                f"  ckpt loaded + averaged in"
                f" \033[1m{fmt_duration(time.time() - load_tic)}\033[0m"
                f" (swa_n={swa.n})"
            )

        # Log swa_n / ckpt_step every iteration (not just eval steps) so
        # the wandb timeline has a continuous swa_n→ckpt_step mapping.
        if global_rank == 0:
            wandb.log(
                {"swa_n": swa.n, "ckpt_step": ckpt_step},
                step=swa.n,
            )

        is_last = ckpt_idx == n_ckpts - 1
        is_eval_step = ((swa.n - 1) % cfg.eval_freq == 0) or is_last
        if is_eval_step:
            swa.sync_to(eval_net.named_parameters())
            evaluator.evaluate(
                nets_with_prefix=[(eval_net, "")],
                eval_ctx_sizes_to_use=cfg.ctx_sizes,
                steps=swa.n,
                reg_metric=cfg.reg_metric,
            )
            if global_rank == 0:
                if save_ckpt_dir_ is not None:
                    save_ckpt_path = save_ckpt_dir_ / f"swa_n={swa.n}.pt"
                    torch.save({"model": eval_net.state_dict()}, save_ckpt_path)
                    print(f"saved SWA checkpoint to {save_ckpt_path}", flush=True)
        ckpt_pbar.update(1)
    ckpt_pbar.close()

    torch.cuda.synchronize()
    if local_rank == 0:
        print(
            f"\nSWA driver complete in"
            f" \033[1m{fmt_duration(time.time() - grand_tic)}\033[0m",
            flush=True,
        )

    if ddp:
        dist.destroy_process_group()
