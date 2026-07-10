from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: rel2tab (repo-only, not in the wheel) pulls heavy deps, so
    # `import rt.config` must not import it at runtime.
    from rel2tab.config import Rel2TabModelConfig


@dataclass
class ModelConfig:
    embedding_model: str
    d_text: int
    num_blocks: int
    d_model: int
    num_heads: int
    d_ff: int
    compile: bool
    materialize_attn_masks: bool
    load_ckpt_path: str | None


@dataclass
class TrainConfig:
    recipe: str
    pre_dir: str
    tokens_per_gpu: int
    num_workers: int
    prefetch_factor: int
    ctx_sizes: list[int]
    local_ctx_sizes: list[int]
    bfs_widths: list[int]
    num_walks: int
    walk_length: int
    prefer_latest: list[bool]
    mask_prob_max: float
    items_per_task: int
    lr: float
    wd: float
    warmup_steps: int
    grad_norm_max: float
    total_bs: int
    total_steps: int
    decay_steps: int
    swa_momentum: float

    swa_loss_freq: int
    seed: int
    save_ckpt_root_dir: str | None
    bool_as_num: bool
    load_optimizer_state: bool
    skip_text_cols: bool
    mmap_populate: bool
    balance_labels: list[bool]
    timeout_per_item: float
    in_order: bool
    # When set, Tier 1 same-table seed selection switches from random
    # walks to FAISS-similarity lookups. Layout is
    # `<vector_db_path>/<db>/<table>.index` and
    # `<vector_db_path>/<db>/<table>_vectors.bin`. When None, behavior
    # is unchanged (random walk + same-table fallback).
    vector_db_path: str | None


@dataclass
class EvalOnlyConfig:
    pass


@dataclass
class EvalConfig:
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
    freq: int | None
    pow2: bool
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
    # See TrainConfig.vector_db_path.
    vector_db_path: str | None


@dataclass
class LoggerConfig:
    project: str
    wandb_run_name: str | None
    wandb_disabled: bool


@dataclass
class Config:
    model: ModelConfig | Rel2TabModelConfig
    train: TrainConfig | EvalOnlyConfig
    eval: EvalConfig
    logger: LoggerConfig
    profile: bool
