# Relational Transformer (RT)

The official implementation of the **Relational Transformer (RT)**, an
architecture for **Relational Foundation Models (RFMs)** that predict directly
over relational databases (tables linked by foreign keys) and generalize
zero-shot to new databases, tasks, and schemas.

| Paper | Venue | Implementation |
|---|---|---|
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | ICLR 2026 | [`rt-v1`](https://github.com/stanford-star/relational-transformer/tree/rt-v1) |
| [PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models](https://arxiv.org/abs/2602.04029) | ICML 2026 | [`main`](https://github.com/stanford-star/relational-transformer) |
| RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions | In progress | [`main`](https://github.com/stanford-star/relational-transformer) |

## Installation

Install the `rt` package (the model plus the native Rust data engine) from
GitHub. It builds the extension from source, so you need a
[Rust toolchain](https://rustup.rs) and Python 3.12+:

```bash
pip install "git+https://github.com/stanford-star/relational-transformer.git"
```

## Quickstart

The quickest way to try a released checkpoint is on a RelBench
database already preprocessed into RT's tensor format on the Hub. The example below predicts whether an F1 driver fails to finish a race (`driver-dnf`) with a released RT-J checkpoint (a PluRel `.pt` from [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel) works too):

```python
import os

# flex_attention's compiled kernel is CUDA-only; run it eager on CPU/MPS
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from huggingface_hub import snapshot_download

from rt import RelationalTransformer
from rt.eval_utils import build_evaluator
from rt.tasks import tasks_from_preprocessed

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 1. download one RelBench database, already preprocessed into RT's tensor format
pre_dir = snapshot_download(
    "stanford-star/relbench-preprocessed",
    repo_type="dataset",
    allow_patterns="rel-f1/*",
)

# 2. load a pretrained checkpoint (RT-J here; a PluRel .pt works too)
model = RelationalTransformer.from_pretrained(
    "stanford-star/rt-j/classification", device=device
).to(torch.bfloat16)
cfg = model.config

# 3. build an evaluator for one task and predict zero-shot for 5 test rows.
#    the evaluator samples each row's context from the preprocessed DB;
#    the same path runs on your own database, any task.
tasks = [
    t for t in tasks_from_preprocessed(pre_dir, splits=("test",), dbs=["rel-f1"])
    if t.table_name == "driver-dnf"
]
ev = build_evaluator(
    tasks, pre_dir,
    embedding_model=cfg["embedding_model"], d_text=cfg["d_text"],
    device=device, ctx_size=128, local_ctx_size=64,
    items_per_task=5, num_workers=0,
)

# evaluate_raw yields one (task, ctx, labels, preds, n) per task
results = ev.evaluate_raw([(model, "")], [128])
_task, _ctx, _labels, out, _n = next(iter(results))
preds = torch.sigmoid(torch.tensor(out[""], dtype=torch.float32))
print("driver-dnf probability:", [round(p, 3) for p in preds.tolist()])
```

> [!NOTE]
> `items_per_task=5` and `ctx_size=128` keep this demo quick (most of the runtime
> is one-time warmup). On a GPU, raise `ctx_size` toward RT-J's training context of
> 8192 (with `local_ctx_size <= ctx_size`) for full accuracy over the whole test
> split.

## Bring your own database

Point RT at your **own** database, define a
prediction task, and infer with a released checkpoint:

- **Colab, no setup**: the [fully worked notebook](examples/byod/colab.ipynb)
  ([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/examples/byod/colab.ipynb))
  runs the whole flow end-to-end on your database (or the bundled demo).
- **Local scripts**: [examples/inference](examples/inference/README.md) runs the
  same from a DuckDB, Postgres, or MySQL database, with any RT-J or PluRel
  checkpoint from the Hugging Face Hub.

## Development

We use [pixi](https://pixi.sh) to manage one self-contained
environment (Python, PyTorch + CUDA, Rust, and all dependencies), built on first use.

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
pixi run test        # or pretrain, eval, preprocess, ...
```

Every task first runs `build-sampler` (`maturin develop`), an editable install of
`rt`, so edits to `src/rt/` take effect live and `import rt` just works.

## Documentation

| Guide | Description |
|---|---|
| [Downloads](docs/downloads.md) | Bulk-download raw data, preprocessed data, and checkpoints from our HuggingFace org |
| [Preprocess](docs/preprocess.md) | Convert RelBench-format databases into RT's on-disk format |
| [Inference](docs/inference.md) | Run a trained checkpoint; evaluate, engineer, tune, and ensemble contexts |
| [Pretrain](docs/pretrain.md) | Train RT from scratch, single-GPU to multi-node |
| [Baselines](docs/baselines.md) | rel2tab tabular baselines through the same eval path |
| [Context visualization](docs/context-visualization.md) | Inspect the contexts sampled for each row |
