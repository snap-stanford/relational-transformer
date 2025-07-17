# Relational Transformer

## Installation:

1. Install pixi [https://pixi.sh/latest/#installation](https://pixi.sh/latest/#installation).

2. Clone and install the repository:
```bash
git clone https://github.com/snap-stanford/relational_transformer
cd relational_transformer/rustler
pixi run maturin develop --uv --release
```

## Data:


1. Download the datasets and tasks from Relbench:
```bash
cd .. # back to the root of the repository
pixi run python scripts/download_relbench_datasets.py
pixi run python scripts/download_relbench_tasks.py
```

2. Link the cache repository
```bash
mkdir ~/scratch
ln -s ~/.cache/relbench ~/scratch/relbench
```

3. Preprocessing (per database):
```bash
cd rustler
pixi run cargo run --release -- pre rel-f1  # preprocesses both database and task tables
```

4. Text embedding (per database):

Find available sentence_transformers models [here](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit?gid=0#gid=0).
```bash
pixi run python -m rt.emb rel-f1 --task False  # to embed text from database tables
pixi run python -m rt.emb rel-f1 --task True  # to embed text from task tables
```
> [!NOTE]
> Steps 3. and 4. should be run for all databases in order for experiments to work. \
> Relbench Databases: `rel-amazon`, `rel-avito`, `rel-event`, `rel-f1`, `rel-hm`, `rel-stack`, `rel-trial`


## Experiments

Pretraining (takes about 5 hours on 8xH100 GPUs, reducing `max_steps` can reduce this runtime):
```bash
pixi run torchrun --standalone --nproc_per_node=8 -m rt.main \
    --pairs='[("rel-hm", "user-churn"), ("rel-stack", "user-badge"), ("rel-stack", "user-engagement"), ("rel-avito", "user-visits"), ("rel-avito", "user-clicks"), ("rel-event", "user-ignore"), ("rel-trial", "study-outcome"), ("rel-f1", "driver-dnf"), ("rel-event", "user-repeat"), ("rel-f1", "driver-top3"), ("rel-hm", "item-sales"), ("rel-stack", "post-votes"), ("rel-trial", "site-success"), ("rel-trial", "study-adverse"), ("rel-event", "user-attendance"), ("rel-f1", "driver-position"), ("rel-avito", "ad-ctr")]' \
    --mask_prob=0.0 \
    --zero_mask_prob_steps=50_000 \
    --mask_db_cells=False \
    --mask_task_cells=True \
    --fake_names=False \
    --batch_size=32 \
    --num_workers=8 \
    --subsample_p2f_edges=256 \
    --isolate_task_tables=False \
    --date=2025-07-02 \
    --profile=False \
    --eval_splits=[] \
    --eval_freq=None \
    --log_eval=False \
    --ckpt_freq=25_000 \
    --load_ckpt_path=None \
    --save_ckpt_dir="~/scratch/ckpts/2025-07-02/leave=rel-amazon__mask_prob=0.0" \
    --lr=1e-3 \
    --lr_schedule=True \
    --max_grad_norm=1.0 \
    --max_steps=200_001 \
    --embedding_model=stsb-distilroberta-base-v2 \
    --d_text=768 \
    --seq_len=1024 \
    --num_layers=12 \
    --d_model=256 \
    --num_heads=8 \
    --d_ff=1024 \
    --loss=huber \
    --compile_=True
```

Finetuning (takes about 1 hour on 8xH100 GPUs)
```bash
pixi run torchrun --standalone --nproc_per_node=8 -m rt.main \
    --pairs='[("rel-amazon", "user-churn")]' \
    --mask_prob=0.0 \
    --zero_mask_prob_steps=0 \
    --mask_db_cells=False \
    --mask_task_cells=True \
    --fake_names=False \
    --batch_size=32 \
    --num_workers=8 \
    --subsample_p2f_edges=256 \
    --isolate_task_tables=False \
    --date=2025-07-02 \
    --profile=False \
    --eval_splits="['val']" \
    --eval_freq=None \
    --log_eval=True \
    --ckpt_freq=None \
    --load_ckpt_path="~/scratch/ckpts/2025-07-02/leave=rel-amazon__mask_prob=0.0/steps=200000.pt" \
    --save_ckpt_dir=None \
    --lr=1e-4 \
    --lr_schedule=False \
    --max_grad_norm=1.0 \
    --max_steps=1025 \
    --embedding_model=stsb-distilroberta-base-v2 \
    --d_text=768 \
    --seq_len=1024 \
    --num_layers=12 \
    --d_model=256 \
    --num_heads=8 \
    --d_ff=1024 \
    --loss=huber \
    --compile_=True
```

