# Inference

Run a trained RT checkpoint on preprocessed data: load the checkpoint, sample an
in-context context for each test row, run the model's forward pass, and (for
RelBench) score the predictions with RelBench's own evaluator. There is no
fine-tuning — RT predicts zero-shot from the context it is given.

A checkpoint is a local path or a Hub model repo such as
`stanford-star/rt-j/classification`; its `config.json` records whether it is a
classifier (`clf`) or regressor (`reg`), so eval automatically restricts to the
matching tasks. A single weights file inside a multi-checkpoint Hub repo also
works — e.g. `stanford-star/rt-plurel/cntd-pretrain_rel-f1_driver-top3.pt` —
and pre-RT-J checkpoints (RT, PluRel) automatically run with the legacy
attention math they were trained with. PluRel checkpoints were trained at
context size 1024, so pass `--ctx-size 1024` when evaluating them.

## Prerequisite: preprocessed data

Inference takes a `--pre-dir` of preprocessed data — either a local path you
produced (see [preprocess.md](preprocess.md)) or a Hub repo such as
`stanford-star/relbench-preprocessed`, downloaded and cached on demand. So you can
reproduce the RelBench numbers with nothing downloaded up front:

```bash
# checkpoint and data both come from the Hub
pixi run eval --checkpoint stanford-star/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed --out-dir eval_out
```

## Inference with default context

The command above runs **simple** inference: one default context config
(`--local-ctx-size 256 --bfs-width 32`, total `--ctx-size 8192`) on the test
split. For each test row the sampler builds a context (a sampled neighborhood of
the relational graph), the model does a single forward pass, and predictions are
keyed back to each row by its seed node index. Eval is single-process (one GPU)
so per-row predictions stay aligned regardless of eval row order.

**Your own database (not RelBench).** `scripts/eval.py` is wired to the RelBench
benchmark, but the pieces compose directly for any database — prediction is just
the model's forward over a sampled context. The [fully worked Colab
notebook](../examples/byod/colab.ipynb) walks through it end-to-end (DuckDB
database → tasks defined in SQL → preprocess → forward pass → map outputs →
score) on the released RT-J checkpoints.

## Inference on a single task

By default `eval` runs every benchmark task of the checkpoint's kind (clf or
reg). Use `--tasks` to restrict to one — either a single `db/task-table` or a
whole `db` (all of its tasks of the checkpoint's kind):

```bash
# just one task
pixi run eval --checkpoint stanford-star/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed \
  --tasks rel-f1/driver-top3 --out-dir eval_out

# every clf task in one database
pixi run eval --checkpoint stanford-star/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed \
  --tasks rel-f1 --out-dir eval_out
```

This downloads only that task's data (the Hub `--pre-dir` is fetched on demand),
so it's the quickest way to try the model end-to-end. `--tasks` accepts several
selectors at once and works in `--mode ensemble` too.

## Evaluate with the RelBench evaluator

`--out-dir` is a valid RelBench **submission directory**: one
`<dataset>__<task>.csv` prediction table per task, scored through **RelBench's own
leaderboard evaluator** (`relbench.leaderboard`). Eval denormalizes regression
predictions to the original target scale (`y = pred*std + mean`, train-split
stats), maps classification logits to probabilities (sigmoid), and keys each
prediction to its relbench `(entity_col, time_col)`. It prints per-task and mean
test metrics — **AUROC** for clf, **NMAE** for reg.

Re-validate / re-score a submission dir any time, and submit it via the [RelBench
leaderboard procedure](https://relbench.stanford.edu):

```bash
pixi run python -m relbench.leaderboard eval_out
```

## Context engineering

Because RT predicts from context alone, the **context sampled for each row** is
the main quality knob. All are CLI flags on `eval`:

| flag | meaning | default |
|---|---|---|
| `--ctx-size` | total context size (cells) the model attends over | 8192 |
| `--local-ctx-size` | max cells collected per BFS expansion around the seed | 256 |
| `--bfs-width` | max DB nodes kept per BFS level (context breadth) | 32 |
| `--num-walks` | random walks used to rank same-table neighbors | 10000 |
| `--walk-length` | max length of each random walk | 20 |
| `--prefer-latest` / `--no-prefer-latest` | order same-table neighbors by recency (latest timestamp first) instead of by frequency | `--prefer-latest` |

Larger `--local-ctx-size` / `--bfs-width` pull more relational neighborhood into
each row's context (more signal, more tokens); `--ctx-size` caps the total.
`--prefer-latest` controls *which* same-table neighbors win that budget — the
most recent rows (default) or the most frequent. The best setting is
task-dependent — which motivates tuning and ensembling below. To *see* the
contexts a config produces, use the [context
visualizer](context-visualization.md).

`--shuffle-seed` (default 0) seeds the per-task subset selection and item
shuffle. Fixing it while capping rows with `--items-per-task N` evaluates the
*same* N validation rows across every config — the basis for a like-for-like
context grid search.

## Context tuning

Rather than fix one config, **tune** the context per task: evaluate a grid of
`(local_ctx_size, bfs_width)` configs on the **validation** split and keep the
best per task. This is the tuning half of `--mode ensemble` (here with a single
test seed, so no averaging yet):

```bash
pixi run eval --mode ensemble \
  --checkpoint stanford-star/rt-j/regression \
  --pre-dir stanford-star/relbench-preprocessed \
  --grid 256,32 512,64 --ensemble-size 1 --out-dir eval_out
```

`--grid` lists the candidate `local_ctx_size,bfs_width` pairs.

## Context ensembling

Context sampling is stochastic, so averaging predictions over several context
**seeds** reduces variance. Add `--ensemble-size`: `--mode ensemble` runs the
per-task tuned config with that many independent context seeds on test and
averages the per-row predictions before scoring:

```bash
pixi run eval --mode ensemble \
  --checkpoint stanford-star/rt-j/regression \
  --pre-dir stanford-star/relbench-preprocessed \
  --grid 256,32 512,64 --ensemble-size 4 --out-dir eval_out
```

So tuning (on validation) and ensembling (on test) are the two halves of one
`--mode ensemble` run: pick the best context config per task, then average that
config over `--ensemble-size` seeds.

For tabular comparisons through this same eval path, see
[baselines.md](baselines.md).

## Optional: FAISS vector-DB sampler

The default sampler is FAISS-free. The opt-in FAISS vector-db sampler (for
nearest-neighbor context retrieval) is built manually and additionally needs
cmake + a BLAS:

```bash
maturin develop --release --features vecdb
```
