#!/usr/bin/env bash
# Single- or multi-node RT pretraining under Slurm (torchrun + pixi).
#
# Infrastructure-agnostic: set ACCOUNT / PARTITION / QOS and the data/out paths
# for your cluster and it runs as-is. Resumes automatically from
# $OUT_DIR/resume.pt and is preemption-safe (SIGTERM -> save resume.pt + exit;
# --requeue relaunches and resumes). Resume is GPU-count flexible: a run
# preempted on, say, 4x8 GPUs can come back on a single 4-GPU node (same OUT_DIR)
# and continue -- pretrain.py is deterministic w.r.t. the resumed step, so the
# data stream is re-seeded consistently regardless of world size.
#
# Run from a clone on STORAGE ALL NODES CAN READ (so every node sees the same
# repo + pixi manifest). The per-node pixi env itself is built node-locally.
#
# --- launch (single node) ---
#   PRE_DIR=stanford-star/the-join-preprocessed \
#   VAL_PRE_DIR=stanford-star/relbench-preprocessed \
#   OUT_DIR=$HOME/ckpts/rtj \
#   sbatch --nodes=1 --gres=gpu:a100:8 \
#          --account=<acct> --partition=<part> --qos=<qos> \
#          scripts/slurm_pretrain.sh
#
# --- launch (multi node, e.g. 4 nodes x 8 GPUs = 32) ---
#   ...same env vars... \
#   sbatch --nodes=4 --gres=gpu:a100:8 \
#          --account=<acct> --partition=<preemptible-part> --qos=<preemptible-qos> \
#          scripts/slurm_pretrain.sh
#   (multi-node typically requires a preemptible queue; --requeue makes that safe)
#
# --- knobs (env vars) ---
#   GPUS_PER_NODE   GPUs (== torchrun procs) per node            [default 8]
#   WANDB_PROJECT   wandb project; if set, run logs online       [unset = offline]
#   WANDB_NAME      wandb run name                                [default rt-pretrain]
#   NCCL_IB_DISABLE set to 1 to force NCCL over TCP/sockets if    [default unset = IB on]
#                   the cluster's InfiniBand is broken (we hit
#                   mlx5 "catastrophic" errors on one fabric;
#                   1 = robust fallback, lower bandwidth)
#   EXTRA_ARGS      extra flags forwarded to pretrain.py
#
#SBATCH --job-name=rt-pretrain
#SBATCH --output=logs/rt-pretrain-%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --requeue
# NOTE: request the node's memory at SUBMIT time, however your cluster expects it
#   --exclusive                 (whole node), or
#   --mem-per-gpu=<X>G          (some schedulers require memory tied to GPUs), or
#   --mem=<X>G                  (absolute).
# Pretraining wants the data resident in RAM (the preprocessed mixture is large),
# so ask for as much of the node as you can. We intentionally do NOT hardcode a
# --mem here: `--mem=0` is rejected on clusters that require --mem-per-gpu.

set -uo pipefail
mkdir -p logs
: "${PRE_DIR:?set PRE_DIR (local path or HF repo of preprocessed pretraining data)}"
: "${VAL_PRE_DIR:=$PRE_DIR}"
: "${OUT_DIR:?set OUT_DIR (checkpoints + resume.pt; must persist across requeue)}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$OUT_DIR"

# wandb: online iff WANDB_PROJECT is set (otherwise pretrain.py runs offline).
WANDB_FLAGS=""
if [ -n "${WANDB_PROJECT:-}" ]; then
  WANDB_FLAGS="--wandb --wandb-project $WANDB_PROJECT --wandb-name ${WANDB_NAME:-rt-pretrain}"
fi

# The rt._rustler sampler extension is built once into the pixi env per node by
# the `build-sampler` task (maturin develop), NOT rebuilt on import. Keep
# MATURIN_IMPORT_HOOK_ENABLED=0 as a guard so a stray global maturin import hook
# can't have every rank trigger a concurrent `cargo build` racing on target/.
export MATURIN_IMPORT_HOOK_ENABLED=0

# Static rendezvous: derive the master from the Slurm nodelist and use a unique
# per-job port. We deliberately avoid torchrun's dynamic c10d store -- under
# load (large multi-node jobs) its TCP store dropped connections and wedged the
# rendezvous; a fixed master-addr/port is robust.
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

# Full node CPU affinity for the step. Without --cpus-per-task the srun step is
# pinned to a small (~8-core) cgroup slice, which starves rayon -- the rustler
# Sampler's parallel mmap-populate AND the dataloader's per-item context
# building run on rayon, so a throttled core count makes data loading the
# bottleneck (slow populate, slow eval). Give the step every core on the node.
NODE_CPUS=$(srun --nodes=1 --ntasks=1 nproc --all 2>/dev/null | tail -1)
NODE_CPUS="${NODE_CPUS:-$(nproc --all)}"

echo "nodes=$SLURM_NNODES gpus/node=$GPUS_PER_NODE cores/node=$NODE_CPUS head=$HEAD port=$PORT out=$OUT_DIR"

# One torchrun per node (ntasks-per-node=1); each spawns GPUS_PER_NODE workers.
# pixi run provides the environment (no env-pack); the editable rt package
# + rustler live inside it.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
     --cpus-per-task="$NODE_CPUS" --mem=0 bash -c '
  set -uo pipefail
  cd "'"$REPO"'"
  export MATURIN_IMPORT_HOOK_ENABLED=0
  export PYTHONUNBUFFERED=1
  # Force NCCL over TCP/sockets when NCCL_IB_DISABLE=1 was passed (broken IB).
  '"${NCCL_IB_DISABLE:+export NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_SOCKET_IFNAME=^lo,docker;}"'
  # Build rustler into this node-local env once (no-op if already present).
  pixi run build-sampler >/dev/null 2>&1 || true
  exec pixi run python -m torch.distributed.run \
    --nnodes='"$SLURM_NNODES"' --nproc-per-node='"$GPUS_PER_NODE"' \
    --node-rank=$SLURM_NODEID --master-addr='"$HEAD"' --master-port='"$PORT"' \
    scripts/pretrain.py \
    --pre-dir "'"$PRE_DIR"'" --val-pre-dir "'"$VAL_PRE_DIR"'" \
    --out-dir "'"$OUT_DIR"'" '"$WANDB_FLAGS"' '"$EXTRA_ARGS"'
'
