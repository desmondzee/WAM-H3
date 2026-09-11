#!/usr/bin/env bash
# 8-GPU rank-128 run on all four LIBERO suites, then LIBERO + LIBERO-Plus eval of the final checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_GPUS=${NUM_GPUS:-8}
[ -f FL2VA/transformer/model.safetensors.index.json ] && [ -d data/text_embeds/libero ] || bash scripts/setup_core.sh
bash scripts/train_fsdp.sh "$NUM_GPUS" task=libero_wamh3 "$@"
CKPT=$(ls -d runs/libero_wamh3/*/checkpoints/step_* | sort | tail -1)
echo "final checkpoint: $CKPT"
bash scripts/eval_libero.sh ckpt="$CKPT" MULTIRUN.num_gpus="$NUM_GPUS"
bash scripts/eval_libero_plus.sh ckpt="$CKPT" MULTIRUN.num_gpus="$NUM_GPUS"
