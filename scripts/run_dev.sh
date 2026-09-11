#!/usr/bin/env bash
# Single-GPU rank-16 dev run on libero_spatial, then LIBERO + LIBERO-Plus eval of the final checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f FL2VA/transformer/model.safetensors.index.json ] && [ -d data/text_embeds/libero ] || bash scripts/setup_core.sh
bash scripts/train_single.sh task=libero_wamh3_dev "$@"
CKPT=$(ls -d runs/libero_wamh3_dev/*/checkpoints/step_* | sort | tail -1)
echo "final checkpoint: $CKPT"
bash scripts/eval_libero.sh ckpt="$CKPT" MULTIRUN.num_gpus=1 MULTIRUN.task_suite_names='[libero_spatial]'
bash scripts/eval_libero_plus.sh ckpt="$CKPT" MULTIRUN.num_gpus=1 MULTIRUN.task_suite_names='[libero_spatial]'
