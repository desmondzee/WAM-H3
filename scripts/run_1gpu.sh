#!/usr/bin/env bash
# One 96 GB GPU: rank-64 LoRA (AdaLN rank 16) on all four LIBERO suites, then LIBERO + LIBERO-Plus eval of the final checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f FL2VA/transformer/model.safetensors.index.json ] && [ -d data/text_embeds/libero ] || bash scripts/setup_core.sh
bash scripts/train_single.sh task=libero_wamh3_1gpu "$@"
CKPT=$(ls -d runs/libero_wamh3_1gpu/*/checkpoints/step_* | sort | tail -1)
echo "final checkpoint: $CKPT"
bash scripts/eval_libero.sh ckpt="$CKPT" MULTIRUN.num_gpus=1
bash scripts/eval_libero_plus.sh ckpt="$CKPT" MULTIRUN.num_gpus=1
