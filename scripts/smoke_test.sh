#!/usr/bin/env bash
# Tiny random DiT + real VAE + 8 real samples: train 8 steps, eval 1 LIBERO trial, measure latency. Needs libero_spatial data + VAE weights.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run scripts/precompute_text_embeds.py --fake --data-dirs data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot --cache-dir data/text_embeds/libero
rm -rf runs/smoke
bash scripts/train_single.sh task=smoke_libero_tiny output_dir=runs/smoke
CKPT=$(ls -d runs/smoke/checkpoints/step_* | sort | tail -1)
printf 'libero_spatial,0\n' > runs/smoke/tasks.txt
bash scripts/eval_libero.sh ckpt="$CKPT" EVALUATION.num_trials=1 MULTIRUN.num_gpus=1 MULTIRUN.task_file=runs/smoke/tasks.txt
uv run scripts/measure_latency.py --task smoke_libero_tiny --ckpt "$CKPT" --iters 3 --warmup 1 --out runs/smoke/latency.json
