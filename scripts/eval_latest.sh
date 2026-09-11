#!/usr/bin/env bash
# usage: scripts/eval_latest.sh [task=libero_wamh3_dev] [libero|libero-plus] [hydra overrides...]
# e.g.   scripts/eval_latest.sh libero_wamh3_dev libero 'MULTIRUN.gpu_ids=[1]' 'MULTIRUN.task_suite_names=[libero_spatial]' EVALUATION.num_trials=10
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:-libero_wamh3_dev}; BENCH=${2:-libero}; shift $(( $# >= 2 ? 2 : $# ))
CKPT=$(ls -d runs/"$TASK"/*/checkpoints/step_* 2>/dev/null | sort | tail -1)
[ -n "$CKPT" ] || { echo "no checkpoint under runs/$TASK" >&2; exit 1; }
echo "evaluating $CKPT on $BENCH"
BENCH=$BENCH bash scripts/eval_libero.sh ckpt="$CKPT" "$@"
