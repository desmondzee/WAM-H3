#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_GPUS=8; case "${1:-}" in ""|*[!0-9]*) ;; *) NUM_GPUS=$1; shift ;; esac
uv run accelerate launch --config_file configs/accelerate/fsdp.yaml --num_processes "$NUM_GPUS" scripts/train.py "$@"
