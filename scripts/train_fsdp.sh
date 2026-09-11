#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_GPUS=${1:-8}; shift || true
uv run accelerate launch --config_file configs/accelerate/fsdp.yaml --num_processes "$NUM_GPUS" scripts/train.py "$@"
