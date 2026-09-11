#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run accelerate launch --config_file configs/accelerate/single.yaml scripts/train.py "$@"
