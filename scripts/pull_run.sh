#!/usr/bin/env bash
# usage: scripts/pull_run.sh <remote_run_dir> <local_name> [step ...]
# e.g.   scripts/pull_run.sh /ephemeral/WAM-H3/runs/libero_wamh3_dev/20260911_140335 test-1 001500 001000
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${BREV_HOST:-h3-wam}
R=$1; L=runs/$(basename "$(dirname "$R")")/$2; shift 2
mkdir -p "$L/checkpoints" "$L/eval"
for f in config.yaml dataset_stats.json wandb_run.json; do brev copy "$HOST:$R/$f" "$L/$f" || true; done
for s in "$@"; do brev copy "$HOST:$R/checkpoints/step_$s" "$L/checkpoints/step_$s"; done
brev copy "$HOST:$R/eval" "$L/eval" || true
echo "pulled into $L"; find "$L" -maxdepth 3 -type d | sort
