#!/usr/bin/env bash
# usage: scripts/eval_libero.sh ckpt=runs/<task>/<run>/checkpoints/step_XXXXXX [MULTIRUN.num_gpus=8 ...]
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH=${BENCH:-libero}
SRC=third_party/LIBERO; CFG=sim_libero
if [ "$BENCH" = libero-plus ]; then SRC=third_party/LIBERO-plus; CFG=sim_libero_plus; fi
export LIBERO_CONFIG_PATH="$PWD/.runtime/$BENCH"
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$PWD/$SRC:$PWD/third_party/FasterWAM:$PWD/third_party/FasterWAM/experiments/libero${PYTHONPATH:+:$PYTHONPATH}"
uv run python third_party/FasterWAM/scripts/setup/configure_libero.py --source-root "$PWD/$SRC" --config-dir "$LIBERO_CONFIG_PATH" >/dev/null
uv run scripts/precompute_text_embeds.py --benchmark-suites libero_spatial libero_object libero_goal libero_10 --cache-dir data/text_embeds/libero
uv run scripts/eval_libero.py --config-name "$CFG" "$@"
