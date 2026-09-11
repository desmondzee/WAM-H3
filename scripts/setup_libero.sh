#!/usr/bin/env bash
# LIBERO + LIBERO-Plus simulators into the project venv (both are package `libero`; selected via PYTHONPATH at eval time).
set -euo pipefail
cd "$(dirname "$0")/.."
clone() { [ -d "$3/.git" ] || git clone "$1" "$3"; git -C "$3" checkout -q "$2"; }
clone https://github.com/Lifelong-Robot-Learning/LIBERO.git 8f1084e third_party/LIBERO
clone https://github.com/sylvestf/LIBERO-plus.git 4976dc3 third_party/LIBERO-plus
uv pip install robosuite==1.4.0 bddl==1.0.1 easydict future==1.0.0 cloudpickle==2.1.0 gym==0.25.2 \
  matplotlib opencv-python-headless mujoco==3.3.2 robomimic==0.2.0 thop h5py
ASSETS=third_party/LIBERO-plus/libero/libero/assets
if [ ! -d "$ASSETS" ]; then
  uv run hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir third_party/LIBERO-plus/libero/libero
  unzip -q -o third_party/LIBERO-plus/libero/libero/assets.zip -d third_party/LIBERO-plus/libero/libero
  NESTED=$(find third_party/LIBERO-plus/libero/libero -type d -path "*LIBERO-plus-0/assets" | head -1)
  [ -n "$NESTED" ] && [ ! -d "$ASSETS" ] && mv "$NESTED" "$ASSETS"
fi
for b in libero libero-plus; do
  src=third_party/LIBERO; [ "$b" = libero-plus ] && src=third_party/LIBERO-plus
  uv run python third_party/FasterWAM/scripts/setup/configure_libero.py --source-root "$PWD/$src" --config-dir "$PWD/.runtime/$b"
done
PYTHONPATH=$PWD/third_party/LIBERO LIBERO_CONFIG_PATH=$PWD/.runtime/libero MUJOCO_GL=${MUJOCO_GL:-egl} \
  uv run python -c "from libero.libero.envs import OffScreenRenderEnv; from libero.libero import benchmark; print('libero ok', sorted(benchmark.get_benchmark_dict()))"
