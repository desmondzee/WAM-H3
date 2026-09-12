set -euo pipefail
cd "$(dirname "$0")/.."
revision=6e3c0bda5a756ec334df449cdc7d4a4685631e91
if [ ! -d .runtime/ComfyUI ]; then
    git clone --no-checkout https://github.com/Comfy-Org/ComfyUI.git .runtime/ComfyUI
    git -C .runtime/ComfyUI checkout "$revision"
fi
[ "$(git -C .runtime/ComfyUI rev-parse HEAD)" = "$revision" ] || { echo "Unexpected ComfyUI revision" >&2; exit 1; }
if [ ! -d .runtime/refined-venv ]; then
    uv venv --python 3.10 .runtime/refined-venv
fi
uv pip install --python .runtime/refined-venv/bin/python --torch-backend cu128 -r requirements-refined.lock
.runtime/refined-venv/bin/python scripts/preflight_refined.py --output .runtime/refined-preflight.json
HF_HUB_DISABLE_TELEMETRY=1 .runtime/refined-venv/bin/hf download Comfy-Org/MiniMax-H3 \
    diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors \
    text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
    vae/minimax_h3_video_vae_fp16.safetensors \
    --revision a98869194787969724c7425d95d0ed73ce9202af --local-dir .runtime/refined-models
.runtime/refined-venv/bin/python -c 'from wam_h3.refined.runtime import verify_models; print(verify_models())'
upstream=third_party/LIBERO
if [ ! -d "$upstream/.git" ]; then
    upstream=.runtime/LIBERO
    if [ ! -d "$upstream" ]; then
        git clone --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git "$upstream"
        git -C "$upstream" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
    fi
fi
HF_HUB_DISABLE_TELEMETRY=1 .runtime/refined-venv/bin/python scripts/download_refined_libero.py --upstream "$upstream"
