#!/usr/bin/env bash
# Fresh clone -> ready to train/eval. Idempotent. Needs: uv, git, a GPU node for the real text-encoder pass.
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init third_party/FasterWAM
uv sync
uv pip install --no-deps -e third_party/FasterWAM
uv run hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/transformer/*" "FL2VA/video_vae/*" "FL2VA/text_encoder/*" "FL2VA/tokenizer/*" --local-dir .
mkdir -p data/libero_mujoco3.3.2
for s in spatial object goal 10; do
  d=data/libero_mujoco3.3.2/libero_${s}_no_noops_lerobot
  if [ ! -d "$d" ]; then
    uv run hf download yuanty/LIBERO-fastwam --repo-type dataset --include "libero_${s}_no_noops_lerobot.tar.gz" --local-dir data/_tar
    tar -xzf "data/_tar/libero_${s}_no_noops_lerobot.tar.gz" -C data/libero_mujoco3.3.2 && rm -f "data/_tar/libero_${s}_no_noops_lerobot.tar.gz"
  fi
done
bash scripts/setup_libero.sh
uv run scripts/precompute_text_embeds.py --data-dirs data/libero_mujoco3.3.2/*_lerobot --cache-dir data/text_embeds/libero
uv run pytest tests -q
