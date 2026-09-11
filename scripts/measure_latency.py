#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from wam_h3.model.dit import patchify_video

ROOT = Path(__file__).resolve().parents[1]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="libero_wamh3_dev")
    ap.add_argument("--ckpt")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="latency.json")
    a = ap.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose("train", overrides=[f"task={a.task}"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = instantiate(cfg.model, device=dev).eval()
    if a.ckpt:
        m.load_checkpoint(a.ckpt)
    c = m.cfg
    h, w = (int(v) for v in cfg.data.train.video_size)
    img = torch.rand(1, 3, h, w, device=dev) * 2 - 1
    ctx = torch.randn(1, c.text_len, c.text_dim, device=dev, dtype=m.torch_dtype)
    valid = torch.ones(1, c.text_len, dtype=torch.bool, device=dev)
    prop = torch.randn(1, c.proprio_dim, device=dev, dtype=m.torch_dtype)
    noise = torch.randn(1, c.num_video_latents * c.frame_rows, c.video_patch_dim, device=dev, dtype=m.torch_dtype)
    sig, _ = m.sched_a.inference_schedule(a.steps)
    tg = torch.tensor([[1.0, c.obs_t, 1 - sig[0].item(), 0.0]], device=dev)
    with torch.no_grad():
        obs = patchify_video(m.vae.encode_image((img + 1) / 2)).to(m.torch_dtype)
        cache = m.dit.prefill(ctx, valid, obs, prop, noise, tg)
        stages = dict(
            vae_obs_encode_ms=timed(lambda: m.vae.encode_image((img + 1) / 2), a.warmup, a.iters),
            prefill_ms=timed(lambda: m.dit.prefill(ctx, valid, obs, prop, noise, tg), a.warmup, a.iters),
            action_denoise_ms=timed(lambda: m.dit.denoise_actions(cache, m.sched_a, a.steps), a.warmup, a.iters),
            end_to_end_ms=timed(lambda: m.infer_action_one_pass_future_cache(
                input_image=img, proprio=prop, context=ctx, context_mask=valid, num_inference_steps=a.steps), a.warmup, a.iters),
        )
    stages.update(steps=a.steps, device=dev, tiny=bool(cfg.model.tiny), rows=m.dit.layout.N)
    Path(a.out).write_text(json.dumps(stages, indent=1))
    print(json.dumps(stages, indent=1))


if __name__ == "__main__":
    main()
