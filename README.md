# WAM-H3

World Action Model on the MiniMax-H3 33B omni-transformer. Actions occupy H3's audio modality slot; the instruction, current observation, noisy future-video latents and action tokens share one joint self-attention sequence with an asymmetric mask (actions read everything, nothing reads actions), so at test time the video is never denoised: one backbone pass fills a K/V cache and 10 action-only Euler steps produce a 32-step chunk (FasterWAM recipe). Video and action flow times are independent. Fine-tuning is LoRA on attention, MLP and AdaLN plus fresh action/proprio heads. Design: `docs/superpowers/specs/2026-09-11-wam-h3-design.md`.

## Node requirements

- Linux x86_64, CUDA 12.8 drivers, `uv`, `git`, `unzip`, EGL (or set `MUJOCO_GL=osmesa`)
- Dev run: 1 GPU with 80 GB. Full run: 8x 80 GB. Text-embedding precompute loads the 66 GB Qwen3-VL encoder once (`--device cpu` needs ~140 GB RAM).
- Disk: ~150 GB (weights 143 GB, LIBERO data 5 GB)

## Quick start

```bash
git clone --recurse-submodules https://github.com/desmondzee/WAM-H3 && cd WAM-H3
bash scripts/setup_core.sh      # venv, weights, LIBERO data, simulators, text embeddings, tests
bash scripts/run_dev.sh         # 1 GPU: rank-16 LoRA on libero_spatial, 3000 steps, then LIBERO + LIBERO-Plus eval
bash scripts/run_full.sh        # 8 GPUs: rank-128 LoRA on all 4 suites, 21.7k steps, then eval
```

Optional: `bash scripts/smoke_test.sh` (tiny random DiT, real VAE, 8 steps + 1 sim trial + latency; also runs on CPU).

## Outputs

```
runs/<task>/<timestamp>/
  config.yaml                       resolved training config (eval reads model/data settings from here)
  dataset_stats.json                action/proprio normalization
  checkpoints/step_XXXXXX/adapter.safetensors   trainable params only (LoRA + heads)
  checkpoints/step_XXXXXX/trainer_state.json    step, epoch, loss, lr
  eval/<benchmark>/step_XXXXXX/
    episodes.jsonl                  one line per episode: suite, task, instruction, success
    suites.json                     per-suite and per-task success rates
    summary.json                    overall numbers
    <suite>/gpu*_task*_results.json raw per-task results, <suite>/videos/ rollouts
```

## Pieces

```bash
bash scripts/train_single.sh task=libero_wamh3_dev                 # or task=libero_wamh3 with train_fsdp.sh 8
bash scripts/train_single.sh task=libero_wamh3_dev train.resume=runs/libero_wamh3_dev/<run>/checkpoints/step_001000
bash scripts/eval_libero.sh ckpt=runs/.../checkpoints/step_003000 MULTIRUN.num_gpus=8
bash scripts/eval_libero_plus.sh ckpt=... MULTIRUN.num_gpus=8
uv run scripts/measure_latency.py --task libero_wamh3_dev --ckpt runs/.../step_003000
uv run pytest tests -q
```

Video loss weight is centred at σ=1 with σ_v=1 sampled 30% of the time (`model.video_weight_center`, `model.video_full_noise_prob`); see the spec for why this departs from FasterWAM.
Hydra overrides work everywhere: `train.batch_size=8 train.grad_accum=2`, `model.lora_r=64`, `EVALUATION.num_trials=20`.
`eval_libero.sh` first caches every task instruction of the benchmark (`precompute_text_embeds.py --benchmark-suites ...`, one encoder load on the eval GPU), so LIBERO-Plus's rewritten instructions are covered; other prompts can be added with `--prompts-file`.

## Layout

`wam_h3/model` DiT, layout/mask, flow schedule, loss, LoRA, VAE encoder, text encoder, facade. `wam_h3/data` dataset + text cache (FasterWAM's LeRobot loader underneath). `wam_h3/train` trainer. `wam_h3/eval` LIBERO worker, results. `configs/` Hydra. `third_party/FasterWAM` submodule (datasets, eval harness). `FL2VA/` MiniMax-H3 weights (downloaded, gitignored).
