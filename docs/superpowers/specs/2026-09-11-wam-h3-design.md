# WAM-H3 design (2026-09-11)
## Context

We want a World Action Model (WAM) built on the MiniMax-H3 33B omni-transformer and benchmarked on LIBERO and RoboTwin, following FasterWAM's training/inference recipe (independent video/action flow time, one-pass future-video cache, ~10 action denoising steps). The repo `/Users/desmondzee/Hardware/WAM-H3` is currently a mirror of the official MiniMax-H3 release: weight index JSONs, configs, VAE source under `FL2VA/video_vae/`, no transformer code, no training code. Everything model-side is brought in from DiffSynth-Studio @ `300e3e4` (its `MiniMaxH3DiT` parameter names match `FL2VA/transformer/model.safetensors.index.json` 1:1), data/eval-side from FasterWAM.

Decisions already made with the user:
- **Option A**: single shared backbone, no action expert. Actions occupy H3's **audio modality slot** (its input/output projections and AdaLN tag 2).
- **No AdaLN language conditioning** — H3 has no cross-attention; the instruction is ordinary rows in the joint self-attention sequence, visible to every token at every layer.
- **Asymmetric attention at all 50 layers**: actions read text/obs/video everywhere; nothing reads actions. This makes the one-pass K/V cache *exact*.
- **Video and action flow time independent**, sampled separately (FasterWAM phi-shift sampler, shift 5 for both).
- **Clip = 5 frames** at env steps 0,8,16,24,32 spanning 32 actions (H3's VAE takes 17k+5 frames → 2 latents: latent0 = frame0, latent1 = frames 1–4).
- **Fine-tuning**: LoRA on `qkv_proj, out_proj, fc1, fc2, adaln_proj.linear` of all 50 blocks + token refiner; full training of `action_in, action_out, proprio_in, final_layer.adaln_proj`. Rank is a config knob (`lora.r`): **dev = rank 16 on a single GPU** to prove the architecture trains and the pipeline works end to end; **final = rank 128 (α=128)** on 8 GPUs (1.19B LoRA params, ~16 GB/GPU under FSDP). Same target set in both so the dev run exercises the real code path.
- **Hardware**: dev on 1× 80 GB (no FSDP: 66 GB bf16 base + ~2 GB rank-16 LoRA/optimizer + <1 GB activations at batch 2 ≈ 69 GB); final on 8× 80 GB, one node, FSDP full-shard, ~80 GPU-hours per LIBERO run (≈10 h).
- **Benchmarks, in order**: LIBERO (4 suites) → LIBERO-Plus (same checkpoint, FasterWAM's `sim_libero_plus` manager config, 15% task sampling) → RoboTwin later.
- **Weights are saved** every `save_every` steps and at the end of training as trainable-only `adapter.safetensors` + `trainer_state.json` + `dataset_stats.json` + resolved `config.yaml`; eval loads them via `load_checkpoint`.
- Code lives in a new package `WAM-H3/wam_h3/`; FasterWAM is a git submodule installed `--no-deps` for datasets + eval harnesses.
- **Git**: commit as work lands; **no Co-Authored-By trailer** on commits. Push to `origin` once local tests pass and the repo is ready to be cloned onto the compute node.
- **This MacBook has no CUDA.** All GPU runs happen on a remote node. Deliverable therefore includes a concise `README.md` and one-shot scripts (`scripts/setup_core.sh`, `scripts/run_dev.sh`, `scripts/run_full.sh`) that take a fresh clone to a finished train+eval run with weights and granular benchmark results saved. Locally: run the CPU unit tests; attempt the tiny smoke test on CPU/MPS (real VAE, tiny DiT, a few real samples); if it needs CUDA, leave it for the node.
- **Code style**: lean, minimal, few or no comments; no speculative abstractions.
- **Results are saved granularly**: per-episode (success, steps, seed, task) JSONL, per-task JSON, per-suite and overall summary JSON, under `runs/<task>/<run_id>/eval/<benchmark>/<step>/`.

## Design summary

**Sequence (per sample, batched `[B, N, 5376]`)**: `[T text (pad to 64) | S obs rows (98 @224×448, 120 @384×320) | P proprio (1) | A actions (32) | V future video (2 latents × rows/frame)]`, N ≈ 390 (LIBERO) / 460 (RoboTwin). Replaces DiffSynth's `[1,S,C]` + `cu_seqlens` varlen loop with true batching, one shared boolean `[N,N]` mask and per-sample text key padding.

**Mask** (query reads key): T→{T,S}; S→{T,S}; V→{T,S,V}; A→{T,S,V,A}. Pad text keys masked for all queries; pad text queries still see S (no NaN).

**Per-row flow time / AdaLN** — 4 timestep groups per sample, modulation index `(b*4+g)*3+tag` so the pretrained `adaln_proj` layout `[modality][6 chunks][hidden]` loads unchanged:

| rows | DiT t (1=clean) | tag |
|---|---|---|
| T | 1.0 | 1 |
| S obs | 0.999 (H3 keyframe noise-aug) | 0 |
| P proprio | 1.0 | 2 |
| V | 1 − σ_v | 0 |
| A | 1 − σ_a | 2 |

**RoPE (t,h,w float)**: text t=0..63; obs t=64 + `_frame_grid`; proprio t=64, h=0, w=w_grid[-1]; video latent k at 64 + `_video_t_grid(k)` (spans (1,4,4,4,4)×5/3); action j at 64 + (j/8)·5/3, h=0, w=w_grid[-1]. Actions and video share the t axis → explicit temporal alignment.

**Loss**: MSE on V rows (all latents, masked by `image_is_pad`) × weight(σ_v) + MSE on A rows (masked by `action_is_pad`) × weight(σ_a), λ=1/1. Convention boundary: scheduler σ∈[0,1] (1 = noise), DiT t = 1−σ, DiT output negated to match target `noise − x`.

**Inference (one_pass_future_cache)**: V = pure noise at the first inference σ; one pass over `[T|S|P|V]` caches per-layer K,V (post q/k-norm + RoPE) for all 50 blocks; 10 Euler steps run the 50 blocks over the 32 A rows only, attending to cache + own K/V. Text embedding cached per instruction; text encoder never loaded in training.

## Package layout (`/Users/desmondzee/Hardware/WAM-H3/`)

```
pyproject.toml                    uv; torch 2.10 cu128, transformers 4.57.3, peft, accelerate, safetensors, hydra-core, omegaconf
                                  + FasterWAM runtime deps (datasets, pyarrow, av, jsonlines, boto3, termcolor, rich, einops, huggingface-hub, torchvision). No torchcodec (pyav fallback).
third_party/FasterWAM             git submodule (desmondzee/FasterWAM), `uv pip install --no-deps -e` (its torch 2.7.1 / transformers 4.49 pins conflict)
wam_h3/model/config.py            WAMH3Config (config.json fields + action_dim, proprio_dim, action_horizon=32, text_len=64, latent_h/w, num_video_latents=2, obs_t=0.999); from_pretrained_dir(); tiny()
wam_h3/model/layout.py            SequenceLayout: slices, row_group/row_tag, position_ids [N,3] (port _axis_from_sqrt_area/_frame_grid/_video_t_grid from DiffSynth pipeline), base_mask(), full_mask(text_valid)->[B,1,N,N], action_query_mask()
wam_h3/model/layers.py            batched ports keeping param names: apply_rope, TimeEmbedder, Attention(project/attend split), MLP, AdalnProj, TokenRefiner(+key mask), DiTBlock(kv_ctx, ctx_insert), FinalLayer(video_out, action_out zero-init)
wam_h3/model/dit.py               WAMH3DiT: video_patch_proj, action_in, proprio_in, condition_proj, time_embedder, rope, token_refiner, blocks, final_layer; embed(), modulation(t_groups[B,4]), forward(), prefill()->KVCache, denoise_action_step(), denoise_actions()
wam_h3/model/flow.py              FlowSchedule (port of FasterWAM scheduler_continuous.py, shift 5): sample_sigma, training_weight, add_noise, training_target, inference_schedule, step; sigma<->dit_t helpers
wam_h3/model/loss.py              training_loss(model, batch, sched_v, sched_a)
wam_h3/model/checkpoint.py        load_pretrained(model, dir|index_json) -> LoadReport(missing, unexpected); trainable_state_dict / load_trainable
wam_h3/model/lora.py              apply_lora(r, alpha=r, target regex `.*blocks\.\d+\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2)|adaln_proj\.linear)$`), mark heads trainable; fsdp_wrap_policy (per DiTBlock/TokenRefinerBlock)
wam_h3/model/text_encoder.py      MiniMaxH3TextEncoder (from DiffSynth te.py: Qwen3-VL, 50 retained layers, LM norm->Identity), presentation_t2va, encode_instructions()->[B,64,5120]+valid; per-string cache
wam_h3/model/vae.py               thin wrapper over FL2VA/video_vae (encode obs as image process_image=True; encode 5-frame clip; per-channel mean/std normalisation)
wam_h3/model/wam.py               WAMH3 facade: .dit/.vae/.text_encoder(optional), build_inputs(sample), training_loss(sample)->(loss, dict), encode_prompt(str) memoised, infer_action_one_pass_future_cache(**kwargs)->{"action":[T,D]}, save/load_checkpoint
wam_h3/model/modulation_cache.py  (later) precompute AdaLN outputs for fixed inference timesteps
wam_h3/data/dataset.py            WAMH3VideoDataset(fasterwam RobotVideoDataset): override _get_cached_text_context (suffix `.qwen3vl_len64.minimaxh3.pt`, in-memory dict, true mask) and _get (undo mask zeroing at robot_video_dataset.py:219-221; proprio -> row 0 [Dp])
wam_h3/data/text_cache.py         cache path/payload helpers
wam_h3/train/runtime.py           run_training (port of fasterwam runtime.py:368-390; call misc.register_work_dir(cfg.output_dir) before dataset)
wam_h3/train/trainer.py           WAMH3Trainer (new, small): accelerate FSDP, AdamW(1e-4, wd 1e-2, betas .9/.95), 5% warmup + cosine to 1% (copy trainer.py:204-253), clip 1.0, ResumableEpochSampler, ETA/logging
wam_h3/train/checkpoint.py        runs/<task>/<run_id>/checkpoints/weights/step_XXXXXX/{adapter.safetensors, trainer_state.json}; dataset_stats.json at run root
wam_h3/eval/libero.py             port of eval_libero_single.py:700-808 calling FasterWAM's run_single_task/_predict_action_chunk via sys.path
wam_h3/eval/robotwin_policy/      deploy_policy.py (get_model/encode_obs/eval/reset_model; keep tiling 271-284), deploy_policy.yml
wam_h3/eval/latency.py            stage breakdown: vae_obs_encode, prefill, action_denoise(10), to_cpu
configs/{train,sim_libero,sim_libero_plus,sim_robotwin}.yaml; configs/data/{libero_2cam,robotwin}.yaml (copied; _target_, num_frames 33, action_video_freq_ratio 8, context_len 64, cache dir); configs/model/{wamh3,wamh3_tiny}.yaml; configs/task/{libero_wamh3_dev (r=16, 1 GPU), libero_wamh3 (r=128, 8 GPU), robotwin_wamh3, smoke_libero_tiny}.yaml; configs/accelerate/{single.yaml, fsdp.yaml}
scripts/{setup_core.sh, precompute_text_embeds.py, train.py, train_single.sh, train_fsdp.sh, eval_libero.sh, eval_libero_plus.sh, eval_robotwin.sh, measure_latency.py, smoke_test.sh, run_dev.sh, run_full.sh}; simulator envs installed with FasterWAM's own scripts/setup/install_{libero,libero_plus,robotwin}.sh (separate uv venvs under third_party/FasterWAM/.venvs/)
  setup_core.sh   uv sync; FasterWAM --no-deps; hf download FL2VA/*; hf download LIBERO-fastwam lerobot_v30; install_libero.sh + install_libero_plus.sh; precompute text embeds
  run_dev.sh      setup (idempotent) -> train_single.sh task=libero_wamh3_dev -> eval_libero.sh + eval_libero_plus.sh on the final checkpoint -> summary printed
  run_full.sh     same with train_fsdp.sh 8 task=libero_wamh3
README.md                         what this is (5 lines), node requirements, `bash scripts/setup_core.sh`, `bash scripts/run_dev.sh`, `bash scripts/run_full.sh`, where weights/results land, how to resume/eval a given step
wam_h3/eval/results.py            write per-episode JSONL, per-task JSON, suite + overall summary JSON; summarize a run dir
tests/                            pytest, CPU, tiny config; tests/reference/minimax_h3_dit.py = vendored DiffSynth reference with SDPA attention + passthrough checkpointing
docs/superpowers/specs/2026-09-11-wam-h3-design.md   this design, committed
```

Reused from FasterWAM **unchanged** (imported): `fasterwam.datasets.lerobot.{base_lerobot_dataset, lerobot/**, processors.fastwam_processor, transforms.*, utils.normalizer}`, `fasterwam.utils.{samplers.ResumableEpochSampler, pytorch_utils, logging_config, config_resolvers, misc, fs}`, `experiments/libero/{libero_utils, action_ensembler}`, and the functions `run_single_task/_predict_action_chunk/_resolve_dataset_stats_path` from `eval_libero_single.py`.
Copied + edited: `robot_video_dataset.py` (subclass only), `precompute_text_embeds.py` (encoder swap, suffix, `--fake`), `trainer.py` (rewritten: line 68 dereferences the DeepSpeed plugin; `_apply_dit_only_train_mode` 286-295 would unfreeze the 33B backbone), config YAMLs, `eval_libero_single.py` main, `run_libero_manager.py` (CHUNK_ENTRY), `eval_robotwin_single.py` (POLICY_NAME, PROJECT_ROOT, drop mot overrides 223-232), `deploy_policy.py`, `measure_latency.py`.

Reference sources: `/private/tmp/claude-501/-Users-desmondzee-Hardware-H3-WAM/dabb3c4e-1d3d-4e15-ab06-4e6639011254/scratchpad/{minimax_h3_dit.py, minimax_h3_audio_video.py, te.py, loss.py}` (DiffSynth @300e3e4; re-fetch from GitHub if the scratchpad is gone), `/Users/desmondzee/Hardware/FasterWAM/src/fasterwam/models/wan22/{jointwam.py, fastwam.py, schedulers/scheduler_continuous.py}`, `/Users/desmondzee/Hardware/H3-WAM/code/diffsynth_h3_action.patch` (mask conventions only).

## Implementation steps (each step: write tests first, implement, run, commit)

### Phase 0 — repo skeleton
1. `pyproject.toml` (uv), `third_party/FasterWAM` submodule, `scripts/setup_core.sh` (`uv sync`; `uv pip install --no-deps -e third_party/FasterWAM`; `hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --local-dir .`), `wam_h3/__init__.py`, `tests/conftest.py`, write + commit the design spec.
   Verify: `uv run python -c "import fasterwam.datasets.lerobot.robot_video_dataset, transformers; assert transformers.__version__=='4.57.3'"`.

### Phase 1 — model package (CPU-testable with `WAMH3Config.tiny()`)
2. `config.py` + `layout.py`. Tests `test_layout.py`: slices/row counts; mask truth table (A columns False for every query; V rows see T,S,V; pad text col False everywhere; every query row has ≥1 True); position ids (obs t==64, video t==[64, 64+5/3], action t==64+j/8·5/3, action w==w_grid[-1]).
3. `layers.py`. Tests `test_layers.py`: batched `Attention` with all-True mask == reference `MiniMaxH3Attention` looped per sample (1e-6, `sdpa_kernel(MATH)`); `state_dict` key sets of `DiTBlock`/`TokenRefiner` equal the reference's.
4. `dit.py` joint forward + `checkpoint.py`. Tests: `test_equivalence.py` (proprio_dim=0, action_dim=32 so audio weights copy across; reference fed rows permuted to `[text|cond|audio|video]`, uniform timestep, all-visible mask → cond+video rows and action rows match, atol 1e-5); `test_weight_load.py` (50-layer tiny model, key names from `FL2VA/transformer/model.safetensors.index.json` → missing == {action_in.*, action_out.*, proprio_in.*}, unexpected == {audio_patch_proj.*, final_layer.audio_out.*}); `test_leakage.py` (perturb A rows and t_a → `torch.equal` on T/S/P/V hidden and video output); `test_independence.py` (vary t_a → only A-row modulation changes; vary t_v → A rows unchanged).
5. `flow.py` + `loss.py`. Tests: `training_weight` matches FasterWAM values at t∈{0,500,999}; Euler with the exact velocity from σ=1→0 recovers x; pad masks zero their contributions; after `apply_lora` grads reach only trainable params.
6. `prefill`/`denoise_action_step`/`denoise_actions`. Test `test_cache.py`: cached action output == joint forward action output (fp32 `allclose` 1e-6; key concat preserves joint order so expect exact); 3-step `denoise_actions` == manual joint-forward Euler loop.
7. `lora.py`. Test `test_lora.py`: LoRA module count = 5·layers + 4·refiner (excludes `final_layer.adaln_proj`); trainable keys ⊆ {`*.lora_A/B*`, 4 head modules}; forward at init == base forward; `trainable_state_dict` round-trips.
8. `text_encoder.py` + `vae.py` + `wam.py` facade (`build_inputs`, `training_loss`, `encode_prompt`, `infer_action_one_pass_future_cache(**kwargs)` accepting FasterWAM's kwarg set incl. `negative_prompt, text_cfg_scale, sigma_shift, rand_device, tiled, num_video_frames`). Tests: pad/valid collation on fake `[L,5120]`; >64 tokens raises; facade smoke with tiny DiT + mocked VAE.

### Phase 2 — data + text cache
9. `wam_h3/data/dataset.py`, `text_cache.py`, `scripts/precompute_text_embeds.py`. Tests `test_dataset_indices.py` (`range(0,33,8)`), `test_text_cache.py` (`--fake` cache → reload `[64,5120]` bf16, mask sum == token count). Verify a real sample: `video [3,5,224,448]` in [-1,1], `action [32,7]`, `proprio [8]`, `context [64,5120]`, `context_mask [64]`.
10. Hydra configs (data/model/task/train/accelerate). Verify `hydra compose train task=libero_wamh3` resolves with no missing interpolations; LIBERO-fastwam has 277,713 frames → 2,170 steps/epoch at global batch 128 → 21,700 steps (matches FasterWAM's released `step_021700`).

### Phase 3 — training (single-GPU first)
11. `trainer.py`, `train/runtime.py`, `train/checkpoint.py`, `scripts/train.py`. The trainer is launcher-agnostic: with `configs/accelerate/single.yaml` it runs one process, no sharding (bf16 base resident, LoRA params fp32); with `configs/accelerate/fsdp.yaml` it runs FULL_SHARD, `fsdp_version: 2` (per-parameter sharding lets bf16 frozen base and fp32 LoRA coexist; fallback FSDP1 with bf16 LoRA), auto-wrap on DiTBlock, `use_orig_params`. Both: activation checkpointing per block; only `model.dit` is prepared (VAE frozen, `no_grad`); optimizer built after `prepare` over `requires_grad` params; `save_every` + final save via `train/checkpoint.py`; `--resume` from a weights dir restores step/epoch/sampler offset. Tests: `test_schedule.py` (LR at 0 / warmup end / final), `test_checkpoint_io.py` (tiny peft round-trip bit-exact; final save present after a 3-step tiny run).
12. `scripts/smoke_test.sh`: `configs/model/wamh3_tiny.yaml` (random 2-layer DiT, real frozen VAE), 8 real LIBERO samples, `max_steps=8 batch_size=1`, then eval 1 trial, then latency. Device auto (cuda > mps > cpu); run the train part locally on the MacBook if the VAE runs without CUDA (download only `FL2VA/video_vae/*` + one LIBERO episode), eval/latency parts on the node. Verify loss finite, trainable count == LoRA+heads, backbone grads None, checkpoint dir written with `adapter.safetensors`.
13. **Dev run** `scripts/train_single.sh task=libero_wamh3_dev`: real 33B weights, rank 16, batch 2 × grad-accum 8 (global 16), `dataset_dirs=[libero_spatial]` (~1/4 of LIBERO), `max_steps=3000`, `save_every=500`, lr 1e-4. Verify: peak memory < 78 GB, loss_video and loss_action both trend down over the run, checkpoints at 500/1000/…/3000 + final, `load_checkpoint` on a fresh process reproduces the saved loss on a fixed batch.

### Phase 4 — eval (LIBERO, then LIBERO-Plus)
14. `wam_h3/eval/libero.py` + `scripts/eval_libero.sh` (wraps `third_party/FasterWAM/scripts/setup/install_libero.sh`'s venv and `run_libero_manager.py` with our `configs/sim_libero.yaml`: `load_text_encoder: true`, replan 10, ensembler off, 10 steps, `CHUNK_ENTRY` → our worker). Verify `EVALUATION.num_trials=1` completes on the smoke checkpoint; then run the dev checkpoint on `libero_spatial` (its training suite) as the first real success-rate number; then all 4 suites.
15. LIBERO-Plus: `configs/sim_libero_plus.yaml` (copy of FasterWAM's; `benchmark_name: libero-plus`, `task_sample_ratio 0.15`) + `scripts/eval_libero_plus.sh` mirroring `eval_fasterwam_libero_plus.sh` (LIBERO-plus @4976dc3 via `install_libero_plus.sh`, `LIBERO_CONFIG_PATH` → `.runtime/libero-plus`). Same checkpoint as step 14. Verify one sampled task completes, then the full sampled set.
16. `wam_h3/eval/latency.py` / `scripts/measure_latency.py`: seeded random weights, 5 warmup / 10 iters, stages above, staged total within 5% of direct.
16b. `README.md`, `scripts/run_dev.sh`, `scripts/run_full.sh`, `wam_h3/eval/results.py`; `bash -n` all scripts; dry-run `run_dev.sh --dry-run` prints the command sequence. Commit and push.

### Phase 5 — full run
17. `scripts/train_fsdp.sh 8 task=libero_wamh3` (rank 128, batch 16 × 8, 21,700 steps, `save_every 2000` + final); LIBERO 4 suites + LIBERO-Plus on the final checkpoint. If OOM at batch 16: `gradient_accumulation_steps: 2, batch_size: 8`.

### Phase 6 — later
18. RoboTwin: `deploy_policy.py`, `scripts/eval_robotwin.sh` (symlink `RoboTwin/policy/wamh3_policy -> wam_h3/eval/robotwin_policy`), replan 28, `configs/task/robotwin_wamh3.yaml`. Risk: FasterWAM's RoboTwin env pins torch 2.4.1 — rebuild on 2.10; fallback is a model-server process.
19. `modulation_cache.py` + test `forward(mod_override) == forward()`; drops 13B AdaLN weights from inference.

## Verification (end to end)
- Locally (MacBook): `uv run pytest tests/` green on CPU (tiny config) — covers equivalence with the reference DiT, mask leakage, cache exactness, timestep independence, weight-key mapping, LoRA targeting, schedule, checkpoint IO, dataset indices, text cache; `bash -n` on all scripts; `run_dev.sh --dry-run`. Then commit + push.
- On the node (not this machine): everything below via `bash scripts/setup_core.sh && bash scripts/run_dev.sh`.
- `scripts/smoke_test.sh` on 1 GPU: train 8 steps → eval 1 LIBERO trial → latency JSON with 4 stages.
- Single-GPU rank-16 dev run on `libero_spatial`: both losses decrease, checkpoints saved every 500 steps + final, reloaded checkpoint reproduces loss; LIBERO `libero_spatial` success rate > 0 with the dev checkpoint; LIBERO-Plus sampled tasks run end to end on the same checkpoint.
- Full 8-GPU rank-128 run reaches 21,700 steps with memory headroom; LIBERO 4-suite and LIBERO-Plus success rates reported; latency script reports ~160–180 ms/chunk on one H100.

## Compute summary (for reference)
Trunk 17.3B (per-token) + AdaLN 13B (per-sample); ~390 rows/sample; ≈40 TFLOP/sample with LoRA + checkpointing; 2.5M samples (10 epochs) ≈ 1e20 FLOP ≈ 80 GPU-h at ~350 TFLOPS effective. Memory/GPU under 8-way FSDP ≈ 16 GB at batch 16.
