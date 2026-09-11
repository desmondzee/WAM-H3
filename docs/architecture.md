# WAM-H3 architecture

A record of the model being trained: a World Action Model built on the MiniMax-H3 33B DiT (reference: `tests/reference/minimax_h3_dit.py`, DiffSynth port of the FL2VA transformer). Code: `wam_h3/model/{dit,layers,layout,loss,lora,wam}.py`. Design rationale: `docs/superpowers/specs/2026-09-11-wam-h3-design.md`.

Numbers below are for LIBERO (`configs/data/libero_2cam.yaml`) and the single-GPU run `configs/task/libero_wamh3_1gpu.yaml` (LoRA rank 64, AdaLN rank 16). The dev run uses rank 16 everywhere; the 8-GPU full run uses rank 128.

**Legend** (used in every diagram)

```mermaid
flowchart LR
  F["Frozen, pretrained H3"]:::frozen
  L["Frozen + LoRA"]:::lora
  N["New, fully trained"]:::new
  M["Changed vs H3<br/>(same weights, new wiring or schedule)"]:::changed
  R["Removed / unused"]:::removed
  classDef frozen fill:#e8edf3,stroke:#6b7f99,color:#1f2d3d
  classDef lora fill:#fde8c8,stroke:#d9822b,color:#3d2a10
  classDef new fill:#d4f4dd,stroke:#2e9e5b,color:#10331f
  classDef changed fill:#eadcf8,stroke:#8e5bd0,color:#2a1540
  classDef removed fill:#fbe1e1,stroke:#c94a4a,color:#401010,stroke-dasharray:5 5
```

## 1. Training forward pass

One LIBERO sample. Both cameras (agentview + wrist, 224×224 each) are concatenated side by side to 224×448. Frame 0 of the clip is the current observation.

```mermaid
flowchart TB
  subgraph IN["Inputs"]
    instr["Instruction text"]
    clip["5-frame clip, 2 cameras, 224×448<br/>env steps 0, 8, 16, 24, 32<br/>frame 0 = current observation"]
    prop["Proprio state, 8-d"]
    act["Action chunk, 32 steps × 7<br/>6 arm + 1 gripper"]
  end

  subgraph ENC["Frozen encoders, outside the DiT"]
    te["Qwen3-VL text encoder<br/>run once offline, cached<br/>→ 64 × 5120"]:::frozen
    vae_o["H3 video VAE, image mode<br/>frame 0 → 1 latent<br/>→ 98 rows × 96"]:::frozen
    vae_v["H3 video VAE, clip mode<br/>5 frames → 2 latents<br/>→ 196 rows × 96"]:::frozen
  end

  instr --> te
  clip -- "frame 0" --> vae_o
  clip --> vae_v

  nv["+ noise at σ_v<br/>(σ_v = 1 with p = 0.3)"]:::changed
  na["+ noise at σ_a<br/>independent of σ_v"]:::changed
  vae_v --> nv
  act --> na

  subgraph EMB["Row embeddings → 5376-d"]
    cp["condition_proj"]:::frozen
    tr["Token refiner, 2 blocks<br/>LoRA on qkv, out, fc1, fc2"]:::lora
    vpo["video_patch_proj"]:::frozen
    pin["proprio_in<br/>init from audio_patch_proj"]:::new
    ain["action_in<br/>init from audio_patch_proj"]:::new
    vpv["video_patch_proj"]:::frozen
  end

  te --> cp --> tr
  vae_o --> vpo
  prop --> pin
  na --> ain
  nv --> vpv

  seq["Joint sequence, N = 391 rows<br/>Text 64 | Obs 98 | Proprio 1 | Actions 32 | Future video 196"]:::changed
  tr --> seq
  vpo --> seq
  pin --> seq
  ain --> seq
  vpv --> seq

  tg["4 timestep groups per sample<br/>proprio 1.0 · obs 0.999 · text+video 1−σ_v · actions 1−σ_a"]:::changed
  temb["time_embedder"]:::frozen
  tg --> temb

  blocks["50 × DiT block<br/>asymmetric mask: nothing reads action rows<br/>LoRA r=64 on qkv, out, fc1, fc2 · r=16 on adaln_proj"]:::lora
  seq --> blocks
  temb --> blocks

  fl["Final layer: RMSNorm + AdaLN"]:::frozen
  blocks --> fl
  vout["video_out<br/>on future-video rows"]:::frozen
  aout["action_out<br/>zero-initialised"]:::new
  fl --> vout
  fl --> aout
  aud_in["audio_patch_proj<br/>only used to init action_in, proprio_in"]:::removed
  aud_out["audio_out<br/>not loaded"]:::removed

  lv["Video flow loss<br/>latent 1 only (latent 0 duplicates obs)<br/>weight peaks at σ_v = 1"]:::changed
  la["Action flow loss<br/>FasterWAM weighting"]:::new
  vout --> lv
  aout --> la

  classDef frozen fill:#e8edf3,stroke:#6b7f99,color:#1f2d3d
  classDef lora fill:#fde8c8,stroke:#d9822b,color:#3d2a10
  classDef new fill:#d4f4dd,stroke:#2e9e5b,color:#10331f
  classDef changed fill:#eadcf8,stroke:#8e5bd0,color:#2a1540
  classDef removed fill:#fbe1e1,stroke:#c94a4a,color:#401010,stroke-dasharray:5 5
```

Loss = video loss + action loss (λ = 1 each). Both are velocity targets `noise − x` (the DiT output is negated to match).

## 2. Sequence layout and per-row conditioning

Every row carries three pieces of conditioning: which AdaLN modality slot it uses (the "tag"), which of the 4 per-sample timesteps it gets, and a 3-D RoPE position. H3's `adaln_proj` produces 3 modality slots × 6 modulation chunks; each row picks its (timestep group, tag) pair, so the pretrained weight layout loads unchanged.

| Block | Rows | Content | AdaLN tag (H3 slot) | Timestep (1 = clean) | RoPE (t, h, w) |
|---|---|---|---|---|---|
| Text | 64 (right-aligned, padding masked) | refined Qwen3-VL embedding | 1 (text) | 1 − σ_v, like stock H3 | t = 0…63 |
| Obs | 98 (7 × 14 patches) | VAE latent of current frame | 0 (video) | 0.999 (H3 keyframe noise-aug) | t = 64, frame grid |
| Proprio | 1 | `proprio_in(state)` | **2 (audio)** | 1.0 | t = 64, h = 0, w = first column |
| Actions | 32 | `action_in(noisy action)` | **2 (audio)** | 1 − σ_a | t = 64 + (j/8)·5/3, h = 0, w = last column |
| Future video | 196 (2 latents × 98) | noisy VAE latents | 0 (video) | 1 − σ_v | t = 64 + H3 video time grid, frame grid |

Actions and video share the RoPE time axis, so action step *j* sits at the same time coordinate as the video frame it lines up with.

## 3. Who attends to what

One boolean N × N mask, shared by all 50 blocks (`wam_h3/model/layout.py`). Rows = queries, columns = keys.

| query ↓ reads key → | Text | Obs | Proprio | Actions | Future video |
|---|---|---|---|---|---|
| **Text** | ✓ | ✓ | ✓ | ✗ | ✓ ¹ |
| **Obs** | ✓ | ✓ | ✓ | ✗ | ✓ ¹ |
| **Proprio** | ✓ | ✓ | ✓ | ✗ | ✓ ¹ |
| **Actions** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Future video** | ✓ | ✓ | ✓ | ✗ | ✓ |

¹ `model.ctx_sees_video=true` (default, like H3 pretraining). Setting it to `false` gives FasterWAM's first-frame-causal variant, where text, obs and proprio read only each other.

Padded text keys are masked for every query. The token refiner has its own text-only attention over the 64 text rows.

```mermaid
flowchart LR
  subgraph CTX["Context rows (read each other)"]
    T["Text 64"]:::frozen
    S["Obs 98"]:::frozen
    P["Proprio 1"]:::new
  end
  V["Future video 196<br/>(reads itself)"]:::frozen
  A["Actions 32<br/>(reads itself)"]:::new
  CTX -- "reads (ctx_sees_video)" --> V
  V -- "reads" --> CTX
  A -- "reads" --> CTX
  A -- "reads" --> V
  X["Nothing outside the action block reads it"]:::changed
  A -.- X
  classDef frozen fill:#e8edf3,stroke:#6b7f99,color:#1f2d3d
  classDef new fill:#d4f4dd,stroke:#2e9e5b,color:#10331f
  classDef changed fill:#eadcf8,stroke:#8e5bd0,color:#2a1540
```

**Why the asymmetry matters.** Because no context or video row depends on the actions, the context and video K/V at every layer are the same whatever the action values are. At inference they are computed once and reused across all action denoising steps (section 5). Stock H3 has no such restriction: text, video and audio all attend to each other bidirectionally within a sample.

## 4. Inside one DiT block (× 50)

```mermaid
flowchart TB
  x["x: N × 5376"]
  subgraph ADA["Per-row AdaLN modulation"]
    t4["t_emb for 4 timestep groups"]:::frozen
    ap["adaln_proj.linear<br/>2688 → 3 slots × 6 × 5376<br/>+ LoRA r=16"]:::lora
    pick["each row picks its (group, tag)<br/>shift1 scale1 gate1 shift2 scale2 gate2"]:::changed
    t4 --> ap --> pick
  end
  n1["RMSNorm"]:::frozen
  m1["modulate: shift1, scale1"]
  qkv["qkv_proj 5376 → 3 × 7168<br/>+ LoRA r=64"]:::lora
  qk["q/k RMSNorm"]:::frozen
  rope["3-D RoPE"]:::frozen
  sdpa["SDPA, 56 heads × 128<br/>boolean mask from section 3"]:::changed
  op["out_proj 7168 → 5376<br/>+ LoRA r=64"]:::lora
  g1["x + gate1 × attn"]
  n2["RMSNorm"]:::frozen
  m2["modulate: shift2, scale2"]
  fc1["fc1 5376 → 2 × 14336<br/>+ LoRA r=64"]:::lora
  swi["SwiGLU"]
  fc2["fc2 14336 → 5376<br/>+ LoRA r=64"]:::lora
  g2["x + gate2 × mlp"]
  out["x: N × 5376"]

  x --> n1 --> m1 --> qkv --> qk --> rope --> sdpa --> op --> g1
  x --> g1
  g1 --> n2 --> m2 --> fc1 --> swi --> fc2 --> g2
  g1 --> g2
  g2 --> out
  pick -.-> m1
  pick -.-> g1
  pick -.-> m2
  pick -.-> g2

  classDef frozen fill:#e8edf3,stroke:#6b7f99,color:#1f2d3d
  classDef lora fill:#fde8c8,stroke:#d9822b,color:#3d2a10
  classDef changed fill:#eadcf8,stroke:#8e5bd0,color:#2a1540
```

`adaln_proj.linear` gets its own, lower rank (`model.lora_adaln_r`) because its only input is the timestep embedding. It holds 13B of the 33B base weights, so at a uniform rank it would take half of all LoRA parameters. The final layer's AdaLN stays frozen because it also modulates the frozen `video_out`.

## 5. Inference: one context pass, then 10 action-only steps

The future video is never denoised. It is fed as pure noise once, only so the context rows see the same input they saw during training at σ_v = 1.

```mermaid
sequenceDiagram
  participant Env as LIBERO env
  participant Enc as VAE + text cache
  participant DiT as 50 DiT blocks
  participant Cache as Per-layer K/V cache
  participant Head as action_out + Euler
  Env->>Enc: current frame (2 cams), proprio, instruction
  Enc->>DiT: prefill rows: Text 64 | Obs 98 | Proprio 1 | Video 196 (pure noise)
  Note over DiT: one pass over 359 rows, action rows absent
  DiT->>Cache: store K, V (after q/k norm + RoPE) for all 50 layers
  loop 10 Euler steps, σ_a from 1 to 0
    Head->>DiT: 32 noisy action rows at timestep 1 − σ_a
    Cache-->>DiT: cached K/V, with the action K/V inserted between proprio and video
    DiT->>Head: action velocity
    Head->>Head: x ← x + Δσ × velocity
  end
  Head->>Env: 32 × 7 action chunk (execute 10, then replan)
```

Because of the mask in section 3, this reproduces a full forward pass exactly: each step runs the 50 blocks over just the 32 action rows.

## 6. What changed vs the reference H3

| | MiniMax-H3 (reference) | WAM-H3 |
|---|---|---|
| Task | text/image → joint video + audio generation | robot action prediction, with future-video prediction as an auxiliary loss |
| Modality slots | video, text, audio | video slot: obs + future video · text slot: instruction · **audio slot: proprio + actions** |
| Input projections | `video_patch_proj`, `condition_proj`, `audio_patch_proj` | first two frozen and reused · `audio_patch_proj` replaced by new `action_in` / `proprio_in` initialised from its weights |
| Output heads | `video_out`, `audio_out` | `video_out` frozen, used for the training loss only · `audio_out` dropped · new zero-initialised `action_out` |
| Attention | full bidirectional per sample, packed varlen (`cu_seqlens`) | batched `[B, N]` with one boolean mask; nothing reads action rows |
| Timesteps | per-row timesteps already supported (`unique_timesteps` + `inverse_indices`): keyframe rows at 0.999, text rows carry the video timestep | same mechanism, fixed to 4 groups per sample; the action rows get their own flow time, sampled independently of the video's (shift 5 for both) |
| Video loss weight | n/a (pretraining) | bump centred at σ_v = 1, with σ_v = 1 forced 30% of the time (FasterWAM centres at 0.5) |
| Inference | iterative denoising of video + audio | one prefill pass, K/V cache, 10 action-only Euler steps; video never denoised |
| Trained parameters | all | LoRA + 3 heads; everything else frozen |

**Parameter counts** (real H3 config, counted on the meta device):

| Config | Frozen base | Trainable | of which AdaLN LoRA | fp32 weights + grads + AdamW |
|---|---|---|---|---|
| Dev (r = 16) | 33.12B | 157M | 80M | 2.5 GB |
| **1-GPU run (r = 64, AdaLN r = 16)** | 33.12B | **390M** | 80M | 6.2 GB |
| Full run (r = 128) | 33.12B | 1,257M | 637M | 20.1 GB |

Trainable parameters of the 1-GPU run: about 298M block LoRA (attention + MLP), 80M AdaLN LoRA, 12M token-refiner LoRA and 0.1M in the three heads.
