# FL2VA KV-Conditioned LIBERO Action Policy

## Goal

Train a policy that predicts LIBERO robot-action chunks from language and a
causal history of the two camera streams. A frozen MiniMax H3 FL2VA model holds
all temporal state. A smaller, stateless action transformer reads K/V features
from a sparse subset of FL2VA layers and predicts the next action chunk.

The policy receives:

- the natural-language task instruction;
- `agentview_rgb` and `eye_in_hand_rgb`, spatially concatenated per timestep;
- no robot proprioception;
- no previous actions;
- no recurrent or persistent state of its own.

The initial or final outcome frame used by the video-generation experiments
must not be supplied as future conditioning. The initial observed frame may be
included in the language/vision conditioning because it is available at
deployment time.

## Prompt augmentation and physical agency

Raw LIBERO instructions describe the desired state change but do not always
say that the robot must cause it. For a video model, that ambiguity can produce
object motion that is not coupled to the arm. Every training prompt should
therefore make the visible robot and gripper the causal agent while preserving
the original high-level task semantics.

Given an original instruction such as:

```text
pick up the black bowl next to the cookie box and place it on the plate
```

sample from controlled variants such as:

```text
The visible robot arm uses its gripper to pick up the black bowl next to the
cookie box and place it on the plate.

Using its gripper, the robot arm physically performs the task: pick up the
black bowl next to the cookie box and place it on the plate.

The robot interacts with the environment to complete the task, grasping and
moving the black bowl next to the cookie box onto the plate.

From the synchronized overhead and wrist-mounted camera views, the same robot
arm uses its gripper to pick up the black bowl next to the cookie box and place
it on the plate.
```

The augmentation should vary wording, not behavior. It must not introduce
low-level motion plans, change object relations, or imply that an external
agent moves an object. Positive descriptions of physical contact are preferred
to long lists of forbidden artifacts.

Because Qwen is frozen, precompute several prompt embeddings per episode (for
example four to eight variants) and sample one during training. This preserves
prompt diversity without putting Qwen in the steady-state training pipeline.

## Frozen models

| Component | Checkpoint | Quantization | File size |
|---|---|---|---:|
| Video transformer | `minimax_h3_fl2va_int8_convrot.safetensors` | INT8 ConvRot | 34.04 GB / 31.70 GiB |
| Qwen3-VL-32B | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | INT8 ConvRot | 27.14 GB / about 25.28 GiB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | FP16 | 5.21 GB / 4.85 GiB |
| Total | | | about 66.4 GB / 61.8 GiB |

Official Comfy-Org repository:
<https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main>

The INT8 Qwen checkpoint is preferable on H100. H100 has compute capability
9.x, whereas ComfyUI's native NVFP4 path requires capability 10.x; NVFP4 would
therefore use an emulated/dequantized computation path.

## FL2VA dimensions

- Transformer layers: 50
- Hidden width: 5,376
- Attention heads: 56
- Attention head dimension: 128
- Width of each projected Q, K, or V: 7,168
- Combined QKV projection width: 21,504
- FFN width: 14,336
- Text-conditioning input width: 5,120
- Video latent channels: 24
- Spatial patch size: `(1, 2, 2)` after VAE compression

A useful initial sparse layer selection is:

```python
fl2va_layers = [7, 15, 23, 31, 39, 49]
```

Capturing full BF16 K and V at 768x384 and 124 frames costs approximately
291 MiB per selected layer per sample. Six uncompressed layers require roughly
1.7 GiB per sample, so the training pipeline should transfer captures promptly
and avoid retaining autograd graphs in frozen FL2VA.

## Video and latent grids

ComfyUI's H3 implementation accepts video lengths on this grid:

```text
pixel frames = 5 + 17k
VAE latents  = 2 +  5k
```

Examples:

```text
frames:   5, 22, 39, 56, 73, 90, 107, 124, 141, ...
latents:  2,  7, 12, 17, 22, 27,  32,  37,  42, ...
```

The VAE internally handles 17-frame temporal clips with fourfold temporal
compression. Action chunks do not have to equal the latent increment, but a
34-action chunk spans exactly two 17-frame video-grid increments.

LIBERO is recorded at 20 Hz while FL2VA assumes 24 fps. We intentionally do not
resample. Frame and action indices remain paired exactly; FL2VA merely perceives
the robot motion as 20% faster. A 34-action chunk lasts 1.7 seconds in LIBERO
and is interpreted as about 1.42 seconds by FL2VA.

## Correct LIBERO transition alignment

LIBERO's processed HDF5 generation calls `env.step(action)` and then stores the
returned camera observation at the same array index. Therefore:

```text
previous state -- action[i] --> stored observation[i]
```

Given the latest stored frame `obs[t]`, the first causally predictable control
is `action[t + 1]`. `action[0]` is omitted because it produced `obs[0]` before
the stored visual sequence begins. It could only be recovered as a training
target by rendering the corresponding pre-action simulator state.

## Initialization and chunk alignment

Repeat the first paired-camera observation five times to satisfy FL2VA's
minimum input grid without waiting for the task to advance:

```text
[obs_0, obs_0, obs_0, obs_0, obs_0]
                 |
                 +--> predict actions 1..34
                      corresponding to observations 1..34
```

After each executed 34-action chunk, append the corresponding 34 real frames:

```text
FL2VA context length 5:
    [obs_0 x 5]
    -> predict actions 1..34
    -> next observed video chunk is obs_1..obs_34

FL2VA context length 39:
    [obs_0 x 5, obs_1..obs_34]
    -> predict actions 35..68
    -> next observed video chunk is obs_35..obs_68

FL2VA context length 73:
    [obs_0 x 5, obs_1..obs_68]
    -> predict actions 69..102
```

General training indexing:

```python
chunk_size = 34
t = chunk_size * chunk_index

video_context = [obs[0]] * 5 + list(obs[1 : t + 1])
action_indices = t + 1 + torch.arange(chunk_size)
action_target = actions[action_indices.clamp_max(len(actions) - 1)]
loss_mask = action_indices < len(actions)
```

The final chunk is not padded with invented actions. Positions without a
corresponding LIBERO frame/action are excluded from the loss:

```python
loss = (per_action_loss * loss_mask).sum() / loss_mask.sum()
```

## Causal FL2VA masking

Both camera images at a timestamp form one spatial frame. Their spatial tokens
may attend freely to each other. Across time, frame `t` may attend only to text
and frames at or before `t`.

```text
Keys ->       Text   F0   F1   F2   ...   Ft
Queries
Text            Y     .    .    .          .
F0              Y     Y    .    .          .
F1              Y     Y    Y    .          .
F2              Y     Y    Y    Y          .
...
Ft              Y     Y    Y    Y    ...   Y
```

Text tokens must remain text-only/static. If text tokens attend to video, a
future frame can contaminate text at one layer and leak back into an earlier
video token at a later layer. The causal restriction must therefore be applied
inside every FL2VA layer, not only when the action head reads the captured K/V.

The VAE must also be causal with respect to the policy decision. Never encode a
temporal clip containing frames later than the decision point. Padding a causal
prefix by repeating its latest available frame is safe; encoding future ground-
truth frames and attempting to hide them only in FL2VA is not safe.

## Action-chunk model

The downstream network is a stateless cross-attention action head. All 34
actions are predicted in parallel from the same FL2VA observation prefix:

```text
text + causal paired-camera history
                 |
      frozen FL2VA, one noise/denoising step
                 |
   sparse K/V taps from selected FL2VA layers
                 |
   34 learned horizon queries in action transformer
                 |
        34 x 7D LIBERO action predictions
```

Suggested starting configuration:

- approximately 250--350M trainable parameters;
- model width 1,024;
- 12 transformer blocks;
- 16 attention heads of width 64;
- SwiGLU FFN width 4,096;
- self-attention among all 34 action-horizon query tokens in every block;
- cross-attention in six blocks, one for each selected FL2VA depth;
- 34 learned horizon-query embeddings;
- BF16 training.

Each query is identified by its horizon offset but receives no previous action.
The 34 queries use bidirectional self-attention so the predicted action chunk
can be internally coordinated. This is safe because the self-attention K/V are
computed only from learned horizon queries and the action transformer's own
hidden states, never from ground-truth or previously executed actions. Selected
blocks then cross-attend to the frozen FL2VA K/V memory:

```text
x = x + self_attention(norm(x))
        Q, K, V all come from the action-transformer horizon tokens

x = x + cross_attention(norm(x), video_k[layer], video_v[layer])
        Q comes from the action transformer; K and V come from FL2VA

x = x + mlp(norm(x))
```

Thus the action transformer attends both to FL2VA's video K/V and to its own
K/V, but it does not retain either action-transformer state or predicted-action
tokens between policy invocations. FL2VA's causal video context contains the
cross-invocation temporal state.

FL2VA's post-RoPE keys are specialized for its own spatial/temporal queries.
Two interfaces should be evaluated:

1. Read the actual selected-layer K/V cache and give the action queries a
   compatible positional treatment.
2. Preferably capture each selected layer's 5,376-wide hidden states and learn
   fresh action-head K/V projections. This is simpler and avoids inheriting
   FL2VA's query-specific RoPE geometry.

## Single-step feature extraction

FL2VA remains frozen and is evaluated for one selected diffusion timestep. The
real video latent prefix should be perturbed according to that timestep's sigma
rather than being passed as a clean latent at an unrelated noise level.

Initially, use a fixed low-to-medium sigma for a stable feature distribution.
Sampling from a narrow sigma range during training is a useful robustness
experiment even if deployment uses one fixed sigma. Forward hooks must detach
captured tensors immediately and must not preserve FL2VA autograd graphs.

## Two-H100 training pipeline

ComfyUI supports component placement and multi-GPU work-unit splitting, but not
tensor-parallel sharding of individual FL2VA matrix multiplications. The useful
training arrangement is therefore pipeline parallelism:

```text
GPU 0: frozen FL2VA single-step forward and sparse feature extraction
GPU 1: trainable action transformer forward/backward/optimizer
```

Qwen conditioning and VAE video latents should be precomputed and cached. They
do not change while FL2VA is frozen, so Qwen and the VAE need not occupy either
GPU in the steady-state training loop.

Double-buffer microbatches:

```text
time --->

GPU 0:  FL2VA(batch 0) | FL2VA(batch 1) | FL2VA(batch 2) | ...
GPU 1:                  | policy(batch 0)| policy(batch 1)| ...
```

Transfer only the selected detached features over GPU peer-to-peer links. The
action model's roughly 300M parameters and BF16/FP32 AdamW training state should
fit comfortably on the second H100; feature/KV volume and attention activation
memory, rather than policy weights, will be the main batching constraint.

## Main experimental risks

- Future leakage through an insufficient FL2VA attention mask.
- Future leakage introduced by noncausal temporal VAE encoding.
- Off-by-one action targets; processed LIBERO frames correspond to the action
  that produced them, so prediction begins at the following action.
- A 34-action chunk is a 1.7-second open-loop horizon at 20 Hz. Training can
  predict all 34 while deployment optionally executes a shorter prefix before
  replanning, although early replanning requires a deliberate strategy for the
  FL2VA input grid.
- Full uncompressed K/V captures are large. Layer count, token reduction, and
  learned adapters should be ablated before scaling the trainable policy.
- Direct FL2VA K reuse may be inferior to learning new projections from hidden
  states because of QK normalization and 3D RoPE.
