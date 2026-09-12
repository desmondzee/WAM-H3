# Refined Action Policy v2 — Changes Only

This document amends `refined_action_policy.md`. Everything not changed below remains as specified there.

## 1. Alternating cross-attention and self-attention blocks

Replace the original arrangement with six pairs of blocks:

| Block | Attention | Following sublayer |
|---|---|---|
| 0 | Cross-attention to FL2VA layer 7 | MLP |
| 1 | Self-attention | MLP |
| 2 | Cross-attention to FL2VA layer 15 | MLP |
| 3 | Self-attention | MLP |
| 4 | Cross-attention to FL2VA layer 23 | MLP |
| 5 | Self-attention | MLP |
| 6 | Cross-attention to FL2VA layer 31 | MLP |
| 7 | Self-attention | MLP |
| 8 | Cross-attention to FL2VA layer 39 | MLP |
| 9 | Self-attention | MLP |
| 10 | Cross-attention to FL2VA layer 49 | MLP |
| 11 | Self-attention | MLP |

Each attention and MLP sublayer has its own pre-normalization and residual connection:

```text
Cross block:
    x = x + cross_attention(cross_norm(x), historical_memory)
    x = x + mlp(mlp_norm(x))

Self block:
    x = x + self_attention(self_norm(x))
    x = x + mlp(mlp_norm(x))
```

The first cross-attention uses the 34 learned horizon embeddings as queries. It grounds the tokens in historical observations and instruction conditioning before any self-attention attempts to coordinate them. There is no initial self-attention over otherwise unconditioned learned embeddings.

Self-attention and cross-attention use separate parameters, softmax operations, and residual paths. Action tokens and video tokens do not compete within a single joint attention softmax. This does not mathematically prevent one residual update from dominating another; ordinary pre-norm residuals are the starting point, with residual scaling or gates optional if training requires them.

### Parameter budget

Keep policy width 1,024, 16 attention heads of width 64, and SwiGLU intermediate width 4,096. Prefer selected FL2VA hidden states of width 5,376 with fresh, per-tap learned K/V projections on the policy GPU.

For policy width `d`, memory width `H`, and SwiGLU intermediate width `f`, the main bias-free parameter counts are:

```text
self-attention:  4 d²
cross-attention: 2 d² + 2 H d
SwiGLU MLP:     3 d f
```

With six self-attention sublayers, six cross-attention sublayers, and twelve MLPs:

| Component | Parameters |
|---|---:|
| Self-attention | 25.2M |
| Cross-attention | 78.6M |
| MLPs | 151.0M |
| Total | approximately 255M |

Normalization, learned horizon embeddings, and the 7D output head add little to this total. These estimates assume independent parameters per block and the hidden-state interface, not direct native FL2VA K/V reuse.

Optional smaller baselines keep width 1,024 and reduce only the SwiGLU intermediate width:

| Intermediate width | Approximate total |
|---|---:|
| 2,048 | 179M |
| 2,560 | 198M |
| 3,072 | 217M |
| 4,096 | 255M |

Start with the 255M configuration. Feature memory and long-memory attention activations still need profiling; parameter count alone does not establish a safe batch size.

## 2. Episode-level feature reuse and two-GPU training

### Hardware correction

The current machine has two H100 PCIe GPUs, each reporting 81,559 MiB of memory. Their topology is `PHB`, not NVLink. NVIDIA topology queries report peer reads and writes as unsupported.

Replace the original assumption of direct GPU peer-to-peer feature transfer with pinned-host-memory staging, unless a runtime capability test establishes another supported transfer path.

### Initial pipeline

```text
Offline: precompute prompt conditioning and verified causal VAE latents

GPU 0: frozen quantized FL2VA extraction
    -> detached selected-layer hidden features
    -> bounded pinned-host-memory queue
GPU 1: learned memory projections + action policy + backward + optimizer
```

Keep Qwen and the VAE off the GPUs during steady-state training. Run FL2VA frozen, in evaluation mode, without constructing autograd graphs. Ensure tensors passed to trainable projections can be saved for backward; detached ordinary tensors from a no-grad extraction path are suitable.

Process an episode once under the causal constraints below and reuse its features for multiple decision points. Do not independently rerun FL2VA on every overlapping prefix when full-episode/prefix equivalence has been established.

```text
one episode's frozen features
    -> decision at t=0
    -> decision at t=34
    -> decision at t=68
    -> ...
```

Transfer each episode's selected features once. On GPU 1, project memory once per training forward and let multiple decision groups read that memory using different historical cutoffs. Avoid physically duplicating every growing prefix. A straightforward batched-prefix implementation is acceptable as a correctness baseline; shared-memory or packed attention is the optimization target.

If decisions must be split across microbatches, account for the lifetime of the projection autograd graph. Trainable projected K/V must be recomputed after optimizer updates; frozen source features can be reused across updates.

### Reduce transfer volume

At equal dtype, transferring 5,376-wide hidden states uses:

```text
5376 / (2 × 7168) = 37.5%
```

of the storage required for native FL2VA K and V together. Using the original document's 124-frame example, six taps decrease from roughly 1.7 GiB of native K/V to roughly 0.64 GiB of hidden states, excluding conditioning tokens and overhead.

Keep learned K/V projections on GPU 1 so no backward traffic crosses devices. Do not retain all-layer outputs when only six taps are required for the policy.

### Scheduling and alternative caching strategy

Overlap extraction, host-staged transfer, and policy training with bounded buffers, asynchronous copies, and explicit completion events before buffer reuse. Bucket episodes by length and exclude padded memory from attention. Measure extraction, transfer, and policy forward/backward times separately; assigning one GPU to each stage does not guarantee balanced utilization.

If frozen extraction dominates and storage permits, an alternative is:

1. Use both GPUs as independent workers to precompute selected features.
2. Train the action policy on both GPUs with data parallelism.

Feature caches must identify the checkpoint, preprocessing, mask mode, positions, selected layers, prompt variant, sigma, and noise realization. Fixed cached features limit prompt/noise augmentation unless several variants are generated. Begin with online episode-level extraction and a bounded cache; choose full offline caching based on measured throughput and storage cost.

Full-episode attention remains expensive even with causal masking. Use an appropriate fused or block-sparse attention implementation rather than materializing full attention-score tensors. Profile long episodes and use bounded episode segments or a deliberately defined history window if needed; training and inference must use compatible context rules.

## 3. Correct masks when training multiple decisions from one video

### Block and target indexing

Define observation blocks and decision query groups as follows:

```text
B0 = [obs0 repeated five times]
B1 = [obs1 ... obs34]
B2 = [obs35 ... obs68]
B3 = [obs69 ... obs102]

Qc = 34 fresh horizon queries for the decision after observing Bc
```

| Query group | Available observation blocks | Targets |
|---|---|---|
| Q0 | B0 | actions 1–34 |
| Q1 | B0, B1 | actions 35–68 |
| Q2 | B0, B1, B2 | actions 69–102 |

The block containing the observations produced by a target action chunk is not visible when predicting that chunk. In particular, Q0 must not read B1.

### A. FL2VA attention: historical features must themselves be causal

The default remains strict latent-frame causality at every FL2VA attention layer:

- Text queries read only static conditioning, not the evolving video.
- Video queries read static conditioning and video tokens at the same or earlier latent time.
- Spatial tokens within one latent frame, including both camera views, may attend bidirectionally.
- No cross-episode attention is allowed when packing examples.

```text
                  Keys
Queries      Text   F0   F1   F2
Text           Y     .    .    .
F0             Y     Y    .    .
F1             Y     Y    Y    .
F2             Y     Y    Y    Y
```

Use latent-time or block-aware masks, not a naive token-wise triangular mask that imposes an unintended raster order within a spatial frame.

Masking only the policy is insufficient: future video can otherwise contaminate historical hidden states or text, then leak through historical K/V. The existing repository's bidirectional video attention mask is not a valid implementation of this proposal.

#### Optional alternative: chunk causality

If decisions occur only after complete 34-frame blocks, FL2VA may instead use block-causal attention:

```text
                  Keys
Queries      Text   B0   B1   B2
Text           Y     .    .    .
B0             Y     Y    .    .
B1             Y     Y    Y    .
B2             Y     Y    Y    Y
```

Within an observed block, video attention is bidirectional. This is safe only because no decision reads a partially observed block. It is an alternative to, not an implementation of, strict frame causality. Keep the chosen semantics identical between training and inference; earlier replanning requires reconsidering block availability and VAE alignment.

### B. Policy cross-attention: one historical cutoff per decision

At every cross-attention block:

```text
allow(q[c, h], memory_token[j])
    iff memory_token[j] is available by decision time 34c
```

All 34 horizon queries in Qc have the same cutoff, regardless of horizon offset `h`:

```text
                  Video memory
Queries        B0   B1   B2   B3
Q0              Y    .    .    .
Q1              Y    Y    .    .
Q2              Y    Y    Y    .
```

Do not give later horizon queries access to later observations within their target chunk. Static conditioning can be readable by every group if included in memory. Exclude invalid/padded memory tokens and tokens from other episodes.

Any token reduction or pooling before cross-attention must preserve availability boundaries. A historical pooled token must not include future features.

### C. Policy self-attention: isolate independent decisions

Self-attention is bidirectional within each 34-query group and block-diagonal across groups:

```text
                  Action-query keys
Queries          Q0   Q1   Q2
Q0                Y    .    .
Q1                .    Y    .
Q2                .    .    Y
```

Allowing Q0 to read Q1 after Q1 has cross-attended to later video would leak future observations. Even allowing later groups to read earlier groups would introduce an action-state dependency absent from the stateless deployment policy. Each group starts from the same learned horizon embeddings and retains no state from another decision.

Putting decision groups in the batch dimension is a simple correctness baseline. Packed execution must explicitly preserve this isolation in every self-attention sublayer and avoid other cross-group mixing operations.

Apply the original target-validity loss mask to incomplete final action chunks. Do not create training decisions with no valid targets; they would give a zero loss denominator.

## 4. VAE prefix equivalence and incremental execution

Refine the original blanket restriction on full-clip VAE encoding: full-episode encoding is safe only when retained historical latents are provably equivalent to causal-prefix encoding. Attention masks cannot undo future information already mixed into a latent.

Associate every retained latent with an availability time: the latest raw observation needed to compute it. Cross-attention may read a latent only when that availability time is no later than the decision boundary. Under chunk-causal FL2VA, a feature's availability also includes the whole block it can attend to.

The local VAE uses causal temporal convolutions and temporally isolated normalization, but also uses 17-frame clip splitting and trailing-token dropping. Verify the exact selected VAE/runtime, padding, and retained-latent indexing rather than inferring prefix equivalence from the word “causal.” Until verified, encode decision prefixes separately.

Appending 34 observations does not automatically permit independently encoding those 34 frames and concatenating their latents. Preserve the VAE's temporal state or clip alignment so incremental inference agrees with the training representation.

For full-episode and prefix execution to agree, also preserve:

- Historical noise realizations and diffusion sigma; adding future tokens must not change historical noisy inputs.
- Prefix-stable positional coordinates, not coordinates rescaled by final episode length.
- The same static conditioning, without future outcome frames or future-derived prompt content.
- Absence of sequence-wide operations that mix future content into historical tokens.

One frozen single-step extraction is not repeated denoising with changing sigma. Changing sigma, historical noise, or conditioning invalidates cached backbone states.

### Sparse policy taps versus backbone caches

Six selected taps are sufficient for the policy interface, but exact incremental FL2VA updates without recomputing history require historical K/V at all 50 backbone layers, plus appropriate VAE state. This cache is distinct from the six features transferred to the policy.

Choose explicitly between:

- Recomputing each causal prefix: more compute, less persistent backbone-cache memory.
- Keeping all-layer K/V: less repeated computation, substantially more persistent memory.
- A bounded-history design: controlled cost, with matching training/inference context semantics and explicitly defined cache eviction behavior.

All-layer incremental caching is not assumed to fit merely because the six policy taps fit.

## 5. Required correctness checks before scaling

1. **Future invariance:** hold the prefix, conditioning, and historical noise fixed; replace later frames with unrelated data. Historical FL2VA features and the earlier decision's predictions must remain unchanged within numerical tolerance.
2. **VAE prefix equivalence:** compare retained historical latents from full-episode and separate-prefix encoding at every decision boundary, including initialization and partial final clips.
3. **Backbone prefix equivalence:** compare selected historical features from full masked execution and prefix-only execution using identical inputs, positions, sigma, and noise.
4. **Policy grouping equivalence:** compare packed multi-decision execution against independent decision forwards. Perturb later groups and verify earlier predictions do not change.
5. **Mask boundaries:** test that Q0 cannot read B1, Q1 cannot read B2, all horizons share one cutoff, padded tokens are excluded, and packed episodes remain isolated.
6. **Incremental equivalence:** if caching is implemented, compare cached execution with full-prefix recomputation at matching boundaries and check cache resets between episodes.
7. **Training integrity and profiling:** confirm FL2VA receives no gradients, learned memory projections do receive gradients, buffers are not reused before copies complete, and peak memory and stage timings are measured on both GPUs.
