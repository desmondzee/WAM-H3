# Refined policy: short-run analysis and optimization recommendations

## Summary

The training pipeline works and the policy is improving, but this run has not demonstrated a meaningful advantage over a constant-action baseline. Priorities are to obtain more useful optimizer updates per unit of frozen-backbone computation, match the learning-rate schedule to the completed update budget, and evaluate pose and gripper behavior against simple baselines.

This analysis concerns the official LIBERO-Spatial, text-only-conditioned run with 450 training and 50 episode-disjoint validation demonstrations. It does not concern the earlier image-conditioned smoke or mixed-suite preparation. LIBERO-Plus remains reserved for later OOD evaluation.

## Run artifacts

- Original scheduled run: `runs/refined_action_policy/spatial_64chunks_cosine/`
- Diagnostic continuation: `runs/refined_action_policy/spatial_64chunks_diagnostics/`
- Original W&B run: https://wandb.ai/wamh3/wam-h3/runs/0728ob0c
- Diagnostic W&B run: https://wandb.ai/wamh3/wam-h3/runs/8pv9hqqo
- Split: `data/refined/spatial_textonly_split.json`
- Shared text caches: `data/refined/spatial_text/`
- Episode caches: `data/refined/spatial_textonly/`
- Long-context feature check: `runs/refined_action_policy/spatial_textonly_longest_smoke.json`
- Earlier stage profile: `runs/refined_action_policy/pipeline_profile_detailed.json`

The shared text-cache directory actually used by preprocessing is `data/refined/spatial_text`. Episode caches reference the individual text files by absolute path.

## 1. Results

Combining the original run through checkpoint 10 with the continuation from checkpoint 10, without double-counting the repeated update 11:

| Metric | Result |
|---|---:|
| Completed optimizer updates | 25 |
| Training episode visits | 399 / 450 |
| Action chunks processed | 1,635 |
| Final training-batch MSE | 0.2311 |
| Validation MSE, update 10 | 0.32687 |
| Validation MSE, update 20 | 0.24336 |
| Final checkpoint | Update 25 |
| Best evaluated checkpoint | Update 20 |

Validation improved 25.5% between updates 10 and 20. The policy nevertheless saw less than one training epoch. Update 25 was not validated because the time budget expired.

The diagnostic continuation finished normally. The original W&B run is marked crashed because it was explicitly stopped for a checkpoint restart, not because of a numerical failure.

The objective averages squared normalized action error over the seven dimensions and valid timesteps within each chunk, then averages equally across chunks. Gradients accumulated across episodes are divided by the actual chunk count before clipping and the optimizer update. Partial final chunks receive equal chunk weight, even when they contain fewer valid timesteps.

## 2. Baseline comparison

Two baselines were evaluated on the same 50 validation episodes, using the same normalization, masking, and equal-chunk weighting. The constant mean action was fitted only on training targets, using equal-chunk weighting.

| Predictor | Validation MSE |
|---|---:|
| Always output raw action zero | 0.25306 |
| Always output the training-set mean action | 0.24411 |
| Policy at update 20 | 0.24336 |

The policy improves on the training-mean baseline by only about 0.3%. Without uncertainty estimates or repeated runs, this is not convincing evidence of useful observation conditioning.

This does not establish that FL2VA features are ineffective: 25 updates is a small optimization budget. It does establish that falling training loss alone is insufficient evidence.

### Component breakdown at update 20

Values below are normalized MSEs with equal-chunk weighting.

| Component | Constant training mean | Policy |
|---|---:|---:|
| Translation 0 | 0.1854 | 0.1612 |
| Translation 1 | 0.1226 | 0.1727 |
| Translation 2 | 0.2996 | 0.3093 |
| Rotation 0 | 0.0365 | 0.0481 |
| Rotation 1 | 0.0386 | 0.1191 |
| Rotation 2 | 0.0265 | 0.0269 |
| Gripper | 0.9996 | 0.8662 |

Much of the improvement comes from the gripper, while several pose components are worse than the constant baseline. Gripper accounts for approximately 51% of the policy's total squared-error sum despite being one of seven dimensions.

Gripper targets are exactly -1/+1 and are nearly balanced under chunk weighting. Its high MSE is not automatically a scaling bug.

### Temporal breakdown

Validation loss by decision index within the episode:

| Decision chunk | MSE | Contributing episodes |
|---|---:|---:|
| 0 | 0.1181 | 50 |
| 1 | 0.3223 | 50 |
| 2 | 0.3027 | 50 |
| 3 | 0.2420 | 41 |
| 4 | 0.1942 | 12 |

The initial decision is easier than the next two. Interpreting this as initial approach versus later manipulation requires task-phase evidence; the loss values alone do not establish that explanation. Later chunk positions also have fewer contributing episodes.

Within a predicted chunk, validation MSE rises from 0.2305 at horizon 0 to 0.2928 at horizon 33, approximately 27%. Partial-target masks change the population contributing to each horizon, so this is not a perfectly controlled comparison.

MAE results are unavailable for this run. MAE support was added after the process started and was not activated by another restart. MAE cannot be reconstructed by taking the square root of logged MSE.

## 3. Training-time optimizations

### A. Prioritize frozen-backbone work

Late non-validation updates averaged approximately 44 seconds, processing around 16 episodes and 65–66 chunks: approximately 2.7 seconds per episode. This estimate uses nine non-validation updates from the diagnostic continuation, excluding its early startup period. Some samples include checkpoint work.

The text-only real-model feature smoke measured approximately:

| Context frames | Frozen extraction |
|---|---:|
| 73 | 1.27 s |
| 107 | 2.16 s |
| 141 | 3.27 s |
| 175 | 4.59 s |

These measurements support frozen extraction as the principal cost. Earlier profiling measured much cheaper policy forward/backward work, but that older benchmark used a synthetic policy loss and is not an exact stage decomposition of this run.

Increasing gradient accumulation does not improve physical GPU batching. The 64-chunk optimizer batch still processes episodes individually; accumulation primarily reduces optimizer-update frequency.

### B. Cache fixed validation features first

Validation uses fixed text, noise, sigma, and frozen weights. Re-extracting those features at each validation pass is unnecessary.

Estimated uncompressed six-tap BF16 hidden-feature storage, excluding metadata:

| Feature set | Storage |
|---|---:|
| All validation episodes, fixed prompt | 28.2 GiB |
| Training episodes, one variant each | 256.3 GiB |
| Training episodes, all four variants | 1,025 GiB |

Validation caching is the clearest immediate reuse opportunity. GPU 1's measured policy-training peak was approximately 10.9 GiB, so retaining validation features there or in host memory is worth testing with explicit memory-headroom checks.

Full training-feature caching is more attractive for multiple epochs or repeated optimizer experiments. It would not have saved repeated training extraction in this run because the first epoch was not completed. Four-variant caching requires substantial storage and an I/O benchmark.

Feature-cache identities must include the model, conditioning, VAE representation, sigma/noise, selected taps, positional conventions, and causal-mask implementation. Learned policy K/V projections must still be recomputed after optimizer updates.

### C. Reduce setup and pipeline overhead

Benchmark these narrow changes:

1. Cache causal masks by text length, latent length, and spatial layout instead of retaining only the last mask.
2. Reuse pinned buffers and CUDA streams.
3. Replace host-side transfer synchronization with event dependencies where safe.
4. Preserve a bounded producer queue across optimizer batches instead of recreating its executor each update.
5. Record extraction, queue wait, transfers, policy compute, validation, and checkpoint time separately.

Preserve fixed-tile causal attention behavior and rerun exact prefix/future-invariance checks after attention-related changes. Do not trade correctness for a faster attention fallback.

### D. Avoid repeated process restarts

The diagnostic continuation's first update took 105 seconds including loading and warmup, versus approximately 39–55 seconds for later ordinary updates. Update 11 was also repeated after restarting from checkpoint 10.

Enable diagnostics before launching future short experiments and support checkpoint-safe stopping. Avoid restarting merely to add metric fields.

## 4. Optimizer setup

### A. Align the schedule with the experiment budget

The schedule was planned for 60 updates, but training ended at 25. The final LR was still 7.37e-5, rather than the intended 1e-5.

This was predominantly a warmup/high-LR experiment, not a completed warmup-and-decay experiment. For the next comparison:

- Either allocate a fixed update budget and complete its schedule;
- Or choose the schedule horizon from measured throughput and the available budget, accounting for validation and startup.

Reserve time to validate the final checkpoint. Do not compare schedule endpoints when one experiment stops halfway through its decay.

### B. Compare smaller chunk batches

Compare 32 versus 64 chunks per optimizer update at matched numbers of processed training chunks, adjusting the LR schedule to each update count.

Rationale:

- The current head received only 25 parameter updates.
- Accumulating 64 chunks does not batch backbone computation.
- A 32-chunk target permits more optimizer updates for approximately the same extraction work, with additional policy/optimizer overhead to measure.

This is an ablation, not a guarantee that smaller batches generalize better. The current evidence does not justify increasing batch size further.

### C. Test a more conservative initialization/LR combination

Early training loss increased from 0.496 to 1.821, then recovered. All 25 updates were clipped; the median pre-clipping gradient norm over the last ten updates was approximately 5.2.

Recommended controlled ablations:

- Peak LR 3e-5 versus 1e-4.
- Small output-head weight initialization, with bias initialized to the training-mean normalized action.
- Retain clipping at 1.0 initially.
- Log parameter-update norm relative to parameter norm, particularly for the output head, learned queries, and memory projections.

The training loss and gradients were correctly averaged before clipping. Clipping bounds the gradient norm, not AdamW's resulting parameter-update norm. The cause of the early spike was not isolated: episodes and prompt variants also vary between updates.

### D. Separate gripper treatment from pose regression

Retain the current MSE objective as the reference, then test:

- Six continuous pose outputs with regression loss;
- A binary gripper logit with BCE;
- An explicit relative weighting between the two losses.

Report gripper accuracy and transition timing alongside pose MAE/MSE. Do not downweight gripper merely to make aggregate loss smaller. This is an objective/head ablation, not a correctness fix to the existing optimizer.

Use explicit AdamW parameter groups: decay matrix weights but exclude biases and normalization parameters. Decide explicitly whether learned query embeddings receive weight decay.

## 5. Recommended next experiment

1. Keep the same 450/50 episode split and text-only conditioning.
2. Add constant, task-conditioned, and decision-position-conditioned baselines fitted only on training data.
3. Cache fixed validation features.
4. Compare 32/64 chunk targets and 3e-5/1e-4 peak LR at matched data exposure.
5. Complete the LR schedule and validate the final checkpoint.
6. Run small-data overfitting and shuffled-observation ablations to establish whether the policy uses visual features.

Use `runs/refined_action_policy/spatial_64chunks_diagnostics/step_000020/` as the best evaluated checkpoint. The latest checkpoint is `step_000025/`, but its validation performance is unknown. Neither checkpoint currently supports a claim of robot task success; simulator evaluation remains separate.
