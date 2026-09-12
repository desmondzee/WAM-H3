import json
import time
from pathlib import Path

import torch

from .conditioning import prompt_variants
from .data import load_episode, save_video
from .runtime import COMFY_REVISION, MODEL_REVISION, load_backbone, load_qwen, load_vae, unload


@torch.no_grad()
def prepare(dataset, episodes, variants, output):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, align_frame_count
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clip, vae = load_qwen(), load_vae()
    for index in episodes:
        episode = load_episode(dataset, index)
        save_video(output / f"episode_{index:06d}_source.mp4", episode.frames, fps=20)
        for variant in variants:
            prompt = prompt_variants(episode.task)[variant]
            frames = align_frame_count(len(episode.frames))
            result = MiniMaxH3ImageToVideo.execute(clip, vae, prompt, 768, 384, frames,
                                                   episode.frames[:1], episode.frames[-1:])
            positive, latent = result.result
            name = f"episode_{index:06d}_{variant}"
            torch.save({"purpose": "endpoint_generation_only", "positive": positive,
                        "video": latent["samples"].tensors[0], "audio": latent["samples"].tensors[1]}, output / f"{name}.pt")
            metadata = dict(purpose="endpoint_generation_only", episode=index, task=episode.task, prompt=prompt,
                            dataset=str(Path(dataset).resolve()), source=episode.source,
                            variant=variant, width=768, height=384, source_frames=len(episode.frames),
                            frames=frames, fps=24, keyframe_indices=[0, frames - 1],
                            model_revision=MODEL_REVISION, comfy_revision=COMFY_REVISION)
            (output / f"{name}.json").write_text(json.dumps(metadata, indent=2))
            print(f"Prepared {name}: {frames} frames, {positive[0][0].shape[1]} conditioning tokens", flush=True)
    del clip, vae
    unload()


@torch.no_grad()
def sample(prepared, steps, seed):
    import comfy.nested_tensor
    import nodes
    path = Path(prepared)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["purpose"] != "endpoint_generation_only":
        raise ValueError("Expected an endpoint-generation diagnostic")
    metadata = json.loads(path.with_suffix(".json").read_text())
    model = load_backbone()
    count = [0]
    def count_forward(module, inputs):
        count[0] += 1
    handle = model.model.diffusion_model.register_forward_pre_hook(count_forward)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result, = nodes.common_ksampler(model, seed, steps, 1.0, "res_multistep", "simple", data["positive"],
                                    data["positive"], {"samples": comfy.nested_tensor.NestedTensor((data["video"], data["audio"]))})
    torch.cuda.synchronize()
    handle.remove()
    video = result["samples"].tensors[0].cpu()
    if not torch.isfinite(video).all():
        raise RuntimeError("Generated latents are not finite")
    name = path.with_name(f"{path.stem}_steps{steps}_seed{seed}")
    torch.save({"video": video, "purpose": "endpoint_generation_only"}, name.with_suffix(".latents.pt"))
    metadata.update(steps=steps, seed=seed, cfg=1.0, sampler="res_multistep", scheduler="simple",
                    model_evaluations=count[0], denoising_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), review_status="pending")
    name.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    del model, result
    unload()
    return name.with_suffix(".latents.pt")


@torch.no_grad()
def decode(latents):
    path = Path(latents)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["purpose"] != "endpoint_generation_only":
        raise ValueError("Expected generated diagnostic latents")
    name = path.with_name(path.name.removesuffix(".latents.pt"))
    metadata = json.loads(name.with_suffix(".json").read_text())
    vae = load_vae()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    images = vae.decode(data["video"]).squeeze(0)
    if images.ndim != 4 or images.shape[1:3] != (384, 768) or not torch.isfinite(images).all():
        raise RuntimeError(f"Invalid decoded video: {images.shape}")
    save_video(name.with_suffix(".mp4"), images)
    metadata.update(decoded_frames=len(images), decode_seconds=time.perf_counter() - start,
                    decode_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    name.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Decoded {name.with_suffix('.mp4')}", flush=True)
    del vae
    unload()
