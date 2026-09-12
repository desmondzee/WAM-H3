import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.cache import identity, load_policy_cache, load_text_cache, save_policy_cache, text_cache_path
from wam_h3.refined.data import decision_indices, load_episode, prefix_frames
from wam_h3.refined.runtime import COMFY_REVISION, MODEL_REVISION, configure, load_vae


def reused_video(path, entry):
    if not path.exists():
        return None
    sample = torch.load(path, map_location="cpu", weights_only=True)
    meta = sample["metadata"]
    source = meta["source"]
    if (sample["identity"] != identity(meta) or meta.get("purpose") != "causal_policy"
            or meta.get("model_revision") != MODEL_REVISION or meta.get("comfy_revision") != COMFY_REVISION
            or source.get("origin") != "official_libero" or source.get("file") != entry["path"]
            or source.get("demo") != entry["demo"] or source.get("episode_sha256") != entry["episode_sha256"]
            or meta.get("action_alignment") != "post_action_verified"
            or len(meta.get("vae_prefix_max_errors", [])) != entry["chunks"] - 1
            or any(error != 0 for error in meta["vae_prefix_max_errors"])):
        raise ValueError(f"Unverified reusable VAE cache: {path}")
    expected = (1, 24, 2 + 10 * (entry["chunks"] - 1), 24, 48)
    if sample["latent"].shape != expected:
        raise ValueError("Reusable latent shape mismatch")
    tensors = {key: sample[key] for key in ("latent", "targets", "valid", "cutoffs")}
    metadata = {key: value for key, value in meta.items() if key not in
                ("schema", "conditioning_images", "prompt", "variant", "initial_image_sha256", "text_cache_identities")}
    return tensors, metadata


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, default=Path("data/refined/spatial_text"))
    parser.add_argument("--reuse-dir", type=Path, default=Path("data/refined/official_spatial_train"))
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid batch/shard")
    split = json.loads(args.split_manifest.read_text())
    root = Path(split["dataset"])
    if hashlib.sha256((root / "provenance.json").read_bytes()).hexdigest() != split["provenance_sha256"]:
        raise ValueError("Dataset provenance changed")
    paths = dict(zip(split["train_episodes"] + split["validation_episodes"], split["train_caches"] + split["validation_caches"]))
    entries = {e["index"]: e for e in split["episodes"]}
    indices = sorted(entries)[args.shard_index::args.num_shards]
    variants = ("minimal", "camera", "constraints", "physical")
    pending = defaultdict(list)
    def save(index, tensors, metadata):
        task = metadata["task"]
        refs = {variant: str(text_cache_path(args.text_dir.resolve(), task, variant)) for variant in variants}
        texts = {variant: load_text_cache(path) for variant, path in refs.items()}
        metadata.update(variant="camera", conditioning_images=0,
                        text_cache_identities={variant: text["identity"] for variant, text in texts.items()})
        tensors["text_cache_paths"] = refs
        save_policy_cache(paths[index], tensors, metadata)
        print(f"Cached episode {index} text-only; latent {tuple(tensors['latent'].shape)}", flush=True)
    for index in indices:
        if Path(paths[index]).exists():
            sample = load_policy_cache(paths[index])
            if sample["metadata"]["source"].get("episode_sha256") != entries[index]["episode_sha256"]:
                raise ValueError("Existing cache belongs to a different source")
            continue
        reused = reused_video(args.reuse_dir / f"episode_{index:06d}_camera.pt", entries[index])
        if reused is not None:
            save(index, *reused)
        else:
            pending[entries[index]["chunks"]].append(index)
    if not pending:
        return
    configure(args.device)
    vae = load_vae()
    import comfy.model_management as mm
    mm.load_models_gpu([vae.patcher], force_full_load=True)
    def encode(videos):
        if len(videos) == 1:
            return vae.encode(videos[0]).cpu()
        pixels = vae.process_input(videos.movedim(-1, 1)).to(vae.vae_dtype)
        out = vae.first_stage_model.encode(pixels, device=vae.device)
        return out.to(device="cpu", dtype=vae.vae_output_dtype())
    tested = set()
    for chunks, group in sorted(pending.items()):
        for start in range(0, len(group), args.batch_size):
            batch = group[start:start + args.batch_size]
            episodes = [load_episode(root, index) for index in batch]
            plans = [decision_indices(len(episode.frames)) for episode in episodes]
            videos = torch.stack([prefix_frames(episode.frames, int(plan[0][-1])) for episode, plan in zip(episodes, plans)])
            latent = encode(videos)
            assert latent.shape == (len(batch), 24, 2 + 10 * (chunks - 1), 24, 48)
            signature = (chunks, len(batch))
            if signature not in tested:
                for i, video in enumerate(videos):
                    native = vae.encode(video).cpu()
                    torch.testing.assert_close(latent[i:i + 1], native, atol=1e-4, rtol=1e-4)
                tested.add(signature)
                print(f"Batch/native equivalence passed: {signature}", flush=True)
            errors = [[] for _ in batch]
            for cutoff in plans[0][0][:-1]:
                prefixes = torch.stack([prefix_frames(episode.frames, int(cutoff)) for episode in episodes])
                prefix = encode(prefixes)
                historical = latent[:, :, :prefix.shape[2]]
                torch.testing.assert_close(prefix, historical, atol=1e-4, rtol=1e-4)
                for i in range(len(batch)):
                    errors[i].append((prefix[i] - historical[i]).abs().max().item())
            for i, (index, episode, plan) in enumerate(zip(batch, episodes, plans)):
                cutoffs, action_indices, valid = plan
                episode.source["episode_sha256"] = entries[index]["episode_sha256"]
                metadata = dict(dataset=str(root), episode=index, task=episode.task, source_frames=len(episode.frames),
                                prefix_frames=videos.shape[1], width=768, height=384, fps=20,
                                resize="per_camera_384_square_bilinear_antialias", camera_order=["agentview_rgb", "eye_in_hand_rgb"],
                                vae_prefix_max_errors=errors[i], action_alignment="post_action_verified",
                                alignment_evidence=episode.source["alignment"], source=episode.source)
                save(index, dict(latent=latent[i:i + 1].clone(), targets=episode.actions[action_indices], valid=valid, cutoffs=cutoffs), metadata)
            print(f"Batch {batch}; prefix errors {errors}; peak_bytes {torch.cuda.max_memory_allocated()}", flush=True)
            del episodes, videos, latent


if __name__ == "__main__":
    main()
