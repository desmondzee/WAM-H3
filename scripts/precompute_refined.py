import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.cache import load_policy_cache, save_policy_cache
from wam_h3.refined.conditioning import prompt_variants
from wam_h3.refined.data import decision_indices, load_episode, prefix_frames
from wam_h3.refined.runtime import configure, load_qwen, load_vae


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/libero_official")
    ap.add_argument("--episodes", type=int, nargs="+")
    ap.add_argument("--variants", nargs="+", default=["camera"])
    ap.add_argument("--output", type=Path, default=Path("data/refined/official_spatial_smoke"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--split-manifest", type=Path)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid preprocessing shard")
    split_entries = {}
    if args.split_manifest:
        split = json.loads(args.split_manifest.read_text())
        if (Path(args.dataset).resolve() != Path(split["dataset"])
                or hashlib.sha256((Path(args.dataset) / "provenance.json").read_bytes()).hexdigest() != split["provenance_sha256"]):
            raise ValueError("Split belongs to different dataset provenance")
        if args.variants != [split["variant"]]:
            raise ValueError("Preprocessing variants must match the split")
        if any(Path(p).parent != args.output.resolve() for p in split["train_caches"] + split["validation_caches"]):
            raise ValueError("Output must match split cache paths")
        available = sorted(split["train_episodes"] + split["validation_episodes"])
        if args.episodes is not None and not set(args.episodes) <= set(available):
            raise ValueError("Requested episodes are outside the split")
        args.episodes = available if args.episodes is None else args.episodes
        split_entries = {e["index"]: e for e in split["episodes"]}
    episodes_to_process = (args.episodes if args.episodes is not None else [0, 1, 2, 3])[args.shard_index::args.num_shards]
    configure(args.device)
    clip, vae = load_qwen(), load_vae()
    for index in episodes_to_process:
        paths = [args.output / f"episode_{index:06d}_{variant}.pt" for variant in args.variants]
        if args.skip_existing and all(path.exists() for path in paths):
            for path, variant in zip(paths, args.variants):
                cached = load_policy_cache(path)["metadata"]
                if (cached["dataset"] != str(Path(args.dataset).resolve()) or cached["episode"] != index
                        or cached["variant"] != variant or cached["source"]["origin"] != "official_libero"):
                    raise ValueError("Existing cache belongs to a different episode")
                if split_entries and cached["source"].get("episode_sha256") != split_entries[index]["episode_sha256"]:
                    raise ValueError("Existing cache fingerprint differs from split")
            print(f"Verified existing episode {index}", flush=True)
            continue
        if any(path.exists() for path in paths):
            raise FileExistsError("Existing cache would be overwritten; use --skip-existing for complete episodes")
        episode = load_episode(args.dataset, index)
        if split_entries:
            episode.source["episode_sha256"] = split_entries[index]["episode_sha256"]
        cutoffs, indices, valid = decision_indices(len(episode.frames))
        video = prefix_frames(episode.frames, int(cutoffs[-1]))
        latent = vae.encode(video).cpu()
        expected_t = 2 + 10 * (int(cutoffs[-1]) // 34)
        assert latent.shape == (1, 24, expected_t, 24, 48), latent.shape
        errors = []
        for cutoff in cutoffs[:-1]:
            prefix = vae.encode(prefix_frames(episode.frames, int(cutoff))).cpu()
            historical = latent[:, :, :prefix.shape[2]]
            error = (prefix - historical).abs().max().item()
            torch.testing.assert_close(prefix, historical, atol=1e-4, rtol=1e-4)
            errors.append(error)
        for variant in args.variants:
            prompt = prompt_variants(episode.task)[variant]
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=[episode.frames[:1]]))
            cond, extra = positive[0]
            tags = extra["minimax_token_tags"]
            metadata = dict(dataset=str(Path(args.dataset).resolve()), episode=index, task=episode.task,
                            prompt=prompt, variant=variant, source_frames=len(episode.frames),
                            prefix_frames=len(video), width=768, height=384, fps=20,
                            resize="per_camera_384_square_bilinear_antialias", camera_order=["agentview_rgb", "eye_in_hand_rgb"],
                            initial_image_sha256=hashlib.sha256(episode.frames[0].numpy().tobytes()).hexdigest(),
                            vae_prefix_max_errors=errors, action_alignment="post_action_verified",
                            alignment_evidence=episode.source["alignment"], source=episode.source)
            save_policy_cache(args.output / f"episode_{index:06d}_{variant}.pt",
                              dict(latent=latent, conditioning=cond.cpu(), tags=tags.cpu(), cutoffs=cutoffs,
                                   targets=episode.actions[indices], valid=valid), metadata)
            print(f"Cached episode {index} {variant}: {latent.shape}, prefix errors {errors}", flush=True)


if __name__ == "__main__":
    main()
