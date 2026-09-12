import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_refined_libero import sha256
from wam_h3.refined.data import ALIGNMENT_EVIDENCE, LIBERO_REVISION, OFFICIAL_REPO, OFFICIAL_SUITES


def partition_episodes(episodes, seed=0):
    unique = {}
    aliases = defaultdict(list)
    for episode in episodes:
        digest = episode["episode_sha256"]
        unique.setdefault(digest, episode)
        aliases[digest].append(episode["index"])
    tasks = defaultdict(list)
    for episode in unique.values():
        tasks[episode["path"]].append(episode)
    train, validation = [], []
    for task, items in sorted(tasks.items()):
        if len(items) < 2:
            raise ValueError(f"Task has fewer than two unique demonstrations: {task}")
        task_seed = int(hashlib.sha256(f"{seed}:{task}".encode()).hexdigest(), 16)
        shuffled = sorted(items, key=lambda e: e["index"])
        random.Random(task_seed).shuffle(shuffled)
        n = max(1, min(len(items) - 1, round(len(items) * 0.1)))
        validation.extend(shuffled[:n])
        train.extend(shuffled[n:])
    return sorted(train, key=lambda e: e["index"]), sorted(validation, key=lambda e: e["index"]), dict(aliases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/libero_official_mixed"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/refined/official_mixed"))
    parser.add_argument("--output", type=Path, default=Path("data/refined/official_mixed_split.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Split already exists; use it unchanged or choose a new output")
    root = args.dataset.resolve()
    manifest_path = root / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["source"] != "official_libero" or manifest["repo_id"] != OFFICIAL_REPO
            or manifest["alignment"] != ALIGNMENT_EVIDENCE or manifest["upstream_revision"] != LIBERO_REVISION):
        raise ValueError("Unaudited official provenance")
    grouped = defaultdict(list)
    for index, entry in enumerate(manifest["episodes"]):
        grouped[entry["path"]].append((index, entry))
    records = {record["path"]: record for record in manifest["files"]}
    episodes = []
    for relative, entries in sorted(grouped.items()):
        path = (root / relative).resolve()
        suite = relative.split("/")[0]
        if not path.is_relative_to(root) or suite not in OFFICIAL_SUITES:
            raise ValueError("Invalid official suite path")
        record = records[relative]
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Official content hash mismatch: {relative}")
        with h5py.File(path, "r") as handle:
            for index, entry in entries:
                demo = handle["data"][entry["demo"]]
                digest = hashlib.sha256()
                for key in ("actions", "obs/agentview_rgb", "obs/eye_in_hand_rgb"):
                    dataset = demo[key]
                    digest.update(json.dumps([key, dataset.shape, str(dataset.dtype)]).encode())
                    for start in range(0, len(dataset), 32):
                        digest.update(dataset[start:start + 32].tobytes())
                length = len(demo["actions"])
                if length < 2:
                    raise ValueError("Episode has no next-action targets")
                episodes.append(dict(index=index, path=relative, demo=entry["demo"], suite=suite,
                                     length=length, chunks=(length - 1 + 33) // 34,
                                     episode_sha256=digest.hexdigest()))
        print(f"Audited {relative}", flush=True)
    train, validation, aliases = partition_episodes(episodes, args.seed)
    cache_dir = args.cache_dir.resolve()
    result = dict(schema=1, seed=args.seed, dataset=str(root), dataset_revision=manifest["revision"],
                  provenance_sha256=sha256(manifest_path), variant="camera",
                  train_episodes=[e["index"] for e in train], validation_episodes=[e["index"] for e in validation],
                  train_caches=[str(cache_dir / f'episode_{e["index"]:06d}_camera.pt') for e in train],
                  validation_caches=[str(cache_dir / f'episode_{e["index"]:06d}_camera.pt') for e in validation],
                  episodes=train + validation, duplicate_groups={k: v for k, v in aliases.items() if len(v) > 1},
                  train_suite_counts=dict(Counter(e["suite"] for e in train)),
                  validation_suite_counts=dict(Counter(e["suite"] for e in validation)),
                  train_chunk_histogram=dict(Counter(e["chunks"] for e in train)),
                  validation_chunk_histogram=dict(Counter(e["chunks"] for e in validation)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ("train_suite_counts", "validation_suite_counts", "train_chunk_histogram", "validation_chunk_histogram")}))


if __name__ == "__main__":
    main()
