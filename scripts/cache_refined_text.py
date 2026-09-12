import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.cache import load_text_cache, save_text_cache, text_cache_path
from wam_h3.refined.conditioning import prompt_variants
from wam_h3.refined.data import ALIGNMENT_EVIDENCE, LIBERO_REVISION, OFFICIAL_REPO, OFFICIAL_SUITES
from wam_h3.refined.runtime import configure, load_qwen


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_prompt(clip, prompt):
    try:
        tokens = clip.tokenize(prompt)
    except Exception as exc:
        raise RuntimeError(f"CLIP tokenize failed for {prompt!r}: {exc}") from exc
    try:
        positive = clip.encode_from_tokens_scheduled(tokens)
    except Exception as exc:
        raise RuntimeError(f"CLIP encode failed for {prompt!r}: {exc}") from exc
    cond, extra = positive[0]
    tags = extra["minimax_token_tags"]
    return cond, tags


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/libero_official")
    ap.add_argument("--output", type=Path, default=Path("data/refined/spatial_text"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=["minimal", "camera", "constraints", "physical"])
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("source") != "official_libero" or manifest.get("repo_id") != OFFICIAL_REPO
            or manifest.get("upstream_revision") != LIBERO_REVISION or manifest.get("alignment") != ALIGNMENT_EVIDENCE):
        raise ValueError("Missing audited official LIBERO provenance/alignment")

    configure(args.device)
    clip = load_qwen()

    seen = set()
    for record in manifest.get("files", []):
        path = dataset / record["path"]
        if not path.is_relative_to(dataset) or path.suffix != ".hdf5":
            continue
        suite = Path(record["path"]).parts[0]
        if suite not in OFFICIAL_SUITES:
            continue
        if "sha256" in record and sha256(path) != record["sha256"]:
            raise ValueError(f"Official HDF5 hash mismatch: {record['path']}")

        with h5py.File(path, "r") as handle:
            problem = json.loads(handle["data"].attrs["problem_info"])
            task = problem.get("language_instruction", "").strip()
            if not task:
                raise ValueError(f"Missing task instruction in {record['path']}")
            if task in seen:
                continue
            seen.add(task)

            variants = prompt_variants(task)
            for variant in args.variants:
                if variant not in variants:
                    raise ValueError(f"Unknown variant {variant}")
                prompt = variants[variant]
                cache_path = text_cache_path(output, task, variant)

                if cache_path.exists():
                    cached = load_text_cache(cache_path)
                    meta = cached["metadata"]
                    if (meta.get("task") != task or meta.get("prompt") != prompt or meta.get("variant") != variant):
                        raise ValueError(f"Existing text cache mismatch at {cache_path}; refusing to overwrite")
                    print(f"Skip existing {cache_path.name}", flush=True)
                    continue

                cond, tags = encode_prompt(clip, prompt)
                save_text_cache(cache_path, cond, tags, task, prompt, variant)
                print(f"Cached {cache_path.name}", flush=True)


if __name__ == "__main__":
    main()
