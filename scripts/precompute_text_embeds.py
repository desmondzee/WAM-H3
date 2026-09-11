#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

from wam_h3.data.text_cache import load_embedding, save_embedding


def collect_prompts(data_dirs, prompts_file):
    prompts = []
    for d in data_dirs:
        for line in (Path(d) / "meta/tasks.jsonl").read_text().splitlines():
            if line.strip():
                prompts.append(DEFAULT_PROMPT.format(task=json.loads(line)["task"]))
    if prompts_file:
        prompts += [DEFAULT_PROMPT.format(task=t.strip()) for t in Path(prompts_file).read_text().splitlines() if t.strip()]
    return list(dict.fromkeys(prompts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="*", default=[])
    ap.add_argument("--prompts-file")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--weights-root", default="FL2VA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    prompts = collect_prompts(a.data_dirs, a.prompts_file)
    todo = [p for p in prompts if a.overwrite or load_embedding(a.cache_dir, p) is None]
    print(f"{len(prompts)} prompts, {len(todo)} to encode")
    if not todo:
        return
    if a.fake:
        g = torch.Generator().manual_seed(0)
        for p in todo:
            save_embedding(a.cache_dir, p, torch.randn(len(p.split()), 5120, generator=g).bfloat16())
        return
    from wam_h3.model.text_encoder import TextEncoder
    enc = TextEncoder.from_pretrained(a.weights_root, device=a.device)
    for p, e in zip(todo, enc.encode(todo)):
        save_embedding(a.cache_dir, p, e)
        print(f"{e.shape[0]:3d} tokens  {p}")


if __name__ == "__main__":
    main()
