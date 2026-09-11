import hashlib
from pathlib import Path

import torch


def cache_path(cache_dir, prompt):
    return Path(cache_dir) / f"{hashlib.sha256(prompt.encode()).hexdigest()}.pt"


def save_embedding(cache_dir, prompt, emb):
    path = cache_path(cache_dir, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"prompt": prompt, "embedding": emb.detach().cpu()}, tmp)
    tmp.replace(path)


def load_embedding(cache_dir, prompt):
    path = cache_path(cache_dir, prompt)
    return torch.load(path, map_location="cpu")["embedding"] if path.exists() else None
