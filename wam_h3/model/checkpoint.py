from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


@dataclass
class LoadReport:
    missing: set
    unexpected: set


def iter_shards(transformer_dir):
    for shard in sorted(Path(transformer_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                yield k, f.get_tensor(k)


def load_pretrained(model, transformer_dir=None, state_dict=None, dtype=None):
    items = state_dict.items() if state_dict is not None else iter_shards(transformer_dir)
    own = model.state_dict()
    loaded, unexpected = {}, set()
    for k, v in items:
        if k in own:
            loaded[k] = v if dtype is None else v.to(dtype)
        else:
            unexpected.add(k)
    model.load_state_dict(loaded, strict=False, assign=True)
    return LoadReport(set(own) - set(loaded), unexpected)


def trainable_state_dict(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}


def save_trainable(model, path):
    save_file(trainable_state_dict(model), str(path))


def load_trainable(model, path):
    sd = load_file(str(path))
    report = model.load_state_dict(sd, strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = trainable - set(sd)
    if missing or report.unexpected_keys:
        raise RuntimeError(f"trainable checkpoint mismatch: missing={sorted(missing)[:5]} unexpected={report.unexpected_keys[:5]}")
    return torch.tensor(0)
