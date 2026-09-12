import hashlib
import json
from pathlib import Path

import torch

from .runtime import COMFY_REVISION, MODEL_REVISION

TEXT_SCHEMA = 1
POLICY_SCHEMA = 2
TEXT_KIND = "text_only"
TEXT_PURPOSE = "text_only_conditioning"
POLICY_PURPOSE = "causal_policy"


def identity(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def text_cache_path(root, task, variant):
    return Path(root) / f"{hashlib.sha256(task.encode()).hexdigest()[:16]}_{variant}.pt"


def _write_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_text_cache(path, conditioning, tags, task, prompt, variant):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite text cache {path}")
    metadata = {
        "schema": TEXT_SCHEMA,
        "purpose": TEXT_PURPOSE,
        "kind": TEXT_KIND,
        "conditioning_images": 0,
        "task": task,
        "prompt": prompt,
        "variant": variant,
        "model_revision": MODEL_REVISION,
        "comfy_revision": COMFY_REVISION,
    }
    payload = {
        "conditioning": conditioning.detach().cpu(),
        "tags": tags.detach().cpu(),
        "metadata": metadata,
        "identity": identity(metadata),
    }
    _write_payload(path, payload)
    return payload


def load_text_cache(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    metadata = data.get("metadata", {})
    if metadata.get("kind") != TEXT_KIND or metadata.get("conditioning_images") != 0:
        raise ValueError("Only text-only conditioning caches are accepted")
    if metadata.get("purpose") != TEXT_PURPOSE or metadata.get("schema") != TEXT_SCHEMA:
        raise ValueError("Text cache purpose/schema mismatch")
    if metadata.get("model_revision") != MODEL_REVISION or metadata.get("comfy_revision") != COMFY_REVISION:
        raise ValueError("Text cache runtime/model revision mismatch")
    if data.get("identity") != identity(metadata):
        raise ValueError("Text cache identity mismatch")
    return data


def save_policy_cache(path, tensors, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite policy cache {path}")
    direct = "conditioning" in tensors or "tags" in tensors
    if direct and metadata.get("conditioning_images") != 0:
        raise ValueError("Direct conditioning/tags require explicit metadata conditioning_images=0")
    metadata = {
        **metadata,
        "schema": POLICY_SCHEMA,
        "purpose": POLICY_PURPOSE,
        "conditioning_images": 0,
        "model_revision": MODEL_REVISION,
        "comfy_revision": COMFY_REVISION,
    }
    payload = {**tensors, "metadata": metadata, "identity": identity(metadata)}
    _write_payload(path, payload)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return payload


def load_policy_cache(path, variant="camera"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    metadata = data.get("metadata", {})
    if metadata.get("purpose") != POLICY_PURPOSE or metadata.get("conditioning_images") != 0:
        raise ValueError("Only text-conditioned causal policy caches are accepted; endpoint generation or image-conditioned caches are forbidden")
    if metadata.get("schema") != POLICY_SCHEMA or metadata.get("model_revision") != MODEL_REVISION or metadata.get("comfy_revision") != COMFY_REVISION:
        raise ValueError("Policy cache runtime/model revision mismatch")
    if data.get("identity") != identity(metadata):
        raise ValueError("Policy cache metadata identity mismatch")
    if "conditioning" in data and "tags" in data:
        return data
    text_cache_paths = data.get("text_cache_paths")
    if not isinstance(text_cache_paths, dict) or variant not in text_cache_paths:
        raise ValueError(f"Policy cache missing text_cache_paths for variant {variant}")
    text_path = Path(text_cache_paths[variant])
    if not text_path.is_absolute():
        raise ValueError(f"Text cache path must be absolute: {text_path}")
    text = load_text_cache(text_path)
    text_metadata = text["metadata"]
    if "task" in metadata and text_metadata.get("task") != metadata["task"]:
        raise ValueError("Text cache task does not match policy cache task")
    if "prompt" in metadata and text_metadata.get("prompt") != metadata["prompt"]:
        raise ValueError("Text cache prompt does not match policy cache prompt")
    if text_metadata.get("variant") != variant:
        raise ValueError("Text cache variant mismatch")
    expected = (metadata.get("text_cache_identities") or data.get("text_cache_identities") or {}).get(variant)
    if expected is not None and text["identity"] != expected:
        raise ValueError(f"Text cache identity mismatch for variant {variant}")
    data = dict(data)
    data["conditioning"] = text["conditioning"]
    data["tags"] = text["tags"]
    return data
