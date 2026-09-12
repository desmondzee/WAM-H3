import hashlib
from pathlib import Path

import pytest
import torch

from wam_h3.refined.cache import identity, load_text_cache, save_text_cache, text_cache_path


def _dummy():
    return torch.ones(1, 12, 64), torch.zeros(1, 12, dtype=torch.long)


def test_text_cache_path(tmp_path):
    task = "pick up the bowl"
    variant = "camera"
    path = text_cache_path(tmp_path, task, variant)
    expected_name = f"{hashlib.sha256(task.encode()).hexdigest()[:16]}_{variant}.pt"
    assert path.name == expected_name
    assert path.parent == Path(tmp_path)


def test_text_cache_roundtrip(tmp_path):
    cond, tags = _dummy()
    path = tmp_path / "text.pt"
    payload = save_text_cache(path, cond, tags, "task text", "prompt text", "camera")
    data = load_text_cache(path)
    assert data["conditioning"].shape == (1, 12, 64)
    assert data["tags"].shape == (1, 12)
    assert data["metadata"]["schema"] == 1
    assert data["metadata"]["purpose"] == "text_only_conditioning"
    assert data["metadata"]["kind"] == "text_only"
    assert data["metadata"]["conditioning_images"] == 0
    assert data["metadata"]["task"] == "task text"
    assert data["metadata"]["prompt"] == "prompt text"
    assert data["metadata"]["variant"] == "camera"
    assert data["identity"] == identity(data["metadata"])
    assert data["identity"] == payload["identity"]


def test_text_cache_identity_mismatch(tmp_path):
    cond, tags = _dummy()
    path = tmp_path / "text.pt"
    save_text_cache(path, cond, tags, "task", "prompt", "camera")
    data = torch.load(path, map_location="cpu", weights_only=True)
    data["metadata"]["prompt"] = "changed"
    torch.save(data, path)
    with pytest.raises(ValueError, match="identity"):
        load_text_cache(path)


def test_text_cache_revision_mismatch(tmp_path):
    cond, tags = _dummy()
    path = tmp_path / "text.pt"
    save_text_cache(path, cond, tags, "task", "prompt", "camera")
    data = torch.load(path, map_location="cpu", weights_only=True)
    data["metadata"]["model_revision"] = "other"
    torch.save(data, path)
    with pytest.raises(ValueError, match="revision"):
        load_text_cache(path)


def test_text_cache_overwrite_guard(tmp_path):
    cond, tags = _dummy()
    path = tmp_path / "t.pt"
    save_text_cache(path, cond, tags, "task", "prompt", "camera")
    with pytest.raises(FileExistsError):
        save_text_cache(path, cond, tags, "task", "prompt", "camera")
