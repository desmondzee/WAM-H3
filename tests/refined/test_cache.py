import pytest
import torch

from wam_h3.refined.cache import (
    identity,
    load_policy_cache,
    load_text_cache,
    save_policy_cache,
    save_text_cache,
    text_cache_path,
)
from wam_h3.refined.runtime import COMFY_REVISION, MODEL_REVISION, MODELS
from wam_h3.refined.trainer import fit_normalizer


def _text_tensors():
    return torch.ones(1, 10, 64), torch.zeros(1, 10, dtype=torch.long)


def test_endpoint_cache_rejected(tmp_path):
    path = tmp_path / "endpoint.pt"
    torch.save({"purpose": "endpoint_generation_only", "positive": []}, path)
    with pytest.raises(ValueError, match="endpoint generation"):
        load_policy_cache(path)


def test_cache_identity_and_roundtrip(tmp_path):
    task = "pick up the black bowl"
    prompt = "A synchronized split-screen recording..."
    cond, tags = _text_tensors()
    text_path = tmp_path / "text.pt"
    save_text_cache(text_path, cond, tags, task, prompt, "camera")
    policy_path = tmp_path / "policy.pt"
    tensors = {
        "latent": torch.ones(1, 24, 2, 2, 2),
        "targets": torch.zeros(2, 3, 7),
        "valid": torch.ones(2, 3, dtype=torch.bool),
        "cutoffs": torch.arange(2),
        "text_cache_paths": {"camera": str(text_path)},
    }
    save_policy_cache(policy_path, tensors, {"episode": 0, "task": task, "variant": "camera", "prompt": prompt})
    data = load_policy_cache(policy_path, variant="camera")
    assert data["metadata"]["conditioning_images"] == 0
    assert data["metadata"]["schema"] == 2
    assert data["metadata"]["model_revision"] == MODEL_REVISION
    assert data["conditioning"].shape == (1, 10, 64)
    assert data["tags"].shape == (1, 10)
    data["metadata"]["episode"] = 1
    torch.save(data, policy_path)
    with pytest.raises(ValueError, match="identity"):
        load_policy_cache(policy_path, variant="camera")


def test_legacy_direct_conditioning_requires_explicit_images_flag(tmp_path):
    cond, tags = _text_tensors()
    path = tmp_path / "legacy.pt"
    save_policy_cache(
        path,
        {
            "latent": torch.ones(1, 24, 2, 2, 2),
            "conditioning": cond,
            "tags": tags,
        },
        {"episode": 0, "conditioning_images": 0},
    )
    data = load_policy_cache(path)
    assert data["metadata"]["conditioning_images"] == 0
    assert data["conditioning"].shape == (1, 10, 64)


def test_direct_conditioning_rejected_without_explicit_flag(tmp_path):
    cond, tags = _text_tensors()
    with pytest.raises(ValueError, match="Direct conditioning"):
        save_policy_cache(
            tmp_path / "bad.pt",
            {
                "latent": torch.ones(1, 24, 2, 2, 2),
                "conditioning": cond,
                "tags": tags,
            },
            {"episode": 0},
        )


def test_old_image_conditioned_cache_rejected(tmp_path):
    path = tmp_path / "old.pt"
    metadata = {
        "purpose": "causal_policy",
        "conditioning_images": 1,
        "schema": 1,
        "model_revision": MODEL_REVISION,
        "comfy_revision": COMFY_REVISION,
    }
    torch.save({"metadata": metadata, "identity": identity(metadata), "latent": torch.ones(1, 24, 2, 2, 2)}, path)
    with pytest.raises(ValueError, match="image-conditioned"):
        load_policy_cache(path)


def test_save_policy_cache_overwrite_guard(tmp_path):
    path = tmp_path / "policy.pt"
    save_policy_cache(path, {"latent": torch.ones(1, 24, 2, 2, 2)}, {"episode": 0})
    with pytest.raises(FileExistsError):
        save_policy_cache(path, {"latent": torch.ones(1, 24, 2, 2, 2)}, {"episode": 1})


def test_text_cache_identity_mismatch_in_policy(tmp_path):
    task = "task"
    prompt = "prompt"
    cond, tags = _text_tensors()
    text_path = tmp_path / "text.pt"
    save_text_cache(text_path, cond, tags, task, prompt, "camera")
    policy_path = tmp_path / "policy.pt"
    save_policy_cache(
        policy_path,
        {
            "latent": torch.ones(1, 24, 2, 2, 2),
            "text_cache_paths": {"camera": str(text_path)},
        },
        {"episode": 0, "task": task, "text_cache_identities": {"camera": "wrong"}},
    )
    with pytest.raises(ValueError, match="Text cache identity"):
        load_policy_cache(policy_path, variant="camera")


def test_normalization_excludes_padding():
    sample = {"targets": torch.tensor([[[-1., 2.], [1., 2.], [100., 100.]]]),
              "valid": torch.tensor([[True, True, False]])}
    stats = fit_normalizer([sample])
    torch.testing.assert_close(stats["center"], torch.tensor([0., 2.]))
    assert torch.isfinite(1 / stats["scale"]).all()


def test_model_hash_lengths():
    assert all(len(digest) == 64 for _, digest in MODELS.values())
