import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from wam_h3.refined.conditioning import prompt_variants
from wam_h3.refined.data import (
    ALIGNMENT_EVIDENCE, LIBERO_REVISION, OFFICIAL_REPO,
    decision_indices, load_episode, paired_frames, prefix_frames,
)


def test_camera_order_and_resize():
    left = torch.zeros(2, 16, 16, 3, dtype=torch.uint8)
    right = torch.full_like(left, 255)
    frames = paired_frames(left, right)
    assert frames.shape == (2, 384, 768, 3)
    assert torch.count_nonzero(frames[:, :, :384]) == 0
    assert torch.all(frames[:, :, 384:] == 1)


def test_next_action_and_partial_tail():
    cutoffs, indices, valid = decision_indices(70)
    assert cutoffs.tolist() == [0, 34, 68]
    assert indices[0].tolist() == list(range(1, 35))
    assert indices[1].tolist() == list(range(35, 69))
    assert valid.sum(1).tolist() == [34, 34, 1]
    assert indices[2, 0] == 69
    with pytest.raises(ValueError):
        decision_indices(1)


def test_repeated_initial_observation_and_prefix():
    frames = torch.arange(70).view(70, 1, 1, 1)
    prefix = prefix_frames(frames, 34)
    assert prefix.shape[0] == 39
    assert prefix[:5].flatten().tolist() == [0] * 5
    assert prefix[5:].flatten().tolist() == list(range(1, 35))


def test_prompt_variants_preserve_task():
    task = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    variants = prompt_variants(task)
    assert len(variants) == 4
    assert all(task in prompt for prompt in variants.values())
    assert "wrist" in variants["camera"]
    assert "No delay" in variants["constraints"]


@pytest.fixture
def official_fixture(tmp_path):
    folder = tmp_path / "libero_spatial"
    folder.mkdir()
    path = folder / "task_demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "pick up the bowl"})
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        data.attrs["macros_image_convention"] = "opengl"
        for index in range(2):
            demo = data.create_group(f"demo_{index}")
            actions = np.arange(28, dtype=np.float32).reshape(4, 7) + index * 100
            actions[1] = 0
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("obs/agentview_rgb", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
            demo.create_dataset("obs/eye_in_hand_rgb", data=np.full((4, 8, 8, 3), 255, dtype=np.uint8))
    relative = path.relative_to(tmp_path).as_posix()
    manifest = dict(source="official_libero", repo_id=OFFICIAL_REPO, revision="synthetic_fixture",
                    upstream_revision=LIBERO_REVISION, alignment=ALIGNMENT_EVIDENCE,
                    files=[dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                size=path.stat().st_size, url="synthetic_fixture")],
                    episodes=[dict(path=relative, demo=f"demo_{index}") for index in range(2)])
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    return tmp_path, path, manifest


def test_official_hdf5_identity_cameras_alignment(official_fixture):
    root, _, _ = official_fixture
    episode = load_episode(root, 1)
    assert episode.task == "pick up the bowl"
    assert episode.frames.shape == (4, 384, 768, 3)
    assert torch.count_nonzero(episode.frames[:, :, :384]) == 0
    assert torch.all(episode.frames[:, :, 384:] == 1)
    assert episode.actions[:, 0].tolist() == [100, 0, 114, 121]
    _, indices, valid = decision_indices(len(episode.frames))
    assert episode.actions[indices][valid][:, 0].tolist() == [0, 114, 121]
    assert episode.source["demo"] == "demo_1"
    assert episode.source["source_frame_indices"] == [0, 1, 2, 3]
    assert episode.source["alignment"]["target_offset"] == 1
    assert not episode.source["noop_filter"]
    with pytest.raises(ValueError, match="outside"):
        load_episode(root, -1)
    with pytest.raises(ValueError, match="outside"):
        load_episode(root, 2)


def test_official_hash_and_provenance_required(official_fixture):
    root, path, manifest = official_fixture
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_episode(root, 0)
    manifest["alignment"] = {"observation": "unverified"}
    (root / "provenance.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance/alignment"):
        load_episode(root, 0)


def test_official_default_rejects_processed_without_opt_in(tmp_path):
    (tmp_path / "meta").mkdir()
    with pytest.raises(FileNotFoundError, match="provenance.json"):
        load_episode(tmp_path, 0)
