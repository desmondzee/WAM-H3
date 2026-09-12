import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image


LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OFFICIAL_REPO = "yifengzhu-hf/LIBERO-datasets"
OFFICIAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10")
ALIGNMENT_EVIDENCE = dict(
    observation="post_action", target_offset=1,
    source_url=f"https://github.com/Lifelong-Robot-Learning/LIBERO/blob/{LIBERO_REVISION}/scripts/create_dataset.py",
    evidence="L175-177 steps action[j]; L192-195 skips first five raw actions and records valid_index; "
             "L220-221 appends resulting camera observations; L226-227 selects actions[valid_index]; "
             "L248-260 writes those observations and actions. Stored obs[t] follows stored action[t], "
             "so the next action is stored action[t+1].",
    upstream_initial_actions_skipped=5, additional_noop_filter=False,
)


@dataclass
class Episode:
    index: int
    task: str
    frames: torch.Tensor
    actions: torch.Tensor
    source: dict = field(default_factory=dict)


def paired_frames(left, right, size=384):
    if left.shape != right.shape or left.ndim != 4 or left.shape[-1] != 3:
        raise ValueError("Camera streams must have matching [T,H,W,3] RGB shapes")
    views = [F.interpolate(x.permute(0, 3, 1, 2).float() / 255, (size, size), mode="bilinear",
                           align_corners=False, antialias=True) for x in (left, right)]
    return torch.cat(views, dim=-1).permute(0, 2, 3, 1).contiguous()


def decode_video(path):
    with av.open(str(path)) as container:
        return torch.from_numpy(np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)]))


@lru_cache(maxsize=16)
def _file_hash(path, size, mtime_ns):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episode(root, index, *, diagnostic_processed=False):
    if diagnostic_processed:
        return load_processed_episode_diagnostic(root, index)
    import h5py
    root = Path(root).resolve()
    if any("libero-plus" in part.lower() or "libero_plus" in part.lower() for part in root.parts):
        raise ValueError("Official LIBERO training must not use LIBERO-Plus")
    manifest = json.loads((root / "provenance.json").read_text())
    if (manifest.get("source") != "official_libero" or manifest.get("repo_id") != OFFICIAL_REPO
            or manifest.get("upstream_revision") != LIBERO_REVISION or manifest.get("alignment") != ALIGNMENT_EVIDENCE):
        raise ValueError("Missing audited official LIBERO provenance/alignment")
    if not isinstance(index, int) or not 0 <= index < len(manifest["episodes"]):
        raise ValueError("Episode index is outside the official manifest")
    entry = manifest["episodes"][index]
    path = (root / entry["path"]).resolve()
    suite = Path(entry["path"]).parts[0]
    if not path.is_relative_to(root) or suite not in OFFICIAL_SUITES:
        raise ValueError("Episode path escapes official LIBERO suites")
    records = [record for record in manifest["files"] if record["path"] == entry["path"]]
    if len(records) != 1:
        raise ValueError("Episode file has no unique provenance record")
    record = records[0]
    stat = path.stat()
    if stat.st_size != record["size"] or _file_hash(str(path), stat.st_size, stat.st_mtime_ns) != record["sha256"]:
        raise ValueError("Official HDF5 content hash mismatch")
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        problem = json.loads(data.attrs["problem_info"])
        task = problem["language_instruction"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Missing official task instruction")
        env = json.loads(data.attrs["env_args"])
        if env["env_kwargs"]["control_freq"] != 20:
            raise ValueError("Expected 20 Hz official recordings")
        demo = data[entry["demo"]]
        actions = torch.from_numpy(demo["actions"][()].astype(np.float32))
        views = [torch.from_numpy(demo[f"obs/{key}"][()]) for key in ("agentview_rgb", "eye_in_hand_rgb")]
        if actions.ndim != 2 or actions.shape[1] != 7 or not torch.isfinite(actions).all():
            raise ValueError("Expected finite [T,7] official actions")
        if len(actions) < 2 or any(len(view) != len(actions) or view.dtype != torch.uint8 for view in views):
            raise ValueError("Official camera/action rows must match without filtering")
        convention = data.attrs.get("macros_image_convention", "unknown")
        if isinstance(convention, bytes):
            convention = convention.decode()
        source = dict(origin="official_libero", repo_id=manifest["repo_id"], revision=manifest["revision"],
                      file=entry["path"], demo=entry["demo"], sha256=record["sha256"], url=record["url"],
                      alignment=manifest["alignment"], source_frame_indices=list(range(len(actions))),
                      source_camera_shape=list(views[0].shape[1:]), image_convention=str(convention),
                      orientation="stored RGB rows preserved; no flip", noop_filter=False, suite=suite)
        if "episode_sha256" in entry:
            source["episode_sha256"] = entry["episode_sha256"]
    return Episode(index, task, paired_frames(*views), actions, source)


def load_processed_episode_diagnostic(root, index):
    root = Path(root)
    if any("libero-plus" in part.lower() or "libero_plus" in part.lower() for part in root.resolve().parts):
        raise ValueError("Training and generation diagnostics require LIBERO, not LIBERO-Plus")
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 20:
        raise ValueError("Expected 20 Hz LIBERO data")
    fields = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
    rows = pq.read_table(root / info["data_path"].format(**fields)).to_pydict()
    n = len(rows["action"])
    if rows["frame_index"] != list(range(n)) or set(rows["episode_index"]) != {index}:
        raise ValueError("Episode rows are not contiguous and isolated")
    if not np.allclose(rows["timestamp"], np.arange(n) / 20, atol=1e-4):
        raise ValueError("Frame timestamps are not aligned at 20 Hz")
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    if len(set(rows["task_index"])) != 1:
        raise ValueError("Episode contains multiple tasks")
    views = [decode_video(root / info["video_path"].format(**fields, video_key=k))
             for k in ("observation.images.image", "observation.images.wrist_image")]
    if any(len(v) != n for v in views):
        raise ValueError("Video frame counts disagree with action rows")
    return Episode(index, tasks[rows["task_index"][0]], paired_frames(*views), torch.tensor(rows["action"]),
                   dict(origin="processed_diagnostic_only", root=str(root.resolve()),
                        alignment="unverified", training_allowed=False))


def decision_indices(length, horizon=34):
    if length < 2:
        raise ValueError("Episode has no predictable action targets")
    cutoffs = torch.arange(0, length - 1, horizon)
    indices = cutoffs[:, None] + 1 + torch.arange(horizon)
    return cutoffs, indices.clamp_max(length - 1), indices < length


def prefix_frames(frames, cutoff):
    if not 0 <= cutoff < len(frames):
        raise ValueError("Decision cutoff is outside the episode")
    return torch.cat([frames[:1].expand(5, -1, -1, -1), frames[1:cutoff + 1]])


def save_video(path, frames, fps=24):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = (frames.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    selected = np.linspace(0, len(images) - 1, 12).round().astype(int)
    tiles = [Image.fromarray(images[i]).resize((384, 192)) for i in selected]
    sheet = Image.new("RGB", (384 * 3, 192 * 4))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (384 * (i % 3), 192 * (i // 3)))
    sheet.save(path.with_suffix(".jpg"))
