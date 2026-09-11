import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from wam_h3.data.text_cache import load_embedding

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="LIBERO-spatial not downloaded")


def test_precompute_fake_writes_one_embedding_per_prompt(tmp_path):
    ds = tmp_path / "ds/meta"
    ds.mkdir(parents=True)
    (ds / "tasks.jsonl").write_text('{"task_index": 0, "task": "open the drawer"}\n{"task_index": 1, "task": "open the drawer"}\n'
                                    '{"task_index": 2, "task": "close it"}\n')
    extra = tmp_path / "extra.txt"
    extra.write_text("wave hello\n")
    cache = tmp_path / "cache"
    subprocess.run([sys.executable, str(ROOT / "scripts/precompute_text_embeds.py"), "--fake", "--data-dirs", str(tmp_path / "ds"),
                    "--prompts-file", str(extra), "--cache-dir", str(cache)], check=True, cwd=ROOT)
    assert len(list(cache.glob("*.pt"))) == 3
    from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    e = load_embedding(cache, DEFAULT_PROMPT.format(task="open the drawer"))
    assert e is not None and e.ndim == 2 and e.shape[1] == 5120 and e.dtype == torch.bfloat16
    assert load_embedding(cache, DEFAULT_PROMPT.format(task="wave hello")) is not None


@needs_data
def test_dataset_sample_layout(tmp_path):
    from fasterwam.utils import misc
    from wam_h3.data.dataset import WAMH3VideoDataset

    misc.register_work_dir(tmp_path)
    cache = tmp_path / "cache"
    subprocess.run([sys.executable, str(ROOT / "scripts/precompute_text_embeds.py"), "--fake", "--data-dirs", str(DATA),
                    "--cache-dir", str(cache)], check=True, cwd=ROOT)
    cfg = OmegaConf.create({"data": OmegaConf.load(ROOT / "configs/data/libero_2cam.yaml")}).data.train
    cfg.dataset_dirs = [str(DATA)]
    cfg.text_embedding_cache_dir = str(cache)
    ds = WAMH3VideoDataset(**{k: v for k, v in cfg.items() if k != "_target_"})
    assert ds.video_sample_indices == [0, 8, 16, 24, 32]
    s = ds[0]
    assert s["video"].shape == (3, 5, 224, 448) and s["video"].min() >= -1 and s["video"].max() <= 1
    assert s["action"].shape == (32, 7) and s["proprio"].shape == (8,)
    assert s["context"].shape == (64, 5120) and s["context_mask"].shape == (64,)
    n = int(s["context_mask"].sum())
    assert 0 < n < 64 and (s["context"][n:] == 0).all()
    assert s["image_is_pad"].shape == (5,) and s["action_is_pad"].shape == (32,)
    assert (tmp_path / "dataset_stats.json").exists()
    assert json.loads((tmp_path / "dataset_stats.json").read_text())
