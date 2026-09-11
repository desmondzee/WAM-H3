from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from safetensors.torch import save_file

from tests.test_dit import inputs
from wam_h3.model.build import build_dit, create_wamh3
from wam_h3.model.config import WAMH3Config
from wam_h3.model.dit import WAMH3DiT

ROOT = Path(__file__).resolve().parents[1]


def test_meta_build_matches_eager_build(tmp_path):
    cfg = WAMH3Config.tiny()
    ref = WAMH3DiT(cfg)
    sd = {k: v.contiguous() for k, v in ref.state_dict().items() if not k.startswith(("action_in", "proprio_in", "final_layer.action_out"))}
    sd["audio_patch_proj.weight"], sd["audio_patch_proj.bias"] = torch.randn(cfg.hidden_size, 32), torch.randn(cfg.hidden_size)
    (tmp_path / "transformer").mkdir()
    save_file(sd, str(tmp_path / "transformer/model.safetensors"))
    dit = build_dit(cfg, tmp_path / "transformer", device="cpu", dtype=torch.float32)
    assert not any(p.is_meta for p in dit.parameters()) and not any(b.is_meta for b in dit.buffers())
    assert torch.equal(dit.blocks[1].mlp.fc1.weight, ref.blocks[1].mlp.fc1.weight)
    assert torch.equal(dit.position_ids, ref.position_ids) and torch.equal(dit.rope.inv_freq, ref.rope.inv_freq)
    assert (dit.final_layer.action_out.weight == 0).all()
    assert torch.equal(dit.action_in.weight, ref.state_dict()["action_in.weight"][:, : cfg.action_dim]) is False
    assert torch.equal(dit.action_in.bias, sd["audio_patch_proj.bias"]) and torch.equal(dit.proprio_in.bias, sd["audio_patch_proj.bias"])
    assert torch.equal(dit.action_in.weight, sd["audio_patch_proj.weight"][:, : cfg.action_dim])
    assert torch.equal(dit.proprio_in.weight, sd["audio_patch_proj.weight"][:, : cfg.proprio_dim])
    ref.load_state_dict({k: v for k, v in dit.state_dict().items() if k.startswith(("action_in", "proprio_in", "final_layer.action_out"))}, strict=False)
    inp, tg = inputs(cfg), torch.rand(2, 4)
    assert torch.allclose(dit(**inp, t_groups=tg).hidden, ref(**inp, t_groups=tg).hidden, atol=1e-5)


def test_build_dit_bf16_dtype(tmp_path):
    cfg = WAMH3Config.tiny()
    sd = {k: v.contiguous() for k, v in WAMH3DiT(cfg).state_dict().items()}
    (tmp_path / "transformer").mkdir()
    save_file(sd, str(tmp_path / "transformer/model.safetensors"))
    dit = build_dit(cfg, tmp_path / "transformer", device="cpu", dtype=torch.bfloat16)
    assert {p.dtype for p in dit.parameters()} == {torch.bfloat16}


def test_create_wamh3_tiny_from_hydra_config():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["task=smoke_libero_tiny"])
    m = instantiate(cfg.model)
    assert m.cfg.action_dim == 7 and m.cfg.proprio_dim == 8
    assert m.cfg.latent_h == 14 and m.cfg.latent_w == 28 and m.cfg.action_horizon == 32 and m.cfg.num_video_latents == 2
    assert m.cfg.text_len == 64 and m.cfg.num_layers == 2
    assert sum(p.numel() for p in m.dit.parameters() if p.requires_grad) > 0
    assert cfg.train.batch_size == 1 and cfg.train.max_steps == 8
    assert cfg.data.train.action_video_freq_ratio == 8


def test_task_configs_compose():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        dev = compose(config_name="train", overrides=["task=libero_wamh3_dev"])
        full = compose(config_name="train", overrides=["task=libero_wamh3"])
    assert dev.model.lora_r == 16 and full.model.lora_r == 128
    assert dev.train.batch_size * dev.train.grad_accum == 128 and full.train.batch_size == 16
    assert list(dev.model.lora_targets) == list(full.model.lora_targets) and dev.model.lora_dropout == 0.0
    assert len(dev.data.train.dataset_dirs) == 1 and len(full.data.train.dataset_dirs) == 4
    assert dev.train.save_every == 100 and full.train.save_every == 2000


def test_single_gpu_task_config():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        one = compose(config_name="train", overrides=["task=libero_wamh3_1gpu"])
        tiny = compose(config_name="train", overrides=["task=smoke_libero_tiny", "model.lora_r=8", "model.lora_adaln_r=2"])
    assert one.model.lora_r == one.model.lora_alpha == 64 and one.model.lora_adaln_r == 16
    assert one.train.batch_size == 2 and one.train.grad_accum == 64 and one.train.max_steps == 900
    assert len(one.data.train.dataset_dirs) == 4
    m = instantiate(tiny.model)
    assert m.dit.blocks[0].adaln_proj.linear.lora_A["default"].weight.shape[0] == 2
    assert m.dit.blocks[0].attn.qkv_proj.lora_A["default"].weight.shape[0] == 8
