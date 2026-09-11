from pathlib import Path

import torch

from .checkpoint import load_pretrained
from .config import WAMH3Config
from .dit import WAMH3DiT
from .lora import apply_lora
from .vae import VideoEncoder
from .wam import WAMH3

FRESH = ("action_in", "proprio_in", "final_layer.action_out")


def build_dit(cfg, transformer_dir, device, dtype):
    with torch.device("meta"):
        dit = WAMH3DiT(cfg)
    dit.to_empty(device=device)
    report = load_pretrained(dit, transformer_dir, dtype=dtype, device=device)
    expected = {k for k in report.missing if k.startswith(FRESH)}
    if report.missing != expected:
        raise RuntimeError(f"pretrained load missing keys: {sorted(report.missing - expected)[:8]}")
    for name in FRESH:
        m = dit.get_submodule(name) if name != "proprio_in" or dit.proprio_in is not None else None
        if m is not None:
            m.reset_parameters()
    torch.nn.init.zeros_(dit.final_layer.action_out.weight)
    torch.nn.init.zeros_(dit.final_layer.action_out.bias)
    dit.init_buffers()
    for p in dit.parameters():
        p.data = p.data.to(dtype)
    return dit


def create_wamh3(weights_root, action_dim, proprio_dim, video_size, num_frames, action_video_freq_ratio, text_len,
                 lora_r=None, lora_alpha=None, tiny=False, text_cache_dir=None, load_text_encoder=False,
                 device=None, dtype="bfloat16", shift=5.0, video_weight_center=1.0, video_full_noise_prob=0.3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    root = Path(weights_root)
    n_video = (num_frames - 1) // action_video_freq_ratio + 1
    dims = dict(action_dim=action_dim, proprio_dim=proprio_dim, action_horizon=num_frames - 1, text_len=text_len,
                latent_h=video_size[0] // 16, latent_w=video_size[1] // 16, num_video_latents=((n_video - 5) // 17) * 5 + 2)
    if tiny:
        cfg = WAMH3Config.tiny(text_dim=5120, **dims)
        dit = WAMH3DiT(cfg).to(device, dtype)
    else:
        cfg = WAMH3Config.from_pretrained_dir(root, **dims)
        dit = build_dit(cfg, root / "transformer", device, dtype)
    assert cfg.num_frames == n_video, (cfg.num_frames, n_video)
    if lora_r:
        apply_lora(dit, lora_r, lora_alpha)
        for p in dit.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    vae = VideoEncoder.from_safetensors(root / "video_vae/source/model.safetensors").to(device)
    text_encoder = None
    if load_text_encoder:
        from .text_encoder import TextEncoder
        text_encoder = TextEncoder.from_pretrained(root, device="cpu")
    return WAMH3(cfg, dit, vae, text_cache_dir=text_cache_dir, text_encoder=text_encoder, shift=shift,
                 video_weight_center=video_weight_center, video_full_noise_prob=video_full_noise_prob)
