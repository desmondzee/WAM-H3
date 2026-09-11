import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class WAMH3Config:
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    action_dim: int = 7
    proprio_dim: int = 8
    action_horizon: int = 32
    text_len: int = 64
    latent_h: int = 14
    latent_w: int = 28
    num_video_latents: int = 2
    obs_t: float = 0.999
    ctx_sees_video: bool = True

    patch_dim = 4

    @property
    def frame_rows(self):
        return (self.latent_h // 2) * (self.latent_w // 2)

    @property
    def video_patch_dim(self):
        return self.latents_dim * self.patch_dim

    @property
    def adaln_out_features(self):
        return 6 * self.hidden_size * 3

    @property
    def final_adaln_out_features(self):
        return 2 * self.hidden_size

    @property
    def num_frames(self):
        return 4 * (self.num_video_latents - 1) + 1

    @property
    def actions_per_frame(self):
        return self.action_horizon // (self.num_frames - 1)

    @classmethod
    def tiny(cls, **kw):
        base = cls(num_layers=2, token_refiner_num_layers=1, hidden_size=64, num_attention_heads=2,
                   attention_head_dim=32, ffn_hidden_size=128, latents_dim=24, text_dim=16,
                   timestep_input_dim=8, time_embed_hidden_size=64, time_embed_dim=32,
                   rope_inv_freq_len=4, action_dim=32, proprio_dim=4, action_horizon=8, text_len=8,
                   latent_h=4, latent_w=4)
        return replace(base, **kw)

    @classmethod
    def from_pretrained_dir(cls, path, **kw):
        raw = json.loads((Path(path) / "transformer" / "config.json").read_text())
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in keys}, **kw)
