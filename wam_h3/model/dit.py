from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import DiTBlock, FinalLayer, Rope, TimeEmbedder, TokenRefiner
from .layout import SequenceLayout

GROUPS = 4


def patchify_video(latent):
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, h // 2, 2, w // 2, 2)
    return torch.einsum("bcthpwq->bthwcpq", x).reshape(b, t * (h // 2) * (w // 2), c * 4).contiguous()


def unpatchify_video(rows, t, h, w, channels=24):
    b = rows.shape[0]
    x = rows.reshape(b, t, h // 2, w // 2, channels, 2, 2)
    return torch.einsum("bthwcpq->bcthpwq", x).reshape(b, channels, t, h, w).contiguous()


@dataclass
class DiTOutput:
    video: torch.Tensor
    action: torch.Tensor
    hidden: torch.Tensor


@dataclass
class KVCache:
    kv: list
    text_valid: torch.Tensor
    t_groups: torch.Tensor


class WAMH3DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        with torch.device("cpu"):
            self.layout = SequenceLayout(cfg)
        self.video_patch_proj = nn.Linear(cfg.video_patch_dim, cfg.hidden_size)
        self.action_in = nn.Linear(cfg.action_dim, cfg.hidden_size)
        self.proprio_in = nn.Linear(cfg.proprio_dim, cfg.hidden_size) if cfg.proprio_dim else None
        self.condition_proj = nn.Linear(cfg.text_dim, cfg.hidden_size)
        self.time_embedder = TimeEmbedder(cfg)
        self.rope = Rope(cfg.rope_inv_freq_len)
        self.token_refiner = TokenRefiner(cfg)
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_layer = FinalLayer(cfg)
        self.init_buffers()

    def init_buffers(self, device=None):
        lay = self.layout
        device = device or self.condition_proj.weight.device
        for name, t in [("position_ids", lay.position_ids.float()), ("row_group", lay.row_group), ("row_tag", lay.row_tag),
                        ("ctx_rows", torch.cat([torch.arange(0, lay.action.start), torch.arange(lay.video.start, lay.video.stop)])),
                        ("action_rows", torch.arange(lay.action.start, lay.action.stop))]:
            self.register_buffer(name, t.to(device), persistent=False)
        self.rope.inv_freq.data = self.rope._inv_freq(device).to(self.rope.inv_freq.dtype)

    def freqs(self, rows=None):
        f = self.rope(self.position_ids[None])
        return f if rows is None else f[rows]

    def row_index(self, B, rows=None):
        g = self.row_group if rows is None else self.row_group[rows]
        t = self.row_tag if rows is None else self.row_tag[rows]
        b = torch.arange(B, device=g.device)[:, None]
        return ((b * GROUPS + g) * 3 + t).reshape(-1), (b * GROUPS + g).reshape(-1)

    def t_emb(self, t_groups, dtype):
        return self.time_embedder(t_groups.reshape(-1), dtype=dtype)

    def row_mods(self, block, t_groups):
        B = t_groups.shape[0]
        idx, _ = self.row_index(B)
        return tuple(m[idx].view(B, self.layout.N, -1) for m in block.adaln_proj(self.t_emb(t_groups, torch.float32)))

    def embed_ctx(self, text, text_valid, obs_rows, proprio):
        dtype = self.condition_proj.weight.dtype
        parts = [self.token_refiner(self.condition_proj(text.to(dtype)), text_valid),
                 self.video_patch_proj(obs_rows.to(dtype))]
        if self.proprio_in is not None:
            parts.append(self.proprio_in(proprio.to(dtype))[:, None])
        return parts

    def forward(self, text, text_valid, obs_rows, proprio, action, video_rows, t_groups, mask=None):
        lay = self.layout
        dtype = self.condition_proj.weight.dtype
        x = torch.cat(self.embed_ctx(text, text_valid, obs_rows, proprio)
                      + [self.action_in(action.to(dtype)), self.video_patch_proj(video_rows.to(dtype))], dim=1)
        B = x.shape[0]
        t_emb = self.t_emb(t_groups, dtype)
        idx, fidx = self.row_index(B)
        freqs = self.freqs()
        mask = lay.full_mask(text_valid) if mask is None else mask
        ckpt = getattr(self, "grad_checkpoint", False) and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if ckpt:
                x, _ = torch.utils.checkpoint.checkpoint(blk, x, t_emb, idx, freqs, mask, use_reentrant=False)
            else:
                x, _ = blk(x, t_emb, idx, freqs, mask)
        h = self.final_layer(x, t_emb, fidx)
        return DiTOutput(self.final_layer.video_out(h[:, lay.video]), self.final_layer.action_out(h[:, lay.action]), x)

    def prefill(self, text, text_valid, obs_rows, proprio, video_rows, t_groups):
        dtype = self.condition_proj.weight.dtype
        x = torch.cat(self.embed_ctx(text, text_valid, obs_rows, proprio) + [self.video_patch_proj(video_rows.to(dtype))], dim=1)
        B, rows = x.shape[0], self.ctx_rows
        t_emb = self.t_emb(t_groups, dtype)
        idx, _ = self.row_index(B, rows)
        freqs = self.freqs(rows)
        mask = self.layout.full_mask(text_valid)[:, :, rows][:, :, :, rows]
        kvs = []
        for blk in self.blocks:
            x, kv = blk(x, t_emb, idx, freqs, mask)
            kvs.append(kv)
        return KVCache(kvs, text_valid, t_groups)

    def denoise_action_step(self, cache, action, t_a):
        dtype = self.condition_proj.weight.dtype
        x = self.action_in(action.to(dtype))
        B, rows = x.shape[0], self.action_rows
        t_groups = torch.cat([cache.t_groups[:, :3], t_a[:, None].to(cache.t_groups)], dim=1)
        t_emb = self.t_emb(t_groups, dtype)
        idx, fidx = self.row_index(B, rows)
        freqs = self.freqs(rows)
        mask = self.layout.action_query_mask(cache.text_valid)
        for blk, kv in zip(self.blocks, cache.kv):
            x, _ = blk(x, t_emb, idx, freqs, mask, kv_ctx=kv, ctx_insert=self.layout.ctx_insert)
        return self.final_layer.action_out(self.final_layer(x, t_emb, fidx))

    def denoise_actions(self, cache, sched, steps=10, generator=None, shift=None):
        B, dev = cache.text_valid.shape[0], cache.text_valid.device
        x = torch.randn(B, self.cfg.action_horizon, self.cfg.action_dim, generator=generator).to(dev)
        sig, dl = sched.inference_schedule(steps, shift)
        for s, d in zip(sig, dl):
            pred = -self.denoise_action_step(cache, x, torch.full((B,), 1 - s.item(), device=dev))
            x = sched.step(pred.float(), d, x)
        return x
