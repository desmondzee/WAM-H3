import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(size, eps):
    return nn.RMSNorm(size, eps=eps)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, freqs):
    rot = freqs.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    cos = torch.cos(freqs).to(x.dtype)[None, None]
    sin = torch.sin(freqs).to(x.dtype)[None, None]
    return torch.cat((xr * cos + _rotate_half(xr) * sin, xp), dim=-1)


def modulate(x, shift, scale, idx):
    B, S, C = x.shape
    return (x * (1.0 + scale[idx].view(B, S, C)) + shift[idx].view(B, S, C)).to(x.dtype)


def gate(x, g, other, idx):
    B, S, C = x.shape
    return (x + g[idx].view(B, S, C) * other).to(x.dtype)


class Rope(nn.Module):
    def __init__(self, inv_freq_len):
        super().__init__()
        self.inv_freq_len = inv_freq_len
        self.inv_freq = nn.Parameter(self._inv_freq(), requires_grad=False)

    def _inv_freq(self, device=None):
        steps = torch.arange(0, self.inv_freq_len, dtype=torch.float32, device=device)
        return 1.0 / (10000.0 ** (steps / self.inv_freq_len))

    def forward(self, position_ids):
        pos = position_ids.to(torch.float32)
        per_axis = pos.unsqueeze(-1) * self._inv_freq(pos.device).view(1, 1, -1)
        half = torch.cat(per_axis.unbind(dim=1), dim=-1)
        return torch.cat((half, half), dim=-1)


class TimeEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.frequency_embedding_size = cfg.timestep_input_dim
        self.proj_in = nn.Linear(cfg.timestep_input_dim, cfg.time_embed_hidden_size)
        self.proj_out = nn.Linear(cfg.time_embed_hidden_size, cfg.time_embed_dim)

    def forward(self, t, dtype):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.proj_out(F.silu(self.proj_in(emb)))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads, self.head_dim = cfg.num_attention_heads, cfg.attention_head_dim
        inner = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(cfg.hidden_size, inner * 3, bias=False)
        self.q_norm = _norm(self.head_dim, cfg.qk_norm_eps)
        self.k_norm = _norm(self.head_dim, cfg.qk_norm_eps)
        self.out_proj = nn.Linear(inner, cfg.hidden_size, bias=False)

    def project(self, x, freqs):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x).view(B, S, self.num_heads, 3, self.head_dim)
        q, k, v = (qkv[:, :, :, i].transpose(1, 2) for i in range(3))
        q, k = self.q_norm(q), self.k_norm(k)
        if freqs is not None:
            q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        return q, k, v

    def attend(self, q, k, v, mask):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        return self.out_proj(out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1))

    def forward(self, x, freqs, mask):
        return self.attend(*self.project(x, freqs), mask)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(cfg.ffn_hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        g, u = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(g) * u)


class AdalnProj(nn.Module):
    def __init__(self, cfg, expand_ratio, modality_num):
        super().__init__()
        self.expand_ratio, self.modality_num, self.hidden_size = expand_ratio, modality_num, cfg.hidden_size
        self.linear = nn.Linear(cfg.time_embed_dim, expand_ratio * cfg.hidden_size * modality_num)

    def forward(self, t_emb):
        x = self.linear(F.silu(t_emb))
        x = x.view(x.shape[0] * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class TokenRefinerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = _norm(cfg.hidden_size, cfg.norm_eps)
        self.norm2 = _norm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), None, mask)
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([TokenRefinerBlock(cfg) for _ in range(cfg.token_refiner_num_layers)])
        self.final_norm = _norm(cfg.hidden_size, cfg.final_norm_eps)

    def forward(self, x, valid):
        mask = valid[:, None, None, :]
        for b in self.blocks:
            x = b(x, mask)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = _norm(cfg.hidden_size, cfg.norm_eps)
        self.norm2 = _norm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.adaln_proj = AdalnProj(cfg, 6, 3)

    def forward(self, x, t_emb, row_idx, freqs, mask, kv_ctx=None, ctx_insert=None, mods=None):
        sh1, sc1, g1, sh2, sc2, g2 = mods if mods is not None else self.adaln_proj(t_emb)
        h = modulate(self.norm1(x), sh1, sc1, row_idx)
        q, k, v = self.attn.project(h, freqs)
        if kv_ctx is not None:
            kc, vc = kv_ctx
            k_all = torch.cat([kc[:, :, :ctx_insert], k, kc[:, :, ctx_insert:]], dim=2)
            v_all = torch.cat([vc[:, :, :ctx_insert], v, vc[:, :, ctx_insert:]], dim=2)
        else:
            k_all, v_all = k, v
        x = gate(x, g1, self.attn.attend(q, k_all, v_all, mask), row_idx)
        h = modulate(self.norm2(x), sh2, sc2, row_idx)
        return gate(x, g2, self.mlp(h), row_idx), (k, v)


class FinalLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = _norm(cfg.hidden_size, cfg.final_norm_eps)
        self.adaln_proj = AdalnProj(cfg, 2, 1)
        self.video_out = nn.Linear(cfg.hidden_size, cfg.video_patch_dim)
        self.action_out = nn.Linear(cfg.hidden_size, cfg.action_dim)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, x, t_emb, row_idx, mods=None):
        shift, scale = mods if mods is not None else self.adaln_proj(t_emb)
        return modulate(self.norm(x), shift, scale, row_idx)
