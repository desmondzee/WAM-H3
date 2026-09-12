from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import policy_mask


@dataclass
class PolicyConfig:
    width: int = 1024
    heads: int = 16
    ffn: int = 4096
    memory_width: int = 5376
    taps: tuple = (7, 15, 23, 31, 39, 49)
    horizon: int = 34
    action_dim: int = 7

    def __post_init__(self):
        if any(v <= 0 for v in (self.width, self.heads, self.ffn, self.memory_width, self.horizon, self.action_dim)):
            raise ValueError("Policy dimensions must be positive")
        if self.width % self.heads or not self.taps or len(set(self.taps)) != len(self.taps):
            raise ValueError("Policy requires divisible heads and unique nonempty taps")


class PolicyBlock(nn.Module):
    def __init__(self, cfg, cross):
        super().__init__()
        self.cross, self.heads = cross, cfg.heads
        d = cfg.width
        self.attn_norm, self.mlp_norm = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, d if cross else d * 3, bias=False)
        self.memory_norm = nn.RMSNorm(cfg.memory_width, eps=1e-6) if cross else None
        self.memory = nn.Linear(cfg.memory_width, d * 2, bias=False) if cross else None
        self.out = nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, cfg.ffn * 2, bias=False), nn.Linear(cfg.ffn, d, bias=False)

    def heads_view(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(-3, -2)

    def forward(self, x, memory=None, mask=None):
        h = self.attn_norm(x)
        if self.cross:
            q = self.heads_view(self.q(h))
            k, v = (self.heads_view(t).unsqueeze(0) for t in self.memory(self.memory_norm(memory)).chunk(2, -1))
        else:
            q, k, v = (self.heads_view(t) for t in self.q(h).chunk(3, -1))
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(h.transpose(-3, -2).flatten(-2))
        gate, value = self.fc1(self.mlp_norm(x)).chunk(2, -1)
        return x + self.fc2(F.silu(gate) * value)


class ActionPolicy(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        self.queries = nn.Parameter(torch.randn(self.cfg.horizon, self.cfg.width) * 0.02)
        self.blocks = nn.ModuleList([PolicyBlock(self.cfg, cross=i % 2 == 0) for i in range(2 * len(self.cfg.taps))])
        self.norm = nn.RMSNorm(self.cfg.width)
        self.output = nn.Linear(self.cfg.width, self.cfg.action_dim)

    def forward(self, memory, memory_times, cutoffs):
        if len(memory) != len(self.cfg.taps):
            raise ValueError("Expected one memory tensor per selected FL2VA layer")
        if memory_times.ndim != 1 or cutoffs.ndim != 1 or not cutoffs.numel():
            raise ValueError("Memory times and nonempty decision cutoffs must be vectors")
        if any(m.shape != (memory_times.numel(), self.cfg.memory_width) for m in memory):
            raise ValueError("Memory shape must match token times and configured width")
        if not torch.isfinite(memory_times).all() or not torch.isfinite(cutoffs).all():
            raise ValueError("Memory times and cutoffs must be finite")
        mask = policy_mask(memory_times, cutoffs)
        if not mask.any(-1).all():
            raise ValueError("Every decision requires historical memory")
        x = self.queries.unsqueeze(0).expand(len(cutoffs), -1, -1)
        for i, block in enumerate(self.blocks):
            x = block(x, memory[i // 2], mask[:, None, None, :]) if block.cross else block(x)
        return self.output(self.norm(x))


def chunk_masked_loss(prediction, target, valid):
    if prediction.ndim != 3 or prediction.shape != target.shape or valid.shape != prediction.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Targets and boolean validity mask must match chunk prediction dimensions")
    counts = valid.sum(-1)
    if not counts.numel() or not (counts > 0).all():
        raise ValueError("Every chunk requires valid action targets")
    selected = (prediction[valid].float() - target[valid].float()).square().mean(-1)
    indices = torch.arange(len(counts), device=valid.device)[:, None].expand_as(valid)[valid]
    return selected.new_zeros(len(counts)).scatter_add(0, indices, selected) / counts


def masked_loss(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != prediction.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Targets and boolean validity mask must match prediction dimensions")
    if not valid.any():
        raise ValueError("No valid action targets")
    return F.mse_loss(prediction[valid].float(), target[valid].float())
