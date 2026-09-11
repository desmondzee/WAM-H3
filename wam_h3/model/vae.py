import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

LATENTS_MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
LATENTS_STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Conv3d(nn.Conv3d):
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0, padding_mode="zeros", causal=True):
        super().__init__(cin, cout, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.pad_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.pad_mode_t = "constant" if causal else "replicate"
        self.causal = causal

    def _temporal_pad(self, x):
        if x.shape[2] > 1:
            p = self.padding[0]
            return F.pad(x, (0, 0, 0, 0, p * 2 if self.causal else p, 0 if self.causal else p), mode=self.pad_mode_t)
        if self.pad_mode_t == "constant":
            return torch.cat([torch.zeros_like(x[:, :, :1]).expand(-1, -1, self.kernel_size[0] - 1, -1, -1), x], dim=2)
        return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)
        if self.padding[2] or self.padding[1]:
            x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0), mode=self.pad_mode)
        return F.conv3d(self._temporal_pad(x), self.weight, self.bias, stride=self.stride, dilation=self.dilation)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        B, C, T, H, W = x.shape
        y = super().forward(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, 1, H, W))
        return y.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


def _gn(ch):
    return TemporalIsolatedGroupNorm(32, ch, eps=1e-6)


class Downsample3D(nn.Module):
    def __init__(self, ch, time_stride, space_stride, **kw):
        super().__init__()
        self.conv = Conv3d(ch, ch, 3, stride=(time_stride, space_stride, space_stride), padding=(1, 0, 0), **kw)
        self.space_stride = space_stride

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.conv.pad_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, cin, cout, **kw):
        super().__init__()
        self.norm1, self.norm2 = _gn(cin), _gn(cout)
        self.conv1 = Conv3d(cin, cout, 3, padding=1, **kw)
        self.conv2 = Conv3d(cout, cout, 3, padding=1, **kw)
        if cin != cout:
            self.nin_shortcut = Conv3d(cin, cout, 1, **kw)

    def forward(self, x):
        h = self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return (self.nin_shortcut(x) if hasattr(self, "nin_shortcut") else x) + h


class Encoder(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, **kw):
        super().__init__()
        mid = [ch * m for m in ch_mult]
        cin = [mid[0]] + mid[:-1]
        self.conv_in = Conv3d(in_channels, cin[0], 3, padding=1, **kw)
        self.down = nn.ModuleList()
        for i in range(len(ch_mult)):
            level = nn.Module()
            level.block = nn.ModuleList([ResnetBlock3D(cin[i] if j == 0 else mid[i], mid[i], **kw) for j in range(num_res_blocks)])
            if space_down[i] * time_down[i] > 1:
                level.downsample = Downsample3D(mid[i], time_down[i], space_down[i], **kw)
            self.down.append(level)
        self.norm_out = _gn(mid[-1])
        self.conv_out = Conv3d(mid[-1], 2 * z_channels, 3, padding=1, **kw)

    def forward(self, x):
        h = self.conv_in(x)
        for level in self.down:
            for b in level.block:
                h = b(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class VideoEncoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 2, 4, 4, 8), space_down=(2, 2, 2, 2, 1, 1), time_down=(1, 2, 2, 1, 1, 1),
                 num_res_blocks=2, z_channels=24, clip_length=17, token_drop=3, tile_size=256, tile_overlap_min=64):
        super().__init__()
        self.ratio, self.ratio_t = math.prod(space_down), math.prod(time_down)
        self.clip_length, self.token_drop = clip_length, token_drop
        self.tile_size, self.tile_overlap_min = tile_size, tile_overlap_min
        self.encoder = Encoder(ch, ch_mult, space_down, time_down, num_res_blocks, 3, z_channels, padding_mode="reflect")
        self.quant_conv = nn.Conv3d(2 * z_channels, 2 * z_channels, 1)
        self.register_buffer("mean", torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(LATENTS_STD).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("px_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("px_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @classmethod
    def from_safetensors(cls, path, **kw):
        m = cls(**kw)
        with safe_open(str(path), framework="pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(("encoder.", "quant_conv."))}
        m.load_state_dict(sd, strict=True)
        return m.eval()

    def _tiles(self, n):
        ts, ov = self.tile_size, self.tile_overlap_min
        if ts >= n:
            return [0], [n], []
        N = math.ceil(n / ts)
        while ts * N - ov * (N - 1) - n < 0:
            N += 1
        overlaps = [ov] * (N - 1)
        for i in range((ts * N - ov * (N - 1) - n) // self.ratio):
            overlaps[i % (N - 1)] += self.ratio
        starts = [0]
        for o in overlaps:
            starts.append(starts[-1] + ts - o)
        return starts, [ts] * N, overlaps

    @staticmethod
    def _blend(a, b, n, dim):
        n = min(a.shape[dim], b.shape[dim], n)
        w = (torch.arange(n, device=b.device, dtype=b.dtype) / n).view([n if d == dim % b.ndim else 1 for d in range(b.ndim)])
        mixed = a.narrow(dim, a.shape[dim] - n, n) * (1 - w) + b.narrow(dim, 0, n) * w
        return torch.cat([mixed, b.narrow(dim, n, b.shape[dim] - n)], dim=dim) if n < b.shape[dim] else mixed

    def _encode_tiled(self, x):
        yi, yl, yo = self._tiles(x.shape[-2])
        xi, xl, xo = self._tiles(x.shape[-1])
        rows = [[self.quant_conv(self.encoder(x[..., i:i + il, j:j + jl])) for j, jl in zip(xi, xl)] for i, il in zip(yi, yl)]
        ly, lx = [o // self.ratio for o in yo], [o // self.ratio for o in xo]
        out = []
        for i, row in enumerate(rows):
            parts = []
            for j, t in enumerate(row):
                if i > 0:
                    t = self._blend(rows[i - 1][j], t, ly[i - 1], -2)
                if j > 0:
                    t = self._blend(row[j - 1], t, lx[j - 1], -1)
                if i < len(rows) - 1:
                    t = t[..., :-ly[i], :]
                if j < len(row) - 1:
                    t = t[..., :, :-lx[j]]
                parts.append(t)
            out.append(torch.cat(parts, dim=-1))
        return torch.cat(out, dim=-2)

    def _prep(self, x):
        return ((x.float() - self.px_mean) / self.px_std).to(self.quant_conv.weight.dtype)

    def _post(self, moments):
        z = moments.chunk(2, dim=1)[0].float()
        return (z - self.mean) / self.std

    @torch.no_grad()
    def encode_image(self, img):
        return self._post(self._encode_tiled(self._prep(img.unsqueeze(2)))[:, :, :1])

    @torch.no_grad()
    def encode_clip(self, video):
        x = self._prep(video)
        pad = (-x.shape[2]) % self.clip_length
        if pad:
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        z = torch.cat([self._encode_tiled(x[:, :, i:i + self.clip_length]) for i in range(0, x.shape[2], self.clip_length)], dim=2)
        return self._post(z[:, :, :-self.token_drop] if self.token_drop else z)
