import numpy as np
import torch

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
INTERP = 32


def _axis(dim, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 0.5
    grid = np.linspace(left, left + ratio, dim // 2, endpoint=False) * INTERP
    return torch.from_numpy(grid).to(torch.float64)


def frame_grid(latent_h, latent_w):
    sqrt_area = np.sqrt(latent_h * latent_w)
    h, w = _axis(latent_h, sqrt_area), _axis(latent_w, sqrt_area)
    hh, ww = torch.meshgrid(h, w, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w


def video_t_grid(n, origin):
    spans = torch.tensor([FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)], dtype=torch.float64)
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


class SequenceLayout:
    def __init__(self, cfg):
        L, F, A, K = cfg.text_len, cfg.frame_rows, cfg.action_horizon, cfg.num_video_latents
        P = 1 if cfg.proprio_dim > 0 else 0
        self.text = slice(0, L)
        self.obs = slice(L, L + F)
        self.proprio = slice(L + F, L + F + P)
        self.action = slice(L + F + P, L + F + P + A)
        self.video = slice(L + F + P + A, L + F + P + A + K * F)
        self.N = self.video.stop
        self.ctx_len = self.N - A
        self.ctx_insert = self.action.start

        g = torch.zeros(self.N, dtype=torch.long)
        t = torch.zeros(self.N, dtype=torch.long)
        g[self.obs], g[self.text], g[self.video], g[self.action] = 1, 2, 2, 3
        t[self.text], t[self.proprio], t[self.action] = 1, 2, 2
        self.row_group, self.row_tag = g, t

        frame, w_grid = frame_grid(cfg.latent_h, cfg.latent_w)
        p = torch.zeros(self.N, 3, dtype=torch.float64)
        p[self.text, 0] = torch.arange(L, dtype=torch.float64)
        p[self.obs, 0] = float(L)
        p[self.obs, 1:] = frame
        p[self.proprio] = torch.tensor([float(L), 0.0, float(w_grid[0])], dtype=torch.float64)
        p[self.video, 0] = video_t_grid(K, float(L)).repeat_interleave(F)
        p[self.video, 1:] = frame.repeat(K, 1)
        p[self.action, 0] = L + torch.arange(A, dtype=torch.float64) / cfg.actions_per_frame * FRAME_RESCALE
        p[self.action, 2] = float(w_grid[-1])
        self.position_ids = p

        m = torch.zeros(self.N, self.N, dtype=torch.bool)
        ctx = slice(0, self.action.start)
        m[ctx, ctx] = True
        m[ctx, self.video] = bool(cfg.ctx_sees_video)
        m[self.video, ctx] = True
        m[self.video, self.video] = True
        m[self.action, :] = True
        self._base_mask = m

    def base_mask(self):
        return self._base_mask.clone()

    def full_mask(self, text_valid):
        m = self._base_mask.to(text_valid.device).expand(text_valid.shape[0], -1, -1).clone()
        m[:, :, self.text] &= text_valid[:, None, :]
        return m[:, None]

    def action_query_mask(self, text_valid):
        return self.full_mask(text_valid)[:, :, self.action]
