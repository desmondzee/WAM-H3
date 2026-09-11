import torch

from wam_h3.model.config import WAMH3Config
from wam_h3.model.layout import SequenceLayout


def test_tiny_config_shapes():
    cfg = WAMH3Config.tiny()
    assert cfg.frame_rows == 4
    assert cfg.adaln_out_features == 6 * cfg.hidden_size * 3
    assert cfg.final_adaln_out_features == 2 * cfg.hidden_size
    assert cfg.num_frames == 5
    assert cfg.actions_per_frame == cfg.action_horizon // 4


def test_slices_tile_sequence(cfg):
    lay = SequenceLayout(cfg)
    L, F, A, K = cfg.text_len, cfg.frame_rows, cfg.action_horizon, cfg.num_video_latents
    assert lay.text == slice(0, L)
    assert lay.obs == slice(L, L + F)
    assert lay.proprio == slice(L + F, L + F + 1)
    assert lay.action == slice(L + F + 1, L + F + 1 + A)
    assert lay.video == slice(L + F + 1 + A, L + F + 1 + A + K * F)
    assert lay.N == L + F + 1 + A + K * F
    assert lay.ctx_len == lay.N - A


def test_groups_and_tags(cfg):
    lay = SequenceLayout(cfg)
    g, t = lay.row_group, lay.row_tag
    assert (g[lay.text] == 2).all() and (t[lay.text] == 1).all()
    assert (g[lay.obs] == 1).all() and (t[lay.obs] == 0).all()
    assert (g[lay.proprio] == 0).all() and (t[lay.proprio] == 2).all()
    assert (g[lay.video] == 2).all() and (t[lay.video] == 0).all()
    assert (g[lay.action] == 3).all() and (t[lay.action] == 2).all()


def test_position_ids(cfg):
    lay = SequenceLayout(cfg)
    p = lay.position_ids
    L = cfg.text_len
    assert p.shape == (lay.N, 3)
    assert torch.equal(p[lay.text, 0], torch.arange(L, dtype=p.dtype))
    assert (p[lay.text, 1:] == 0).all()
    assert (p[lay.obs, 0] == L).all()
    frame = torch.tensor([[0, 0], [0, 16], [16, 0], [16, 16]], dtype=p.dtype)
    assert torch.equal(p[lay.obs, 1:], frame)
    assert torch.equal(p[lay.proprio], torch.tensor([[L, 0, 16]], dtype=p.dtype))
    vt = p[lay.video, 0].view(cfg.num_video_latents, cfg.frame_rows)[:, 0]
    assert torch.allclose(vt, torch.tensor([L, L + 5 / 3], dtype=p.dtype))
    assert torch.equal(p[lay.video, 1:].view(-1, cfg.frame_rows, 2)[1], frame)
    j = torch.arange(cfg.action_horizon, dtype=p.dtype)
    assert torch.allclose(p[lay.action, 0], L + j / cfg.actions_per_frame * 5 / 3)
    assert (p[lay.action, 1] == 0).all() and (p[lay.action, 2] == 16).all()


def test_base_mask_rules(cfg):
    lay = SequenceLayout(cfg)
    m = lay.base_mask()
    assert m.shape == (lay.N, lay.N) and m.dtype == torch.bool
    assert not m[:, lay.action].any() or m[lay.action, lay.action].all()
    assert not m[lay.text, lay.action].any()
    assert not m[lay.obs, lay.action].any()
    assert not m[lay.video, lay.action].any()
    assert m[lay.action].all()
    assert m[lay.video, lay.text].all() and m[lay.video, lay.obs].all() and m[lay.video, lay.video].all()
    assert not m[lay.text, lay.video].any() and not m[lay.obs, lay.video].any()
    assert m[lay.text, lay.text].all() and m[lay.text, lay.obs].all() and m[lay.text, lay.proprio].all()
    assert m[lay.obs, lay.text].all() and m[lay.obs, lay.obs].all()
    assert m.any(dim=1).all()


def test_full_mask_pads_text_keys(cfg):
    lay = SequenceLayout(cfg)
    valid = torch.ones(2, cfg.text_len, dtype=torch.bool)
    valid[1, 3:] = False
    m = lay.full_mask(valid)
    assert m.shape == (2, 1, lay.N, lay.N)
    assert torch.equal(m[0, 0], lay.base_mask())
    assert not m[1, 0, :, 3:cfg.text_len].any()
    assert m[1, 0, :, :3].all()
    assert m[1, 0].any(dim=1).all()


def test_action_query_mask(cfg):
    lay = SequenceLayout(cfg)
    valid = torch.ones(1, cfg.text_len, dtype=torch.bool)
    valid[0, 5:] = False
    q = lay.action_query_mask(valid)
    assert q.shape == (1, 1, cfg.action_horizon, lay.N)
    assert torch.equal(q[0, 0], lay.full_mask(valid)[0, 0, lay.action])
