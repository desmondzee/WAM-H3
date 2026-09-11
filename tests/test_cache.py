import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from wam_h3.model.dit import WAMH3DiT
from wam_h3.model.flow import FlowSchedule


def ctx(cfg, B=2):
    F, K = cfg.frame_rows, cfg.num_video_latents
    return dict(
        text=torch.randn(B, cfg.text_len, cfg.text_dim),
        text_valid=torch.rand(B, cfg.text_len) < 0.7,
        obs_rows=torch.randn(B, F, cfg.video_patch_dim),
        proprio=torch.randn(B, cfg.proprio_dim),
        video_rows=torch.randn(B, K * F, cfg.video_patch_dim),
    )


def test_cached_action_step_equals_joint_forward(cfg):
    m = WAMH3DiT(cfg)
    torch.nn.init.normal_(m.final_layer.action_out.weight)
    c = ctx(cfg)
    c["text_valid"][:, 0] = True
    tg = torch.rand(2, 4)
    a = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    with sdpa_kernel(SDPBackend.MATH):
        cache = m.prefill(**c, t_groups=tg)
        assert len(cache.kv) == cfg.num_layers
        assert cache.kv[0][0].shape == (2, cfg.num_attention_heads, m.layout.ctx_len, cfg.attention_head_dim)
        step = m.denoise_action_step(cache, a, tg[:, 3])
        joint = m(**c, action=a, t_groups=tg).action
    assert torch.allclose(step, joint, atol=1e-6)


def test_cached_step_ignores_prefill_action_time(cfg):
    m = WAMH3DiT(cfg)
    torch.nn.init.normal_(m.final_layer.action_out.weight)
    c = ctx(cfg)
    c["text_valid"][:, 0] = True
    tg = torch.rand(2, 4)
    tg2 = tg.clone()
    tg2[:, 3] = 0.123
    a = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    with sdpa_kernel(SDPBackend.MATH):
        s1 = m.denoise_action_step(m.prefill(**c, t_groups=tg), a, tg[:, 3])
        s2 = m.denoise_action_step(m.prefill(**c, t_groups=tg2), a, tg[:, 3])
    assert torch.equal(s1, s2)


def test_denoise_actions_matches_manual_joint_euler(cfg):
    m = WAMH3DiT(cfg)
    torch.nn.init.normal_(m.final_layer.action_out.weight)
    c = ctx(cfg)
    c["text_valid"][:, 0] = True
    f = FlowSchedule(shift=5.0)
    sig, dl = f.inference_schedule(3)
    tg = torch.stack([torch.ones(2), torch.full((2,), cfg.obs_t), torch.full((2,), 1 - sig[0].item()), torch.zeros(2)], 1)
    with sdpa_kernel(SDPBackend.MATH):
        out = m.denoise_actions(m.prefill(**c, t_groups=tg), f, steps=3, generator=torch.Generator().manual_seed(7))
        x = torch.randn(2, cfg.action_horizon, cfg.action_dim, generator=torch.Generator().manual_seed(7))
        for s, d in zip(sig, dl):
            tg[:, 3] = 1 - s
            x = f.step(-m(**c, action=x, t_groups=tg).action, d, x)
    assert torch.allclose(out, x, atol=1e-5)
