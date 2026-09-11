import torch
from fasterwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

from wam_h3.model.dit import WAMH3DiT
from wam_h3.model.flow import FlowSchedule
from wam_h3.model.loss import training_loss


def test_training_weight_matches_fasterwam():
    ref, mine = WanContinuousFlowMatchScheduler(shift=5.0), FlowSchedule(shift=5.0)
    sigma = torch.tensor([0.0, 0.1, 0.5, 0.999, 1.0])
    assert torch.allclose(mine.training_weight(sigma), ref.training_weight(sigma * 1000).float(), atol=1e-6)


def test_inference_schedule_matches_fasterwam():
    ref, mine = WanContinuousFlowMatchScheduler(shift=5.0), FlowSchedule(shift=5.0)
    ts, ds = ref.build_inference_schedule(10, torch.device("cpu"), torch.float32)
    sig, dl = mine.inference_schedule(10)
    assert torch.allclose(sig, ts / 1000) and torch.allclose(dl, ds)
    assert sig[0] == 1.0 and (dl < 0).all()


def test_sample_sigma_in_unit_interval_and_shifted():
    s = FlowSchedule(shift=5.0).sample_sigma(4096, torch.device("cpu"))
    assert s.shape == (4096,) and (s > 0).all() and (s <= 1).all()
    assert s.mean() > 0.6


def test_euler_with_exact_velocity_recovers_sample():
    f = FlowSchedule(shift=5.0)
    x0 = torch.randn(3, 5)
    noise = torch.randn(3, 5)
    sig, dl = f.inference_schedule(50)
    x = f.add_noise(x0, noise, torch.ones(3))
    for s, d in zip(sig, dl):
        x = f.step(f.training_target(x0, noise), d, x)
    assert torch.allclose(x, x0, atol=1e-5)


def batch(cfg, B=2):
    F, K = cfg.frame_rows, cfg.num_video_latents
    return dict(
        text=torch.randn(B, cfg.text_len, cfg.text_dim),
        text_valid=torch.ones(B, cfg.text_len, dtype=torch.bool),
        obs_rows=torch.randn(B, F, cfg.video_patch_dim),
        proprio=torch.randn(B, cfg.proprio_dim),
        action=torch.randn(B, cfg.action_horizon, cfg.action_dim),
        video_rows=torch.randn(B, K * F, cfg.video_patch_dim),
        image_is_pad=torch.zeros(B, K, dtype=torch.bool),
        action_is_pad=torch.zeros(B, cfg.action_horizon, dtype=torch.bool),
    )


def test_loss_finite_and_split(cfg):
    m = WAMH3DiT(cfg)
    loss, parts = training_loss(m, batch(cfg), FlowSchedule(), FlowSchedule())
    assert torch.isfinite(loss) and loss.requires_grad
    assert set(parts) == {"loss_video", "loss_action", "loss_video_full_noise"}
    assert abs(loss.item() - parts["loss_video"] - parts["loss_action"]) < 1e-5


def test_latent0_and_padded_rows_excluded_from_loss(cfg):
    m = WAMH3DiT(cfg)
    b = batch(cfg)
    b["action_is_pad"][:, 4:] = True
    fv, fa = FlowSchedule(), FlowSchedule()
    loss, _ = training_loss(m, b, fv, fa, generator=torch.Generator().manual_seed(1))
    g = torch.Generator().manual_seed(1)
    sv, sa = fv.sample_sigma(2, "cpu", g), fa.sample_sigma(2, "cpu", g)
    nv = torch.randn(b["video_rows"].shape, generator=g)
    na = torch.randn(b["action"].shape, generator=g)
    tg = torch.stack([torch.ones(2), torch.full((2,), cfg.obs_t), 1 - sv, 1 - sa], 1)
    out = m(b["text"], b["text_valid"], b["obs_rows"], b["proprio"], fa.add_noise(b["action"], na, sa),
            fv.add_noise(b["video_rows"], nv, sv), t_groups=tg)
    ev = ((-out.video - (nv - b["video_rows"])) ** 2).mean(-1)[:, cfg.frame_rows:].mean(1)
    ea = ((-out.action - (na - b["action"])) ** 2).mean(-1)[:, :4].mean(1)
    expected = (ev * fv.training_weight(sv)).mean() + (ea * fa.training_weight(sa)).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_loss_uses_independent_video_and_action_sigmas(cfg, monkeypatch):
    m = WAMH3DiT(cfg)
    seen = {}
    captured = m.forward

    def spy(*a, **k):
        seen["t_groups"] = k["t_groups"]
        return captured(*a, **k)

    monkeypatch.setattr(m, "forward", spy)
    training_loss(m, batch(cfg), FlowSchedule(), FlowSchedule())
    tg = seen["t_groups"]
    assert (tg[:, 0] == 1).all() and torch.allclose(tg[:, 1], torch.full((2,), cfg.obs_t))
    assert not torch.equal(tg[:, 2], tg[:, 3])


def test_video_weight_peaks_at_full_noise():
    f = FlowSchedule(shift=5.0, weight_center=1.0, subtract_min=False)
    grid = f.sample_sigma(20000, "cpu", torch.Generator().manual_seed(0))
    w = f.training_weight(grid)
    assert (w > 0).all() and w.max() <= f.training_weight(torch.ones(1))[0] and f.training_weight(torch.ones(1))[0] > 1.0
    assert abs(w.mean() - 1.0) < 0.05


def test_video_sigma_hits_one_with_full_noise_prob(cfg, monkeypatch):
    m = WAMH3DiT(cfg)
    seen = {}
    captured = m.forward

    def spy(*a, **k):
        seen["t_groups"] = k["t_groups"]
        return captured(*a, **k)

    monkeypatch.setattr(m, "forward", spy)
    training_loss(m, batch(cfg, B=8), FlowSchedule(), FlowSchedule(), full_noise_prob=1.0)
    assert (seen["t_groups"][:, 2] == 0).all()
    training_loss(m, batch(cfg, B=8), FlowSchedule(), FlowSchedule(), full_noise_prob=0.0)
    assert (seen["t_groups"][:, 2] > 0).all()


def test_loss_reports_full_noise_video_bucket(cfg):
    m = WAMH3DiT(cfg)
    _, parts = training_loss(m, batch(cfg, B=4), FlowSchedule(), FlowSchedule(), full_noise_prob=1.0)
    assert abs(parts["loss_video_full_noise"] - parts["loss_video"]) < 1e-6
    _, parts = training_loss(m, batch(cfg, B=4), FlowSchedule(), FlowSchedule(), full_noise_prob=0.0)
    assert parts["loss_video_full_noise"] != parts["loss_video_full_noise"]
