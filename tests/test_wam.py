import torch

from wam_h3.data.text_cache import save_embedding
from wam_h3.model.config import WAMH3Config
from wam_h3.model.dit import WAMH3DiT
from wam_h3.model.lora import apply_lora
from wam_h3.model.vae import VideoEncoder
from wam_h3.model.wam import WAMH3

H, W = 64, 96


def make(tmp_path, r=None):
    cfg = WAMH3Config.tiny(latent_h=H // 16, latent_w=W // 16, action_dim=7, proprio_dim=8)
    dit = WAMH3DiT(cfg)
    torch.nn.init.normal_(dit.final_layer.action_out.weight, std=0.1)
    if r:
        apply_lora(dit, r)
    vae = VideoEncoder(ch=32, ch_mult=(1, 1, 1, 1, 1, 1), num_res_blocks=1)
    return WAMH3(cfg, dit, vae, text_cache_dir=tmp_path / "text")


def sample(cfg, B=2):
    return dict(
        video=torch.rand(B, 3, cfg.num_frames, H, W) * 2 - 1,
        action=torch.randn(B, cfg.action_horizon, cfg.action_dim),
        proprio=torch.randn(B, cfg.proprio_dim),
        context=torch.randn(B, cfg.text_len, cfg.text_dim),
        context_mask=torch.ones(B, cfg.text_len, dtype=torch.bool),
        image_is_pad=torch.zeros(B, cfg.num_frames, dtype=torch.bool),
        action_is_pad=torch.zeros(B, cfg.action_horizon, dtype=torch.bool),
    )


def test_build_inputs_shapes_and_latent_pad(tmp_path):
    m = make(tmp_path)
    cfg = m.cfg
    s = sample(cfg)
    s["image_is_pad"][1, 1:] = True
    inp = m.build_inputs(s)
    F = cfg.frame_rows
    assert inp["obs_rows"].shape == (2, F, cfg.video_patch_dim)
    assert inp["video_rows"].shape == (2, cfg.num_video_latents * F, cfg.video_patch_dim)
    assert inp["text"].shape == (2, cfg.text_len, cfg.text_dim) and inp["text_valid"].dtype == torch.bool
    assert inp["image_is_pad"].tolist() == [[False, False], [False, True]]
    assert torch.equal(inp["action_is_pad"], s["action_is_pad"])
    assert inp["proprio"].shape == (2, cfg.proprio_dim)


def test_training_loss_runs_and_backprops_only_trainable(tmp_path):
    m = make(tmp_path, r=4)
    loss, parts = m.training_loss(sample(m.cfg))
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in m.dit.named_parameters():
        assert (p.grad is not None) == p.requires_grad, n


def test_infer_uses_cached_prompt_and_returns_chunk(tmp_path):
    m = make(tmp_path)
    cfg = m.cfg
    save_embedding(tmp_path / "text", "put the bowl on the plate", torch.randn(5, cfg.text_dim))
    out = m.infer_action_one_pass_future_cache(
        prompt="put the bowl on the plate", input_image=torch.rand(1, 3, H, W) * 2 - 1,
        proprio=torch.randn(1, cfg.proprio_dim), action_horizon=cfg.action_horizon,
        num_inference_steps=3, seed=0, negative_prompt="", text_cfg_scale=1.0, sigma_shift=None,
        rand_device="cpu", tiled=False)
    a = out["action"]
    assert a.shape == (cfg.action_horizon, cfg.action_dim) and a.dtype == torch.float32 and a.device.type == "cpu"
    again = m.infer_action_one_pass_future_cache(
        prompt="put the bowl on the plate", input_image=torch.rand(1, 3, H, W) * 2 - 1,
        proprio=torch.randn(1, cfg.proprio_dim), action_horizon=cfg.action_horizon, num_inference_steps=3, seed=0)
    assert again["action"].shape == a.shape


def test_infer_matches_manual_prefill_denoise(tmp_path):
    m = make(tmp_path)
    cfg = m.cfg
    ctx, mask = torch.randn(1, cfg.text_len, cfg.text_dim), torch.ones(1, cfg.text_len, dtype=torch.bool)
    img, prop = torch.rand(1, 3, H, W) * 2 - 1, torch.randn(1, cfg.proprio_dim)
    out = m.infer_action_one_pass_future_cache(context=ctx, context_mask=mask, input_image=img, proprio=prop,
                                               action_horizon=cfg.action_horizon, num_inference_steps=2, seed=3)
    inp = m.build_inputs(dict(video=img[:, :, None].expand(-1, -1, cfg.num_frames, -1, -1), proprio=prop,
                              context=ctx, context_mask=mask))
    sig, _ = m.sched_a.inference_schedule(2)
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(inp["video_rows"].shape, generator=g)
    tg = torch.tensor([[1.0, cfg.obs_t, 1 - sig[0].item(), 0.0]])
    cache = m.dit.prefill(inp["text"], inp["text_valid"], inp["obs_rows"], inp["proprio"], noise, tg)
    exp = m.dit.denoise_actions(cache, m.sched_a, steps=2, generator=g)
    assert torch.allclose(out["action"], exp[0], atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    m = make(tmp_path, r=4)
    for p in m.dit.parameters():
        if p.requires_grad:
            p.data.normal_()
    m.save_checkpoint(tmp_path / "ckpt", step=7, extra={"lr": 1e-4})
    assert (tmp_path / "ckpt" / "adapter.safetensors").exists()
    m2 = make(tmp_path, r=4)
    m2.dit.load_state_dict({k: v for k, v in m.dit.state_dict().items() if "lora" not in k}, strict=False)
    m2.vae.load_state_dict(m.vae.state_dict())
    state = m2.load_checkpoint(tmp_path / "ckpt")
    assert state["step"] == 7 and state["lr"] == 1e-4
    s = sample(m.cfg)
    torch.manual_seed(0)
    a, _ = m.training_loss(s)
    torch.manual_seed(0)
    b, _ = m2.training_loss(s)
    assert torch.allclose(a, b)


def test_infer_bf16_base_with_fp32_heads(tmp_path):
    m = make(tmp_path, r=4)
    m.dit.to(torch.bfloat16)
    for p in m.dit.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    assert m.torch_dtype == torch.bfloat16
    out = m.infer_action_one_pass_future_cache(input_image=torch.rand(1, 3, H, W) * 2 - 1, proprio=torch.randn(1, m.cfg.proprio_dim),
                                               context=torch.randn(1, m.cfg.text_len, m.cfg.text_dim),
                                               context_mask=torch.ones(1, m.cfg.text_len, dtype=torch.bool), num_inference_steps=2, seed=0)
    assert out["action"].shape == (m.cfg.action_horizon, m.cfg.action_dim) and torch.isfinite(out["action"]).all()


def test_build_inputs_chunked_vae_matches_per_sample(tmp_path):
    m = make(tmp_path)
    s = sample(m.cfg, B=6)
    inp = m.build_inputs(s)
    one = m.build_inputs({k: v[5:6] for k, v in s.items()})
    assert torch.allclose(inp["video_rows"][5], one["video_rows"][0], atol=1e-5)
    assert torch.allclose(inp["obs_rows"][5], one["obs_rows"][0], atol=1e-5)
