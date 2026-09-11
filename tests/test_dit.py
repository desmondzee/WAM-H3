import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.reference import minimax_h3_dit as ref
from wam_h3.model.checkpoint import load_pretrained
from wam_h3.model.config import WAMH3Config
from wam_h3.model.dit import WAMH3DiT, patchify_video, unpatchify_video

ROOT = Path(__file__).resolve().parents[1]


def MATH():
    return sdpa_kernel(SDPBackend.MATH)


def ref_dit(cfg):
    return ref.MiniMaxH3DiT(
        num_layers=cfg.num_layers, token_refiner_num_layers=cfg.token_refiner_num_layers, hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads, attention_head_dim=cfg.attention_head_dim,
        ffn_hidden_size=cfg.ffn_hidden_size, latents_dim=cfg.latents_dim, audio_latents_dim=cfg.action_dim,
        text_dim=cfg.text_dim, timestep_input_dim=cfg.timestep_input_dim,
        time_embed_hidden_size=cfg.time_embed_hidden_size, time_embed_dim=cfg.time_embed_dim,
        adaln_out_features=cfg.adaln_out_features, final_adaln_out_features=cfg.final_adaln_out_features,
        rope_inv_freq_len=cfg.rope_inv_freq_len)


def inputs(cfg, B=2):
    lay_F, K = cfg.frame_rows, cfg.num_video_latents
    return dict(
        text=torch.randn(B, cfg.text_len, cfg.text_dim),
        text_valid=torch.ones(B, cfg.text_len, dtype=torch.bool),
        obs_rows=torch.randn(B, lay_F, cfg.video_patch_dim),
        proprio=torch.randn(B, cfg.proprio_dim) if cfg.proprio_dim else None,
        action=torch.randn(B, cfg.action_horizon, cfg.action_dim),
        video_rows=torch.randn(B, K * lay_F, cfg.video_patch_dim),
    )


def test_patchify_matches_reference_and_roundtrips(cfg):
    lat = torch.randn(2, cfg.latents_dim, cfg.num_video_latents, cfg.latent_h, cfg.latent_w)
    rows = patchify_video(lat)
    assert rows.shape == (2, cfg.num_video_latents * cfg.frame_rows, cfg.video_patch_dim)
    assert torch.equal(rows[1], ref.patchify_video(lat[1:2]))
    assert torch.equal(unpatchify_video(rows, cfg.num_video_latents, cfg.latent_h, cfg.latent_w), lat)


def test_joint_forward_matches_reference():
    cfg = WAMH3Config.tiny(proprio_dim=0)
    r, m = ref_dit(cfg), WAMH3DiT(cfg)
    sd = r.state_dict()
    for a, b in [("audio_patch_proj", "action_in"), ("final_layer.audio_out", "final_layer.action_out")]:
        for s in ("weight", "bias"):
            sd[f"{b}.{s}"] = sd.pop(f"{a}.{s}")
    m.load_state_dict(sd)
    inp = inputs(cfg, B=1)
    lay = m.layout
    t = 0.37
    x = torch.zeros(1, lay.N, cfg.video_patch_dim)
    x[:, lay.obs], x[:, lay.video] = inp["obs_rows"], inp["video_rows"]
    ax = torch.zeros(1, lay.N, cfg.action_dim)
    ax[:, lay.action] = inp["action"]
    img_pos = torch.cat([torch.arange(lay.obs.start, lay.obs.stop), torch.arange(lay.video.start, lay.video.stop)])
    with MATH():
        exp_v, exp_a = r(
            x=x, audio_x=ax, img_position_ids=lay.position_ids[None], unique_timesteps=torch.tensor([t]),
            inverse_indices=torch.zeros(lay.N, dtype=torch.long), update_mask=None, token_tags=lay.row_tag,
            prompt_embeds=inp["text"][0], img_pos_info={"position_ids": img_pos},
            audio_pos_info={"position_ids": torch.arange(lay.action.start, lay.action.stop)},
            text_pos_info={"position_ids": torch.arange(cfg.text_len)},
            img_pos_for_infer_output_info={"position_ids": img_pos},
            packed_seq_params={"cu_seqlens_q": torch.tensor([0, lay.N, lay.N], dtype=torch.int32), "max_seqlen_q": lay.N},
            refiner_packed_seq_params={"cu_seqlens_q": torch.tensor([0, cfg.text_len, cfg.text_len], dtype=torch.int32),
                                       "max_seqlen_q": cfg.text_len},
            skip_mask_out_condition=True)
        out = m(**inp, t_groups=torch.full((1, 4), t), mask=torch.ones(1, 1, lay.N, lay.N, dtype=torch.bool))
    assert torch.allclose(out.video[0], exp_v[cfg.frame_rows:], atol=1e-5)
    assert torch.allclose(out.action[0], exp_a, atol=1e-5)


def test_load_pretrained_reports_head_swap():
    cfg = WAMH3Config.tiny(num_layers=50, token_refiner_num_layers=2)
    keys = set(json.loads((ROOT / "FL2VA/transformer/model.safetensors.index.json").read_text())["weight_map"])
    sd = ref_dit(cfg).state_dict()
    assert set(sd) == keys
    m = WAMH3DiT(cfg)
    before = m.blocks[3].attn.qkv_proj.weight.clone()
    report = load_pretrained(m, state_dict=sd)
    assert report.missing == {"action_in.weight", "action_in.bias", "final_layer.action_out.weight",
                              "final_layer.action_out.bias", "proprio_in.weight", "proprio_in.bias"}
    assert report.unexpected == {"audio_patch_proj.weight", "audio_patch_proj.bias",
                                 "final_layer.audio_out.weight", "final_layer.audio_out.bias"}
    assert torch.equal(m.blocks[3].attn.qkv_proj.weight, sd["blocks.3.attn.qkv_proj.weight"])
    assert not torch.equal(before, m.blocks[3].attn.qkv_proj.weight)


def test_actions_do_not_leak_into_context(cfg):
    m = WAMH3DiT(cfg)
    inp = inputs(cfg)
    tg = torch.rand(2, 4)
    inp2 = dict(inp, action=inp["action"] + torch.randn_like(inp["action"]))
    tg2 = tg.clone()
    tg2[:, 3] = torch.rand(2)
    lay = m.layout
    with MATH():
        a, b = m(**inp, t_groups=tg), m(**inp2, t_groups=tg2)
    assert torch.equal(a.hidden[:, :lay.action.start], b.hidden[:, :lay.action.start])
    assert torch.equal(a.hidden[:, lay.video], b.hidden[:, lay.video])
    assert torch.equal(a.video, b.video)
    assert not torch.equal(a.hidden[:, lay.action], b.hidden[:, lay.action])


def test_action_modulation_independent_of_video_time(cfg):
    m = WAMH3DiT(cfg)
    lay = m.layout
    tg = torch.rand(2, 4)
    tg2 = tg.clone()
    tg2[:, 2] = torch.rand(2)
    mods = [m.row_mods(m.blocks[0], t) for t in (tg, tg2)]
    for i in range(6):
        assert torch.equal(mods[0][i][:, lay.action], mods[1][i][:, lay.action])
        assert torch.equal(mods[0][i][:, lay.text], mods[1][i][:, lay.text])
        assert not torch.equal(mods[0][i][:, lay.video], mods[1][i][:, lay.video])
