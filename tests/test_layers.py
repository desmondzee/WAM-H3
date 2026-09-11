import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.reference import minimax_h3_dit as ref
from wam_h3.model import layers as L

def MATH():
    return sdpa_kernel(SDPBackend.MATH)


def ref_attention(cfg):
    return ref.MiniMaxH3Attention(cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.qk_norm_eps)


def ref_block(cfg):
    return ref.MiniMaxH3DiTBlock(cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.ffn_hidden_size,
                                 cfg.time_embed_dim, cfg.adaln_out_features, cfg.norm_eps, cfg.qk_norm_eps)


def ref_refiner(cfg):
    return ref.MiniMaxH3TokenRefiner(cfg.token_refiner_num_layers, cfg.hidden_size, cfg.num_attention_heads,
                                     cfg.attention_head_dim, cfg.ffn_hidden_size, cfg.norm_eps, cfg.qk_norm_eps, cfg.final_norm_eps)


def test_apply_rope_matches_reference(cfg):
    B, S, H, D = 2, 5, cfg.num_attention_heads, cfg.attention_head_dim
    x = torch.randn(B, S, H, D)
    freqs = torch.randn(S, 2 * 3 * cfg.rope_inv_freq_len)
    out = L.apply_rope(x.transpose(1, 2), freqs)
    for b in range(B):
        assert torch.allclose(out[b].transpose(0, 1), ref._apply_rope(x[b], freqs), atol=1e-6)


def test_attention_matches_reference_per_sample(cfg):
    B, S = 2, 6
    r, m = ref_attention(cfg), L.Attention(cfg)
    m.load_state_dict(r.state_dict())
    x = torch.randn(B, S, cfg.hidden_size)
    freqs = torch.randn(S, 2 * 3 * cfg.rope_inv_freq_len)
    mask = torch.ones(B, 1, S, S, dtype=torch.bool)
    with MATH():
        out = m(x, freqs, mask)
        for b in range(B):
            exp = r(x[b], rope_freqs=freqs, cu_seqlens=torch.tensor([0, S], dtype=torch.int32))
            assert torch.allclose(out[b], exp, atol=1e-5)


def test_attention_mask_blocks_keys(cfg):
    B, S = 1, 6
    m = L.Attention(cfg)
    x = torch.randn(B, S, cfg.hidden_size)
    mask = torch.ones(B, 1, S, S, dtype=torch.bool)
    mask[:, :, :3, 3:] = False
    with MATH():
        a = m(x, None, mask)
        x2 = x.clone()
        x2[:, 3:] += 1.0
        b = m(x2, None, mask)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)
    assert not torch.allclose(a[:, 3:], b[:, 3:])


def test_block_matches_reference_uniform_timestep(cfg):
    B, S, G = 2, 6, 4
    r, m = ref_block(cfg), L.DiTBlock(cfg)
    m.load_state_dict(r.state_dict())
    assert set(m.state_dict()) == set(r.state_dict())
    x = torch.randn(B, S, cfg.hidden_size)
    freqs = torch.randn(S, 2 * 3 * cfg.rope_inv_freq_len)
    t_emb = torch.randn(1, cfg.time_embed_dim)
    tags = torch.tensor([1, 1, 0, 0, 2, 0])
    row_idx = torch.cat([(b * G + 0) * 3 + tags for b in range(B)])
    mask = torch.ones(B, 1, S, S, dtype=torch.bool)
    with MATH():
        out, _ = m(x, t_emb.repeat(B * G, 1), row_idx, freqs, mask)
        for b in range(B):
            exp = r(x[b], t_emb=t_emb, combined_indices=tags, rope_freqs=freqs,
                    cu_seqlens=torch.tensor([0, S], dtype=torch.int32), max_seqlen=S)
            assert torch.allclose(out[b], exp, atol=1e-5)


def test_block_with_kv_ctx_equals_joint(cfg):
    B, S, G = 1, 7, 4
    m = L.DiTBlock(cfg)
    x = torch.randn(B, S, cfg.hidden_size)
    freqs = torch.randn(S, 2 * 3 * cfg.rope_inv_freq_len)
    t_emb = torch.randn(B * G, cfg.time_embed_dim)
    row_idx = torch.zeros(B * S, dtype=torch.long)
    ins, A = 3, 2
    sel = torch.ones(S, dtype=torch.bool)
    sel[ins:ins + A] = False
    mask = torch.ones(B, 1, S, S, dtype=torch.bool)
    mask[:, :, sel, ins:ins + A] = False
    with MATH():
        joint, _ = m(x, t_emb, row_idx, freqs, mask)
        _, (k, v) = m(x[:, sel], t_emb, row_idx[sel], freqs[sel], mask[:, :, sel][:, :, :, sel])
        part, _ = m(x[:, ins:ins + A], t_emb, row_idx[ins:ins + A], freqs[ins:ins + A],
                    mask[:, :, ins:ins + A], kv_ctx=(k, v), ctx_insert=ins)
    assert torch.allclose(part, joint[:, ins:ins + A], atol=1e-6)


def test_refiner_matches_reference_and_masks_pad(cfg):
    B, Lt = 2, 6
    r, m = ref_refiner(cfg), L.TokenRefiner(cfg)
    m.load_state_dict(r.state_dict())
    assert set(m.state_dict()) == set(r.state_dict())
    x = torch.randn(B, Lt, cfg.hidden_size)
    valid = torch.ones(B, Lt, dtype=torch.bool)
    valid[1, 4:] = False
    with MATH():
        out = m(x, valid)
        exp0 = r(x[0], cu_seqlens=torch.tensor([0, Lt], dtype=torch.int32), max_seqlen=Lt)
        exp1 = r(x[1, :4], cu_seqlens=torch.tensor([0, 4], dtype=torch.int32), max_seqlen=4)
    assert torch.allclose(out[0], exp0, atol=1e-5)
    assert torch.allclose(out[1, :4], exp1, atol=1e-5)


def test_final_layer_keys_and_zero_action_head(cfg):
    f = L.FinalLayer(cfg)
    keys = set(f.state_dict())
    assert {"norm.weight", "adaln_proj.linear.weight", "adaln_proj.linear.bias", "video_out.weight", "video_out.bias",
            "action_out.weight", "action_out.bias"} == keys
    assert (f.action_out.weight == 0).all() and (f.action_out.bias == 0).all()
    assert f.video_out.out_features == cfg.video_patch_dim and f.action_out.out_features == cfg.action_dim
