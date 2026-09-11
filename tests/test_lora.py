import torch

from wam_h3.model.checkpoint import load_trainable, save_trainable, trainable_state_dict
from wam_h3.model.dit import WAMH3DiT
from wam_h3.model.lora import HEADS, apply_lora
from tests.test_dit import inputs


def test_lora_targets_and_trainable_set(cfg):
    m = apply_lora(WAMH3DiT(cfg), r=4)
    n_lora = sum(1 for mod in m.modules() if hasattr(mod, "lora_A"))
    assert n_lora == 5 * cfg.num_layers + 4 * cfg.token_refiner_num_layers
    assert not hasattr(m.final_layer.adaln_proj.linear, "lora_A")
    assert not hasattr(m.final_layer.adaln_proj.linear, "lora_A")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable
    for n in trainable:
        assert ".lora_" in n or n.startswith(HEADS), n
    assert {n for n in trainable if n.startswith(HEADS)} >= {"action_in.weight", "proprio_in.weight",
                                                            "final_layer.action_out.weight"}
    assert not m.final_layer.adaln_proj.linear.weight.requires_grad
    assert not m.blocks[0].attn.qkv_proj.base_layer.weight.requires_grad


def test_lora_init_preserves_forward(cfg):
    m = WAMH3DiT(cfg)
    inp = inputs(cfg)
    tg = torch.rand(2, 4)
    before = m(**inp, t_groups=tg)
    apply_lora(m, r=4)
    after = m(**inp, t_groups=tg)
    assert torch.allclose(before.hidden, after.hidden, atol=1e-6)


def test_trainable_checkpoint_roundtrip(cfg, tmp_path):
    base = WAMH3DiT(cfg).state_dict()

    def fresh():
        d = WAMH3DiT(cfg)
        d.load_state_dict(base)
        return apply_lora(d, r=4)

    m = fresh()
    for p in m.parameters():
        if p.requires_grad:
            p.data.normal_()
    sd = trainable_state_dict(m)
    assert set(sd) == {n for n, p in m.named_parameters() if p.requires_grad}
    save_trainable(m, tmp_path / "adapter.safetensors")
    m2 = fresh()
    inp, tg = inputs(cfg), torch.rand(2, 4)
    assert not torch.allclose(m(**inp, t_groups=tg).hidden, m2(**inp, t_groups=tg).hidden)
    load_trainable(m2, tmp_path / "adapter.safetensors")
    assert torch.allclose(m(**inp, t_groups=tg).hidden, m2(**inp, t_groups=tg).hidden)
