import torch

from wam_h3.refined.layout import availability_times, causal_mask
from wam_h3.refined.policy import ActionPolicy, PolicyConfig, masked_loss


def test_availability_and_backbone_mask():
    times = availability_times(22, 2)
    assert (times <= 0).sum() == 4
    assert (times <= 34).sum() == 24
    assert (times <= 68).sum() == 44
    mask = causal_mask(2, 3, 2)
    assert mask.shape == (8, 8)
    assert not mask[:2, 2:].any()
    assert mask[2:4, :4].all()
    assert not mask[2:4, 4:].any()
    assert mask[-2:].all()


def test_future_and_decision_isolation():
    cfg = PolicyConfig(width=32, heads=4, ffn=64, memory_width=48, taps=(1, 3), horizon=4)
    model = ActionPolicy(cfg).eval()
    memory = [torch.randn(6, 48) for _ in cfg.taps]
    times = torch.tensor([0, 0, 34, 34, 68, 68])
    cutoffs = torch.tensor([0, 34, 68])
    actual = model(memory, times, cutoffs)
    independent = torch.cat([model(memory, times, c[None]) for c in cutoffs])
    torch.testing.assert_close(actual, independent, atol=1e-6, rtol=1e-5)
    changed = [m.clone() for m in memory]
    for m in changed:
        m[2:] += 100
    torch.testing.assert_close(model(changed, times, cutoffs)[0], actual[0])
    assert actual.shape == (3, 4, 7)
    assert [b.cross for b in model.blocks] == [True, False, True, False]
    loss = masked_loss(actual, torch.zeros_like(actual), torch.ones(3, 4, dtype=torch.bool))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.blocks[0].memory.weight.grad.abs().sum() > 0


def test_invalid_targets_do_not_affect_loss():
    prediction = torch.zeros(1, 4, 7)
    target = torch.zeros_like(prediction)
    target[:, 2:] = float("nan")
    loss = masked_loss(prediction, target, torch.tensor([[True, True, False, False]]))
    assert loss == 0


def test_chunk_loss_fp32_partial_nan_and_empty():
    import pytest
    from wam_h3.refined.policy import chunk_masked_loss
    prediction = torch.tensor([[[1., 3.], [float("nan"), float("nan")]],
                               [[2., 4.], [4., 6.]]], dtype=torch.bfloat16, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, 1] = float("nan")
    valid = torch.tensor([[True, False], [True, True]])
    losses = chunk_masked_loss(prediction, target, valid)
    assert losses.dtype == torch.float32
    torch.testing.assert_close(losses, torch.tensor([5., 18.]))
    losses.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 1].eq(0).all()
    with pytest.raises(ValueError, match="Every chunk"):
        chunk_masked_loss(prediction, target, torch.tensor([[False, False], [True, True]]))
