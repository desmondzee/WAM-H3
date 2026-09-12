import math

import pytest
import torch

from wam_h3.refined.metrics import LossAccumulator
from wam_h3.refined.policy import chunk_masked_loss


def handcrafted():
    components = torch.arange(1, 8).float()
    episodes = []
    for amplitudes, mask in (
        ([[1, 2, 0, 0], [3, 0, 0, 0]], [[True, True, False, False], [True, False, False, False]]),
        ([[4, 0, 6, 0]], [[True, False, True, False]]),
    ):
        prediction = torch.tensor(amplitudes).float().unsqueeze(-1) * components
        target = torch.zeros_like(prediction)
        valid = torch.tensor(mask)
        prediction[~valid] = float("nan")
        target[~valid] = float("nan")
        episodes.append((prediction, target, valid))
    return episodes


@pytest.mark.parametrize("prefix", ["train", "validation"])
def test_unequal_chunks_masks_nan_counts_components_and_raw_scale(prefix):
    scale = torch.arange(1, 8).float()
    accumulator = LossAccumulator(scale)
    episodes = handcrafted()
    for episode in episodes:
        accumulator.update(*episode)
    metrics = accumulator.finish(prefix)
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics[f"{prefix}/count_by_chunk/action_chunk_0"] == 2
    assert metrics[f"{prefix}/count_by_chunk/action_chunk_1"] == 1
    assert f"{prefix}/loss_by_chunk/action_chunk_2" not in metrics
    assert metrics[f"{prefix}/loss_by_chunk/action_chunk_0"] == pytest.approx(285)
    assert metrics[f"{prefix}/loss_by_chunk/action_chunk_1"] == pytest.approx(180)
    for horizon, count, loss in ((0, 3, 520 / 3), (1, 1, 80), (2, 1, 720), (3, 0, None)):
        assert metrics[f"{prefix}/count_by_horizon/action_unit_{horizon}"] == count
        if count:
            assert metrics[f"{prefix}/loss_by_horizon/action_unit_{horizon}"] == pytest.approx(loss)
        else:
            assert f"{prefix}/loss_by_horizon/action_unit_{horizon}" not in metrics
    expected_components = [12.5 * i ** 2 for i in range(1, 8)]
    expected_raw = [value * (i + 1) ** 2 for i, value in enumerate(expected_components)]
    for root, expected in ((prefix, expected_components), (f"{prefix}/raw_action_mse", expected_raw)):
        for i, value in enumerate(expected):
            assert metrics[f"{root}/loss_by_component/component_{i}"] == pytest.approx(value)
        for group, indices in (("translation", range(3)), ("rotation", range(3, 6)), ("gripper", [6])):
            assert metrics[f"{root}/loss_groups/{group}"] == pytest.approx(sum(expected[i] for i in indices) / len(indices))
    objective = torch.cat([chunk_masked_loss(*episode) for episode in episodes]).mean().item()
    assert sum(metrics[f"{prefix}/loss_by_component/component_{i}"] for i in range(7)) / 7 == pytest.approx(objective)
    assert metrics[f"{prefix}/component_chunk_count"] == 3


@pytest.mark.parametrize("all_valid", [False, True])
def test_partition_invariance_dynamic_positions_and_horizon_33(all_valid):
    generator = torch.Generator().manual_seed(7)
    episodes = []
    for chunks in (2, 16, 5):
        prediction = torch.randn(chunks, 34, 7, generator=generator)
        target = torch.randn(chunks, 34, 7, generator=generator)
        valid = torch.ones(chunks, 34, dtype=torch.bool)
        if not all_valid:
            valid[:, 2::3] = False
            target[~valid] = float("nan")
        episodes.append((prediction, target, valid))
    whole, partitioned = LossAccumulator(torch.ones(7)), LossAccumulator(torch.ones(7))
    for episode in episodes:
        whole.update(*episode)
    for episode in reversed(episodes):
        for start in range(0, len(episode[0]), 2):
            partitioned.update(*(tensor[start:start + 2] for tensor in episode), chunk_start=start)
    metrics = whole.finish("train")
    assert partitioned.finish("train") == pytest.approx(metrics)
    assert metrics["train/count_by_chunk/action_chunk_15"] == 1
    assert metrics["train/count_by_chunk/action_chunk_2"] == 2
    assert metrics["train/count_by_chunk/action_chunk_0"] == 3
    assert metrics["train/count_by_horizon/action_unit_33"] == 23
    objective = torch.cat([chunk_masked_loss(*episode) for episode in episodes]).mean().item()
    assert sum(metrics[f"train/loss_by_component/component_{i}"] for i in range(7)) / 7 == pytest.approx(objective)
    if all_valid:
        assert sum(metrics[f"train/loss_by_horizon/action_unit_{i}"] for i in range(34)) / 34 == pytest.approx(objective)


def test_empty_padding_absent_positions_and_fresh_accumulator():
    accumulator = LossAccumulator(torch.ones(7))
    prediction = torch.full((3, 4, 7), float("nan"))
    valid = torch.zeros(3, 4, dtype=torch.bool)
    prediction[1, 0] = 2
    valid[1, 0] = True
    accumulator.update(prediction, torch.zeros_like(prediction), valid)
    metrics = accumulator.finish("train")
    assert metrics["train/component_chunk_count"] == 1
    assert metrics["train/count_by_chunk/action_chunk_0"] == 0
    assert "train/loss_by_chunk/action_chunk_0" not in metrics
    assert metrics["train/loss_by_chunk/action_chunk_1"] == 4
    assert metrics["train/count_by_chunk/action_chunk_2"] == 0
    fresh = LossAccumulator(torch.ones(7)).finish("train")
    assert fresh == {"train/component_chunk_count": 0}


def test_validation_aggregates_all_fifty_episodes():
    episodes = handcrafted() * 25
    accumulator = LossAccumulator(torch.ones(7))
    for episode in episodes:
        accumulator.update(*episode)
    metrics = accumulator.finish("validation")
    assert metrics["validation/count_by_chunk/action_chunk_0"] == 50
    assert metrics["validation/count_by_chunk/action_chunk_1"] == 25
    assert metrics["validation/count_by_horizon/action_unit_0"] == 75
    assert metrics["validation/component_chunk_count"] == 75
    objective = torch.cat([chunk_masked_loss(*episode) for episode in episodes]).mean().item()
    assert sum(metrics[f"validation/loss_by_component/component_{i}"] for i in range(7)) / 7 == pytest.approx(objective)


def test_fp32_detached_state_and_unchanged_model_gradients():
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 7)
    reference = torch.nn.Linear(3, 7)
    reference.load_state_dict(model.state_dict())
    scale = torch.arange(1, 8).float().requires_grad_()
    accumulator = LossAccumulator(scale)
    for chunks in (2, 3):
        inputs = torch.randn(chunks, 4, 3)
        target = torch.randn(chunks, 4, 7)
        valid = torch.ones(chunks, 4, dtype=torch.bool)
        valid[-1, 1:] = False
        target[~valid] = float("nan")
        prediction = model(inputs).to(torch.bfloat16)
        losses = chunk_masked_loss(prediction, target, valid)
        accumulator.update(prediction, target, valid)
        losses.sum().backward()
        chunk_masked_loss(reference(inputs).to(torch.bfloat16), target, valid).sum().backward()
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
    for value in vars(accumulator).values():
        assert value.dtype == torch.float32
        assert not value.requires_grad
        assert value.grad_fn is None
    assert scale.grad is None
    assert all(isinstance(value, float) for value in accumulator.finish("train").values())
    large = torch.full((1, 1, 7), 300, dtype=torch.float16, requires_grad=True)
    fp32 = LossAccumulator(torch.ones(7))
    fp32.update(large, torch.zeros_like(large), torch.ones(1, 1, dtype=torch.bool))
    assert fp32.finish("train")["train/loss_by_component/component_0"] == 90000
    assert fp32.finish("train")["train/mae_by_component/component_0"] == 300


@pytest.mark.parametrize("prefix", ["train", "validation"])
def test_mae_signed_residuals_variable_lengths_masks_and_absolute_raw_scale(prefix):
    components = torch.arange(1, 8).float()
    scale = torch.tensor([-2, 3, -4, 0.5, -0.25, 0, 7]).requires_grad_()
    accumulator = LossAccumulator(scale)
    partitioned = LossAccumulator(scale)
    expected_chunks = []
    for amplitudes, mask in (
        ([[-1, 3, 0, 0], [-6, 0, 0, 0], [0, 0, 0, 0]],
         [[True, True, False, False], [True, False, False, False], [False] * 4]),
        ([[2, 0, -4]], [[True, False, True]]),
    ):
        residual = torch.tensor(amplitudes).float().unsqueeze(-1) * components
        target = torch.full_like(residual, 5)
        prediction = target + residual
        valid = torch.tensor(mask)
        prediction[~valid] = float("nan")
        target[~valid] = float("nan")
        prediction.requires_grad_()
        target.requires_grad_()
        accumulator.update(prediction, target, valid)
        for start in range(len(prediction)):
            partitioned.update(prediction[start:start + 1], target[start:start + 1],
                               valid[start:start + 1], chunk_start=start)
            if valid[start].any():
                expected_chunks.append(residual[start, valid[start]].abs().mean().item())
        assert prediction.grad is None
        assert target.grad is None
    metrics = accumulator.finish(prefix)
    assert partitioned.finish(prefix) == pytest.approx(metrics)
    assert all(key.startswith(f"{prefix}/") for key in metrics)
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics[f"{prefix}/component_chunk_count"] == 3
    normalized = components * (11 / 3)
    raw = normalized * scale.detach().abs()
    for root, expected in ((prefix, normalized), (f"{prefix}/raw_action_mae", raw)):
        for i, value in enumerate(expected.tolist()):
            assert metrics[f"{root}/mae_by_component/component_{i}"] == pytest.approx(value)
        for group, indices in (("translation", slice(0, 3)), ("rotation", slice(3, 6)), ("gripper", slice(6, 7))):
            assert metrics[f"{root}/mae_groups/{group}"] == pytest.approx(expected[indices].mean().item())
        assert metrics[f"{root}/mae"] == pytest.approx(expected.mean().item())
    assert metrics[f"{prefix}/mae"] == pytest.approx(sum(expected_chunks) / len(expected_chunks))
    assert metrics[f"{prefix}/mae_by_component/component_0"] != pytest.approx(
        math.sqrt(metrics[f"{prefix}/loss_by_component/component_0"]))
    assert scale.grad is None
    for value in vars(accumulator).values():
        assert value.dtype == torch.float32
        assert not value.requires_grad
        assert value.grad_fn is None


@pytest.mark.parametrize("prefix", ["train", "validation"])
def test_mae_all_empty_chunks_omit_values(prefix):
    accumulator = LossAccumulator(torch.ones(7))
    prediction = torch.full((2, 3, 7), float("nan"))
    accumulator.update(prediction, prediction, torch.zeros(2, 3, dtype=torch.bool))
    metrics = accumulator.finish(prefix)
    assert metrics[f"{prefix}/component_chunk_count"] == 0
    assert all("/mae" not in key and "/loss_" not in key for key in metrics)
    assert all(value == 0 for value in metrics.values())
