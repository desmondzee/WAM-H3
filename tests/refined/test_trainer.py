import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from scripts import train_refined
from wam_h3.refined.policy import ActionPolicy, PolicyConfig, masked_loss
from wam_h3.refined.trainer import load_checkpoint, save_checkpoint


def test_checkpoint_exact_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    cfg = PolicyConfig(width=16, heads=2, ffn=32, memory_width=24, taps=(1,), horizon=3)
    model = ActionPolicy(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    memory = [torch.randn(4, 24) * 10000]
    times, cutoffs = torch.tensor([0, 0, 34, 34]), torch.tensor([0, 34])
    target, valid = torch.randn(2, 3, 7), torch.ones(2, 3, dtype=torch.bool)
    def step(m, opt):
        opt.zero_grad(set_to_none=True)
        loss = masked_loss(m(memory, times, cutoffs), target, valid)
        loss.backward()
        opt.step()
        return loss.detach()
    step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, 1, {"center": torch.zeros(7), "scale": torch.ones(7)}, ["source"])
    expected_random = (random.random(), torch.rand(1))
    expected_loss = step(model, optimizer)
    restored = ActionPolicy(cfg)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=5e-2)
    state = load_checkpoint(tmp_path, restored, restored_opt, ["source"])
    assert state["step"] == 1
    assert restored_opt.param_groups[0]["lr"] == 1e-3
    assert random.random() == expected_random[0]
    torch.testing.assert_close(torch.rand(1), expected_random[1], atol=0, rtol=0)
    torch.testing.assert_close(step(restored, restored_opt), expected_loss, atol=0, rtol=0)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    with pytest.raises(ValueError, match="identities"):
        load_checkpoint(tmp_path, restored, restored_opt, ["different"])


@pytest.fixture
def wandb_entrypoint(tmp_path, monkeypatch):
    cfg = OmegaConf.load("configs/refined_train.yaml")
    cfg.output_dir = str(tmp_path)
    run = SimpleNamespace(
        id="test-id", project="wam-h3", entity="test-entity",
        url="https://wandb.ai/test-entity/wam-h3/runs/test-id",
        settings=SimpleNamespace(mode="online"), log=Mock(), finish=Mock(),
    )
    wandb = SimpleNamespace(init=Mock(return_value=run), Settings=Mock())
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setattr(train_refined, "configure", Mock())
    monkeypatch.setattr(train_refined, "train", Mock())
    return cfg, wandb, run


@pytest.mark.parametrize("missing", [False, True])
def test_wandb_disabled_does_not_init(wandb_entrypoint, missing):
    cfg, wandb, run = wandb_entrypoint
    assert cfg.wandb.enabled is False
    if missing:
        del cfg.wandb
    train_refined.main.__wrapped__(cfg)
    wandb.init.assert_not_called()
    wandb.Settings.assert_not_called()
    train_refined.train.assert_called_once_with(cfg)
    run.finish.assert_not_called()


def test_wandb_online_logs_metrics_and_identity(wandb_entrypoint, tmp_path):
    cfg, wandb, run = wandb_entrypoint
    cfg.wandb.enabled = True
    metrics = dict(step=11, loss=0.5, grad_norm=0.2, elapsed=3.0, source_peak=123, target_peak=456)
    train_refined.train.side_effect = lambda cfg, log_metrics: log_metrics(metrics, 11)
    train_refined.main.__wrapped__(cfg)
    wandb.init.assert_called_once_with(
        project="wam-h3", name="refined-training-smoke", mode="online",
        config=OmegaConf.to_container(cfg, resolve=True), dir=str(tmp_path),
        settings=wandb.Settings.return_value,
    )
    wandb.Settings.assert_called_once_with(disable_code=True, disable_git=True, console="off")
    run.log.assert_called_once_with(metrics, step=11)
    run.finish.assert_called_once_with(exit_code=0)
    assert json.loads((tmp_path / "wandb_run.json").read_text()) == dict(
        id=run.id, project=run.project, entity=run.entity, url=run.url, mode="online",
    )
    assert OmegaConf.load(tmp_path / "config.yaml") == cfg


@pytest.mark.parametrize("failure", [RuntimeError("training failed"), KeyboardInterrupt()])
def test_wandb_finishes_failed_training(wandb_entrypoint, failure):
    cfg, wandb, run = wandb_entrypoint
    cfg.wandb.enabled = True
    train_refined.train.side_effect = failure
    with pytest.raises(type(failure)) as caught:
        train_refined.main.__wrapped__(cfg)
    assert caught.value is failure
    run.finish.assert_called_once_with(exit_code=1)


def test_wandb_auth_failure_propagates(wandb_entrypoint):
    cfg, wandb, run = wandb_entrypoint
    cfg.wandb.enabled = True
    wandb.init.side_effect = RuntimeError("authentication failed")
    with pytest.raises(RuntimeError, match="authentication failed"):
        train_refined.main.__wrapped__(cfg)
    train_refined.train.assert_not_called()
    run.finish.assert_not_called()


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_wandb_rejects_non_online_run(wandb_entrypoint, mode):
    cfg, wandb, run = wandb_entrypoint
    cfg.wandb.enabled = True
    run.settings.mode = mode
    with pytest.raises(RuntimeError, match="must be online"):
        train_refined.main.__wrapped__(cfg)
    train_refined.train.assert_not_called()
    run.finish.assert_called_once_with(exit_code=1)


def test_wandb_logging_failure_finishes_failed(wandb_entrypoint):
    cfg, wandb, run = wandb_entrypoint
    cfg.wandb.enabled = True
    run.log.side_effect = RuntimeError("logging failed")
    train_refined.train.side_effect = lambda cfg, log_metrics: log_metrics({"loss": 1.0}, 1)
    with pytest.raises(RuntimeError, match="logging failed"):
        train_refined.main.__wrapped__(cfg)
    run.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("overlap", [False, True])
def test_pipeline_variant_paths_and_camera_default(monkeypatch, overlap):
    from wam_h3.refined import cache, pipeline
    calls = []
    def load(path, variant="camera"):
        calls.append((path, variant))
        return dict(latent=variant, conditioning=variant, tags={}, identity="base-camera")
    monkeypatch.setattr(cache, "load_policy_cache", load)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(pipeline, "to_host", lambda features, device: features)
    monkeypatch.setattr(pipeline, "to_device", lambda features, device: features)
    def extract(latent, conditioning, tags):
        assert latent == conditioning
        return [torch.zeros(1)], torch.tensor([0])
    paths = [("train.pt", variant) for variant in ("minimal", "camera", "constraints", "physical")] + ["validation.pt"]
    rng = random.getstate()
    results = list(pipeline.feature_batches(extract, paths, len(paths), "cpu", "cpu", overlap))
    assert random.getstate() == rng
    assert calls == paths[:-1] + [("validation.pt", "camera")]
    assert all(sample["identity"] == "base-camera" for _, _, sample in results)


def test_prompt_variants_default_config():
    from wam_h3.refined.trainer import validate_config
    cfg = OmegaConf.load("configs/refined_train.yaml")
    assert list(cfg.prompt_variants) == ["minimal", "camera", "constraints", "physical"]
    validate_config(cfg)


def test_unequal_episode_chunk_gradient_equivalence():
    from wam_h3.refined.policy import chunk_masked_loss
    from wam_h3.refined.trainer import scale_gradients
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 7)
    reference = torch.nn.Linear(3, 7)
    reference.load_state_dict(model.state_dict())
    episodes = []
    for chunks in (1, 4, 2):
        x, y = torch.randn(chunks, 5, 3), torch.randn(chunks, 5, 7)
        valid = torch.ones(chunks, 5, dtype=torch.bool)
        valid[-1, 2:] = False
        y[~valid] = float("nan")
        episodes.append((x, y, valid))
        chunk_masked_loss(model(x), y, valid).sum().backward()
    scale_gradients(model, 7)
    x, y, valid = (torch.cat([e[i] for e in episodes]) for i in range(3))
    chunk_masked_loss(reference(x), y, valid).mean().backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    for m in (model, reference):
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
        torch.optim.AdamW(m.parameters(), lr=0.01).step()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)


def test_sampler_whole_episode_overshoot_and_resume():
    from wam_h3.refined.trainer import EpisodeSampler
    records = [{"chunks": n} for n in (20, 30, 40)]
    sampler = EpisodeSampler(3, 11)
    visits = []
    for _ in range(5):
        batch = sampler.batch(records, 64)
        assert sum(records[i]["chunks"] for i in batch) >= 64
        assert sum(records[i]["chunks"] for i in batch[:-1]) < 64
        visits.extend(batch)
    for i in range(0, len(visits) - 2, 3):
        assert sorted(visits[i:i + 3]) == [0, 1, 2]
    resumed = EpisodeSampler(**sampler.state_dict())
    assert [sampler.next() for _ in range(20)] == [resumed.next() for _ in range(20)]
    assert EpisodeSampler(3, 11).batch(records, 64) == visits[:len(EpisodeSampler(3, 11).batch(records, 64))]


def cache_sample(demo, chunks=2, value=1.0, episode_sha256=None):
    source = dict(origin="official_libero", revision="revision", sha256="file-hash", demo=demo, suite="libero_spatial")
    metadata = dict(source=source, action_alignment="post_action_verified")
    if episode_sha256:
        metadata["episode_sha256"] = episode_sha256
    from wam_h3.refined.cache import identity
    return dict(metadata=metadata, identity=identity(metadata), targets=torch.full((chunks, 3, 7), value),
                valid=torch.ones(chunks, 3, dtype=torch.bool), cutoffs=torch.arange(chunks))


def split_config(tmp_path, train_path, validation_path):
    manifest = dict(schema=1, train_caches=[str(train_path)], validation_caches=[str(validation_path)],
                    seed=0, dataset_revision="revision", train_episodes=1, validation_episodes=1)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    return SimpleNamespace(split_manifest=str(path), seed=0)


@pytest.mark.parametrize("same_demo", [True, False])
def test_split_rejects_physical_leakage(tmp_path, monkeypatch, same_demo):
    from wam_h3.refined import trainer
    train_path, validation_path = tmp_path / "train.pt", tmp_path / "validation.pt"
    samples = {str(train_path): cache_sample("demo_0", episode_sha256="same"),
               str(validation_path): cache_sample("demo_0" if same_demo else "demo_1", episode_sha256="same")}
    monkeypatch.setattr(trainer, "load_policy_cache", samples.__getitem__)
    with pytest.raises(ValueError, match="overlap"):
        trainer.load_splits(split_config(tmp_path, train_path, validation_path))


@pytest.mark.parametrize("variants", [None, ["minimal", "camera", "constraints", "physical"]])
@pytest.mark.parametrize("budget", [7, 10, None])
def test_train_normalizer_validation_and_timed_checkpoint(tmp_path, monkeypatch, budget, variants):
    from wam_h3.refined import trainer
    train_path, validation_path = tmp_path / "train.pt", tmp_path / "validation.pt"
    samples = {str(train_path): cache_sample("demo_0", chunks=2, value=1.0),
               str(validation_path): cache_sample("demo_1", chunks=1, value=100.0)}
    samples[str(train_path)]["targets"][0] = -1
    cfg = split_config(tmp_path, train_path, validation_path)
    cfg.__dict__.update(source_device="cpu:1", target_device="cpu", max_steps=1 if budget is None else 5, target_chunks=3,
                        max_seconds=budget, validation_every=10 if budget is None else 1, checkpoint_every=1, lr=0.01,
                        weight_decay=0.01, max_grad_norm=1.0, resume=None, overlap=True,
                        output_dir=str(tmp_path / "run"), generation_review=str(tmp_path / "review.json"))
    if variants is not None:
        cfg.prompt_variants = variants
    (tmp_path / "review.json").write_text(json.dumps(dict(status="approved", model_revision=trainer.MODEL_REVISION)))
    monkeypatch.setattr(trainer, "load_policy_cache", samples.__getitem__)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    clock = [0.0]
    monkeypatch.setattr(trainer.time, "perf_counter", lambda: clock[0])
    class TinyPolicy(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.value = torch.nn.Parameter(torch.tensor(0.2))
        def forward(self, features, times, cutoffs):
            return self.value.expand(len(cutoffs), 3, 7)
    monkeypatch.setattr(trainer, "ActionPolicy", TinyPolicy)
    monkeypatch.setitem(sys.modules, "wam_h3.refined.backbone", SimpleNamespace(FrozenFeatures=lambda: SimpleNamespace(model=torch.nn.Linear(1, 1))))
    visits = []
    def features(extractor, paths, steps, source, target, overlap):
        assert len(paths) == steps
        for path in paths:
            visits.append((path, torch.is_grad_enabled()))
            clock[0] += 4
            yield [], torch.tensor([0]), samples[path[0] if isinstance(path, tuple) else path]
    monkeypatch.setattr(trainer, "feature_batches", features)
    metrics = []
    result = trainer.train(cfg, lambda m, s: metrics.append(dict(m)))
    assert result["actual_step"] == 1
    def expected_visits(start, count):
        return [((str(train_path), random.Random(cfg.seed + ordinal).choice(variants)) if variants else str(train_path), True)
                for ordinal in range(start, start + count)]
    assert visits == expected_visits(0, 2) + ([] if budget == 7 else [(str(validation_path), False)])
    batch = next(m for m in metrics if "loss" in m)
    assert (batch["chunks"], batch["episodes"], batch["valid_timesteps"]) == (4, 2, 12)
    if budget != 7:
        assert batch["validation/chunks"] == 1
        assert batch["validation/loss"] > 9000
    else:
        assert "validation/loss" not in batch
    assert metrics[-1]["stop_reason"] == ("max_steps" if budget is None else "max_seconds")
    checkpoints = list((tmp_path / "run").glob("step_*"))
    assert len(checkpoints) == 1 and checkpoints[0].name == "step_000001"
    state = torch.load(checkpoints[0] / "training.pt", weights_only=True)
    assert state["sampler"]["visited"] == 2
    assert state["train_config"].get("prompt_variants") == variants
    torch.testing.assert_close(state["normalizer"]["center"], torch.zeros(7))
    torch.testing.assert_close(state["normalizer"]["scale"], torch.ones(7))
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(checkpoints[0], TinyPolicy(PolicyConfig()), None, 1, {}, [])
    if budget is None:
        from safetensors.torch import load_file
        cfg.resume, cfg.max_steps, cfg.output_dir = str(checkpoints[0]), 2, str(tmp_path / "resumed")
        visits.clear()
        resumed = trainer.train(cfg)
        assert resumed["sampler"]["visited"] == 4
        assert visits == expected_visits(2, 2) + [(str(validation_path), False)]
        if variants:
            cfg.prompt_variants = list(reversed(variants))
            with pytest.raises(ValueError, match="configuration differs"):
                trainer.train(cfg)
            cfg.prompt_variants = variants
        cfg.resume, cfg.output_dir = None, str(tmp_path / "reference")
        visits.clear()
        trainer.train(cfg)
        assert [visit for visit in visits if visit[1]] == expected_visits(0, 4)
        assert [visit for visit in visits if not visit[1]] == [(str(validation_path), False)]
        actual = load_file(str(tmp_path / "resumed" / "step_000002" / "policy.safetensors"))
        expected = load_file(str(tmp_path / "reference" / "step_000002" / "policy.safetensors"))
        torch.testing.assert_close(actual["value"], expected["value"], atol=0, rtol=0)


@pytest.mark.parametrize("name,value", [("target_chunks", 0), ("max_seconds", -1), ("validation_every", 0), ("lr", float("nan")),
                                       ("prompt_variants", []), ("prompt_variants", ["unknown"]),
                                       ("prompt_variants", "camera"), ("prompt_variants", ["camera", "camera"]),
                                       ("prompt_variants", None), ("prompt_variants", 42)])
def test_invalid_config_precedes_model_load(name, value, monkeypatch):
    from wam_h3.refined import trainer
    cfg = OmegaConf.load("configs/refined_train.yaml")
    setattr(cfg, name, value)
    model = Mock(side_effect=AssertionError("model must not load"))
    monkeypatch.setattr(trainer, "ActionPolicy", model)
    with pytest.raises(ValueError):
        trainer.train(cfg)
    model.assert_not_called()


def test_audited_index_split_identity_validation(tmp_path, monkeypatch):
    from wam_h3.refined import trainer
    paths = [tmp_path / "train.pt", tmp_path / "validation.pt"]
    samples, entries = {}, []
    for index, path in enumerate(paths):
        sample = cache_sample(f"demo_{index}")
        sample["metadata"].update(episode=index, variant="camera")
        sample["metadata"]["source"].update(episode_sha256=f"hash-{index}", file="libero_spatial/task.hdf5")
        samples[str(path)] = sample
        entries.append(dict(index=index, path="libero_spatial/task.hdf5", demo=f"demo_{index}",
                            episode_sha256=f"hash-{index}", chunks=2))
    monkeypatch.setattr(trainer, "load_policy_cache", samples.__getitem__)
    cfg = split_config(tmp_path, *paths)
    manifest = json.loads((tmp_path / "split.json").read_text())
    manifest.update(train_episodes=[0], validation_episodes=[1], episodes=entries, variant="camera")
    (tmp_path / "split.json").write_text(json.dumps(manifest))
    training, validation, _ = trainer.load_splits(cfg)
    assert len(training) == len(validation) == 1
    entries[1]["episode_sha256"] = "wrong"
    (tmp_path / "split.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identities differ"):
        trainer.load_splits(cfg)
