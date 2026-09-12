import json
import math
import random
import time
import uuid
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .cache import load_policy_cache
from .metrics import LossAccumulator
from .pipeline import feature_batches
from .policy import ActionPolicy, PolicyConfig, chunk_masked_loss
from .runtime import MODEL_REVISION


def fit_normalizer(samples):
    low = high = None
    for sample in samples:
        targets = sample["targets"][sample["valid"]].float()
        if not targets.numel() or not torch.isfinite(targets).all():
            raise ValueError("Training targets must be nonempty and finite")
        lo, hi = targets.amin(0), targets.amax(0)
        low, high = (lo, hi) if low is None else (torch.minimum(low, lo), torch.maximum(high, hi))
    if low is None:
        raise ValueError("No training episodes")
    return {"center": (low + high) / 2, "scale": ((high - low) / 2).clamp_min(1e-6)}


def save_checkpoint(path, model, optimizer, step, normalizer, identities, sampler=None, train_config=None, best_loss=None, run_config=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("policy.safetensors", "training.pt", "policy_config.json")):
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path / "policy.safetensors"))
    torch.save(dict(optimizer=optimizer.state_dict(), step=step, normalizer=normalizer, identities=identities,
                    sampler=sampler, train_config=train_config, best_loss=best_loss, run_config=run_config,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate()),
               path / "training.pt")
    (path / "policy_config.json").write_text(json.dumps(asdict(model.cfg), indent=2))


def load_checkpoint(path, model, optimizer, identities):
    path = Path(path)
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    if state["identities"] != identities:
        raise ValueError("Resume cache identities differ from the checkpoint")
    model.load_state_dict(load_file(str(path / "policy.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return state


class EpisodeSampler:
    def __init__(self, size, seed, visited=0):
        if size <= 0 or visited < 0:
            raise ValueError("Sampler requires episodes and a nonnegative visit count")
        self.size, self.seed, self.visited = size, seed, visited
        self._epoch, self._order = None, None

    def next(self):
        epoch, offset = divmod(self.visited, self.size)
        if self._epoch != epoch:
            self._order = list(range(self.size))
            random.Random(self.seed + epoch).shuffle(self._order)
            self._epoch = epoch
        self.visited += 1
        return self._order[offset]

    def batch(self, records, target_chunks):
        if target_chunks <= 0 or any(r["chunks"] <= 0 for r in records):
            raise ValueError("Batch and episode chunk counts must be positive")
        indices, chunks = [], 0
        while chunks < target_chunks:
            index = self.next()
            indices.append(index)
            chunks += records[index]["chunks"]
        return indices

    def state_dict(self):
        return dict(seed=self.seed, size=self.size, visited=self.visited)


def physical_identities(metadata):
    source = metadata.get("source", {})
    keys = set()
    if source.get("sha256") and source.get("demo"):
        keys.add(("source", source["sha256"], source["demo"]))
    episode = metadata.get("episode_sha256") or source.get("episode_sha256")
    if episode:
        keys.add(("episode", episode))
    if not keys:
        raise ValueError("Cache requires a physical source identity")
    return keys


def suite_name(metadata):
    source = metadata.get("source", {})
    if source.get("suite") or metadata.get("suite"):
        return source.get("suite") or metadata["suite"]
    for part in Path(source.get("file", "")).parts:
        if part.startswith("libero_"):
            return part
    return "unknown"


def load_records(paths):
    records = []
    for path in paths:
        sample = load_policy_cache(path)
        metadata = sample["metadata"]
        if metadata.get("action_alignment") != "post_action_verified":
            raise ValueError("Processed LIBERO post-action alignment must be verified before training")
        if metadata.get("source", {}).get("origin") != "official_libero":
            raise ValueError("Training requires official LIBERO provenance")
        valid, targets = sample["valid"], sample["targets"]
        if valid.dtype != torch.bool or valid.ndim != 2 or targets.shape != (*valid.shape, 7):
            raise ValueError("Invalid episode targets/mask")
        if not len(valid) or not valid.any(-1).all() or not torch.isfinite(targets[valid]).all():
            raise ValueError("Every episode chunk must contain finite valid targets")
        if sample["cutoffs"].shape != (len(valid),):
            raise ValueError("Episode cutoffs must match chunk count")
        records.append(dict(path=str(Path(path).resolve()), identity=sample["identity"], metadata=metadata,
                            physical=physical_identities(metadata), suite=suite_name(metadata),
                            targets=targets, valid=valid, chunks=len(valid), valid_timesteps=int(valid.sum())))
        del sample
    return records


def load_splits(cfg):
    manifest = None
    if getattr(cfg, "split_manifest", None):
        manifest = json.loads(Path(cfg.split_manifest).read_text())
        required = {"schema", "train_caches", "validation_caches", "seed", "dataset_revision", "train_episodes", "validation_episodes"}
        if not required <= manifest.keys() or manifest["schema"] != 1:
            raise ValueError("Invalid split manifest schema")
        if manifest["seed"] != cfg.seed or not manifest["dataset_revision"]:
            raise ValueError("Split seed/revision mismatch")
        for key in ("train_caches", "validation_caches"):
            if not isinstance(manifest[key], list) or not manifest[key] or any(not isinstance(p, str) or not Path(p).is_absolute() for p in manifest[key]):
                raise ValueError("Split caches must be nonempty absolute path lists")
        train_paths, validation_paths = manifest["train_caches"], manifest["validation_caches"]
    else:
        train_paths, validation_paths = cfg.caches, []
    training, validation = load_records(train_paths), load_records(validation_paths)
    if not training:
        raise ValueError("No training episodes")
    train_keys = set().union(*(r["physical"] for r in training))
    validation_keys = set().union(*(r["physical"] for r in validation))
    if train_keys & validation_keys:
        raise ValueError("Train/validation physical episode overlap")
    if manifest:
        for name, records in (("train", training), ("validation", validation)):
            keys = set()
            for r in records:
                if r["metadata"]["source"].get("revision") != manifest["dataset_revision"]:
                    raise ValueError("Split dataset revision differs from cache identity")
                keys.update(r["physical"])
            declared = manifest[name + "_episodes"]
            if isinstance(declared, int) and declared != len({tuple(sorted(r["physical"])) for r in records}):
                raise ValueError("Split episode count mismatch")
            if isinstance(declared, list):
                if not declared:
                    raise ValueError("Empty split episode identities")
                if all(isinstance(entry, int) and not isinstance(entry, bool) for entry in declared):
                    entries = manifest.get("episodes", [])
                    by_index = {entry["index"]: entry for entry in entries}
                    if len(by_index) != len(entries) or len(set(declared)) != len(declared) or len(declared) != len(records):
                        raise ValueError("Split episode index count mismatch")
                    for index, record in zip(declared, records):
                        entry = by_index.get(index, {})
                        metadata, source = record["metadata"], record["metadata"]["source"]
                        if (metadata.get("episode") != index or not entry.get("episode_sha256")
                                or source.get("episode_sha256") != entry["episode_sha256"]
                                or source.get("file") != entry.get("path") or source.get("demo") != entry.get("demo")
                                or record["chunks"] != entry.get("chunks")
                                or (manifest.get("variant") and metadata.get("variant") != manifest["variant"])):
                            raise ValueError("Split episode identities differ from caches")
                else:
                    declared_keys = set()
                    for entry in declared:
                        if not isinstance(entry, dict):
                            raise ValueError("Split episode identities must be records or audited indices")
                        declared_keys.update(physical_identities(entry if "source" in entry else {"source": entry}))
                    if not all(r["physical"] & declared_keys for r in records) or not declared_keys <= keys:
                        raise ValueError("Split episode identities differ from caches")
            elif not isinstance(declared, int):
                raise ValueError("Invalid split episode identities")
    return training, validation, manifest


def lr_schedule_config(cfg):
    schedule = getattr(cfg, "lr_schedule", None)
    if schedule is None:
        return None
    from collections.abc import Mapping
    if not isinstance(schedule, Mapping):
        raise ValueError("lr_schedule must be a mapping or null")
    if not isinstance(schedule.get("enabled", True), bool):
        raise ValueError("lr_schedule.enabled must be boolean")
    if not schedule.get("enabled", True):
        return None
    schedule = dict(schedule)
    for name in ("warmup_steps", "total_steps"):
        value = schedule.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"lr_schedule.{name} must be an integer")
    if not 2 <= schedule["warmup_steps"] < schedule["total_steps"]:
        raise ValueError("lr_schedule requires total_steps > warmup_steps >= 2")
    for name in ("start_lr", "min_lr"):
        value = schedule.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= cfg.lr:
            raise ValueError(f"lr_schedule.{name} must be finite, positive, and <= lr")
    if cfg.max_steps > schedule["total_steps"]:
        raise ValueError("max_steps must not exceed lr_schedule.total_steps")
    return {name: schedule[name] for name in ("warmup_steps", "start_lr", "min_lr", "total_steps")}


def learning_rate_for_update(peak_lr, schedule, update):
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError("Optimizer update must be a positive integer")
    if schedule is None:
        return peak_lr
    warmup, total = schedule["warmup_steps"], schedule["total_steps"]
    if update <= warmup:
        return schedule["start_lr"] + (peak_lr - schedule["start_lr"]) * (update - 1) / (warmup - 1)
    if update >= total:
        return schedule["min_lr"]
    return schedule["min_lr"] + 0.5 * (peak_lr - schedule["min_lr"]) * (1 + math.cos(math.pi * (update - warmup) / (total - warmup)))


def validate_config(cfg):
    for name, default in (("max_steps", None), ("target_chunks", 64), ("validation_every", 10), ("checkpoint_every", 10)):
        value = getattr(cfg, name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("lr", "max_grad_norm"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    lr_schedule_config(cfg)
    budget = getattr(cfg, "max_seconds", None)
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        raise ValueError("max_seconds must be finite and positive or null")
    source, target = torch.device(cfg.source_device), torch.device(cfg.target_device)
    if source == target:
        raise ValueError("Training requires distinct devices")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if hasattr(cfg, "prompt_variants"):
        variants = cfg.prompt_variants
        if (not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)) or not variants
                or any(v not in ("minimal", "camera", "constraints", "physical") for v in variants)
                or len(set(variants)) != len(variants)):
            raise ValueError("prompt_variants must be a nonempty list of distinct supported variants")
    return source, target


def scale_gradients(model, chunks):
    if chunks <= 0:
        raise ValueError("No chunks accumulated")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(chunks)


def train(cfg, log_metrics=None):
    source, target = validate_config(cfg)
    started = time.perf_counter()
    budget = getattr(cfg, "max_seconds", None)
    def has_time():
        return budget is None or time.perf_counter() - started < budget

    review = json.loads(Path(cfg.generation_review).read_text())
    if review.get("status") != "approved" or review.get("model_revision") != MODEL_REVISION:
        raise ValueError("Native generation requires an approved review for this checkpoint before training")
    training, validation, manifest = load_splits(cfg)
    normalizer = fit_normalizer(training)
    identities = dict(train=[r["identity"] for r in training], validation=[r["identity"] for r in validation], split=manifest)
    train_config = dict(seed=cfg.seed, target_chunks=getattr(cfg, "target_chunks", 64), lr=cfg.lr,
                        weight_decay=cfg.weight_decay, max_grad_norm=cfg.max_grad_norm)
    schedule = lr_schedule_config(cfg)
    if schedule is not None:
        train_config["lr_schedule"] = schedule
    if hasattr(cfg, "prompt_variants"):
        train_config["prompt_variants"] = list(cfg.prompt_variants)
    from omegaconf import OmegaConf
    run_config = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else vars(cfg).copy()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = ActionPolicy(PolicyConfig()).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = EpisodeSampler(len(training), cfg.seed)
    step, best_loss = 0, float("inf")
    from .backbone import FrozenFeatures
    extractor = FrozenFeatures()
    if cfg.resume:
        state = load_checkpoint(cfg.resume, model, optimizer, identities)
        step, normalizer = state["step"], state["normalizer"]
        if (state.get("train_config") is not None and state["train_config"] != train_config
                or schedule is not None and state.get("train_config") is None):
            raise ValueError("Resume training configuration differs from checkpoint")
        if state.get("sampler"):
            sampler = EpisodeSampler(**state["sampler"])
        elif step:
            raise ValueError("Resume checkpoint has no episode sampler state")
        best_loss = state.get("best_loss") if state.get("best_loss") is not None else best_loss
    if step >= cfg.max_steps:
        raise ValueError("max_steps must exceed the resumed step")
    center, scale = (normalizer[k].to(target) for k in ("center", "scale"))
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex[:12]
    saved, last_validation = {}, None
    initial = {name: p.detach().flatten()[:16].cpu().clone() for name, p in model.named_parameters()}
    initial_step = step

    def peak(device):
        return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    def checkpoint():
        if step not in saved:
            path = output / f"step_{step:06d}"
            if path.exists():
                path = output / f"step_{step:06d}_{session}"
            save_checkpoint(path, model, optimizer, step, normalizer, identities, sampler.state_dict(), train_config, best_loss, run_config)
            saved[step] = str(path)
        return saved[step]

    def episode_losses(features, times, sample, diagnostics):
        valid = sample["valid"].to(target)
        targets = (sample["targets"].to(target) - center) / scale
        with torch.autocast("cuda", dtype=torch.bfloat16) if target.type == "cuda" else nullcontext():
            predicted = model(features, times, sample["cutoffs"].to(target))
            losses = chunk_masked_loss(predicted, targets, valid)
        if not torch.isfinite(losses).all():
            raise RuntimeError("Nonfinite policy loss")
        diagnostics.update(predicted, targets, valid)
        return losses

    def validate():
        nonlocal last_validation, best_loss
        model.eval()
        sums, chunks, timesteps = {}, {}, 0
        diagnostics = LossAccumulator(scale)
        with torch.no_grad():
            paths = [r["path"] for r in validation]
            for index, (features, times, sample) in enumerate(feature_batches(extractor, paths, len(paths), source, target, cfg.overlap)):
                if sample["identity"] != validation[index]["identity"]:
                    raise ValueError("Cache identity changed during validation")
                losses = episode_losses(features, times, sample, diagnostics)
                suite = suite_name(sample["metadata"])
                sums[suite] = sums.get(suite, 0.0) + losses.sum().item()
                chunks[suite] = chunks.get(suite, 0) + len(losses)
                timesteps += int(sample["valid"].sum())
                del features, losses, sample
        model.train()
        last_validation = step
        loss = sum(sums.values()) / sum(chunks.values())
        metrics = {"validation/loss": loss, "validation/episodes": len(validation),
                   "validation/chunks": sum(chunks.values()), "validation/valid_timesteps": timesteps}
        metrics.update(diagnostics.finish("validation"))
        for suite in sums:
            metrics[f"validation/{suite}/loss"] = sums[suite] / chunks[suite]
            metrics[f"validation/{suite}/chunks"] = chunks[suite]
            metrics[f"validation/{suite}/episodes"] = sum(r["suite"] == suite for r in validation)
            metrics[f"validation/{suite}/valid_timesteps"] = sum(r["valid_timesteps"] for r in validation if r["suite"] == suite)
        if loss < best_loss:
            best_loss = loss
            (output / f"best_{session}.json").write_text(json.dumps(dict(step=step, loss=loss, checkpoint=checkpoint())))
        return metrics

    with (output / "metrics.jsonl").open("a") as log:
        def emit(metrics):
            elapsed = time.perf_counter() - started
            metrics.update(step=step, elapsed=elapsed, steps_per_second=(step - initial_step) / max(elapsed, 1e-9),
                           source_peak=peak(source), target_peak=peak(target))
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)
            if log_metrics is not None:
                log_metrics(metrics, step)

        while step < cfg.max_steps and has_time():
            first_visit = sampler.visited
            indices = sampler.batch(training, train_config["target_chunks"])
            paths = [training[i]["path"] for i in indices]
            if "prompt_variants" in train_config:
                paths = [(path, random.Random(cfg.seed + first_visit + offset).choice(train_config["prompt_variants"]))
                         for offset, path in enumerate(paths)]
            optimizer.zero_grad(set_to_none=True)
            loss_sum, chunks, timesteps, suites = 0.0, 0, 0, {}
            diagnostics = LossAccumulator(scale)
            for index, (features, times, sample) in zip(indices, feature_batches(extractor, paths, len(paths), source, target, cfg.overlap)):
                if sample["identity"] != training[index]["identity"]:
                    raise ValueError("Cache identity changed during training")
                losses = episode_losses(features, times, sample, diagnostics)
                if len(losses) != training[index]["chunks"]:
                    raise ValueError("Cache chunk count changed during training")
                losses.sum().backward()
                loss_sum += losses.detach().sum().item()
                chunks += len(losses)
                timesteps += int(sample["valid"].sum())
                suite = training[index]["suite"]
                counts = suites.setdefault(suite, dict(episodes=0, chunks=0, valid_timesteps=0, loss_sum=0.0))
                counts["loss_sum"] += losses.detach().sum().item()
                counts["episodes"] += 1
                counts["chunks"] += len(losses)
                counts["valid_timesteps"] += int(sample["valid"].sum())
                del features, losses, sample
            scale_gradients(model, chunks)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            applied_lr = learning_rate_for_update(cfg.lr, schedule, step + 1)
            for group in optimizer.param_groups:
                group["lr"] = applied_lr
            optimizer.step()
            step += 1
            metrics = dict(lr=applied_lr, loss=loss_sum / chunks, grad_norm=norm.item(), episodes=len(indices), chunks=chunks,
                           valid_timesteps=timesteps, sampler_visited=sampler.visited, sampler_epoch=sampler.visited // sampler.size)
            metrics.update(diagnostics.finish("train"))
            for suite, counts in suites.items():
                counts["loss"] = counts.pop("loss_sum") / counts["chunks"]
                metrics.update({f"train/{suite}/{key}": value for key, value in counts.items()})
            if validation and (step % getattr(cfg, "validation_every", 10) == 0 or step == cfg.max_steps) and has_time():
                metrics.update(validate())
            if step % getattr(cfg, "checkpoint_every", 10) == 0:
                checkpoint()
            emit(metrics)
        if validation and last_validation != step and has_time():
            emit(validate())
        checkpoint()
        emit(dict(stopped=True, stop_reason="max_steps" if step >= cfg.max_steps else "max_seconds",
                  final_validation_step=last_validation))
    if any(p.grad is not None for p in extractor.model.parameters()):
        raise RuntimeError("Frozen FL2VA accumulated gradients")
    changed = [name for name, p in model.named_parameters() if not torch.equal(initial[name], p.detach().flatten()[:16].cpu())]
    if step > initial_step and "queries" in initial and ("queries" not in changed or not any(".memory.weight" in name for name in changed)):
        raise RuntimeError("Policy queries and memory projections did not both update")
    verification = dict(frozen_backbone=True, changed_parameter_samples=changed, actual_step=step,
                        parameters=sum(p.numel() for p in model.parameters()), cache_identities=identities,
                        source_peak=peak(source), target_peak=peak(target), sampler=sampler.state_dict(),
                        time_boundary="Budget includes setup and validation; complete active optimizer batch or entire validation pass, then save. No new work after deadline.")
    (output / "verification.json").write_text(json.dumps(verification, indent=2))
    return verification
