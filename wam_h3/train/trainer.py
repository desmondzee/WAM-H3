import json
import time
from math import ceil
from pathlib import Path

import torch
from fasterwam.utils.samplers import ResumableEpochSampler
from safetensors.torch import save_file
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader


def build_scheduler(opt, total_steps, warmup_ratio, min_lr_ratio):
    warmup = min(max(int(total_steps * warmup_ratio), 0), total_steps - 1)
    lr = opt.param_groups[0]["lr"]
    main = CosineAnnealingLR(opt, T_max=max(total_steps - warmup, 1), eta_min=lr * min_lr_ratio)
    if warmup <= 0:
        return main
    return SequentialLR(opt, [LinearLR(opt, start_factor=1.0 / warmup, total_iters=warmup), main], milestones=[warmup])


class Trainer:
    def __init__(self, cfg, model, dataset, accelerator, log=None):
        c, acc = cfg.train, accelerator
        self.c, self.acc, self.model, self.log = c, acc, model, log or (lambda m, s: None)
        self.out = Path(cfg.output_dir)
        self.trainable = {n for n, p in model.dit.named_parameters() if p.requires_grad}
        self.step = self.epoch = self.batch_in_epoch = 0
        if c.resume:
            st = model.load_checkpoint(c.resume)
            self.step, self.epoch, self.batch_in_epoch = st["step"], st["epoch"], st["batch_in_epoch"]
        model.dit.train()
        model.dit.grad_checkpoint = bool(c.get("grad_checkpoint", True))
        opt = torch.optim.AdamW([p for p in model.dit.parameters() if p.requires_grad], lr=c.lr,
                                weight_decay=c.weight_decay, betas=tuple(c.betas))
        self.dit, self.opt = acc.prepare(model.dit, opt)
        model.dit = self.dit
        self.sampler = ResumableEpochSampler(dataset, cfg.seed, c.batch_size, acc.num_processes)
        self.loader = acc.prepare(DataLoader(dataset, batch_size=c.batch_size, sampler=self.sampler,
                                             num_workers=c.num_workers, pin_memory=torch.cuda.is_available()))
        per_epoch = ceil(ceil(len(dataset) / (c.batch_size * acc.num_processes)) / c.grad_accum)
        self.total_steps = int(c.max_steps) if c.max_steps else per_epoch * c.num_epochs
        self.sched = build_scheduler(self.opt, self.total_steps, c.warmup_ratio, c.min_lr_ratio)
        self.probe_batch = next(iter(self.loader)) if c.get("probe_every") else None
        if c.resume:
            self.sampler.set_epoch(self.epoch)
            self.sampler.set_resume_batch_offset(self.batch_in_epoch)
            for _ in range(self.step):
                self.sched.step()

    def save(self, loss):
        sd = {}
        for n, p in self.dit.named_parameters():
            if n in self.trainable:
                t = p.detach()
                sd[n] = (t.full_tensor() if hasattr(t, "full_tensor") else t).cpu().contiguous()
        if self.acc.is_main_process:
            d = self.out / "checkpoints" / f"step_{self.step:06d}"
            d.mkdir(parents=True, exist_ok=True)
            save_file(sd, str(d / "adapter.safetensors"))
            (d / "trainer_state.json").write_text(json.dumps(dict(
                step=self.step, epoch=self.epoch, batch_in_epoch=self.batch_in_epoch, loss=loss,
                lr=self.opt.param_groups[0]["lr"]), indent=1))
        self.acc.wait_for_everyone()

    @torch.no_grad()
    def probe(self):
        g = torch.Generator(device=self.acc.device).manual_seed(0)
        with self.acc.autocast():
            _, parts = self.model.training_loss(self.probe_batch, generator=g)
        vals = torch.tensor([parts[k] for k in ("loss_video", "loss_action", "loss_video_full_noise")], device=self.acc.device)
        means = torch.nanmean(self.acc.gather(vals.float().reshape(1, -1)), dim=0)
        return {f"probe/{k}": float(v) for k, v in zip(("loss_video", "loss_action", "loss_video_full_noise"), means)}

    def train(self):
        it, t0, start = iter(self.loader), time.time(), self.step
        loss_val, acc_parts, prev, inc_sq = None, [], None, 0.0
        params = [p for p in self.dit.parameters() if p.requires_grad]
        track_var = self.acc.num_processes == 1
        while self.step < self.total_steps:
            try:
                batch = next(it)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.sampler.clear_resume_batch_offset()
                self.sampler.set_epoch(self.epoch)
                it = iter(self.loader)
                continue
            with self.acc.accumulate(self.dit):
                with self.acc.autocast():
                    loss, parts = self.model.training_loss(batch)
                self.acc.backward(loss)
                acc_parts.append(dict(parts, loss=loss.item()))
                if track_var:
                    cur = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                    inc_sq += sum(float(((c if prev is None else c - q) ** 2).sum()) for c, q in zip(cur, prev or [None] * len(cur)) if c is not None)
                    prev = cur
                if self.acc.sync_gradients:
                    n_acc = len(acc_parts)
                    g_sq = sum(float((p.grad.detach() ** 2).sum()) for p in params if p.grad is not None) if track_var else 0.0
                    grad_var = max(n_acc * inc_sq - g_sq, 0.0) if track_var else float("nan")
                    noise_scale = self.c.batch_size * grad_var / max(g_sq, 1e-12) if track_var else float("nan")
                    prev, inc_sq = None, 0.0
                    gn = self.acc.clip_grad_norm_(self.dit.parameters(), self.c.max_grad_norm)
                    self.opt.step()
                    self.sched.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.step += 1
                    stacked = torch.tensor([[p[k] for k in acc_parts[0]] for p in acc_parts], device=loss.device)
                    means = torch.nanmean(self.acc.gather(stacked.float()), dim=0)
                    parts = {k: float(v) for k, v in zip(acc_parts[0], means)}
                    loss_val, acc_parts = parts.pop("loss"), []
                    probe = self.probe() if self.probe_batch is not None and self.step % self.c.probe_every == 0 else {}
                    if self.step % self.c.log_every == 0 and self.acc.is_main_process:
                        rate = (self.step - start) / max(time.time() - t0, 1e-6)
                        eta = (self.total_steps - self.step) / max(rate, 1e-9)
                        lr = self.opt.param_groups[0]["lr"]
                        print(f"step {self.step}/{self.total_steps} ep {self.epoch} loss {loss_val:.4f} "
                              f"v {parts['loss_video']:.4f} v1 {parts['loss_video_full_noise']:.4f} a {parts['loss_action']:.4f} gn {float(gn):.3f} "
                              f"lr {lr:.2e} {rate:.2f} it/s eta {eta / 60:.1f} min", flush=True)
                        self.log({"train/loss": loss_val, "train/loss_video": parts["loss_video"], "train/loss_action": parts["loss_action"],
                                  "train/loss_video_full_noise": parts["loss_video_full_noise"], "train/grad_norm": float(gn),
                                  "train/lr": lr, "train/epoch": self.epoch, "train/steps_per_sec": rate,
                                  "train/grad_var": grad_var, "train/grad_noise_scale": noise_scale, **probe}, self.step)
                    if self.c.save_every and self.step % self.c.save_every == 0:
                        self.save(loss_val)
        if not self.c.save_every or self.step % self.c.save_every:
            self.save(loss_val)
