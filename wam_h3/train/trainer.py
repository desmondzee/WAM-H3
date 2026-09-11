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
    def __init__(self, cfg, model, dataset, accelerator):
        c, acc = cfg.train, accelerator
        self.c, self.acc, self.model = c, acc, model
        self.out = Path(cfg.output_dir)
        self.trainable = {n for n, p in model.dit.named_parameters() if p.requires_grad}
        model.dit.train()
        model.dit.grad_checkpoint = bool(c.get("grad_checkpoint", True))
        self.dit = model.dit = acc.prepare(model.dit)
        self.opt = torch.optim.AdamW([p for p in self.dit.parameters() if p.requires_grad], lr=c.lr,
                                     weight_decay=c.weight_decay, betas=tuple(c.betas))
        self.sampler = ResumableEpochSampler(dataset, cfg.seed, c.batch_size, acc.num_processes)
        self.loader = acc.prepare(DataLoader(dataset, batch_size=c.batch_size, sampler=self.sampler,
                                             num_workers=c.num_workers, pin_memory=torch.cuda.is_available()))
        per_epoch = ceil(ceil(len(dataset) / (c.batch_size * acc.num_processes)) / c.grad_accum)
        self.total_steps = int(c.max_steps) if c.max_steps else per_epoch * c.num_epochs
        self.sched = build_scheduler(self.opt, self.total_steps, c.warmup_ratio, c.min_lr_ratio)
        self.step = self.epoch = self.batch_in_epoch = 0
        if c.resume:
            st = model.load_checkpoint(c.resume)
            self.step, self.epoch, self.batch_in_epoch = st["step"], st["epoch"], st["batch_in_epoch"]
            self.sampler.set_epoch(self.epoch)
            self.sampler.set_resume_batch_offset(self.batch_in_epoch)
            for _ in range(self.step):
                self.sched.step()

    def save(self, loss):
        sd = self.acc.get_state_dict(self.dit)
        if self.acc.is_main_process:
            d = self.out / "checkpoints" / f"step_{self.step:06d}"
            d.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().cpu().contiguous() for k, v in sd.items() if k in self.trainable}, str(d / "adapter.safetensors"))
            (d / "trainer_state.json").write_text(json.dumps(dict(
                step=self.step, epoch=self.epoch, batch_in_epoch=self.batch_in_epoch, loss=loss,
                lr=self.opt.param_groups[0]["lr"]), indent=1))
        self.acc.wait_for_everyone()

    def train(self):
        it, t0, start = iter(self.loader), time.time(), self.step
        loss_val = None
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
                if self.acc.sync_gradients:
                    gn = self.acc.clip_grad_norm_(self.dit.parameters(), self.c.max_grad_norm)
                    self.opt.step()
                    self.sched.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.step += 1
                    loss_val = self.acc.gather(loss.detach().float().reshape(1)).mean().item()
                    if self.step % self.c.log_every == 0 and self.acc.is_main_process:
                        rate = (self.step - start) / max(time.time() - t0, 1e-6)
                        eta = (self.total_steps - self.step) / max(rate, 1e-9)
                        print(f"step {self.step}/{self.total_steps} ep {self.epoch} loss {loss_val:.4f} "
                              f"v {parts['loss_video']:.4f} a {parts['loss_action']:.4f} gn {float(gn):.3f} "
                              f"lr {self.opt.param_groups[0]['lr']:.2e} {rate:.2f} it/s eta {eta / 60:.1f} min", flush=True)
                    if self.c.save_every and self.step % self.c.save_every == 0:
                        self.save(loss_val)
        if not self.c.save_every or self.step % self.c.save_every:
            self.save(loss_val)
