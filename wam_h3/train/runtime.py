from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from fasterwam.utils import misc
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .trainer import Trainer


def run_training(cfg):
    misc.register_work_dir(cfg.output_dir)
    set_seed(cfg.seed, device_specific=True)
    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=cfg.train.grad_accum)
    if acc.is_main_process:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, Path(cfg.output_dir) / "config.yaml", resolve=True)
    model = instantiate(cfg.model, device=str(acc.device))
    dataset = instantiate(cfg.data.train)
    if acc.is_main_process:
        n = sum(p.numel() for p in model.dit.parameters() if p.requires_grad)
        print(f"trainable params {n / 1e6:.1f}M, dataset {len(dataset)} samples, device {acc.device}", flush=True)
    Trainer(cfg, model, dataset, acc).train()
    if acc.is_main_process and torch.cuda.is_available():
        print(f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
