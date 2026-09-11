import json

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from tests.test_wam import make, sample
from wam_h3.train.trainer import Trainer, build_scheduler


def train_cfg(tmp_path, **kw):
    base = dict(batch_size=2, grad_accum=1, num_workers=0, lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95], warmup_ratio=0.25,
                min_lr_ratio=0.1, max_grad_norm=1.0, num_epochs=1, max_steps=4, save_every=2, log_every=1, resume=None)
    base.update(kw)
    return OmegaConf.create(dict(train=base, output_dir=str(tmp_path / "run"), seed=0, wandb=dict(mode="disabled", project="wam-h3", name=None)))


def test_scheduler_warmup_then_cosine():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    sch = build_scheduler(opt, total_steps=100, warmup_ratio=0.05, min_lr_ratio=0.01)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert abs(lrs[0] - 0.2) < 1e-6 and abs(lrs[5] - 1.0) < 1e-6
    assert lrs[6] < lrs[5] and abs(lrs[-1] - 0.01) < 2e-3


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, n=6):
        self.items = [{k: v[0] for k, v in sample(cfg, B=1).items()} for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_trainer_saves_checkpoints_and_only_updates_trainable(tmp_path):
    model = make(tmp_path, r=4)
    frozen = model.dit.blocks[0].attn.qkv_proj.base_layer.weight.clone()
    lora = model.dit.blocks[0].attn.qkv_proj.lora_B["default"].weight.clone()
    t = Trainer(train_cfg(tmp_path), model, ListDataset(model.cfg), Accelerator(cpu=True))
    assert t.total_steps == 4
    t.train()
    ck = tmp_path / "run/checkpoints"
    assert sorted(p.name for p in ck.iterdir()) == ["step_000002", "step_000004"]
    st = json.loads((ck / "step_000004/trainer_state.json").read_text())
    assert st["step"] == 4 and st["epoch"] == 1 and "loss" in st
    assert torch.equal(model.dit.blocks[0].attn.qkv_proj.base_layer.weight, frozen)
    assert not torch.equal(model.dit.blocks[0].attn.qkv_proj.lora_B["default"].weight, lora)


def test_trainer_resumes_from_checkpoint(tmp_path):
    model = make(tmp_path, r=4)
    t = Trainer(train_cfg(tmp_path, max_steps=3, save_every=2), model, ListDataset(model.cfg), Accelerator(cpu=True))
    t.train()
    model2 = make(tmp_path, r=4)
    model2.dit.load_state_dict({k: v for k, v in model.dit.state_dict().items() if "lora" not in k}, strict=False)
    model2.vae.load_state_dict(model.vae.state_dict())
    t2 = Trainer(train_cfg(tmp_path, max_steps=5, save_every=2, resume=str(tmp_path / "run/checkpoints/step_000002")),
                 model2, ListDataset(model2.cfg), Accelerator(cpu=True))
    assert t2.step == 2
    from safetensors.torch import load_file
    ck = load_file(str(tmp_path / "run/checkpoints/step_000002/adapter.safetensors"))
    sd2 = model2.dit.state_dict()
    assert all(torch.equal(sd2[k], v) for k, v in ck.items()) and "blocks.0.attn.qkv_proj.lora_B.default.weight" in ck
    t2.train()
    assert (tmp_path / "run/checkpoints/step_000005").exists()
    assert json.loads((tmp_path / "run/checkpoints/step_000004/trainer_state.json").read_text())["step"] == 4


def test_trainer_logs_metrics_each_step(tmp_path):
    model = make(tmp_path, r=4)
    seen = []
    t = Trainer(train_cfg(tmp_path, max_steps=3, save_every=0, log_every=1), model, ListDataset(model.cfg), Accelerator(cpu=True),
                log=lambda d, step: seen.append((step, d)))
    t.train()
    assert [s for s, _ in seen] == [1, 2, 3]
    assert {"train/loss", "train/loss_video", "train/loss_video_full_noise", "train/loss_action", "train/grad_norm", "train/lr",
            "train/epoch", "train/steps_per_sec"} <= set(seen[0][1])
