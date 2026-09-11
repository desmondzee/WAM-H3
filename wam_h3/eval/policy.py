from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


def run_dir_of(ckpt):
    return Path(ckpt).resolve().parents[1]


def load_eval_model(ckpt, device, load_text_encoder=False):
    run_cfg = OmegaConf.load(run_dir_of(ckpt) / "config.yaml")
    model = instantiate(run_cfg.model, device=device, load_text_encoder=load_text_encoder)
    state = model.load_checkpoint(ckpt)
    return model.eval(), run_cfg, state
