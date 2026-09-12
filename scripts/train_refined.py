import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.runtime import configure
from wam_h3.refined.trainer import train


@hydra.main(config_path="../configs", config_name="refined_train", version_base=None)
def main(cfg):
    configure(torch.device(cfg.source_device).index)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    w = cfg.get("wandb") or {}
    if not w.get("enabled", False):
        train(cfg)
        return
    import wandb
    run = wandb.init(
        project=w.get("project", "wam-h3"), name=w.get("name", "refined-training-smoke"),
        mode="online", config=OmegaConf.to_container(cfg, resolve=True), dir=str(output),
        settings=wandb.Settings(disable_code=True, disable_git=True, console="off"),
    )
    if run is None:
        raise RuntimeError("W&B did not initialize an online run")
    try:
        if run.settings.mode != "online" or not run.url:
            raise RuntimeError("W&B must be online for refined training")
        (output / "wandb_run.json").write_text(json.dumps(dict(
            id=run.id, project=run.project, entity=run.entity, url=run.url, mode="online",
        ), indent=2))
        train(cfg, log_metrics=lambda metrics, step: run.log(metrics, step=step))
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


if __name__ == "__main__":
    main()
