import json
from pathlib import Path

from omegaconf import OmegaConf


def init_train(cfg):
    w = cfg.get("wandb") or {}
    if w.get("mode", "online") == "disabled":
        return (lambda metrics, step: None), (lambda: None)
    import wandb
    run = wandb.init(project=w.get("project", "wam-h3"), name=w.get("name") or cfg.task_name, mode=w.get("mode", "online"),
                     config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.output_dir)
    (Path(cfg.output_dir) / "wandb_run.json").write_text(json.dumps(dict(id=run.id, project=run.project, entity=run.entity)))
    return (lambda metrics, step: run.log(metrics, step=step)), run.finish


def eval_metrics(benchmark, step, summary):
    m = {f"eval/{benchmark}/{k}": v for k, v in summary["suites"].items()}
    m[f"eval/{benchmark}/mean"] = summary["success_rate_mean_over_suites"]
    m[f"eval/{benchmark}/episodes"] = summary["episodes"]
    m[f"eval/{benchmark}/successes"] = summary["successes"]
    m["eval/step"] = step
    return m


def log_eval(run_dir, benchmark, step, summary):
    f = Path(run_dir) / "wandb_run.json"
    if not f.exists():
        return
    import wandb
    r = json.loads(f.read_text())
    run = wandb.init(id=r["id"], project=r["project"], entity=r["entity"], resume="allow")
    run.log(eval_metrics(benchmark, step, summary))
    run.finish()
