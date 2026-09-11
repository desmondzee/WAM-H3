import json

from omegaconf import OmegaConf

from wam_h3.train import wandb_log


def test_disabled_mode_returns_noop_logger(tmp_path):
    cfg = OmegaConf.create(dict(task_name="t", output_dir=str(tmp_path), wandb=dict(mode="disabled", project="p", name=None)))
    log, finish = wandb_log.init_train(cfg)
    log({"train/loss": 1.0}, 1)
    finish()
    assert not (tmp_path / "wandb_run.json").exists()


def test_eval_metrics_flatten_summary():
    summary = dict(suites={"libero_spatial": 0.5, "libero_goal": 1.0}, success_rate_mean_over_suites=0.75, episodes=10, successes=6)
    m = wandb_log.eval_metrics("libero", 3000, summary)
    assert m == {"eval/libero/libero_spatial": 0.5, "eval/libero/libero_goal": 1.0, "eval/libero/mean": 0.75,
                 "eval/libero/episodes": 10, "eval/libero/successes": 6, "eval/step": 3000}


def test_log_eval_without_run_file_is_noop(tmp_path):
    wandb_log.log_eval(tmp_path, "libero", 1, dict(suites={}, success_rate_mean_over_suites=0.0, episodes=0, successes=0))
