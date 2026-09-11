#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT / "third_party/FasterWAM"
for p in (FW, FW / "experiments/libero"):
    sys.path.insert(0, str(p))

import torch
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from wam_h3.eval.policy import load_eval_model, run_dir_of


def main():
    ap = argparse.ArgumentParser()
    for k in ("config-dir", "config-name", "task", "ckpt", "task-chunk-file", "first-suite", "first-task-id", "gpu-id",
              "num-trials", "output-dir"):
        ap.add_argument(f"--{k}", required=True)
    a, overrides = ap.parse_known_args()
    overrides = [o for o in overrides if o != "--"]
    with initialize_config_dir(version_base=None, config_dir=str(Path(a.config_dir).resolve())):
        cfg = compose(a.config_name, overrides=[f"ckpt={a.ckpt}", f"gpu_id={a.gpu_id}", f"EVALUATION.num_trials={a.num_trials}",
                                                f"EVALUATION.output_dir={a.output_dir}", *overrides])
    device = str(cfg.EVALUATION.device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(0)
    model, run_cfg, state = load_eval_model(cfg.ckpt, device, bool(cfg.EVALUATION.load_text_encoder))
    OmegaConf.set_struct(cfg, False)
    cfg.data = run_cfg.data
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run_dir_of(cfg.ckpt) / "dataset_stats.json")))
    horizon = cfg.EVALUATION.action_horizon or int(cfg.data.train.num_frames) - 1
    h, w = (int(v) for v in cfg.data.train.video_size)

    import eval_libero_single as single
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    chunk = [l.split(",") for l in Path(a.task_chunk_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    out = Path(a.output_dir)
    for suite, task_id in chunk:
        task_id, t0 = int(task_id), time.time()
        cfg.EVALUATION.task_suite_name, cfg.EVALUATION.task_id = suite, task_id
        ts = suites[suite]()
        init = list(ts.get_task_init_states(task_id))
        while len(init) < int(cfg.EVALUATION.num_trials):
            init += init[: int(cfg.EVALUATION.num_trials) - len(init)]
        (out / suite / "videos").mkdir(parents=True, exist_ok=True)
        r = dict(benchmark=str(cfg.benchmark_name), task_suite=suite, task_id=task_id, checkpoint=str(cfg.ckpt),
                 step=state.get("step"), total_episodes=int(cfg.EVALUATION.num_trials), gpu_id=int(a.gpu_id),
                 start_time=time.strftime("%Y-%m-%d %H:%M:%S"))
        r.update(single.run_single_task(task=ts.get_task(task_id), initial_states=init, model=model, processor=processor,
                                        cfg=cfg, video_dir=out / suite / "videos", predicted_video_dir=out / suite / "predicted",
                                        action_horizon=horizon, input_w=w, input_h=h, model_device=device))
        r["duration"] = time.time() - t0
        (out / suite / f"gpu{a.gpu_id}_task{task_id}_results.json").write_text(json.dumps(r, indent=1, cls=single.NumpyEncoder))
        print(f"{suite} task {task_id}: {r['successes']}/{r['total_episodes']} in {r['duration']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
