#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "third_party/FasterWAM"
for p in (FW, FW / "experiments/libero"):
    sys.path.insert(0, str(p))

import hydra
from hydra.core.hydra_config import HydraConfig

from omegaconf import OmegaConf

from wam_h3.eval.policy import run_dir_of
from wam_h3.eval.results import summarize


@hydra.main(config_path="../configs", config_name="sim_libero", version_base=None)
def main(cfg):
    import run_libero_manager as m

    m.CHUNK_ENTRY, m.PROJECT_ROOT = ROOT / "wam_h3/eval/libero_worker.py", ROOT
    ckpt = Path(cfg.ckpt).resolve()
    out = Path(cfg.EVALUATION.output_dir or run_dir_of(ckpt) / "eval" / cfg.benchmark_name / ckpt.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mr = cfg.MULTIRUN
    cache = OmegaConf.load(run_dir_of(ckpt) / "config.yaml").model.text_cache_dir
    subprocess.run([sys.executable, str(ROOT / "scripts/precompute_text_embeds.py"), "--benchmark-suites", *mr.task_suite_names,
                    "--cache-dir", str(cache)], check=True, cwd=ROOT)
    task_file = Path(mr.task_file).resolve() if mr.task_file else m.create_task_file(
        out / "tasks.txt", list(mr.task_suite_names), benchmark_name=str(cfg.benchmark_name),
        sample_ratio=m._resolve_sample_ratio(mr.task_sample_ratio), sample_seed=int(mr.task_sample_seed))[0]
    m.run_evaluation(task_file=task_file, config_name=str(HydraConfig.get().job.config_name), task_choice="-", ckpt=ckpt,
                     num_trials=int(cfg.EVALUATION.num_trials), max_chunks_per_gpu=int(mr.max_tasks_per_gpu),
                     chunk_size=int(mr.chunk_size), output_dir=out, extra_overrides=m.collect_worker_overrides(),
                     gpu_ids=m._parse_gpu_ids(mr), dry_run=bool(mr.dry_run))
    if not mr.dry_run:
        print(json.dumps(summarize(out), indent=1))
        print(f"results: {out}")


if __name__ == "__main__":
    main()
