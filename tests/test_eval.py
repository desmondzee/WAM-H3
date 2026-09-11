from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from wam_h3.eval.policy import load_eval_model

ROOT = Path(__file__).resolve().parents[1]
SMOKE = sorted((ROOT / "runs/smoke/checkpoints").glob("step_*")) if (ROOT / "runs/smoke/checkpoints").exists() else []


def test_sim_configs_compose():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        a = compose("sim_libero", overrides=["ckpt=x", "MULTIRUN.num_gpus=1"])
        b = compose("sim_libero_plus", overrides=["ckpt=x"])
    assert a.EVALUATION.action_infer_mode == "one_pass_future_cache" and a.EVALUATION.replan_steps == 10
    assert a.EVALUATION.num_inference_steps == 10 and a.benchmark_name == "libero"
    assert b.benchmark_name == "libero-plus" and b.MULTIRUN.task_sample_ratio == 0.15 and b.EVALUATION.num_trials == 1
    assert b.EVALUATION.replan_steps == 10 and len(b.MULTIRUN.task_suite_names) == 4


@pytest.mark.skipif(not SMOKE, reason="no smoke checkpoint")
def test_load_eval_model_from_smoke_checkpoint():
    model, run_cfg, state = load_eval_model(SMOKE[-1], device="cpu")
    assert state["step"] == int(SMOKE[-1].name.split("_")[1])
    assert run_cfg.model.tiny and not model.dit.training
    out = model.infer_action_one_pass_future_cache(
        input_image=torch.rand(1, 3, 224, 448) * 2 - 1, proprio=torch.randn(1, 8),
        context=torch.randn(1, 64, 5120), context_mask=torch.ones(1, 64, dtype=torch.bool), num_inference_steps=2, seed=0)
    assert out["action"].shape == (32, 7) and torch.isfinite(out["action"]).all()


def test_facade_provides_every_model_attribute_the_libero_loop_uses():
    import ast
    src = (ROOT / "third_party/FasterWAM/experiments/libero/eval_libero_single.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_predict_action_chunk")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "model"}
    from wam_h3.model.wam import WAMH3
    missing = {a for a in attrs if a not in ("infer_joint",) and not hasattr(WAMH3, a) and a != "torch_dtype"}
    assert "infer_action" in attrs and not missing, missing
