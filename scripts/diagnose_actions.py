#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasterwam.utils import misc
from wam_h3.eval.policy import load_eval_model, run_dir_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, run_cfg, state = load_eval_model(a.ckpt, dev)
    misc.register_work_dir(run_dir_of(a.ckpt))
    ds = instantiate(run_cfg.data.train)
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(len(ds), generator=g)[: a.n].tolist()
    gt, pred, zero, gt_all = [], [], [], []
    for i in idx:
        s = ds[i]
        out = model.infer_action_one_pass_future_cache(
            input_image=s["video"][:, 0][None], proprio=s["proprio"][None], context=s["context"][None],
            context_mask=s["context_mask"][None], num_inference_steps=a.steps, seed=a.seed)
        gt.append(s["action"]); pred.append(out["action"]); gt_all.append(s["action"])
    gt, pred = torch.stack(gt), torch.stack(pred)
    mean = gt.mean(dim=(0, 1), keepdim=True)
    mse = ((pred - gt) ** 2).mean(dim=(0, 1))
    base = ((mean - gt) ** 2).mean(dim=(0, 1))
    r = {
        "ckpt": a.ckpt, "step": state.get("step"), "n": a.n,
        "mse_per_dim": mse.tolist(), "mean_baseline_per_dim": base.tolist(),
        "mse_over_baseline": (mse.sum() / base.sum()).item(),
        "pred_std_across_samples_per_dim": pred.std(dim=0).mean(0).tolist(),
        "gt_std_across_samples_per_dim": gt.std(dim=0).mean(0).tolist(),
        "gripper_sign_acc": ((pred[..., -1] > 0) == (gt[..., -1] > 0)).float().mean().item(),
        "corr_per_dim": [torch.corrcoef(torch.stack([pred[..., d].flatten(), gt[..., d].flatten()]))[0, 1].item() for d in range(gt.shape[-1])],
        "pred_first_vs_last_step_change": (pred[:, -1] - pred[:, 0]).abs().mean().item(),
    }
    print(json.dumps(r, indent=1))
    Path(a.ckpt, "diagnose_actions.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
