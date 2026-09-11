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
from wam_h3.model.layout import SequenceLayout
from wam_h3.model.loss import training_loss


def open_loop(model, samples, steps, seed):
    gt, pred = [], []
    for s in samples:
        out = model.infer_action_one_pass_future_cache(
            input_image=s["video"][:, 0][None], proprio=s["proprio"][None], context=s["context"][None],
            context_mask=s["context_mask"][None], num_inference_steps=steps, seed=seed)
        gt.append(s["action"]); pred.append(out["action"])
    gt, pred = torch.stack(gt), torch.stack(pred)
    base = ((gt.mean(dim=(0, 1), keepdim=True) - gt) ** 2).mean()
    return dict(
        mse_over_baseline=(((pred - gt) ** 2).mean() / base).item(),
        corr_per_dim=[torch.corrcoef(torch.stack([pred[..., d].flatten(), gt[..., d].flatten()]))[0, 1].item() for d in range(gt.shape[-1])],
        gripper_sign_acc=((pred[..., -1] > 0) == (gt[..., -1] > 0)).float().mean().item(),
        pred_std=pred.std(dim=0).mean().item())


def with_layout(model, **kw):
    from dataclasses import replace
    dit = model.dit
    cfg = replace(dit.cfg, **kw)
    old = dit.layout
    dit.layout = SequenceLayout(cfg)
    dit.init_buffers(model.device)
    return old


def restore(model, old):
    model.dit.layout = old
    model.dit.init_buffers(model.device)


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
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(a.seed))[: a.n].tolist()
    samples = [ds[i] for i in idx]
    r = {"ckpt": a.ckpt, "step": state.get("step"), "n": a.n, "baseline": open_loop(model, samples, a.steps, a.seed)}

    old = with_layout(model, action_sees_video=False)
    r["ablate_action_sees_video"] = open_loop(model, samples, a.steps, a.seed)
    restore(model, old)

    tag = model.dit.row_tag.clone()
    model.dit.row_tag[model.dit.layout.action] = 0
    r["action_rows_with_video_adaln_slot"] = open_loop(model, samples, a.steps, a.seed)
    model.dit.row_tag.copy_(tag)

    batch = {k: torch.stack([s[k] for s in samples[:8]]) for k in ("video", "action", "proprio", "context", "context_mask", "image_is_pad", "action_is_pad")}
    inp = model.build_inputs(batch)
    sweep = {}
    with torch.no_grad(), torch.autocast(model.device.type, dtype=model.torch_dtype, enabled=model.torch_dtype != torch.float32):
        for sa in (1.0, 0.9, 0.7, 0.5, 0.3, 0.1):
            g = torch.Generator(device=dev).manual_seed(a.seed)
            fixed = type(model.sched_a)(model.sched_a.shift, subtract_min=False)
            fixed.sample_sigma = lambda B, device, generator=None, _s=sa: torch.full((B,), _s, device=device)
            _, parts = training_loss(model.dit, inp, fixed, fixed, generator=g, full_noise_prob=0.0)
            w = fixed.training_weight(torch.tensor([sa]))[0].item()
            sweep[str(sa)] = dict(action=parts["loss_action"] / w, video=parts["loss_video"] / w)
    r["loss_at_fixed_sigma_unweighted"] = sweep

    drift = {}
    for n, p in model.dit.named_parameters():
        if ".lora_B.default" in n and "adaln_proj.linear" in n:
            blk = n.split(".")[1]
            A = model.dit.get_parameter(n.replace("lora_B", "lora_A"))
            d = (p.float() @ A.float()).view(3, -1).norm(dim=1)
            drift[blk] = [x.item() for x in d]
    r["adaln_lora_delta_norm_per_slot_video_text_audio"] = {k: drift[k] for k in sorted(drift, key=int)[:: max(1, len(drift) // 10)]}
    print(json.dumps(r, indent=1))
    Path(a.ckpt, "diagnose_actions.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
