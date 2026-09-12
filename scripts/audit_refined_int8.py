import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.cache import load_policy_cache
from wam_h3.refined.runtime import configure


def difference(a, b):
    delta = a.float() - b.float()
    return {"max_abs": delta.abs().max().item(), "relative_rmse": (delta.square().mean() / a.float().square().mean().clamp_min(1e-20)).sqrt().item(),
            "different_fraction": (delta != 0).float().mean().item(), "reference_rms": a.float().square().mean().sqrt().item()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/refined_action_policy/int8_audit.json"))
    args = ap.parse_args()
    configure()
    from wam_h3.refined.backbone import FrozenFeatures
    data = load_policy_cache(args.input)
    extractor = FrozenFeatures(taps=(0, 1, 2, 7, 15, 23, 31, 39, 49))
    rows = data["conditioning"].shape[1] + 576
    snapshots = {}
    def hook(name):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            snapshots[name] = value[:576 if name == "video_patch_proj" else rows].detach().float().cpu().clone()
        return capture
    for name in ("video_patch_proj", "condition_proj", "blocks.0.norm1", "blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2"):
        extractor.model.get_submodule(name).register_forward_hook(hook(name))
    def input_hook(module, inputs):
        snapshots["attention_output_before_projection"] = inputs[0][:rows].detach().float().cpu().clone()
    extractor.model.blocks[0].attn.out_proj.register_forward_pre_hook(input_hook)
    original_attend = extractor.attend
    def audit_attention(q, k, v, **kwargs):
        if "attention_q" not in snapshots:
            for name, value in (("attention_q", q), ("attention_k", k), ("attention_v", v)):
                snapshots[name] = value[:, :, :rows].detach().float().cpu().clone()
        return original_attend(q, k, v, **kwargs)
    extractor.attend = audit_attention
    def run(z):
        features, _ = extractor(z, data["conditioning"], data["tags"])
        result = {**snapshots, **{f"tap_{i}": f[:576].float().cpu().clone() for i, f in zip(extractor.taps, features)}}
        snapshots.clear()
        return result
    baseline = run(data["latent"])
    cases = {"same_full": data["latent"], "prefix": data["latent"][:, :, :2]}
    altered = data["latent"].clone()
    altered[:, :, 2:] = torch.randn_like(altered[:, :, 2:]) * 10
    cases["changed_future"] = altered
    report = {}
    for label, latent in cases.items():
        candidate = run(latent)
        report[label] = {name: difference(base[:candidate[name].shape[0]], candidate[name]) for name, base in baseline.items()}
        print(label, json.dumps(report[label], indent=2), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
