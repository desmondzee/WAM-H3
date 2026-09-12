import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.cache import load_policy_cache
from wam_h3.refined.runtime import configure


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/refined_action_policy/feature_smoke.json"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    configure(args.device)
    from wam_h3.refined.backbone import FrozenFeatures
    data = load_policy_cache(args.input)
    extractor = FrozenFeatures()
    report = {"cache_identity": data["identity"], "shapes": [], "causal_checks": []}
    latent = data["latent"]
    for t in sorted({2, 12, 22, 32, 42, latent.shape[2]}):
        real = t <= latent.shape[2]
        z = latent[:, :, :t] if real else torch.cat([latent, latent[:, :, -1:].expand(-1, -1, t - latent.shape[2], -1, -1)], 2)
        for _ in range(2):
            features, times = extractor(z, data["conditioning"], data["tags"])
            del features, times
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings, idle = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            features, times = extractor(z, data["conditioning"], data["tags"])
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            capture_bytes = sum(f.numel() * f.element_size() for f in features)
            del features, times
            idle.append(torch.cuda.memory_allocated())
        item = dict(latents=t, frames=5 + (t - 2) // 5 * 17, real_prefix=real,
                    seconds=timings, capture_bytes=capture_bytes, idle_allocated=idle,
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                    free_bytes=torch.cuda.mem_get_info()[0])
        report["shapes"].append(item)
        print(json.dumps(item), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        if item["free_bytes"] < 8 * 1024 ** 3:
            raise RuntimeError("Feature extraction has less than 8 GiB memory headroom")
        if max(idle) - min(idle) > 64 * 1024 ** 2:
            raise RuntimeError("Live feature-extraction allocations grew after warmup")
    full, _ = extractor(latent, data["conditioning"], data["tags"])
    reference = [f.cpu() for f in full]
    del full
    for t in sorted({2, 12, max(2, latent.shape[2] - 10)}):
        if t >= latent.shape[2]:
            continue
        prefix, _ = extractor(latent[:, :, :t], data["conditioning"], data["tags"])
        prefix = [f.cpu() for f in prefix]
        changed = latent.clone()
        changed[:, :, t:] = torch.randn_like(changed[:, :, t:]) * 10
        perturbed, _ = extractor(changed, data["conditioning"], data["tags"])
        errors = []
        for original, short, altered in zip(reference, prefix, perturbed):
            n = short.shape[0]
            torch.testing.assert_close(original[:n], short, atol=0, rtol=0)
            torch.testing.assert_close(original[:n], altered[:n].cpu(), atol=0, rtol=0)
            errors.append((original[:n] - short).abs().max().item())
        report["causal_checks"].append({"latents": t, "prefix_max_abs": errors, "future_invariant": True})
        del perturbed
    report["frozen"] = all(p.grad is None and not p.requires_grad for p in extractor.model.parameters())
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
