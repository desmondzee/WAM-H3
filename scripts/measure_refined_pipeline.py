import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.cache import load_policy_cache
from wam_h3.refined.pipeline import feature_batches, to_device, to_host
from wam_h3.refined.policy import ActionPolicy
from wam_h3.refined.runtime import configure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/refined_action_policy/pipeline_profile.json"))
    args = ap.parse_args()
    configure()
    from wam_h3.refined.backbone import FrozenFeatures
    sample = load_policy_cache(args.input)
    extractor = FrozenFeatures()
    policy = ActionPolicy().to("cuda:1")
    features, _ = extractor(sample["latent"], sample["conditioning"], sample["tags"])
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    host = to_host(features, torch.device("cuda:0"))
    d2h_seconds = time.perf_counter() - start
    start = time.perf_counter()
    transferred = to_device(host, torch.device("cuda:1"))
    h2d_seconds = time.perf_counter() - start
    for reference, actual in zip(host, transferred):
        torch.testing.assert_close(reference, actual.cpu(), atol=0, rtol=0)
    report = dict(cache_identity=sample["identity"], transfer_bytes=sum(f.numel() * f.element_size() for f in features),
                  d2h_seconds=d2h_seconds, h2d_seconds=h2d_seconds, transfer_exact=True, optimizer_updates=0, modes={})
    del features, transferred, host
    for mode, overlap, steps in (("warmup", False, 2), ("serial", False, 5), ("overlap", True, 5)):
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        start = time.perf_counter()
        timings = []
        for features, times, data in feature_batches(extractor, [sample], steps, torch.device("cuda:0"), torch.device("cuda:1"), overlap, timings):
            policy_start = time.perf_counter()
            policy.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = policy(features, times, data["cutoffs"].to("cuda:1"))
                loss = prediction.float().square().mean()
            loss.backward()
            torch.cuda.synchronize(1)
            timings[-1]["policy_forward_backward_seconds"] = time.perf_counter() - policy_start
            del prediction, loss, features
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        report["modes"][mode] = dict(steps=steps, seconds=time.perf_counter() - start, timings=timings)
        print(mode, report["modes"][mode], flush=True)
    report.update(source_peak=torch.cuda.max_memory_allocated(0), target_peak=torch.cuda.max_memory_allocated(1),
                  source_free_bytes=torch.cuda.mem_get_info(0)[0], target_free_bytes=torch.cuda.mem_get_info(1)[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
