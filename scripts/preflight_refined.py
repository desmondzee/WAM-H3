import argparse
import json
import time
from pathlib import Path

import comfy_kitchen as ck
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ck.disable_backend("cuda")
    torch.manual_seed(0)
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "backend": "triton", "devices": []}
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_device(device)
        x = torch.randn(576, 5376, device=device, dtype=torch.bfloat16)
        weight = torch.randint(-127, 128, (7168, 5376), device=device, dtype=torch.int8)
        scale = torch.tensor(0.001, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with ck.use_backend("triton"):
            actual = ck.int8_linear(x, weight, scale, convrot=True)
            for _ in range(2):
                ck.int8_linear(x, weight, scale, convrot=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(5):
                ck.int8_linear(x, weight, scale, convrot=True)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) / 5
        with ck.use_backend("eager"):
            expected = ck.int8_linear(x, weight, scale, convrot=True)
        relative_rmse = ((actual.float() - expected.float()).square().mean() / expected.float().square().mean()).sqrt().item()
        if not torch.isfinite(actual).all() or relative_rmse > 0.03:
            raise RuntimeError(f"ConvRot numerical check failed: {relative_rmse=}")
        report["devices"].append({"index": device, "name": torch.cuda.get_device_name(device),
                                  "seconds_per_linear": elapsed, "relative_rmse": relative_rmse,
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
        del x, weight, scale, actual, expected
    report["p2p"] = torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
