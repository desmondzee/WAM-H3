import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.runtime import configure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "sample", "decode"])
    ap.add_argument("--dataset", default="data/libero_official")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", choices=["minimal", "camera", "constraints", "physical"], default=["camera"])
    ap.add_argument("--output", type=Path, default=Path("runs/refined_generation/official_smoke"))
    ap.add_argument("--input", type=Path)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    configure(args.device)
    from wam_h3.refined.generation import decode, prepare, sample
    if args.stage == "prepare":
        prepare(args.dataset, args.episodes, args.variants, args.output)
    elif args.input is None:
        ap.error("--input is required for sampling and decoding")
    elif args.stage == "sample":
        sample(args.input, args.steps, args.seed)
    else:
        decode(args.input)


if __name__ == "__main__":
    main()
