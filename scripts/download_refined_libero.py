import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_h3.refined.data import ALIGNMENT_EVIDENCE, LIBERO_REVISION, OFFICIAL_REPO, OFFICIAL_SUITES


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/libero_official"))
    parser.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parents[1] / "third_party/LIBERO")
    parser.add_argument("--suites", nargs="+", choices=(*OFFICIAL_SUITES, "all"), default=["libero_spatial"])
    args = parser.parse_args()
    suites = OFFICIAL_SUITES if "all" in args.suites else tuple(dict.fromkeys(args.suites))
    if (args.output / "provenance.json").exists():
        existing = json.loads((args.output / "provenance.json").read_text())
        existing_suites = {record["path"].split("/")[0] for record in existing["files"]}
        if existing_suites != set(suites):
            raise ValueError("Use a new output directory when changing suites to preserve episode identities")
    upstream = args.upstream
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != LIBERO_REVISION:
        raise ValueError("Re-audit upstream recording code before changing the source revision")
    downloader = upstream / "libero/libero/utils/download_utils.py"
    if f'HF_REPO_ID = "{OFFICIAL_REPO}"' not in downloader.read_text():
        raise ValueError("Dataset repository is not referenced by the official downloader")
    info = HfApi().dataset_info(OFFICIAL_REPO, revision="f13aa24a3da8c43c7225569f28c562979fa0e35a", files_metadata=True)
    files = sorted((f for f in info.siblings if f.rfilename.split("/")[0] in suites and f.rfilename.endswith(".hdf5")), key=lambda f: f.rfilename)
    for suite in suites:
        expected = 90 if suite == "libero_90" else 10
        if sum(f.rfilename.startswith(suite + "/") for f in files) != expected:
            raise ValueError(f"Expected {expected} official task files for {suite}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, source="official_libero", repo_id=OFFICIAL_REPO, revision=info.sha,
                    upstream_revision=revision, alignment=ALIGNMENT_EVIDENCE,
                    downloader_url=f"https://github.com/Lifelong-Robot-Learning/LIBERO/blob/{revision}/libero/libero/utils/download_utils.py#L108",
                    original_archive_url="https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
                    distribution="official upstream HF mirror; original Box archive returned HTTP 404", files=[], episodes=[])
    for item in files:
        path = Path(hf_hub_download(OFFICIAL_REPO, item.rfilename, repo_type="dataset", revision=info.sha, local_dir=args.output))
        digest = sha256(path)
        if item.lfs is None or digest != item.lfs.sha256:
            raise ValueError(f"Official LFS hash mismatch: {item.rfilename}")
        manifest["files"].append(dict(path=item.rfilename, sha256=digest, size=path.stat().st_size,
                                      url=hf_hub_url(OFFICIAL_REPO, item.rfilename, repo_type="dataset", revision=info.sha)))
        import h5py
        with h5py.File(path, "r") as data:
            for demo in sorted(data["data"], key=lambda name: int(name.removeprefix("demo_"))):
                manifest["episodes"].append(dict(path=item.rfilename, demo=demo))
        temporary = args.output / "provenance.partial.json"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Verified {item.rfilename}: {digest}", flush=True)
    manifest["smoke_episodes"] = [0, 1, 2, 3]
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {len(manifest['episodes'])} official episodes; smoke indices 0,1,2,3", flush=True)


if __name__ == "__main__":
    main()
