import gc
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
COMFY_REVISION = "6e3c0bda5a756ec334df449cdc7d4a4685631e91"
MODEL_REVISION = "a98869194787969724c7425d95d0ed73ce9202af"
MODELS = {
    "backbone": ("diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors", "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5"),
    "qwen": ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"),
    "vae": ("vae/minimax_h3_video_vae_fp16.safetensors", "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
}


def configure(device=0):
    path = ROOT / ".runtime/ComfyUI"
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMFY_REVISION:
        raise RuntimeError(f"Expected ComfyUI {COMFY_REVISION}, found {revision}")
    sys.path.insert(0, str(path))
    from comfy.cli_args import args
    args.enable_triton_backend = True
    args.disable_cuda_graphs = True
    args.disable_comfy_compiler = True
    args.use_pytorch_cross_attention = True
    args.reserve_vram = 8.0
    torch.cuda.set_device(device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import comfy.quant_ops
    comfy.quant_ops.ck.disable_backend("cuda")
    comfy.quant_ops.ck.set_backend_priority(["triton", "eager"])
    if not comfy.quant_ops.ck.list_backends()["triton"]["available"]:
        raise RuntimeError("Accelerated INT8 Triton backend is required")


def model_path(kind):
    return ROOT / ".runtime/refined-models" / MODELS[kind][0]


def verify_models():
    report = {}
    for kind, (_, expected) in MODELS.items():
        path = model_path(kind)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual}")
        report[kind] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    return report


def load_qwen():
    import comfy.sd
    return comfy.sd.load_clip([str(model_path("qwen"))], clip_type=comfy.sd.CLIPType.MINIMAX,
                              model_options={"dtype": torch.bfloat16}, disable_dynamic=True)


def load_vae():
    import comfy.sd
    import comfy.utils
    state, metadata = comfy.utils.load_torch_file(str(model_path("vae")), return_metadata=True)
    return comfy.sd.VAE(sd=state, metadata=metadata, dtype=torch.float16)


def load_backbone():
    import comfy.sd
    return comfy.sd.load_diffusion_model(str(model_path("backbone")),
                                        model_options={"dtype": torch.bfloat16}, disable_dynamic=True)


def unload():
    import comfy.model_management
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()
