import torch

from tests.reference import minimax_h3_video_vae as refmod
from wam_h3.model.vae import VideoEncoder

TINY = dict(ch=32, ch_mult=(1, 1, 1, 1, 1, 1), num_res_blocks=1)


def pair():
    refmod.ViT3DDecoder = lambda **kw: torch.nn.Identity()
    ref = refmod.MiniMaxH3VideoVAE(**TINY)
    enc = VideoEncoder(**TINY)
    sd = {k: v for k, v in ref.state_dict().items() if k.startswith(("encoder.", "quant_conv."))}
    assert enc.load_state_dict(sd, strict=True)
    return ref, enc


def test_encode_clip_matches_reference():
    ref, enc = pair()
    video = torch.rand(1, 3, 5, 64, 96)
    with torch.no_grad():
        exp = ref.encode_video(video)
        out = enc.encode_clip(video)
    assert out.shape == (1, 24, 2, 4, 6)
    assert torch.allclose(out, exp, atol=1e-5)


def test_encode_image_matches_reference():
    ref, enc = pair()
    img = torch.rand(2, 3, 64, 96)
    with torch.no_grad():
        exp = ref.encode_video(img, process_image=True)
        out = enc.encode_image(img)
    assert out.shape == (2, 24, 1, 4, 6)
    assert torch.allclose(out, exp, atol=1e-5)


def test_encode_tiles_wide_frames_like_reference():
    ref, enc = pair()
    enc.tile_size, enc.tile_overlap_min = 64, 16
    video = torch.rand(1, 3, 5, 64, 160)
    with torch.no_grad():
        assert torch.allclose(enc.encode_clip(video), ref.encode_video(video, tile_size=64, tile_overlap=16), atol=1e-5)


def test_load_weights_reads_only_encoder_keys(tmp_path):
    from safetensors.torch import save_file
    ref, _ = pair()
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(tmp_path / "model.safetensors"))
    enc = VideoEncoder.from_safetensors(tmp_path / "model.safetensors", **TINY)
    assert torch.equal(enc.encoder.conv_in.weight, ref.encoder.conv_in.weight)
    assert not any(k.startswith("decoder") for k in enc.state_dict())
