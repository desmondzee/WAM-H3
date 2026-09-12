from types import MethodType

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .layout import availability_times
from .runtime import load_backbone


def _fixed_tile_attention(q, k, v, block_mask):
    if not torch.compiler.is_compiling():
        raise RuntimeError("Frozen attention requires compiled fixed-tile execution")
    return flex_attention(q, k, v, block_mask=block_mask,
                          kernel_options={"BACKEND": "TRITON", "BLOCK_M": 64, "BLOCK_N": 64})


def _compile_attention(backend="inductor"):
    return torch.compile(_fixed_tile_attention, backend=backend, dynamic=True, fullgraph=True)


class FrozenFeatures:
    def __init__(self, taps=(7, 15, 23, 31, 39, 49)):
        import comfy.model_management as mm
        self.patcher = load_backbone()
        mm.load_models_gpu([self.patcher], force_full_load=True)
        self.model = self.patcher.model.diffusion_model
        self.model.eval().requires_grad_(False)
        self.taps = tuple(taps)
        self.captures = {}
        self.mask = None
        self.signature = None
        self.attend = _compile_attention()
        for block in self.model.blocks:
            block.attn.forward = MethodType(self._attention, block.attn)
        self.handles = [self.model.blocks[i].register_forward_hook(self._capture(i)) for i in taps]

    def _capture(self, index):
        def capture(module, inputs, output):
            self.captures[index] = output[self.video_start:].detach().clone()
        return capture

    def _attention(self, attention, x, rope_freqs=None, transformer_options=None):
        import comfy.model_management as mm
        import comfy.quant_ops as qo
        s, heads, dim = x.shape[0], attention.heads, attention.head_dim
        q, k, v = attention.qkv_proj(x).split(heads * dim, dim=-1)
        q, k = q.view(1, s, heads, dim), k.view(1, s, heads, dim)
        q, k = qo.ck.rms_rope_split_half(q, k, rope_freqs,
                                        mm.cast_to(attention.q_norm.weight, device=x.device),
                                        mm.cast_to(attention.k_norm.weight, device=x.device),
                                        epsilon=attention.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
        v = v.view(1, s, heads, dim)
        out = self.attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=self.mask)
        return attention.out_proj(out.transpose(1, 2).reshape(s, heads * dim))

    @torch.no_grad()
    def __call__(self, latent, conditioning, tags, sigma=0.2, seed=0):
        if not 0 < sigma < 1:
            raise ValueError("Feature sigma must be between zero and one")
        device = next(self.model.parameters()).device
        latent = latent.to(device=device, dtype=torch.bfloat16)
        conditioning = conditioning.to(device=device, dtype=torch.bfloat16)
        _, channels, t, h, w = latent.shape
        frame_tokens = (h // 2) * (w // 2)
        self.video_start = conditioning.shape[1]
        n = self.video_start + t * frame_tokens
        signature = (self.video_start, t, frame_tokens, device)
        if signature != self.signature:
            text = self.video_start
            def allowed(b, head, q, k):
                return (k < text) | ((q >= text) & (k >= text) & ((k - text) // frame_tokens <= (q - text) // frame_tokens))
            raw_mask = create_block_mask(allowed, 1, 1, n, n, device=str(device))
            blocks = raw_mask.to_dense().bool()
            counts = blocks.sum(-1).to(torch.int32)
            indices = blocks.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            self.mask = BlockMask.from_kv_blocks(counts, indices, BLOCK_SIZE=raw_mask.BLOCK_SIZE,
                                                mask_mod=allowed, seq_lengths=(n, n))
            self.signature = signature
        noise = torch.stack([torch.randn(channels, h, w, generator=torch.Generator().manual_seed(seed + i))
                             for i in range(t)], dim=1).unsqueeze(0).to(latent)
        noisy = (1 - sigma) * latent + sigma * noise
        audio = torch.empty(1, 32, 2, 0, device=device, dtype=latent.dtype)
        self.captures.clear()
        try:
            self.model([noisy, audio], torch.tensor([sigma * 1000], device=device), conditioning,
                       transformer_options={}, minimax_payload={"text_token_tags": tags.to(device)})
            features = [self.captures[i] for i in self.taps]
            if any(not torch.isfinite(f).all() for f in features):
                raise RuntimeError("Nonfinite frozen features")
            return features, availability_times(t, frame_tokens, device)
        finally:
            self.captures.clear()
