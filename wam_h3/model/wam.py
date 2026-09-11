import json
from pathlib import Path

import torch
import torch.nn as nn

from ..data.text_cache import load_embedding, save_embedding
from .checkpoint import load_trainable, save_trainable
from .dit import patchify_video
from .flow import FlowSchedule
from .loss import training_loss
from .text_encoder import collate_instructions


class WAMH3(nn.Module):
    def __init__(self, cfg, dit, vae, text_cache_dir=None, text_encoder=None, shift=5.0, video_weight_center=1.0,
                 video_full_noise_prob=0.3):
        super().__init__()
        self.cfg, self.dit, self.vae = cfg, dit, vae
        self.text_cache_dir, self.text_encoder = text_cache_dir, text_encoder
        self.sched_v = FlowSchedule(shift, weight_center=video_weight_center, subtract_min=False)
        self.sched_a = FlowSchedule(shift)
        self.video_full_noise_prob = video_full_noise_prob
        self._prompts = {}

    @property
    def device(self):
        return self.dit.condition_proj.weight.device

    @property
    def torch_dtype(self):
        return self.dit.condition_proj.weight.dtype

    def encode_prompt(self, prompt):
        if prompt not in self._prompts:
            emb = load_embedding(self.text_cache_dir, prompt) if self.text_cache_dir else None
            if emb is None:
                if self.text_encoder is None:
                    raise RuntimeError(f"no cached embedding for {prompt!r}; run scripts/precompute_text_embeds.py")
                emb = self.text_encoder.encode([prompt])[0]
                if self.text_cache_dir:
                    save_embedding(self.text_cache_dir, prompt, emb)
            self._prompts[prompt] = collate_instructions([emb], self.cfg.text_len)
        return self._prompts[prompt]

    def build_inputs(self, s, video=True):
        cfg, dev, dt = self.cfg, self.device, self.torch_dtype
        frames = (s["video"].to(dev, torch.float32) + 1) / 2
        out = dict(text=s["context"].to(dev, dt), text_valid=s["context_mask"].to(dev).bool(),
                   obs_rows=patchify_video(self.vae.encode_image(frames[:, :, 0])).to(dt))
        if video:
            out["video_rows"] = patchify_video(self.vae.encode_clip(frames)).to(dt)
        p = s.get("proprio")
        out["proprio"] = None if p is None else (p[:, 0] if p.ndim == 3 else p).to(dev, dt)
        if "action" in s:
            out["action"] = s["action"].to(dev, dt)
        if "image_is_pad" in s:
            pad = s["image_is_pad"].to(dev)
            out["image_is_pad"] = torch.cat([pad[:, :1], pad[:, 1:].view(pad.shape[0], cfg.num_video_latents - 1, 4).all(-1)], 1)
        if "action_is_pad" in s:
            out["action_is_pad"] = s["action_is_pad"].to(dev)
        return out

    def training_loss(self, s):
        return training_loss(self.dit, self.build_inputs(s), self.sched_v, self.sched_a, full_noise_prob=self.video_full_noise_prob)

    @torch.no_grad()
    def infer_action_one_pass_future_cache(self, input_image, proprio=None, prompt=None, context=None, context_mask=None,
                                           action_horizon=None, num_inference_steps=10, seed=None, sigma_shift=None, **_):
        cfg = self.cfg
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        s = dict(video=input_image[:, :, None], context=context, context_mask=context_mask, proprio=proprio)
        inp = self.build_inputs(s, video=False)
        g = None if seed is None else torch.Generator().manual_seed(seed)
        noise = torch.randn(1, cfg.num_video_latents * cfg.frame_rows, cfg.video_patch_dim, generator=g).to(self.device, self.torch_dtype)
        sig, _ = self.sched_a.inference_schedule(num_inference_steps, sigma_shift)
        tg = torch.tensor([[1.0, cfg.obs_t, 1 - sig[0].item(), 0.0]], device=self.device)
        with torch.autocast(self.device.type, dtype=self.torch_dtype, enabled=self.torch_dtype != torch.float32):
            cache = self.dit.prefill(inp["text"], inp["text_valid"], inp["obs_rows"], inp["proprio"], noise, tg)
            a = self.dit.denoise_actions(cache, self.sched_a, num_inference_steps, generator=g, shift=sigma_shift)
        return {"action": a[0].float().cpu()}

    def save_checkpoint(self, path, step, extra=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_trainable(self.dit, path / "adapter.safetensors")
        (path / "trainer_state.json").write_text(json.dumps({"step": step, **(extra or {})}, indent=1))

    def load_checkpoint(self, path):
        path = Path(path)
        load_trainable(self.dit, path / "adapter.safetensors")
        return json.loads((path / "trainer_state.json").read_text())
