import torch
import torch.nn.functional as F


def _masked_mean(per_row, is_pad):
    valid = (~is_pad).to(per_row.dtype)
    return (per_row * valid).sum(1) / valid.sum(1).clamp(min=1.0)


def training_loss(dit, batch, sched_v, sched_a, generator=None, full_noise_prob=0.0):
    cfg = getattr(dit, "module", dit).cfg
    video, action = batch["video_rows"], batch["action"]
    B, dev = video.shape[0], video.device
    sv, sa = sched_v.sample_sigma(B, dev, generator), sched_a.sample_sigma(B, dev, generator)
    if full_noise_prob > 0:
        sv = torch.where(torch.rand(B, device=dev, generator=generator) < full_noise_prob, torch.ones_like(sv), sv)
    nv = torch.randn(video.shape, device=dev, generator=generator, dtype=video.dtype)
    na = torch.randn(action.shape, device=dev, generator=generator, dtype=action.dtype)
    t_groups = torch.stack([torch.ones_like(sv), torch.full_like(sv, cfg.obs_t), 1 - sv, 1 - sa], dim=1)
    out = dit(batch["text"], batch["text_valid"], batch["obs_rows"], batch.get("proprio"),
              sched_a.add_noise(action, na, sa), sched_v.add_noise(video, nv, sv), t_groups=t_groups)
    lv = F.mse_loss(-out.video.float(), sched_v.training_target(video, nv).float(), reduction="none").mean(-1)
    la = F.mse_loss(-out.action.float(), sched_a.training_target(action, na).float(), reduction="none").mean(-1)
    F_ = cfg.frame_rows
    lv = _masked_mean(lv[:, F_:], batch["image_is_pad"][:, 1:].repeat_interleave(F_, dim=1)) * sched_v.training_weight(sv)
    full = sv >= 1.0
    lv_full = lv[full].mean().item() if full.any() else float("nan")
    lv = lv.mean()
    la = (_masked_mean(la, batch["action_is_pad"]) * sched_a.training_weight(sa)).mean()
    return lv + la, {"loss_video": lv.item(), "loss_action": la.item(), "loss_video_full_noise": lv_full}
