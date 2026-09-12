import torch


def availability_times(latents, frame_tokens=288, device=None):
    j = torch.arange(latents, device=device)
    return (17 * (j // 5) + 4 * (j % 5) - 4).clamp_min(0).repeat_interleave(frame_tokens)


def causal_mask(text_tokens, latents, frame_tokens, device=None):
    times = torch.cat([torch.full((text_tokens,), -1, device=device),
                       torch.arange(latents, device=device).repeat_interleave(frame_tokens)])
    return times[None, :] <= times[:, None]


def policy_mask(memory_times, cutoffs):
    return memory_times[None, :] <= cutoffs[:, None]
