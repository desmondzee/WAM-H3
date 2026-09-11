import torch


def _phi(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


class FlowSchedule:
    def __init__(self, shift=5.0, num_train_timesteps=1000, eps=1e-10):
        self.shift, self.n, self.eps = float(shift), int(num_train_timesteps), eps
        u = torch.linspace(1.0, 0.0, self.n + 1, dtype=torch.float64)[:-1]
        y = self._bump(_phi(u, self.shift))
        self.y_min = float(y.min())
        self.norm = float((y - self.y_min).mean())

    def _bump(self, sigma):
        return torch.exp(-2.0 * (sigma.to(torch.float64) - 0.5) ** 2)

    def sample_sigma(self, B, device, generator=None):
        return _phi(torch.rand(B, device=device, generator=generator), self.shift)

    def training_weight(self, sigma):
        return ((self._bump(sigma) - self.y_min) / (self.norm + self.eps)).to(torch.float32)

    @staticmethod
    def add_noise(x, noise, sigma):
        s = sigma.view(-1, *([1] * (x.ndim - 1))).to(x.dtype)
        return (1 - s) * x + s * noise

    @staticmethod
    def training_target(x, noise):
        return noise - x

    def inference_schedule(self, steps, shift=None):
        s = _phi(torch.linspace(1.0, 0.0, steps + 1), self.shift if shift is None else float(shift))
        return s[:-1], s[1:] - s[:-1]

    @staticmethod
    def step(pred, delta, x):
        return x + pred * delta.to(x.dtype)
