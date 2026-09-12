import torch


class LossAccumulator:
    """Detached diagnostics for one optimizer batch or complete validation pass.

    Update once per episode, or with disjoint slices and their chunk_start.
    Each chunk position must occur at most once per observed episode. Empty
    padded chunks do not count; unobserved losses are omitted, counts are zero.
    """

    def __init__(self, scale):
        self.scale_squared = scale.detach().float().square()
        if self.scale_squared.shape != (7,):
            raise ValueError("Action normalizer scale must have seven components")
        self.chunk_sums = self.scale_squared.new_zeros(0)
        self.chunk_counts = self.scale_squared.new_zeros(0)
        self.horizon_sums = self.scale_squared.new_zeros(0)
        self.horizon_counts = self.scale_squared.new_zeros(0)
        self.component_sums = self.scale_squared.new_zeros(7)

    @staticmethod
    def _grow(values, size):
        if size > len(values):
            return torch.cat((values, values.new_zeros(size - len(values))))
        return values

    @torch.no_grad()
    def update(self, prediction, target, valid, chunk_start=0):
        if (prediction.ndim != 3 or prediction.shape != target.shape
                or prediction.shape[-1] != 7 or valid.shape != prediction.shape[:-1]
                or valid.dtype != torch.bool or chunk_start < 0):
            raise ValueError("Expected chunk/horizon/seven-component predictions and boolean validity")
        invalid = ~valid.unsqueeze(-1)
        predicted = prediction.detach().float().masked_fill(invalid, 0)
        targets = target.detach().float().masked_fill(invalid, 0)
        error = (predicted - targets).square()
        counts = valid.sum(-1)
        components = error.sum(1) / counts.clamp_min(1).unsqueeze(-1)
        end = chunk_start + len(prediction)
        self.chunk_sums = self._grow(self.chunk_sums, end)
        self.chunk_counts = self._grow(self.chunk_counts, end)
        self.chunk_sums[chunk_start:end].add_(components.mean(-1))
        self.chunk_counts[chunk_start:end].add_(counts > 0)
        horizon = prediction.shape[1]
        self.horizon_sums = self._grow(self.horizon_sums, horizon)
        self.horizon_counts = self._grow(self.horizon_counts, horizon)
        self.horizon_sums[:horizon].add_(error.mean(-1).sum(0))
        self.horizon_counts[:horizon].add_(valid.sum(0))
        self.component_sums.add_(components.sum(0))

    @torch.no_grad()
    def finish(self, prefix):
        """Return scalars with a single device-to-CPU transfer; retain no inputs."""
        chunks = self.chunk_counts.sum()
        components = self.component_sums / chunks.clamp_min(1)
        raw_components = components * self.scale_squared
        names, values = [], []
        for axis, label, sums, counts in (
            ("chunk", "action_chunk", self.chunk_sums, self.chunk_counts),
            ("horizon", "action_unit", self.horizon_sums, self.horizon_counts),
        ):
            names.extend(f"{prefix}/loss_by_{axis}/{label}_{i}" for i in range(len(sums)))
            values.append(sums / counts.clamp_min(1))
            names.extend(f"{prefix}/count_by_{axis}/{label}_{i}" for i in range(len(counts)))
            values.append(counts)
        for root, component_values in ((prefix, components), (f"{prefix}/raw_action_mse", raw_components)):
            names.extend(f"{root}/loss_by_component/component_{i}" for i in range(7))
            values.append(component_values)
            names.extend(f"{root}/loss_groups/{group}" for group in ("translation", "rotation", "gripper"))
            values.append(torch.stack((component_values[:3].mean(), component_values[3:6].mean(), component_values[6])))
        names.append(f"{prefix}/component_chunk_count")
        values.append(chunks.unsqueeze(0))
        metrics = dict(zip(names, torch.cat(values).cpu().tolist()))
        for axis, label, counts in (("chunk", "action_chunk", self.chunk_counts),
                                    ("horizon", "action_unit", self.horizon_counts)):
            for i in range(len(counts)):
                if metrics[f"{prefix}/count_by_{axis}/{label}_{i}"] == 0:
                    del metrics[f"{prefix}/loss_by_{axis}/{label}_{i}"]
        if metrics[f"{prefix}/component_chunk_count"] == 0:
            metrics = {key: value for key, value in metrics.items() if "/loss_" not in key}
        return metrics
