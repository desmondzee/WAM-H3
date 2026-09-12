import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch


def to_host(features, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    buffers = [torch.empty_like(f, device="cpu", pin_memory=True) for f in features]
    with torch.cuda.stream(stream):
        for host, feature in zip(buffers, features):
            host.copy_(feature, non_blocking=True)
        ready = stream.record_event()
    ready.synchronize()
    return buffers


def to_device(buffers, device):
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        features = [b.to(device, non_blocking=True) for b in buffers]
        ready = stream.record_event()
    ready.synchronize()
    return features


def feature_batches(extractor, samples, steps, source_device, target_device, overlap=True, timings=None):
    def extract(index):
        torch.cuda.set_device(source_device)
        sample = samples[index % len(samples)]
        if isinstance(sample, (str, Path, tuple)):
            from .cache import load_policy_cache
            sample = load_policy_cache(*sample) if isinstance(sample, tuple) else load_policy_cache(sample)
        start = time.perf_counter()
        features, times = extractor(sample["latent"], sample["conditioning"], sample["tags"])
        if timings is not None:
            torch.cuda.synchronize(source_device)
        extracted = time.perf_counter()
        buffers = to_host(features, source_device)
        measured = dict(extraction_seconds=extracted - start, d2h_seconds=time.perf_counter() - extracted)
        return buffers, times.cpu(), sample, measured
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract, 0) if overlap and steps else None
        for step in range(steps):
            start = time.perf_counter()
            buffers, times, sample, measured = future.result() if overlap else extract(step)
            measured["queue_wait_seconds"] = time.perf_counter() - start if overlap else 0.0
            if overlap and step + 1 < steps:
                future = executor.submit(extract, step + 1)
            start = time.perf_counter()
            features = to_device(buffers, target_device)
            measured["h2d_seconds"] = time.perf_counter() - start
            measured["host_pinned"] = all(buffer.is_pinned() for buffer in buffers)
            if timings is not None:
                timings.append(measured)
            del buffers
            yield features, times.to(target_device), sample
