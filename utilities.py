"""Small helpers shared by PyTorch experiments."""

import sys
import time

import torch


def force_cpu_requested() -> bool:
    """Return whether the command line includes the --cpu flag."""
    return "--cpu" in sys.argv[1:]


def get_device(force_cpu: bool = False) -> str:
    """Return the best available device name: cuda, mps, or cpu."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def time_function(
    fn, *args, device: str = "cpu", iterations: int = 1,
    warmup: int = 0, **kwargs
):
    """Print wall time excluding warmup and return the last result."""
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be +ve and warmup cannot be -ve")

    device_type = torch.device(device).type

    def synchronize():
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        fn(*args, **kwargs)
    synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn(*args, **kwargs)
    synchronize()
    elapsed = time.perf_counter() - start

    print(
        f"{iterations} runs: {elapsed:.6f} s total, "
        f"{elapsed / iterations * 1000:.3f} ms/run"
    )
    return result
