import time
import torch
from sin_cos_tensor import sin_cos


def benchmark(device, repeats=100):
    x = torch.randn(1_000_000, device=device)

    # Warm up
    for _ in range(10):
        sin_cos(x)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        y = sin_cos(x)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / repeats


def main():
    # Benchmark
    print("CPU:", benchmark("cpu"), "ms/call")
    if torch.backends.mps.is_available():
        print("MPS:", benchmark("mps"), "ms/call")
    if torch.cuda.is_available():
        print("CUDA:", benchmark("cuda"), "ms/call")


if __name__ == "__main__":
    main()
