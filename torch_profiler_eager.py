import torch
from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext
from sin_cos_tensor import sin_cos
from utilities import get_device


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    device = get_device()
    print(f"Using device: {device}")

    x = torch.randn(1_000_000, device=device)

    # Warm up before recording.
    for _ in range(5):
        sin_cos(x)
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # MPS signposts can be viewed in Xcode Instruments.
    mps_profile = (
        torch.mps.profiler.profile(mode="interval")
        if device == "mps"
        else nullcontext()
    )
    with mps_profile:
        with profile(activities=activities, record_shapes=True) as prof:
            y = sin_cos(x)  # eager execution
            synchronize(device)

    print(prof.key_averages().table(sort_by="cpu_time_total"))
    print(y[:5])


if __name__ == "__main__":
    main()
