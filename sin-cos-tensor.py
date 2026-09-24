import torch
from utilities import get_device, time_function


def sin_cos(x):
    a = torch.cos(x)
    b = torch.sin(a)
    return b


def main():
    device_string = get_device()
    print(f"Using device: {device_string}")
    new_fn = torch.compile(sin_cos, backend="inductor")
    input_tensor = torch.randn(10000, device=device_string)
    r = time_function(new_fn, input_tensor, device=device_string)
    print(r[:10])


if __name__ == "__main__":
    main()
