import torch


if not torch.cuda.is_available():
    if not torch.backends.cuda.is_built():
        print("CUDA is unavailable: this PyTorch installation was not built with CUDA support.")
    else:
        print("CUDA is unavailable: no usable CUDA device was found.")
else:
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = x * 2
    print(f"CUDA is available on {torch.cuda.get_device_name(device)}")
    print(f"x = {x}, y = {y}")
