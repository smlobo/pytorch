"""Study torch.compile on ResNet-50."""

import argparse

import torch

from utilities import get_device, time_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU execution"
    )
    parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    from torchvision.models import resnet50

    model = resnet50(weights=None).eval().to(device)
    compiled_model = torch.compile(model)
    input_tensor = torch.randn(1, 3, 64, 64, device=device)

    with torch.inference_mode():
        print("First call (includes compilation):")
        output = time_function(compiled_model, input_tensor, device=device)
        print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
