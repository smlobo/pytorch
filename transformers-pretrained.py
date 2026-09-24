"""Benchmark torch.compile on a pretrained BERT model."""

import argparse

import torch

from utilities import get_device, time_function


MODEL_ID = "bert-base-uncased"
TEXT = "Replace me by any text you'd like."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU execution"
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    device = get_device()
    print(f"Using device: {device}")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    compiled_model = torch.compile(model, backend="inductor")
    inputs = tokenizer(
        TEXT, return_tensors="pt", padding="max_length",
        truncation=True, max_length=32
    ).to(device)

    with torch.inference_mode():
        print("First call (includes compilation):")
        output = time_function(
            compiled_model, device=device, return_dict=False, **inputs
        )
        print(f"Last hidden state shape: {tuple(output[0].shape)}")

        print("Repeated calls:")
        time_function(
            compiled_model, device=device, return_dict=False,
            iterations=args.iterations, warmup=2, **inputs
        )


if __name__ == "__main__":
    main()
