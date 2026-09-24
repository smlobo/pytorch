import math
import torch
from pathlib import Path
from utilities import get_device, draw_dot_svg_graph


# B == batch size
# T == tokens per sequence
# D == input features per token
# H == total projected (hidden) feature width
# num_heads == number of atention heads
B, T, D, H, num_heads = 2, 8, 8, 16, 2

# head dimension == 8
head_dim = H // num_heads


class ManualAttention(torch.nn.Module):
    def forward(self, q, k, v):
        scores = q @ k.transpose(-2, -1)        # [B, num_heads, T, T]
        weights = torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1)
        return weights @ v                      # [B, num_heads, T, head_dim]


def generate(device="cpu"):
    q = torch.randn(B, num_heads, T, head_dim, device=device)
    k = torch.randn(B, num_heads, T, head_dim, device=device)
    v = torch.randn(B, num_heads, T, head_dim, device=device)
    return q, k, v


def main():
    device_string = get_device()
    print(f"Using device: {device_string}")
    q, k, v = generate(device=device_string)
    model = ManualAttention()
    exported = torch.export.export(model, (q, k, v))

    print(exported.graph_module.print_readable(print_output=False))
    draw_dot_svg_graph(
        exported.graph_module, Path(__file__).stem, (q, k, v)
    )
    torch.testing.assert_close(exported.module()(q, k, v), model(q, k, v))


if __name__ == "__main__":
    main()
