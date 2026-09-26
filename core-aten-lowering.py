import math
import torch
from pathlib import Path
from utilities import get_device, draw_dot_svg_graph


# B == batch size
# T == tokens per sequence (seq_len)
# D == input features per token
# H == total projected (hidden) feature width
# num_heads == number of atention heads
B, T, D, H, num_heads = 2, None, 8, 16, 2

# head dimension == 8
head_dim = H // num_heads


class ManualAttention(torch.nn.Module):
    def forward(self, q, k, v):
        scores = q @ k.transpose(-2, -1)        # [B, num_heads, T, T]
        weights = torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1)
        return weights @ v                      # [B, num_heads, T, head_dim]


def generate(T, device="cpu"):
    q = torch.randn(B, num_heads, T, head_dim, device=device)
    k = torch.randn(B, num_heads, T, head_dim, device=device)
    v = torch.randn(B, num_heads, T, head_dim, device=device)
    return q, k, v


def main():
    device_string = get_device()
    print(f"Using device: {device_string}")

    q4, k4, v4 = generate(T=4, device=device_string)
    q8, k8, v8 = generate(T=8, device=device_string)
    q20, k20, v20 = generate(T=20, device=device_string)

    seq_len = torch.export.Dim("seq_len", min=2, max=16)
    dynamic_shapes = {
        "q": {2: seq_len},
        "k": {2: seq_len},
        "v": {2: seq_len},
    }

    model = ManualAttention()
    exported = torch.export.export(
        model, (q4, k4, v4), dynamic_shapes=dynamic_shapes
    )

    # print(exported.graph_module.print_readable(print_output=False))
    print(f"Graph constraints: {exported.range_constraints}")

    core = exported.run_decompositions(decomp_table=None)

    print("Original operators:")
    print([str(n.target) for n in exported.graph_module.graph.nodes
           if n.op == "call_function"])

    print("Core ATen operators:")
    print([str(n.target) for n in core.graph_module.graph.nodes
           if n.op == "call_function"])

    print(core.graph_module.print_readable(print_output=False))
    print(f"Graph constraints: {core.range_constraints}")

    for inputs in ((q4, k4, v4), (q8, k8, v8)):
        torch.testing.assert_close(core.module()(*inputs), model(*inputs))

    draw_dot_svg_graph(
        core.graph_module, Path(__file__).stem, (q4, k4, v4)
    )


if __name__ == "__main__":
    main()
