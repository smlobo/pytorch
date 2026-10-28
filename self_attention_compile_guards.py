import torch
import argparse
from pathlib import Path
from functools import partial
from utilities import inspect_backend, draw_dot_svg_graph


# B == batch size
# T == tokens per sequence
# D == input features per token / input_dim
# H == total projected (hidden) feature width / projection_dim
# num_heads == number of attention heads
B, T, D, H, num_heads = 2, 4, 8, 16, 2


class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, projection_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        assert projection_dim % num_heads == 0
        self.head_dim = projection_dim // num_heads

        # Input layers
        # (Linear() stores weights as projection_dim x input_dim)
        self.q_proj = torch.nn.Linear(input_dim, projection_dim)    # [16,8]
        self.k_proj = torch.nn.Linear(input_dim, projection_dim)
        self.v_proj = torch.nn.Linear(input_dim, projection_dim)

        # Output layer
        self.out_proj = torch.nn.Linear(projection_dim, input_dim)  # [8,16]

    def forward(self, x):
        batch_size, tokens, _ = x.shape

        q = self.q_proj(x)                              # [2,4,16]
        q_heads = q.reshape(
            batch_size, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)                               # [2,2,4,8]
        k = self.k_proj(x)
        k_heads = k.reshape(
            batch_size, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x)
        v_heads = v.reshape(
            batch_size, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_heads = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=True
        )                                               # [2,2,4,8]

        rearranged_heads = attention_heads.transpose(1, 2)      # [2,4,2,8]
        combined_attention = rearranged_heads.reshape(
            batch_size, tokens, self.num_heads*self.head_dim
        )                                                       # [2,4,16]

        output = self.out_proj(combined_attention)              # [2,4,8]

        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamic",
        choices=("false", "none", "true"),
        default="false",
        help="Dynamic shapes for torch.compile (default: false)",
    )
    args = parser.parse_args()
    dynamic_compile = {"false": False, "none": None, "true": True}[
        args.dynamic
    ]

    device = "cpu"
    print(f"Using device: {device}")

    # My model
    model = SelfAttention(D, H, num_heads).to(device)

    # switch to model evaluation
    model.eval()

    # Check when recompilation is done for different sequence lengths
    # Use the inductor (default PyTorch graph optimizer) backend
    # - modes dynamic=False, dynamic=True (--dynamic true), dynamic=None (none)
    # - run with TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCH_COMPILE_DEBUG=1
    #     TORCH_LOGS="guards,recompiles" INDUCTOR_POST_FUSION_SVG=1
    #     TORCH_LOGS_FORMAT="[%(levelname)s]
    #       [%(pathname)s/%(filename)s:%(lineno)s] [%(name)s] %(message)s"
    cpu_compiled_model = torch.compile(
        model,
        backend="inductor",
        fullgraph=True,
        dynamic=dynamic_compile
    )

    # Sequence lengths to try
    sequence_lengths = (4, 4, 6, 8, 4)
    for sequence_length in sequence_lengths:

        print(f'Sequence length: {sequence_length}')

        # Input [2,SL,8]
        input_tensor = torch.randn(B, sequence_length, D, device=device)

        # inference mode - the operations will not need gradients
        # leads to simpler compiled code
        with torch.inference_mode():
            # Output
            output_orig = model(input_tensor)                     # [2,SL,8]
            output_compiled = cpu_compiled_model(input_tensor)    # [2,SL,8]

        # print(f"PyTorch causal self attention:\n{output_orig[:, :, :5]}")
        # print(f"Compiled causal self attention:\n{output_compiled[:, :, :5]}")
        torch.testing.assert_close(output_orig, output_compiled)

    # ----------------------

    # Input (inspect compile & exported comparison)
    input_tensor = torch.randn(B, T, D, device=device)      # [2,4,8]

    # compile for graph inspection
    compiled_model = torch.compile(
        model,
        backend=partial(
                inspect_backend,
                filename=Path(__file__).stem,
                dump=False
            ),
        fullgraph=True
    )
    output_compiled = compiled_model(input_tensor)          # [2,4,8]

    # export with symbolic dimension
    seq_len = torch.export.Dim("seq_len", min=2, max=16)
    dynamic_shapes = {
        "x": {1: seq_len},
    }
    exported_model = torch.export.export(
        model,
        (input_tensor,),
        dynamic_shapes=dynamic_shapes,
    )
    # print(exported_model.graph_module.print_readable(print_output=False))
    print(f"Graph constraints: {exported_model.range_constraints}")
    draw_dot_svg_graph(
        exported_model.graph_module,
        f'{Path(__file__).stem}-export',
        (input_tensor,)
    )
    output_exported = exported_model.module()(input_tensor)

    torch.testing.assert_close(output_compiled, output_exported)


if __name__ == "__main__":
    main()
