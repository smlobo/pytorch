import torch
import copy
from causal_attention import manual_attention

# B -> sequences in a batch size
# T -> tokens per sequence
# D -> input features per token / input_dim
# H -> total projected feature width / projection_dim
# num_heads -> number of attention heads
B, T, D, H, num_heads = 2, 4, 8, 16, 2


class ComparableTinySelfAttention(torch.nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, num_heads: int):

        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        assert projection_dim % num_heads == 0
        self.head_dim = projection_dim // num_heads

        # Input layers
        # (Linear() stores weights as projection_dim x input_dim)
        self.q_proj = torch.nn.Linear(input_dim, projection_dim,
                                      dtype=torch.float64)          # [16,8]
        self.k_proj = torch.nn.Linear(input_dim, projection_dim,
                                      dtype=torch.float64)          # [16,8]
        self.v_proj = torch.nn.Linear(input_dim, projection_dim,
                                      dtype=torch.float64)          # [16,8]

        # Output layer
        self.out_proj = torch.nn.Linear(projection_dim, input_dim,
                                        dtype=torch.float64)  # [8,16]

    def forward(self, x, pytorch_sdpa: bool = False):
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

        # Use PyTorch if set
        if pytorch_sdpa:
            attention_heads = torch.nn.functional.scaled_dot_product_attention(
                q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=True
            )                                               # [2,2,4,8]
        else:
            attention_heads = manual_attention(
                q_heads, k_heads, v_heads, causal=True
            )                                               # [2,2,4,8]

        rearranged_heads = attention_heads.transpose(1, 2)      # [2,4,2,8]
        combined_attention = rearranged_heads.reshape(
            batch_size, tokens, self.num_heads*self.head_dim
        )                                                       # [2,4,16]

        output = self.out_proj(combined_attention)              # [2,4,8]

        return output


def main():
    device = "cpu"

    # Input
    x = torch.randn(B, T, D, device=device, dtype=torch.float64)  # [2,4,8]

    # My model for using manual SDPA
    model_manual = ComparableTinySelfAttention(D, H, num_heads).to(device)
    # Copy for using PyTorch SDPA
    model_pytorch = copy.deepcopy(model_manual)

    output_manual = model_manual(x)                               # [2,4,8]
    output_pytorch = model_pytorch(x, pytorch_sdpa=True)          # [2,4,8]

    print(f"My causal self attention:\n{output_manual[:, :, :5]}")
    print(f"Torch causal self attention:\n{output_pytorch[:, :, :5]}")

    # Check that manual & PyTorch SDPA have the same self attention
    torch.testing.assert_close(output_manual, output_pytorch)

    target = x

    loss_manual = torch.nn.functional.mse_loss(output_manual, target)
    loss_pytorch = torch.nn.functional.mse_loss(output_pytorch, target)

    print(f"Manual loss: {loss_manual}")
    print(f"PyTorch loss: {loss_pytorch}")
    print("Manual loss == PyTorch loss: ",
          torch.equal(loss_manual, loss_pytorch))
    torch.testing.assert_close(loss_manual, loss_pytorch)

    loss_manual.backward()
    loss_pytorch.backward()

    print(f"Manual q_proj weight:\n{model_manual.q_proj.weight.grad[:2]}")
    print(f"Manual q_proj bias:\n{model_manual.q_proj.bias.grad}")
    print(f"Manual k_proj weight:\n{model_manual.k_proj.weight.grad[:2]}")
    print(f"Manual k_proj bias:\n{model_manual.k_proj.bias.grad}")
    print(f"Manual v_proj weight:\n{model_manual.v_proj.weight.grad[:2]}")
    print(f"Manual v_proj bias:\n{model_manual.v_proj.bias.grad}")
    print(f"Manual out_proj weight:\n{model_manual.out_proj.weight.grad[:, :5]}")
    print(f"Manual out_proj bias:\n{model_manual.out_proj.bias.grad}")

    # Compare manual to PyTorch
    torch.testing.assert_close(model_manual.q_proj.weight.grad,
                               model_pytorch.q_proj.weight.grad)
    torch.testing.assert_close(model_manual.q_proj.bias.grad,
                               model_pytorch.q_proj.bias.grad)
    torch.testing.assert_close(model_manual.k_proj.weight.grad,
                               model_pytorch.k_proj.weight.grad)
    torch.testing.assert_close(model_manual.k_proj.bias.grad,
                               model_pytorch.k_proj.bias.grad)
    torch.testing.assert_close(model_manual.v_proj.weight.grad,
                               model_pytorch.v_proj.weight.grad)
    torch.testing.assert_close(model_manual.v_proj.bias.grad,
                               model_pytorch.v_proj.bias.grad)
    torch.testing.assert_close(model_manual.out_proj.weight.grad,
                               model_pytorch.out_proj.weight.grad)
    torch.testing.assert_close(model_manual.out_proj.bias.grad,
                               model_pytorch.out_proj.bias.grad)


if __name__ == "__main__":
    main()
