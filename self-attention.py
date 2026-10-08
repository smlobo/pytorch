import torch
from utilities import get_device

# B == batch size
# T == tokens per sequence
# D == input features per token / input_dim
# H == total projected (hidden) feature width / projection_dim
# num_heads == number of atention heads
B, T, D, H, num_heads = 2, 4, 8, 16, 2


class TinySelfAttention(torch.nn.Module):
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
        # rearranged_heads = attention_heads.permute(0, 2, 1, 3)# [2,4,2,8]
        combined_attention = rearranged_heads.reshape(
            batch_size, tokens, self.num_heads*self.head_dim
        )                                                       # [2,4,16]

        output = self.out_proj(combined_attention)              # [2,4,8]

        return output


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Input
    x = torch.randn(B, T, D, device=device)             # [2,4,8]

    # My model
    model = TinySelfAttention(D, H, num_heads).to(device)
    output = model(x)                                   # [2,4,8]

    # Check that the same input results in the same output
    output_redo = model(x)
    torch.testing.assert_close(output, output_redo)

    # Single training step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = x

    weight_before = model.q_proj.weight.detach().clone()

    optimizer.zero_grad()
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()       # Calculate gradients for the model's parameters
    optimizer.step()       # Update the parameters

    print("loss:", loss.item())
    print("q_proj gradient norm:", model.q_proj.weight.grad.norm().item())
    print("q_proj weight changed:",
          not torch.equal(weight_before, model.q_proj.weight))


if __name__ == "__main__":
    main()
