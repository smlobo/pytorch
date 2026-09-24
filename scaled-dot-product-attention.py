import torch
import math
from utilities import get_device


# B == batch size
# T == tokens per sequence
# D == input features per token
# H == total projected (hidden) feature width
# num_heads == number of atention heads
B, T, D, H, num_heads = 2, 4, 8, 16, 2

# head width == 8
head_width = H // num_heads


def generate(device="cpu"):
    # Input token features
    x = torch.randn(B, T, D, device=device)     # [2, 4, 8]
    # Queries
    Wq = torch.randn(D, H, device=device)       # [8, 16]
    queries = (x @ Wq).reshape(
        B, T, num_heads, head_width
    ).transpose(1, 2)                           # [2, 2, 4, 8]
    # Keys
    Wk = torch.randn(D, H, device=device)       # [8, 16]
    keys = (x @ Wk).reshape(
        B, T, num_heads, head_width
    ).transpose(1, 2)                           # [2, 2, 4, 8]
    # Values
    Wv = torch.randn(D, H, device=device)       # [8, 16]
    values = (x @ Wv).reshape(
        B, T, num_heads, head_width
    ).transpose(1, 2)                           # [2, 2, 4, 8]
    return queries, keys, values


def scaled_dot_product_attention(queries, keys, values):
    scores = queries @ keys.transpose(-2, -1)       # [2, 2, 4, 4]
    scaled_scores = scores / math.sqrt(head_width)
    weights = torch.softmax(scaled_scores, dim=-1)  # [2, 2, 4, 4]
    output = weights @ values                       # [2, 2, 4, 8]
    return output


def main():
    device_string = get_device()
    print(f"Using device: {device_string}")
    q, k, v = generate(device=device_string)
    this = scaled_dot_product_attention(q, k, v)
    that = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0
    )
    print(f"My scaled dot product:\n{this[:, :, :2, :4]}")
    print(f"Torch scaled dot product:\n{that[:, :, :2, :4]}")
    torch.testing.assert_close(this, that)


if __name__ == "__main__":
    main()
