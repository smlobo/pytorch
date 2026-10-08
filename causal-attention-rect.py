import torch
import math
from utilities import get_device


# B == batch size
# T == tokens per sequence
# D == input features per token
# H == total projected (hidden) feature width
# num_heads == number of atention heads
B, T, D, H, num_heads = 2, 4, 8, 16, 2
qT = T
kvT = T + 2

# head dimension == 8
head_dim = H // num_heads


def generate(device="cpu"):
    q = torch.randn(B, num_heads, qT, head_dim, device=device)
    k = torch.randn(B, num_heads, kvT, head_dim, device=device)
    v = torch.randn_like(k)                         # [2,2,6,8]
    return q, k, v


def manual_attention(q, k, v, causal=False):
    drvd_qT = q.shape[2]
    drvd_kvT = k.shape[2]
    drvd_dev = q.device

    scores = q @ k.transpose(-2, -1)                # [2,2,4,6]
    scores = scores / math.sqrt(q.shape[-1])        # [2,2,4,6]

    if causal:
        # Create an inverted Boolean lower-triangular mask.
        # True means this query/key pair is disallowed.
        # Replace disallowed scores with float("-inf").
        mask_dim = (drvd_qT, drvd_kvT)
        mask_2d = ~torch.ones(
            mask_dim, dtype=torch.bool, device=drvd_dev).tril()
        scores = scores.masked_fill(mask_2d, float('-inf'))

    probabilities = torch.softmax(scores, dim=-1)   # [2,2,4,6]

    # Verify
    if causal:
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                for i in range(drvd_qT):
                    row_sum = 0.0
                    for j in range(drvd_kvT):
                        # Verify row i has probability 0 for every key j > i
                        if j > i:
                            assert probabilities[b, h, i, j].item() == 0.0
                        row_sum += probabilities[b, h, i, j].item()
                    # Verify row sums to 1.0
                    assert math.isclose(row_sum, 1.0, rel_tol=1e-6)

    print(f"Probabilities:\n{probabilities[:, :, :, :]}")

    return probabilities @ v                        # [2,2,4,8]


def main():
    device_string = get_device()
    print(f"Using device: {device_string}")
    q, k, v = generate(device=device_string)
    this = manual_attention(q, k, v, causal=True)
    that = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=True
    )
    print(f"My causal attention:\n{this[:, :, :, :5]}")
    print(f"Torch SDPA causal:\n{that[:, :, :2, :4]}")
    torch.testing.assert_close(this, that)

    # Earlier query outputs are unaffected by changes to future keys and values
    for i in range(q.shape[-2]):
        changed_k = k.clone()
        changed_v = v.clone()
        changed_k[:, :, i + 1:, :] += 10
        changed_v[:, :, i + 1:, :] += 10

        changed_output = manual_attention(q, changed_k, changed_v, causal=True)
        torch.testing.assert_close(
            changed_output[:, :, i, :],
            this[:, :, i, :],
        )


if __name__ == "__main__":
    main()
