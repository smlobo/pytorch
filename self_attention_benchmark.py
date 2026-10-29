import time
import torch
from utilities import get_device
from scaled_dot_product_attention import scaled_dot_product_attention

# B == batch size
# T == tokens per sequence
# D == input features per token / input_dim
# H == total projected (hidden) feature width / projection_dim
# num_heads == number of attention heads
# B, T, D, H, num_heads = 2, 4, 8, 16, 2
# GPT-3 sizes (except for B)
# B, T, D, H, num_heads = 2, 2048, 12288, 12288, 96
B, T, D, H, num_heads = 2, 1024, 2048, 2048, 32


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
        self.q_proj = torch.nn.Linear(input_dim, projection_dim)    # [D,H]
        self.k_proj = torch.nn.Linear(input_dim, projection_dim)
        self.v_proj = torch.nn.Linear(input_dim, projection_dim)

        # Output layer
        self.out_proj = torch.nn.Linear(projection_dim, input_dim)  # [H,D]

    def forward(self, x):
        batch_size, tokens, _ = x.shape

        q = self.q_proj(x)                           # [B,T,H]
        q_heads = q.reshape(
            batch_size, tokens, self.num_heads,
            self.head_dim                            # [B,T,num_heads,head_dim]
        ).transpose(1, 2)                            # [B,num_heads,T,head_dim]
        k = self.k_proj(x)
        k_heads = k.reshape(
            batch_size, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x)
        v_heads = v.reshape(
            batch_size, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_heads = scaled_dot_product_attention(
            q_heads, k_heads, v_heads,               # [B,num_heads,T,head_dim]
        )

        rearranged_heads = attention_heads.transpose(
            1, 2
        )                                            # [B,T,num_heads,head_dim]
        combined_attention = rearranged_heads.reshape(
            batch_size, tokens, self.num_heads*self.head_dim
        )                                            # [B,T,H]

        output = self.out_proj(combined_attention)   # [B,T,D]

        return output


def benchmark(model, device):
    # Input
    x = torch.randn(B, T, D, device=device)             # [B,T,D]

    # Warm up
    for _ in range(10):
        model(x)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    repeats = 100
    start = time.perf_counter()
    for count in range(repeats):
        # if count % 10 == 0:
        #     print(f'{count},', end="")
        y = model(x)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / repeats


def main():
    device = get_device()
    print(f"Using device: {device}")

    # My model
    model = TinySelfAttention(D, H, num_heads).to(device)

    # compile
    compiled_model = torch.compile(
        model,
        backend="inductor",
        fullgraph=True,
    )

    with torch.inference_mode():
        # Benchmark
        print("Running eager ...")
        print(f'Eager: {benchmark(model, device):.4f} ms/call')
        print("Running compiled ...")
        print(f'Compiled: {benchmark(compiled_model, device):.4f} ms/call')


if __name__ == "__main__":
    main()
