import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
B, T, D, H = 2, 4, 8, 16  # batch, tokens, input width, hidden width

x = torch.randn(B, T, D, device=device)
weight = torch.randn(D, H, device=device)
bias = torch.randn(H, device=device)

projected = x @ weight + bias
scores = projected @ projected.transpose(-2, -1)
probabilities = torch.softmax(scores, dim=-1)

for name, tensor in [
    ("x", x),
    ("projected", projected),
    ("scores", scores),
    ("probabilities", probabilities),
]:
    print(name, tensor.shape, tensor.dtype, tensor.device, tensor.stride())

print("row sums:", probabilities.sum(dim=-1))

# float16 vs float32
projected16 = x.half() @ weight.half() + bias.half()
print(f"float16 vs float32 projected: "
      f"{(projected16.float() - projected).abs().max()}")

# stride + transpose.stride
print(f"projected.stride() = {projected.stride()}")
print(f"projected.transpose(-2, -1).stride() = "
      f"{projected.transpose(-2, -1).stride()}")

# bias.shape == (1, T, 1)
print()

bias = torch.randn(1, T, 1, device=device, dtype=torch.float16)

projected = x @ weight + bias
scores = projected @ projected.transpose(-2, -1)
probabilities = torch.softmax(scores, dim=-1)

for name, tensor in [
    ("x", x),
    ("projected", projected),
    ("scores", scores),
    ("probabilities", probabilities),
]:
    print(f"bias[1,{T},1]: {name}, {tensor.shape}, {tensor.dtype}, "
          f"{tensor.device}, {tensor.stride()}")

print(f"row sums (bias[1,{T},1]): {probabilities.sum(dim=-1)}")

# data type 16
print()

x = torch.randn(B, T, D, device=device, dtype=torch.float16)
weight = torch.randn(D, H, device=device, dtype=torch.float16)
bias = torch.randn(H, device=device, dtype=torch.float16)

projected = x @ weight + bias
scores = projected @ projected.transpose(-2, -1)
probabilities = torch.softmax(scores, dim=-1)

for name, tensor in [
    ("x", x),
    ("projected", projected),
    ("scores", scores),
    ("probabilities", probabilities),
]:
    print(f"float16: {name}, {tensor.shape}, {tensor.dtype}, {tensor.device}, "
          f"{tensor.stride()}")

print("row sums (float16):", probabilities.sum(dim=-1))
