from pathlib import Path

import torch
from utilities import draw_dot_svg_graph


def step(x):
    y = x @ x.T + 1
    if x.shape[0] <= 128:
        return torch.relu(y)
    else:
        return torch.sigmoid(y)


def inspect_backend(graph_module, example_inputs):
    print(graph_module.graph)
    print(graph_module.print_readable(print_output=False))
    draw_dot_svg_graph(
        graph_module, Path(__file__).stem, example_inputs
    )
    return graph_module.forward


compiled = torch.compile(step, backend=inspect_backend, fullgraph=True)

# random 128x128 tensor
x = torch.randn(128, 128)
torch.testing.assert_close(compiled(x), step(x))

# random 256x256 tensor
y = torch.randn(256, 256)
torch.testing.assert_close(compiled(y), step(y))
