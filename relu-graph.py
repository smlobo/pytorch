from pathlib import Path

import torch

from utilities import draw_dot_svg_graph


def step(x):
    return torch.relu(x @ x.T + 1)


def inspect_backend(graph_module, example_inputs):
    print(graph_module.graph)
    draw_dot_svg_graph(
        graph_module, Path(__file__).stem, example_inputs
    )
    return graph_module.forward


x = torch.randn(128, 128)
compiled = torch.compile(step, backend=inspect_backend, fullgraph=True)
torch.testing.assert_close(compiled(x), step(x))
