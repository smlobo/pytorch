from pathlib import Path

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer


def step(x):
    return torch.sigmoid(x @ x.T + 1)


def inspect_backend(graph_module, example_inputs):
    print(graph_module.graph)
    dot = FxGraphDrawer(graph_module, "sigmoid_graph").get_dot_graph()
    output_base = Path(__file__).with_suffix("")
    output_base.with_suffix(".dot").write_text(dot.to_string())
    dot.write_svg(str(output_base.with_suffix(".svg")))
    return graph_module.forward


x = torch.randn(128, 128)
compiled = torch.compile(step, backend=inspect_backend, fullgraph=True)
torch.testing.assert_close(compiled(x), step(x))
