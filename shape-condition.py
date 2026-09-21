from pathlib import Path

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer


shape_counts = {}


def step(x):
    y = x @ x.T + 1
    if x.shape[0] <= 128:
        return torch.relu(y)
    else:
        return torch.sigmoid(y)


def inspect_backend(graph_module, example_inputs):
    print(graph_module.graph)
    tensor = next(x for x in example_inputs if isinstance(x, torch.Tensor))
    shape = "x".join(str(size) for size in tensor.shape)
    shape_counts[shape] = shape_counts.get(shape, 0) + 1
    suffix = f"-{shape_counts[shape]}" if shape_counts[shape] > 1 else ""
    name = f"{Path(__file__).stem}-{shape}{suffix}"

    dot = FxGraphDrawer(graph_module, name.replace("-", "_")).get_dot_graph()
    output_base = Path(__file__).with_name(name)
    dot_path = output_base.with_suffix(".dot")
    svg_path = output_base.with_suffix(".svg")
    dot_path.write_text(dot.to_string())
    dot.write_svg(str(svg_path))
    print(f"Wrote {dot_path.name} and {svg_path.name}")
    return graph_module.forward


compiled = torch.compile(step, backend=inspect_backend, fullgraph=True)
compiled(torch.randn(128, 128))
compiled(torch.randn(256, 256))
