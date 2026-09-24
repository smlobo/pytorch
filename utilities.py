"""Small helpers shared by PyTorch experiments."""

import sys
import time
from pathlib import Path

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer


def force_cpu_requested() -> bool:
    """Return whether the command line includes the --cpu flag."""
    return "--cpu" in sys.argv[1:]


def get_device() -> str:
    """Return the best available device name: cuda, mps, or cpu."""
    if force_cpu_requested():
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def time_function(
    fn, *args, device: str = "cpu", iterations: int = 1,
    warmup: int = 0, **kwargs
):
    """Print wall time excluding warmup and return the last result."""
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be +ve and warmup cannot be -ve")

    device_type = torch.device(device).type

    def synchronize():
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        fn(*args, **kwargs)
    synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn(*args, **kwargs)
    synchronize()
    elapsed = time.perf_counter() - start

    print(
        f"{iterations} runs: {elapsed:.6f} s total, "
        f"{elapsed / iterations * 1000:.3f} ms/run"
    )
    return result


def draw_dot_svg_graph(graph_module, name, example_inputs=None):
    """Write DOT and SVG graphs, including an input shape when provided."""
    output_name = name
    if example_inputs is not None:
        tensor = next(
            value
            for value in example_inputs
            if isinstance(value, torch.Tensor)
        )
        shape = "x".join(str(size) for size in tensor.shape)
        output_name = f"{name}-{shape}"

    dot = FxGraphDrawer(
        graph_module, output_name.replace("-", "_")
    ).get_dot_graph()
    output_base = Path(__file__).with_name(output_name)
    dot_path = output_base.with_suffix(".dot")
    svg_path = output_base.with_suffix(".svg")
    dot_path.write_text(dot.to_string())
    dot.write_svg(str(svg_path))
    print(f"Wrote {dot_path.name} and {svg_path.name}")
