from pathlib import Path

from torch.fx import symbolic_trace
from torchvision.models import resnet50

from utilities import draw_dot_svg_graph

graph = symbolic_trace(resnet50(weights=None).eval())
draw_dot_svg_graph(graph, Path(__file__).stem)
