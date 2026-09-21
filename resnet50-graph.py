from pathlib import Path

from torch.fx import symbolic_trace
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torchvision.models import resnet50

graph = symbolic_trace(resnet50(weights=None).eval())
dot = FxGraphDrawer(graph, "resnet50").get_dot_graph()

Path("resnet50.dot").write_text(dot.to_string())
dot.write_svg("resnet50.svg")
