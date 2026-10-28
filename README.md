# PyTorch compilation exercises

Small experiments with PyTorch tensors, devices, and `torch.compile`.

## macOS (Apple silicon)

From the repository root, create a Python 3.12 environment and install the
packages used by the examples:

```sh
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision transformers numpy pydot
```

PyTorch uses MPS for GPU work on a supported Mac. Check that it is available
with `python check-mps.py`.

## Linux

```sh
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision transformers numpy pydot --torch-backend=auto
```

The `auto` backend selects a compatible PyTorch build for an available GPU,
or a CPU build when no supported GPU is found.

## Learning examples

### Basic compilation

```sh
python sin-cos-tensor.py
python sin-cos-tensor.py --cpu
python resnet50.py
python transformers-pretrained.py
python tensor-shape.py
python scaled-dot-product-attention.py
```

`sin-cos-tensor.py` compiles a cosine and sine function on a 10,000-element
tensor, times its first call, and prints the first 10 results.

`resnet50.py` times its first compiled call. The model uses random weights,
so no pretrained weights are downloaded.

`transformers-pretrained.py` uses pretrained `bert-base-uncased` weights. The
first run downloads and caches the model and tokenizer. It times model calls,
excluding tokenization, with a fixed 32-token input. It accepts `--iterations N`
for repeated-call timing.

See [Draw computation graphs](#draw-computation-graphs) below.

### Tensor shapes

`tensor-shape.py` examines tensor shapes, data types, devices, and strides
through projection, transposition, matrix multiplication, broadcasting, and
softmax. It also compares float16 and float32 results.

### Scaled Dot Product Attention

`scaled-dot-product-attention.py` builds queries, keys, and values for two
attention heads. It implements scaled dot product attention from matrix
multiplication, scaling, and softmax, then checks its output against
`torch.nn.functional.scaled_dot_product_attention`.

### Backend and device checks

`list-backends.py` prints the backends registered for `torch.compile`.
`check-cuda.py` reports whether this PyTorch installation can use CUDA and,
when a CUDA device is available, runs a small tensor operation on it.

### Causal attention and training

`causal_attention.py` implements scaled dot product attention with an optional
causal mask, then compares its causal result with PyTorch's SDPA operation.
`causal-attention-rect.py` extends the example to different query and
key/value sequence lengths and checks that changing future keys and values
does not affect earlier causal outputs.

`self-attention.py` builds a small causal self-attention module from linear
projections and SDPA, then runs a single AdamW training step and reports a
gradient norm and whether a weight changed.

`self_attention_gradient_comparison.py` compares manual attention with
PyTorch SDPA in float64, checking both forward outputs, losses, and parameter
gradients.

`self_attention_compile_guards.py` runs a small self-attention model at
several sequence lengths and compares eager execution with `torch.compile`.
Its `--dynamic` option accepts `false` (the default), `none`, or `true` and
passes the corresponding value to `torch.compile`. It also exports the model
with a symbolic sequence dimension and compares the exported module's output.
For example:

```sh
python self_attention_compile_guards.py --dynamic true
```

## Draw computation graphs

The graph examples use `pydot` and the Graphviz `dot` executable to write
`.dot` and `.svg` files. On macOS, install Graphviz with
`brew install graphviz`. On Linux, install the `graphviz` package with your
distribution's package manager.

```sh
python relu-graph.py
python sigmoid-graph.py
python shape-condition.py
python resnet50-graph.py
python attention-graph.py
python attention-graph-symbolic-dimension.py
```

`relu-graph.py` and `sigmoid-graph.py` capture small FX graphs through a custom
`torch.compile` backend. They print each graph, write a DOT and SVG pair, and
check that the compiled result matches eager execution.

`shape-condition.py` compiles a function that selects ReLU or sigmoid based on
the input shape. It runs with 128 × 128 and 256 × 256 tensors and writes a
separate DOT and SVG pair for each backend invocation, named after the shape
(for example, `shape-condition-128x128.svg`).

`resnet50-graph.py` traces the ResNet-50 model with `torch.fx.symbolic_trace`
and writes `resnet50.dot` and `resnet50.svg`. The generated graph files are
ignored by Git.

`attention-graph.py` defines a manual scaled dot product attention module and
captures it with `torch.export.export`. It prints the normalized ATen graph,
writes `attention-graph-2x2x4x8.dot` and `.svg`, and checks that the exported
module produces the same result as eager execution.

`attention-graph-symbolic-dimension.py` exports the same attention module with
a symbolic sequence length constrained to the range 2 through 16. It prints
the normalized ATen graph and its range constraints, writes a shape-named DOT
and SVG pair, and checks the exported module with sequence lengths 4 and 8.

`core-aten-lowering.py` continues the previous examples adding PyTorch lowering
via the `run_decompositions()` method. Operations like `matmul` & `transpose`
are expanded & transformed.

## Inspect compilation output

Set `TORCH_COMPILE_DEBUG=1` when running an example to see compiler debug
messages and save compilation artifacts:

```sh
TORCH_COMPILE_DEBUG=1 python sin-cos-tensor.py
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCH_COMPILE_DEBUG=1 TORCH_LOGS="guards,recompiles" python self_attention_compile_guards.py 2> file.log
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCH_COMPILE_DEBUG=1 TORCH_LOGS="guards,recompiles" TORCH_LOGS_FORMAT="[%(levelname)s] [%(pathname)s/%(filename)s:%(lineno)s] [%(name)s] %(message)s" python self_attention_compile_guards.py 2> file.log
```

TORCH_LOGS options:
- output_code
- guards
- recompiles

PyTorch writes a `torch_compile_debug/run_.../` directory in the current
directory. Look under its `torchinductor/` subdirectories for
`fx_graph_runnable.py` and `fx_graph_transformed.py` (the captured and
transformed graphs), `ir_pre_fusion.txt` and `ir_post_fusion.txt` (compiler
intermediate representations), and `output_code.py` (generated code, including
kernel code when applicable). The terminal output also shows the debug trace
path. Available files depend on the graph and device. For more details, see
the PyTorch compiler debugging guide:

<https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html>
