# PyTorch compilation exercises

Small experiments with PyTorch tensors, devices, and `torch.compile`.

## macOS (Apple silicon)

From the repository root, create a Python 3.12 environment and install the
packages used by the examples:

```sh
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch numpy
```

PyTorch uses MPS for GPU work on a supported Mac. Check that it is available
with `python check-mps.py`.

## Linux

```sh
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch numpy --torch-backend=auto
```

The `auto` backend selects a compatible PyTorch build for an available GPU,
or a CPU build when no supported GPU is found.

## Run the example

```sh
python sin-cos-tensor.py
python sin-cos-tensor.py --cpu
```

The second command forces CPU execution. Run the commands from the repository
root so Python can import `utilities.py`.
