# hopfieldpc-memory-core

Standalone modern Hopfield memory core for the HopfieldPC project.

<p align="justify">
This directory contains a modern Hopfield memory core integrated with
FabricPC. It focuses on beta-controlled associative retrieval, tensor-shape
validation, diagnostics, synthetic benchmarks, and unit tests. Use
`fabricpc.nodes.HopfieldMemoryNode` to add fixed prototype memories to a
FabricPC graph; the node's memory is not part of graph parameters.

## FabricPC PC MNIST comparison

The original Task 13 runner remains available and trains its custom JAX
baseline. To run the corresponding CNN as a FabricPC graph trained with
predictive coding (PC), run this separate benchmark from the repository root:

```powershell
python -m fabricpc.hopfield.benchmarks.mnist_pc_prototype_retrieval --quick
python -m fabricpc.hopfield.benchmarks.mnist_pc_prototype_retrieval
```

It reuses the Task 13 data splits, prototype construction, beta-selection
sets, corruption grids, and reports. The CNN is expressed with FabricPC
`ConvNode`, `MaxPool`, and `Linear` nodes and trained using
`train(..., algorithm="pc")`. At evaluation, it compares an identity bridge
with a frozen `HopfieldMemoryNode` bridge using the same trained graph
parameters. Evaluation uses the same feed-forward pass as Task 13; PC inference
settling is used during training only. This is a PC-trained comparison, not a
byte-for-byte reproduction of the original backprop-trained baseline. The new runner writes to
`reports/mnist_pc_prototype_retrieval/` (or its `_quick` sibling) to avoid
overwriting Task 13 reports.

The runner prints stage updates, a minibatch progress bar during training,
beta-selection progress, and test-evaluation progress. The first message
appears before MNIST is loaded; if the dataset is not cached, that step may
include downloading the MNIST files. Cached gzip files are validated; truncated
or invalid files are automatically replaced with complete downloads.

PC inference defaults to 10 steps with `eta_infer=0.05`. Adjust these without
changing the benchmark config using `--infer-steps` and `--eta-infer`.

## Setup

Clone the repository:

```bash
git clone https://github.com/iCog-Labs-Dev/hopfieldpc-memory-core.git
cd hopfieldpc-memory-core
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import jax; print(jax.__version__)"
python -c "import hopfieldpc; print('hopfieldpc import OK')"
```

## Current Implementation Stage

Current task:

```text
Task 2: Implement one-step Hopfield retrieval
```

Implemented so far:

```text
src/hopfieldpc/api.py
src/hopfieldpc/functional.py
src/hopfieldpc/memory.py
src/hopfieldpc/__init__.py
tests/test_api_contract.py
tests/test_one_step_retrieval.py
tests/test_beta_behavior.py
```

The module now validates tensor shapes and performs one-step modern Hopfield retrieval.

## Expected Tensor Contract

```text
query:     [batch_size, dim]
memory:    [num_memories, dim]
retrieved: [batch_size, dim]
weights:   [batch_size, num_memories]
```

## Core Retrieval Formula

```python
scores = beta * query @ memory.T
weights = softmax(scores)
retrieved = weights @ memory
```

Equivalent mathematical form:

```text
p = softmax(β qMᵀ)
h_memory = pM
```

## Run Tests

Run all tests:

```bash
pytest -q
```

Run only the API contract tests:

```bash
pytest tests/test_api_contract.py -q
```

Run only one-step retrieval tests:

```bash
pytest tests/test_one_step_retrieval.py tests/test_beta_behavior.py -q
```

## Next Planned Tasks

```text
Task 3: Add memory modes and prepare module extension points.
Task 4: Add optional multi-step retrieval.
Task 5: Add diagnostics such as entropy, top-k mass, and retrieval distance.
Task 6: Add synthetic retrieval benchmarks.
```

## Task 13: MNIST prototype robustness experiment

The JAX implementation provides a controlled Baseline-vs-Hopfield MNIST CNN
experiment using the frozen prototype bridge at layer L8:

```bash
python -m benchmarks.mnist_prototype_retrieval --config configs/mnist_prototype_retrieval.yaml
python -m benchmarks.mnist_prototype_retrieval --quick
python -m benchmarks.mnist_baseline --config configs/mnist_prototype_retrieval.yaml --quick
```

The experiment uses official MNIST IDX downloads, seeded corruption grids,
k-means prototypes per class, beta and update-step sweeps, placement
ablations, paired seed comparisons, cost measurements, CSV/JSON reports, and
Markdown/Python plot outputs. It is implemented entirely with JAX arrays and
JAX autodiff; PyTorch and TensorFlow dependencies are not used.
