# FabricPC baseline protocol

## Purpose and scope

This document defines the control experiment for Track A (HopfieldReadout).
The control is the unmodified MNIST predictive-coding example in
[`examples/mnist_demo.py`](../../examples/mnist_demo.py). A readout experiment
must use the same graph, checkpoint, input preprocessing, random seeds, and
evaluation procedure unless its protocol explicitly says otherwise.

This protocol describes source/configuration facts that are fixed by the
repository. Accuracy, runtime, and checkpoint files are run artifacts and must
be recorded from the environment in which the experiment is executed; they
must not be copied from the example's illustrative output.

## Source and dependency pin

| Item | Baseline value |
| --- | --- |
| Repository | `iCog-Labs-Dev/HopfieldPC` |
| Baseline commit | `8406e6a838442391fd3089958e1a2c1b44c57eda` |
| Upstream relationship | The package metadata identifies the upstream project as `trueagi-io/FabricPC`; this fork's Git remote is `https://github.com/iCog-Labs-Dev/HopfieldPC` |
| FabricPC package version | `0.6.0` |
| Python | `>=3.11` (record the exact interpreter version for every run) |
| Declared core dependencies | `jax>=0.7.0`, `optax>=0.1.7`, `orbax-checkpoint>=0.4.0`, `chex>=0.1.84`, `jaxtyping>=0.2.23`, `numpy>=1.24.0`, `tqdm>=4.65.0` |

The declared dependency floors are in
[`pyproject.toml`](../../pyproject.toml). This repository does not contain a
lockfile, so a completed baseline run must save exact installed versions and
the selected JAX backend alongside its checkpoint. A suitable capture command
is:

```powershell
python --version
python -m pip freeze
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

The source tree must be clean apart from explicitly recorded experiment
artifacts, and the baseline commit must be recorded in the run manifest:

```powershell
git rev-parse HEAD
git status --short
```

## Model and representation protocol

The graph has four nodes and three directed edges:

```text
pixels(784) -> hidden1(256) -> hidden2(64) -> class(10)
 Identity       Linear/Sigmoid   Linear/Sigmoid  Linear/Softmax + CE
```

The exact construction is in
[`examples/mnist_demo.py`](../../examples/mnist_demo.py):

| Node | Type/configuration | Role |
| --- | --- | --- |
| `pixels` | `IdentityNode`, shape `(784,)` | Input image, flattened |
| `hidden1` | `Linear`, shape `(256,)`, `SigmoidActivation`, `XavierInitializer` | First hidden representation |
| `hidden2` | `Linear`, shape `(64,)`, `SigmoidActivation`, `XavierInitializer` | **Canonical readout representation** |
| `class` | `Linear`, shape `(10,)`, `SoftmaxActivation`, `CrossEntropyEnergy`, `XavierInitializer` | Classification prediction |

The task map is `x=pixels`, `y=class`. The exact representation extraction
point for Track A is the settled `hidden2` state after inference:

```python
representation = final_state.nodes["hidden2"].z_latent
```

If a downstream experiment needs the model's prediction rather than its
latent representation, it must use `final_state.nodes["class"].z_mu` and
state that choice explicitly. A post-inference readout must not replace,
modify, or feed back into the baseline graph.

## Initialization, training, and learning

The script sets:

- master PRNG seed: `0`;
- graph, training, and evaluation keys: `jax.random.split(master_key, 3)`;
- graph parameter initialization: `initialize_params(structure, graph_key)`;
- training loader: MNIST `train`, `batch_size=200`, `tensor_format="flat"`,
  shuffled with seed `42`;
- evaluation loader: MNIST `test`, `batch_size=200`, flat tensors, no
  shuffling;
- epochs: `20`;
- optimizer: `optax.adamw(0.001, weight_decay=0.1)`;
- training algorithm: predictive coding (`train(..., algorithm="pc")`, the
  default);
- checkpoint source: the returned `TrainResult.params` after epoch 20.

The current example prints a parameter count of 218,058 for the initialized
graph. Later experiments must reference a saved checkpoint produced from
these exact settings rather than relying on an in-memory result.

For each training batch, the trainer:

1. Converts the loader batch to the task-keyed dictionary `{x, y}`.
2. Clamps both `pixels` and `class`.
3. Initializes a `GraphState` using the graph's state initializer and the
   batch PRNG key.
4. Runs the graph's `InferenceSGD(eta_infer=0.05, infer_steps=20)` solver.
5. Computes the graph energy over internal prediction nodes.
6. Computes local, node-wise weight gradients and divides them by the
   prediction count.
7. Applies the AdamW update.

The training flow is implemented in
[`training/trainer.py`](../../fabricpc/training/trainer.py), and local
gradients are computed in
[`core/learning.py`](../../fabricpc/core/learning.py).

Source nodes have no incoming weights. For this graph, `pixels` therefore
receives an empty parameter/gradient dictionary; `hidden1`, `hidden2`, and
`class` own the trainable parameters.

## Latent initialization and clamping

The graph uses the default graph state initializer,
`FeedforwardStateInit`, unless one is supplied to `graph(...)`. It requires
the current parameters and propagates predictions in graph order. Unclamped
nodes start with `z_latent = z_mu`; clamped nodes retain their clamp as
`z_latent` while their prediction and error are computed from upstream inputs.

During training, both the input and target are clamped. During evaluation,
only the input is clamped; the target remains free and is used only to
calculate metrics. This distinction is implemented by
`build_clamps(..., clamp_target=True)` in training and
`build_clamps(..., clamp_target=False)` in evaluation.

The state initializer and clamp rules are defined in
[`graph_initialization/state_initializer.py`](../../fabricpc/graph_initialization/state_initializer.py)
and [`training/trainer.py`](../../fabricpc/training/trainer.py).

## Inference, derivatives, and muPC

The baseline inference solver is `InferenceSGD` with 20 steps and learning
rate `0.05`. Each step performs a forward pass, accumulates local latent
gradients, and updates every unclamped latent. Clamped nodes are not updated.
The node contract computes predictions, prediction errors, energy, and
derivatives through pure JAX-compatible functions.

muPC is **disabled** in this baseline: the call to `graph(...)` does not pass a
`MuPCConfig` through its `scaling` argument, so each node's
`node_info.scaling_config` is `None`. Consequently:

- [`core/scaling.py`](../../fabricpc/core/scaling.py) returns inputs and
  gradients unchanged;
- no forward input, latent gradient, or weight gradient receives muPC
  scaling;
- the baseline must not be compared with a muPC-enabled run under the same
  identifier.

If a later control enables muPC, it is a different baseline variant and must
record its `MuPCConfig`, graph topology, and scaling factors.

## Evaluation protocol

Evaluation calls:

```python
evaluate(trained_params, structure, test_loader, train_config, eval_key)
```

Because the algorithm remains `"pc"`, evaluation initializes states with only
the `pixels` clamp and runs the same 20-step iterative inference procedure.
It does not use a separate feedforward-only evaluation path.

The default metrics are derived from the target node's energy functional:

- `accuracy`;
- `target_energy`;
- `cross_entropy` for the `CrossEntropyEnergy` target;
- `energy` for the PC objective.

The test loader is deterministic (`shuffle=False`). Record all returned
metrics, the number of test samples, and the exact evaluation key in the run
manifest.

## Checkpoint and run manifest contract

The example currently keeps `trained_params` in memory and does not write a
checkpoint. Before using the baseline as a dependency of A-02 or A-03, save
the returned `TrainResult.params` with a checkpoint mechanism such as
Orbax, and store a manifest next to it. The manifest must include:

```text
baseline_commit
python_version
dependency_versions
jax_devices
master_seed
graph_seed
train_seed
eval_seed
graph definition and node shapes
representation_node = hidden2
parameter_initialization = XavierInitializer for hidden1/hidden2/class
latent_initialization = FeedforwardStateInit
inference = InferenceSGD(eta_infer=0.05, infer_steps=20)
optimizer = AdamW(learning_rate=0.001, weight_decay=0.1)
batch_size = 200
epochs = 20
train_loader_seed = 42
training_algorithm = pc
muPC = disabled
training metrics by epoch
evaluation metrics
checkpoint path and checksum
```

The checkpoint must contain trained parameters only (plus the optimizer state
if resuming is required). It must not contain Hopfield keys, values, readout
parameters, or test-derived data. Later readout experiments should load this
checkpoint, verify the checksum, and assert that the baseline parameters and
inference trajectory are unchanged when the readout is disabled.

## Baseline run checklist

- [ ] Run from commit `8406e6a838442391fd3089958e1a2c1b44c57eda`.
- [ ] Capture exact Python, package, JAX, backend, and device versions.
- [ ] Confirm four nodes, three edges, and 218,058 initialized parameters.
- [ ] Save the `hidden2` representation contract and checkpoint manifest.
- [ ] Confirm training clamps `pixels` and `class`.
- [ ] Confirm evaluation clamps `pixels` only and runs iterative inference.
- [ ] Confirm `scaling=None` and `node_info.scaling_config is None`.
- [ ] Record epoch metrics, final evaluation metrics, runtime, and checkpoint
      checksum.
- [ ] Use the saved trained checkpoint as the sole baseline input to later
      HopfieldReadout experiments.
