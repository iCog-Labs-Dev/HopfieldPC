# Step 2 — Fixed-bank Hopfield adapter

A fixed bank is a collection of prototype vectors that a hidden node consults during inference. The caller picks the vectors before the fixed-bank run. **The bank is used during both training and testing.** Its rows and active mask stay fixed while the hidden latent moves and the retrieval weights are recomputed. Step 2 connects the [standalone JAX Hopfield core](step1.md) to FabricPC through the `HopfieldLinear` node.

## Implementation changes

- **Carry the bank through inference.** The graph initializer attaches the validated bank and mask to the hidden node before its first forward pass. The node carries those buffers through each inference step.
- **Add Hopfield energy to a linear hidden node.** `HopfieldLinear` reuses the linear predictor and adds the Hopfield term at the current hidden latent. Zero coupling selects ordinary `Linear` behavior.
- **Pass the bank through training and evaluation.** Both entry points accept the same runtime buffers. Each device on a data mesh gets a full copy, and AdamW updates the model parameters separately.
- **Resume training with the same bank.** The Orbax example restores the bank alongside the training state and checks the next inference and optimizer step against uninterrupted training.

## 1. Choose the size and select the prototypes

| Quantity | How to choose it |
| --- | --- |
| Hidden width `D` | Use the hidden node's width. A node with `shape=(128,)` needs 128 features per memory vector. The memory and latent must share a floating dtype. |
| Capacity `N` | Allocate one row per prototype you want to store, plus any spare rows. Ten classes with one prototype per class need 10 rows. Four prototypes per class need 40. |
| Active slots | Supply a boolean mask of shape `[N]`. A false entry disables that row. At least one row must be active. Omitting the mask activates every row. |

The bank has shape `[N, D]`. Every sample in the batch consults the same bank. For a batch of size `B`, scoring all allocated rows costs approximately `O(B N D)`. Spare rows still occupy storage and take part in the array computation.

**In the demo**, the three vectors are hand-picked. The hidden width is 2, the bank has capacity 3, and only its first and third rows are active.

**With a dataset**, one way to choose prototypes is to train a plain `Linear` baseline first. With its parameters fixed, run inference on training examples and collect the latents at the hidden layer you plan to replace. For classification, average the latents within each class to get one prototype per class, or cluster them within each class to get several.

Freeze those prototypes before starting the adapter run. If you start the adapter from the baseline weights, the bank and the model begin in the same hidden representation. Prototype selection is a separate preparation stage. The adapter begins using the supplied vectors on the first training batch. [Demo bank](../../examples/hopfield_fixed_bank.py)

## 2. Attach the bank to the hidden node

In the demo, the graph is input → `HopfieldLinear` hidden node → linear output. The adapter inherits its predictor and parameter initialization from `Linear`. Initialize and connect it as you would a linear hidden node, then prepare the bank outside JIT and pass it under the node's finalized name.

~~~python
import jax.numpy as jnp

from fabricpc.hopfield import prepare_hopfield_bank
from fabricpc.nodes import HopfieldLinear

hidden = HopfieldLinear(
    shape=(2,), name="hidden", hopfield_strength=0.75, beta=2.0
)
memory = jnp.asarray([[1.0, 0.25], [4.0, -2.0], [-0.5, 0.75]])
node_buffers = {
    "hidden": prepare_hopfield_bank(memory, active_mask=[True, False, True]),
}
~~~

`prepare_hopfield_bank` checks the shape and dtype, and verifies that active rows contain finite values. It packages the supplied values as `hopfield_memory` and `hopfield_active_mask`. Pass `node_buffers=node_buffers` to `train` and `evaluate`, to the step returned by `make_train_step` for a custom training loop, or to `initialize_graph_state` for manual inference. With a data mesh, each device gets the full bank and mask. [Bank preparation](../../fabricpc/hopfield/_validation.py) · [Training and evaluation API](../../fabricpc/training/trainer.py)

The graph initializer puts the bank in `state.nodes["hidden"].buffers`. It initializes `state.nodes["hidden"].z_latent` separately. That latent has shape `[B, D]` and provides one query per sample. The demo draws its starting latents from a zero-mean normal distribution with standard deviation 0.2. The ordinary linear predictor produces a separate value, `z_mu`, from the incoming activation. [State initialization](../../fabricpc/graph_initialization/state_initializer.py) · [Linear predictor](../../fabricpc/nodes/linear.py)

## 3. Use the current latent to read the bank

At inference step $t$, a sample's current hidden latent $z_t$ scores each active memory $m_i$ by dot product. With active set $A$ and inverse temperature $\beta$, the scores $s_i$, weights $p_i$, and retrieved vector $R(z_t)$ are

$$
s_i=\beta\,m_i^\top z_t,\qquad
p_i=\frac{\exp(s_i)}{\sum_{j\in A}\exp(s_j)}\quad(i\in A),\qquad
R(z_t)=\sum_{i\in A}p_i m_i.
$$

Masked rows have weight zero. A larger `beta` sharpens the preference for higher dot products. With one active row, it receives all the weight and the retrieved vector is that row for every query. [Retrieval implementation](../../fabricpc/hopfield/functional.py)

The adapter adds Hopfield energy $E_H$ to the hidden node's ordinary predictive-coding energy. For the default Gaussian energy, the ordinary term is $\tfrac12\lVert z_t-z_\mu\rVert^2$. The coupling $\lambda$ is `hopfield_strength`.

$$
\begin{aligned}
E_H(z_t) &= \tfrac12\lVert z_t\rVert^2
  -\tfrac1\beta\log\sum_{i\in A}\exp(\beta m_i^\top z_t),\\
E_{\mathrm{node}} &= E_{\mathrm{PC,node}}+\lambda E_H(z_t),\\
\nabla_{z_t}\bigl(\lambda E_H(z_t)\bigr) &= \lambda\bigl(z_t-R(z_t)\bigr).
\end{aligned}
$$

The solver combines this gradient with the usual predictive-coding gradient $g_{\mathrm{PC}}$, which includes the hidden node's own prediction error and downstream contributions. For the demo's `InferenceSGD`, a free hidden latent takes this step at inference rate $\eta$.

$$
z_{t+1}=z_t-\eta\,g_{\mathrm{PC}}
             +\eta\lambda\bigl(R(z_t)-z_t\bigr).
$$

The last term moves the latent toward the retrieved vector as part of the full solver update. The solver then uses the new latent to score the same bank. A clamped latent stays at its assigned value. Setting $\lambda=0$ gives ordinary `Linear` behavior and lets the node run without a bank. [Energy hook](../../fabricpc/nodes/hopfield_linear.py) · [Inference solver](../../fabricpc/core/inference.py)

**Worked example.** For the demo's first training sample, $z_0\approx(-0.0258,-0.1546)$ gives retrieval weights $(0.5193,0,0.4807)$ and $R(z_0)\approx(0.2790,0.4903)$. With $\eta=0.1$ and $\lambda=0.75$, the bank contributes about $(+0.0229,+0.0484)$ to the update. After the ordinary predictive-coding update is included, the solver reaches $z_1\approx(0.0355,-0.1279)$. The next inference step reads the same rows with this new query. [Runnable example](../../examples/hopfield_fixed_bank.py)

## 4. Use the same bank in training and testing

| Stage | What happens |
| --- | --- |
| Training batch | With inputs and targets clamped, predictive-coding inference uses the bank to update the free hidden latents. The trainer then computes weight gradients from the inferred state, and AdamW updates the model weights and biases. |
| Testing or evaluation batch | The inputs are clamped and the targets are left free. The same bank guides inference of the hidden and output latents. The model parameters stay fixed. |
| Next batch or epoch | The trainer initializes fresh latents for each batch and reuses the supplied memory rows and mask. To use different prototypes, the caller prepares and supplies a different bank explicitly. |

The bank and mask are runtime buffers outside `GraphParams`. The energy hook stops gradients into the memory, and Optax updates only the parameter tree. The bank can influence the learned weights through the hidden states that inference produces. [Trainer](../../fabricpc/training/trainer.py) · [Fixed-bank training tests](../../tests/test_hopfield_training.py)

## 5. Run and resume the example

Run the synthetic CPU example from the repository root with FabricPC and its dependencies installed.

~~~bash
python examples/hopfield_fixed_bank.py
~~~

The example saves after the first epoch, restores from disk, and runs the second epoch. It checks that the restored checkpoint matches the saved parameters, optimizer state, bank and mask, settings, and resume position exactly. It then checks that the next inference state, optimizer update, and metrics match uninterrupted training on CPU. Finally, it evaluates with free targets. By default it uses a temporary checkpoint. To keep the checkpoint, supply a new path.

~~~bash
python examples/hopfield_fixed_bank.py \
    --checkpoint-dir /tmp/hopfield-step2-checkpoint
~~~

The Orbax checkpoint contains the inputs needed to resume at the epoch boundary.

| Field | What it restores |
| --- | --- |
| `params` | Model weights and biases. |
| `opt_state` | AdamW moments and update count. |
| `node_buffers` | The fixed bank and boolean active mask. |
| `settings` | The adapter, solver, latent initializer, and optimizer configuration. |
| `rng_key` and `next_epoch` | The latent-initialization random stream for the next epoch. |

Build the restore target with the same parameter and optimizer tree layout and the expected bank shapes and dtypes. The template supplies types, shapes, and device placement for the CPU restore. Orbax fills in the saved values. Validate the restored bank again with `prepare_hopfield_bank`, rebuild the graph and optimizer from the restored settings, and resume with the saved optimizer state, random key, and node buffers and with `start_epoch=next_epoch`. Matching an uninterrupted run also requires the same batch order. The synthetic example uses a fixed batch. A data pipeline can preserve the loader or sampler state needed to reproduce its order. [Checkpoint restore code](../../examples/hopfield_fixed_bank.py)

## Verification

The CPU checkpoint example and 160 focused tests passed in the Step 2 verification run, including checks with two forced CPU devices.

### Bank preparation and inference

- `test_prepare_hopfield_bank_retains_inactive_storage_and_dtype`\
  Verifies that preparation preserves the supplied dtype and stored values, including nonfinite values in inactive rows. It checks both the supplied boolean mask and the default all-active mask.
- `test_prepare_hopfield_bank_rejects_invalid_inputs`\
  Checks errors for malformed or empty banks, integer memories, a mask with the wrong shape or dtype, an all-inactive mask, and nonfinite values in active rows.
- `test_prepare_hopfield_bank_requires_concrete_inputs`\
  Checks that preparation rejects a traced memory array or mask inside JIT. This keeps value validation at the host boundary.
- `test_hopfield_bank_reaches_node_under_jit_and_survives_inference`\
  Exercises all three built-in state initializers. Changing the runtime mask changes the probe energy under JIT. Within each run, the bank stays fixed while the hidden latent moves, the target clamp stays fixed, and the weight gradients match the parameter tree layout.
- `test_initialize_graph_state_rejects_unknown_buffer_node`\
  Checks that the initializer reports the offending node name when a buffer entry targets a node missing from the graph. [Bank tests](../../tests/test_hopfield_bank.py)

### Hidden-state energy and the solver

- `test_zero_strength_matches_linear_through_inference_and_learning`\
  Compares a zero-coupling adapter with an ordinary `Linear` graph. It checks exact equality for the initialized parameters, the full compiled inference trajectory, and the local weight gradients.
- `test_positive_strength_adds_energy_and_gradient_at_current_latent`\
  Checks the per-sample energy and the analytical added latent gradient at two current latent states. It also checks that prediction, error, and input gradients match the baseline at those states and that the gradient into the fixed bank is zero.
- `test_standard_solver_applies_added_gradient_and_preserves_clamps`\
  Checks that one `InferenceSGD` step applies the expected additional gradient to a free hidden latent. It also checks the bank and every clamp, including the case where the hidden node itself is clamped.
- `test_positive_strength_requires_matching_bank`\
  Checks that a coupled internal node gives an error for a missing bank or a bank whose feature width differs from the hidden latent's.
- `test_source_node_can_use_baseline_energy_without_bank`\
  Exercises a source node with no incoming edges. Without a bank, it keeps the input clamp and reports the baseline energy.
- `test_invalid_configuration_is_rejected`\
  Checks that construction rejects invalid coupling values and inverse temperatures.
- `test_requires_vector_hidden_shape`\
  Checks that the adapter rejects scalar, empty, and non-vector hidden shapes. [Adapter tests](../../tests/test_hopfield_linear.py)

### Training and evaluation

- `test_make_train_step_accepts_different_runtime_banks`\
  Passes different masks to the same compiled training step and checks that the hidden states differ. On both single-device and two-device paths, it checks exact buffer contents, and on the mesh it checks full replication.
- `test_train_keeps_bank_fixed_through_adamw_updates`\
  Checks that hidden weights change after each batch while the bank, mask, and input and target clamps stay fixed. It also checks that training with and without a callback yields matching parameters.
- `test_evaluate_uses_fixed_bank_with_free_targets`\
  Compares the evaluation metric with a manual inference run where the targets are free. The two-device case includes a smaller final batch that requires padding. [Training tests](../../tests/test_hopfield_training.py) · [Sharding tests](../../tests/test_sharding.py)

### Checkpoint continuation

- `test_restored_fixed_bank_matches_uninterrupted_training`\
  Runs the example through a disk checkpoint round trip. It checks exact agreement with uninterrupted training for the next inference state, parameter update, optimizer state, and metrics. It also reopens the checkpoint and verifies the saved bank, boolean mask, and epoch offset. [Checkpoint test](../../tests/test_hopfield_checkpoint.py)

### Run the tests

~~~bash
python -m pytest tests/test_hopfield_bank.py tests/test_hopfield_linear.py \
    tests/test_hopfield_training.py tests/test_hopfield_checkpoint.py -q
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
    python -m pytest tests/test_hopfield_training.py tests/test_sharding.py -q
~~~
