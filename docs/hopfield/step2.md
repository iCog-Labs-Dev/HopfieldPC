# Step 2 Fixed-bank Hopfield adapter

Step 2 connects the [standalone Hopfield core](step1.md) to a vector-valued FabricPC hidden node. `HopfieldLinear` inherits prediction and parameter initialization from `Linear`. At each inference evaluation it adds a per-sample Hopfield term at the node's current latent $z$.

$$
E_{\mathrm{total}}(z,z_\mu)=E_{\mathrm{PC}}(z,z_\mu)+\lambda E_0(z;M,\beta),\qquad
\nabla_z E_{\mathrm{added}}=\lambda\bigl(z-R(z)\bigr).
$$

Here `hopfield_strength` is the coupling $\lambda$ and `beta` is the inverse temperature. The existing inference solver evaluates the node, accumulates the latent gradient, and applies its update and clamp rules. The adapter uses the same active mask as the standalone core. At zero coupling the energy hook takes the baseline path, which makes a matched `Linear` graph a direct control. [Adapter implementation](../../fabricpc/nodes/hopfield_linear.py)

## Carry a fixed bank through an episode

Prepare a bank with the same feature dimension and floating dtype as the hidden latent. Its shape is `[N, D]`, where `N` is capacity and `D` is the node's feature dimension. The boolean mask has shape `[N]`, with at least one active slot. Run host validation before passing the prepared buffers into compiled code.

~~~python
from fabricpc.hopfield import prepare_hopfield_bank

node_buffers = {
    "hidden": prepare_hopfield_bank(memory, active_mask=active_mask),
}
~~~

Use the finalized node name as the key. Pass `node_buffers=node_buffers` to `initialize_graph_state` for an explicit inference run, to the callable returned by `make_train_step` for a custom loop, or to `train` and `evaluate` for the standard paths. The bank and mask reach the energy hook in the hidden node's `NodeState`. Optax receives the learned `GraphParams`, leaving the bank and mask outside the optimizer tree. With a data mesh, each device receives the full bank and mask. [Training and evaluation API](../../fabricpc/training/trainer.py)

## Run the checkpoint example

From the repository root, run the synthetic CPU example with FabricPC and its standard dependencies installed.

~~~bash
python examples/hopfield_fixed_bank.py
~~~

The example trains a small input–hidden–output graph with `InferenceSGD` and AdamW, saves after the first epoch, restores from disk, and runs the second epoch. It compares the result with a continuous two-epoch run. The default uses a temporary checkpoint. To keep it, pass a new path.

~~~bash
python examples/hopfield_fixed_bank.py --checkpoint-dir /tmp/hopfield-step2-checkpoint
~~~

The run checks exact equality on CPU for the restored settings, parameters, optimizer state, bank, and mask. It also checks the next batch's inference state, optimizer update, and metrics against the continuous run, then evaluates the restored model with targets free. [Runnable example](../../examples/hopfield_fixed_bank.py)

## Save and restore the whole episode input

The example uses one Orbax checkpoint with these fields.

| Field | Purpose at restore |
| --- | --- |
| `params` | Restores the model weights and biases. |
| `opt_state` | Restores AdamW moments and the update count. |
| `node_buffers` | Restores the fixed bank and boolean active mask for the hidden node. |
| `settings` | Rebuilds the adapter, inference solver, latent initializer, and optimizer with the saved settings. |
| `rng_key` and `next_epoch` | Reproduce the trainer's latent-initialization RNG stream for the next epoch. |

Build a fresh restore target with the same graph and optimizer tree layout and the expected bank shapes and dtypes. The target provides the types and placement for this example's CPU restore. Orbax fills in the saved values. Prepare the restored bank again at the host boundary to validate its active slots, rebuild the graph and optimizer from `settings`, and resume with the saved `opt_state`, `rng_key`, `node_buffers`, and `start_epoch=next_epoch`. The example resumes at an epoch boundary with the same ordered synthetic batches. In a data pipeline, retain the loader or sampler state needed to reproduce that order. [Checkpoint restore code](../../examples/hopfield_fixed_bank.py)

## Verification

The tests cover host bank validation, runtime transport through the initializers, exact baseline parity at zero coupling, the analytical added latent gradient, clamp preservation, and AdamW updates with a fixed bank. Sharding tests exercise replicated buffers, including evaluation with a padded final batch. The checkpoint test performs a disk round trip and checks the next inference and training step. [Bank tests](../../tests/test_hopfield_bank.py) · [Adapter tests](../../tests/test_hopfield_linear.py) · [Training tests](../../tests/test_hopfield_training.py) · [Checkpoint test](../../tests/test_hopfield_checkpoint.py)

~~~bash
python -m pytest tests/test_hopfield_bank.py tests/test_hopfield_linear.py tests/test_hopfield_training.py tests/test_hopfield_checkpoint.py -q
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
    python -m pytest tests/test_hopfield_training.py tests/test_sharding.py -q
~~~
