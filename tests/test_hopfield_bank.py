"""Checks for fixed Hopfield banks carried in FabricPC runtime state."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import (
    FeedforwardStateInit,
    GlobalStateInit,
    NodeDistributionStateInit,
    initialize_graph_state,
    initialize_params,
)
from fabricpc.hopfield import prepare_hopfield_bank
from fabricpc.nodes import Linear


class BankEnergyProbe(Linear):
    """Make the bank visible through the existing energy hook."""

    @staticmethod
    def energy(params, inputs, state, aux, node_info):
        memory = state.buffers["hopfield_memory"]
        mask = state.buffers["hopfield_active_mask"]
        contribution = jnp.sum(jnp.where(mask[:, None], memory, 0))
        return Linear.energy(params, inputs, state, aux, node_info) + contribution


def make_graph(initializer):
    source = Linear(shape=(2,), name="input")
    hidden = BankEnergyProbe(shape=(2,), name="hidden")
    output = Linear(shape=(2,), name="output")
    return graph(
        nodes=[source, hidden, output],
        edges=[
            Edge(source=source, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=source, y=output),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
        graph_state_initializer=initializer,
    )


def test_prepare_hopfield_bank_retains_inactive_storage_and_dtype():
    memory = jnp.asarray([[1.0, 2.0], [np.nan, np.inf]], jnp.float16)
    mask = jnp.asarray([True, False])
    buffers = prepare_hopfield_bank(memory, active_mask=mask)
    np.testing.assert_array_equal(buffers["hopfield_memory"], memory)
    assert buffers["hopfield_memory"].dtype == jnp.float16
    np.testing.assert_array_equal(buffers["hopfield_active_mask"], mask)
    assert buffers["hopfield_active_mask"].dtype == jnp.bool_

    default_buffers = prepare_hopfield_bank(memory[:1])
    np.testing.assert_array_equal(default_buffers["hopfield_active_mask"], [True])


@pytest.mark.parametrize(
    ("memory", "mask", "error", "message"),
    [
        ([1.0], None, ValueError, "memory must have shape"),
        (np.empty((0, 2)), None, ValueError, "memory must have shape"),
        (np.empty((2, 0)), None, ValueError, "memory must have shape"),
        ([[1, 2]], None, TypeError, "Inputs must use float"),
        ([[1.0]], [True, False], ValueError, "active_mask must have shape"),
        ([[1.0]], [1], TypeError, "active_mask must have boolean dtype"),
        ([[1.0]], [False], ValueError, "at least one active memory"),
        ([[np.nan]], None, ValueError, "Active memory values must be finite"),
        ([[np.inf]], None, ValueError, "Active memory values must be finite"),
    ],
)
def test_prepare_hopfield_bank_rejects_invalid_inputs(memory, mask, error, message):
    with pytest.raises(error, match=message):
        prepare_hopfield_bank(memory, active_mask=mask)


def test_prepare_hopfield_bank_requires_concrete_inputs():
    memory = jnp.ones((1, 2))
    with pytest.raises(TypeError, match="outside JAX transformations"):
        jax.jit(prepare_hopfield_bank)(memory)
    with pytest.raises(TypeError, match="outside JAX transformations"):
        jax.jit(lambda mask: prepare_hopfield_bank(memory, active_mask=mask))(
            jnp.asarray([True])
        )


@pytest.mark.parametrize(
    "initializer",
    [FeedforwardStateInit(), GlobalStateInit(), NodeDistributionStateInit()],
)
def test_hopfield_bank_reaches_node_under_jit_and_survives_inference(
    initializer, rng_key
):
    structure = make_graph(initializer)
    params = initialize_params(structure, rng_key)
    clamps = {
        "input": jnp.asarray([[1.0, 0.5], [0.0, 1.0]]),
        "output": jnp.asarray([[0.0, -0.5], [0.25, 0.0]]),
    }
    memory = jnp.asarray([[1.0, 2.0], [np.nan, np.inf], [3.0, 4.0]])
    first_bank = prepare_hopfield_bank(memory, active_mask=[True, False, False])
    second_bank = prepare_hopfield_bank(memory, active_mask=[False, False, True])

    @jax.jit
    def run(bank):
        initial = initialize_graph_state(
            structure, 2, rng_key, clamps, params=params, node_buffers={"hidden": bank}
        )
        final = run_inference(params, initial, clamps, structure)
        return initial, final, compute_local_weight_gradients(params, final, structure)

    initial, final, grads = run(first_bank)
    other_initial, other_final, _ = run(second_bank)
    for state, bank in (
        (initial, first_bank),
        (final, first_bank),
        (other_initial, second_bank),
        (other_final, second_bank),
    ):
        np.testing.assert_array_equal(state.nodes["output"].z_latent, clamps["output"])
        for name, array in bank.items():
            np.testing.assert_array_equal(state.nodes["hidden"].buffers[name], array)
        assert state.nodes["input"].buffers is None

    assert not np.allclose(
        initial.nodes["hidden"].z_latent, final.nodes["hidden"].z_latent
    )
    np.testing.assert_allclose(
        other_final.nodes["hidden"].energy - final.nodes["hidden"].energy, 4.0
    )
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(params)


def test_initialize_graph_state_rejects_unknown_buffer_node(rng_key):
    structure = make_graph(GlobalStateInit())
    buffers = prepare_hopfield_bank(jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="unknown nodes.*missing"):
        initialize_graph_state(structure, 1, rng_key, node_buffers={"missing": buffers})
