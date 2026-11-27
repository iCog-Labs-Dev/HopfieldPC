"""Verify the fixed-bank Hopfield energy on a FabricPC linear node."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD, gather_inputs, run_inference
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_graph_state, initialize_params
from fabricpc.hopfield import hopfield_energy, hopfield_retrieve, prepare_hopfield_bank
from fabricpc.nodes import HopfieldLinear, Linear


def make_structure(hidden_type=Linear, *, infer_steps=3, **options):
    source = Linear(shape=(2,), name="input")
    hidden = hidden_type(
        shape=(2,),
        name="hidden",
        activation=TanhActivation(),
        energy=GaussianEnergy(precision=1.7),
        weight_init=NormalInitializer(std=0.3),
        **options,
    )
    output = Linear(shape=(2,), name="output", weight_init=NormalInitializer(std=0.3))
    return graph(
        nodes=[source, hidden, output],
        edges=[
            Edge(source=source, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=source, y=output),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=infer_steps),
    )


@pytest.fixture
def clamps():
    return {
        "input": jnp.asarray([[1.0, 0.5], [0.0, 1.0]]),
        "output": jnp.asarray([[0.0, -0.5], [0.25, 0.0]]),
    }


@pytest.fixture
def bank():
    memory = jnp.asarray([[1.0, 0.25], [np.nan, np.inf], [-0.5, 0.75]])
    return prepare_hopfield_bank(memory, active_mask=[True, False, True])


def assert_same_tree(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_zero_strength_matches_linear_through_inference_and_learning(rng_key, clamps):
    baseline = make_structure()
    adapted = make_structure(HopfieldLinear, hopfield_strength=0.0)
    baseline_params = initialize_params(baseline, rng_key)
    adapted_params = initialize_params(adapted, rng_key)
    assert_same_tree(adapted_params, baseline_params)

    def episode(structure, params):
        def run(p):
            state = initialize_graph_state(structure, 2, rng_key, clamps, params=p)
            trajectory = [state]
            solver = structure.config["inference"]
            for _ in range(solver.config["infer_steps"]):
                state = InferenceSGD.inference_step(
                    p, state, clamps, structure, solver.config
                )
                trajectory.append(state)
            return trajectory, compute_local_weight_gradients(p, state, structure)

        return jax.jit(run)(params)

    assert_same_tree(
        episode(adapted, adapted_params), episode(baseline, baseline_params)
    )


def test_positive_strength_adds_energy_and_gradient_at_current_latent(
    rng_key, clamps, bank
):
    strength, beta = 0.7, 2.5
    baseline = make_structure()
    adapted = make_structure(HopfieldLinear, hopfield_strength=strength, beta=beta)
    params = initialize_params(baseline, rng_key)
    initial = initialize_graph_state(baseline, 2, rng_key, clamps, params=params)
    info = baseline.nodes["hidden"].node_info
    adapted_info = adapted.nodes["hidden"].node_info
    inputs = gather_inputs(info, baseline, initial)
    hidden_params = params.nodes["hidden"]

    @jax.jit
    def evaluate(z, buffers):
        state = initial.nodes["hidden"]._replace(z_latent=z)
        return (
            Linear.forward_and_latent_grads(hidden_params, inputs, state, info, False),
            HopfieldLinear.forward_and_latent_grads(
                hidden_params,
                inputs,
                state._replace(buffers=buffers),
                adapted_info,
                False,
            ),
        )

    memory, mask = bank["hopfield_memory"], bank["hopfield_active_mask"]
    for z in (jnp.asarray([[0.8, 0.2], [-0.3, 0.6]]), jnp.zeros((2, 2))):
        (base, base_inputs, base_grad), (state, input_grads, grad) = evaluate(z, bank)
        expected_energy = hopfield_energy(z, memory, beta=beta, active_mask=mask)
        retrieved, _ = hopfield_retrieve(z, memory, beta=beta, active_mask=mask)
        np.testing.assert_array_equal(state.z_mu, base.z_mu)
        np.testing.assert_array_equal(state.error, base.error)
        assert state.energy.shape == (2,)
        np.testing.assert_allclose(
            state.energy - base.energy, strength * expected_energy, atol=1e-6
        )
        np.testing.assert_allclose(
            grad - base_grad, strength * (z - retrieved), atol=1e-6
        )
        assert_same_tree(input_grads, base_inputs)

    def energy_for_memory(memory):
        state = initial.nodes["hidden"]._replace(
            buffers={"hopfield_memory": memory, "hopfield_active_mask": mask}
        )
        return HopfieldLinear.forward(
            hidden_params, inputs, state, adapted_info
        ).energy.sum()

    np.testing.assert_array_equal(
        jax.grad(energy_for_memory)(memory), jnp.zeros_like(memory)
    )


@pytest.mark.parametrize("clamp_hidden", [False, True])
def test_standard_solver_applies_added_gradient_and_preserves_clamps(
    rng_key, clamps, bank, clamp_hidden
):
    strength, beta = 0.8, 2.0
    baseline = make_structure(infer_steps=1)
    adapted = make_structure(
        HopfieldLinear, infer_steps=1, hopfield_strength=strength, beta=beta
    )
    params = initialize_params(baseline, rng_key)
    if clamp_hidden:
        clamps = {**clamps, "hidden": jnp.asarray([[0.8, 0.2], [-0.3, 0.6]])}

    base_initial = initialize_graph_state(baseline, 2, rng_key, clamps, params=params)
    initial = initialize_graph_state(
        adapted, 2, rng_key, clamps, params=params, node_buffers={"hidden": bank}
    )
    base = run_inference(params, base_initial, clamps, baseline)
    run = jax.jit(lambda state: run_inference(params, state, clamps, adapted))
    final = run(initial)

    z = initial.nodes["hidden"].z_latent
    memory, mask = bank["hopfield_memory"], bank["hopfield_active_mask"]
    retrieved, _ = hopfield_retrieve(z, memory, beta=beta, active_mask=mask)
    added_gradient = strength * (z - retrieved)
    np.testing.assert_allclose(
        final.nodes["hidden"].latent_grad - base.nodes["hidden"].latent_grad,
        added_gradient,
        atol=1e-6,
    )
    for name, value in clamps.items():
        np.testing.assert_array_equal(final.nodes[name].z_latent, value)
    if not clamp_hidden:
        step_size = adapted.config["inference"].config["eta_infer"]
        np.testing.assert_allclose(
            final.nodes["hidden"].z_latent,
            base.nodes["hidden"].z_latent - step_size * added_gradient,
            atol=1e-6,
        )
    assert_same_tree(final.nodes["hidden"].buffers, bank)


def test_positive_strength_requires_matching_bank(rng_key, clamps):
    structure = make_structure(HopfieldLinear)
    params = initialize_params(structure, rng_key)
    with pytest.raises(ValueError, match="hidden.*prepare_hopfield_bank"):
        initialize_graph_state(structure, 2, rng_key, clamps, params=params)
    wrong_bank = prepare_hopfield_bank(jnp.ones((3, 4)))
    with pytest.raises(ValueError, match="same feature dimension"):
        initialize_graph_state(
            structure,
            2,
            rng_key,
            clamps,
            params=params,
            node_buffers={"hidden": wrong_bank},
        )


def test_source_node_can_use_baseline_energy_without_bank(rng_key, clamps):
    source = HopfieldLinear(shape=(2,), name="input")
    output = Linear(shape=(2,), name="output")
    structure = graph(
        nodes=[source, output],
        edges=[Edge(source=source, target=output.slot("in"))],
        task_map=TaskMap(x=source, y=output),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=1),
    )
    params = initialize_params(structure, rng_key)
    initial = initialize_graph_state(structure, 2, rng_key, clamps, params=params)
    final = run_inference(params, initial, clamps, structure)
    np.testing.assert_array_equal(final.nodes["input"].z_latent, clamps["input"])
    np.testing.assert_array_equal(final.nodes["input"].energy, jnp.zeros(2))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"hopfield_strength": -0.25}, "hopfield_strength"),
        ({"hopfield_strength": float("inf")}, "hopfield_strength"),
        ({"hopfield_strength": float("nan")}, "hopfield_strength"),
        ({"hopfield_strength": True}, "hopfield_strength"),
        ({"beta": 0.0}, "beta"),
        ({"beta": float("inf")}, "beta"),
    ],
)
def test_invalid_configuration_is_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        HopfieldLinear(shape=(2,), name="hidden", **options)


@pytest.mark.parametrize("shape", [(), (0,), (2, 2)])
def test_requires_vector_hidden_shape(shape):
    with pytest.raises(ValueError, match="vector shape"):
        HopfieldLinear(shape=shape, name="hidden")
