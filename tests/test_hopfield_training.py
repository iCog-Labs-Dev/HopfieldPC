"""Integration checks for fixed Hopfield banks in the training API."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from conftest import ListLoader, max_param_diff
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_graph_state, initialize_params
from fabricpc.hopfield import prepare_hopfield_bank
from fabricpc.nodes import HopfieldLinear, Linear
from fabricpc.training import evaluate, make_train_step, train


@pytest.fixture
def structure():
    source = Linear(shape=(2,), name="input")
    hidden = HopfieldLinear(
        shape=(2,),
        name="hidden",
        beta=2.0,
        hopfield_strength=0.75,
        weight_init=NormalInitializer(std=0.3),
    )
    output = Linear(shape=(2,), name="output", weight_init=NormalInitializer(std=0.3))
    return graph(
        nodes=[source, hidden, output],
        edges=[
            Edge(source=source, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=source, y=output),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
    )


@pytest.fixture
def banks():
    memory = jnp.asarray([[1.0, 0.25], [np.nan, np.inf], [-0.5, 0.75]])
    return (
        prepare_hopfield_bank(memory, active_mask=[True, False, False]),
        prepare_hopfield_bank(memory, active_mask=[False, False, True]),
    )


@pytest.fixture
def batches():
    return [
        {
            "x": jnp.asarray([[0.7, 0.25], [-0.5, 1.0]]),
            "y": jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        },
        {
            "x": jnp.asarray([[-0.75, -0.3], [0.5, 0.6]]),
            "y": jnp.asarray([[0.0, 1.0], [1.0, 0.0]]),
        },
    ]


@pytest.fixture(
    params=[
        pytest.param(None, id="single-device"),
        pytest.param(
            2,
            id="two-device-mesh",
            marks=pytest.mark.skipif(
                jax.device_count() < 2, reason="needs two devices"
            ),
        ),
    ]
)
def mesh(request):
    if request.param is None:
        return None
    return jax.make_mesh((request.param,), ("data",))


def assert_bank_equal(actual, expected, mesh):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
        assert actual[key].dtype == value.dtype
        if mesh is not None:
            sharding = actual[key].sharding
            assert sharding.is_fully_replicated
            assert sharding.device_set == set(mesh.devices.flat)


def test_make_train_step_accepts_different_runtime_banks(
    structure, rng_key, batches, banks, mesh
):
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(learning_rate=0.01, weight_decay=0.2)
    opt_state = optimizer.init(params)
    step = make_train_step(structure, optimizer, mesh=mesh)
    _, _, _, first = step(
        params, opt_state, batches[0], rng_key, node_buffers={"hidden": banks[0]}
    )
    _, _, _, second = step(
        params, opt_state, batches[0], rng_key, node_buffers={"hidden": banks[1]}
    )

    assert_bank_equal(first.nodes["hidden"].buffers, banks[0], mesh)
    assert_bank_equal(second.nodes["hidden"].buffers, banks[1], mesh)
    assert not np.allclose(
        first.nodes["hidden"].z_latent, second.nodes["hidden"].z_latent
    )


def test_train_keeps_bank_fixed_through_adamw_updates(
    structure, rng_key, batches, banks, mesh
):
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(learning_rate=0.01, weight_decay=0.2)
    bank = banks[0]
    bank_snapshot = {key: np.array(value, copy=True) for key, value in bank.items()}
    hidden_weights = [
        np.array(next(iter(params.nodes["hidden"].weights.values())), copy=True)
    ]

    def check_step(ctx):
        assert_bank_equal(ctx.state.nodes["hidden"].buffers, bank_snapshot, mesh)
        np.testing.assert_array_equal(ctx.state.nodes["input"].z_latent, ctx.batch["x"])
        np.testing.assert_array_equal(
            ctx.state.nodes["output"].z_latent, ctx.batch["y"]
        )
        current = np.array(
            next(iter(ctx.params.nodes["hidden"].weights.values())), copy=True
        )
        assert not np.array_equal(current, hidden_weights[-1])
        hidden_weights.append(current)

    options = dict(mesh=mesh, verbose=False, node_buffers={"hidden": bank})
    args = (
        params,
        structure,
        ListLoader(batches),
        optimizer,
        {"num_epochs": 1},
        rng_key,
    )
    with_callbacks = train(*args, iter_callback=check_step, **options)
    assert with_callbacks.step == len(batches)
    assert len(hidden_weights) == len(batches) + 1
    assert_bank_equal(bank, bank_snapshot, None)

    without_callbacks = train(*args, **options)
    assert max_param_diff(without_callbacks.params, with_callbacks.params) < 1e-7
    assert_bank_equal(bank, bank_snapshot, None)


def test_evaluate_uses_fixed_bank_with_free_targets(
    structure, rng_key, batches, banks, mesh
):
    params = initialize_params(structure, rng_key)
    batches = [batches[0], {key: value[:1] for key, value in batches[1].items()}]
    buffers = {"hidden": banks[0]}

    def output_latent(state, batch, structure):
        value = state.nodes["output"].z_latent[:, 0]
        return value, jnp.ones_like(value)

    measured = evaluate(
        params,
        structure,
        ListLoader(batches),
        {},
        rng_key,
        mesh=mesh,
        metrics={"output_latent": output_latent},
        node_buffers=buffers,
    )

    expected = []
    for batch_idx, batch in enumerate(batches):
        clamps = {"input": batch["x"]}
        key = jax.random.fold_in(rng_key, batch_idx)
        initial = initialize_graph_state(
            structure, len(batch["x"]), key, clamps, params=params, node_buffers=buffers
        )
        final = run_inference(params, initial, clamps, structure)
        expected.append(final.nodes["output"].z_latent[:, 0])

    np.testing.assert_allclose(
        measured["output_latent"], jnp.concatenate(expected).mean(), atol=1e-6
    )
