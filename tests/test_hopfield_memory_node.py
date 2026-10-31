"""Tests for integrating the modern Hopfield core as a FabricPC node."""

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.hopfield import multi_step_retrieve
from fabricpc.nodes import HopfieldMemoryNode, IdentityNode, Linear
from fabricpc.training import make_train_step


def _build_graph(shape=(6,), prototypes=None, center=None):
    if prototypes is None:
        prototypes = jax.random.normal(jax.random.PRNGKey(9), (4, 6))
    source = IdentityNode(shape=shape, name="input")
    memory = HopfieldMemoryNode(
        shape=shape,
        name="memory",
        prototypes=prototypes,
        beta=1.5,
        num_updates=2,
        center=center,
    )
    structure = graph(
        nodes=[source, memory],
        edges=[Edge(source=source, target=memory.slot("in"))],
        task_map=TaskMap(x=source, y=memory),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
    )
    return structure


def _node_state(query):
    return NodeState(
        z_latent=query,
        z_mu=jnp.zeros_like(query),
        error=jnp.zeros_like(query),
        energy=jnp.zeros((query.shape[0],)),
        latent_grad=jnp.zeros_like(query),
    )


def test_hopfield_memory_node_matches_core_retrieval(rng_key):
    prototypes = jax.random.normal(rng_key, (4, 6))
    structure = _build_graph(prototypes=prototypes)
    params = initialize_params(structure, rng_key)
    batch_size = 3
    query = jax.random.normal(rng_key, (batch_size, 6))
    node = structure.nodes["memory"]
    info = node.node_info
    node_params = params.nodes["memory"]
    edge_key = info.in_edges[0]

    actual, _ = type(node).predict(
        node_params,
        {edge_key: query},
        state=_node_state(query),
        node_info=info,
    )
    expected, _ = multi_step_retrieve(
        query=query,
        memory=node.prototypes,
        beta=1.5,
        num_updates=2,
        normalize=True,
    )

    assert actual.shape == (batch_size, 6)
    assert jnp.allclose(actual, expected)
    assert node_params.weights == {}
    assert node_params.biases == {}

    compiled = jax.jit(
        lambda value: type(node).predict(
            node_params,
            {edge_key: value},
            state=_node_state(value),
            node_info=info,
        )[0]
    )
    assert jnp.allclose(compiled(query), expected)


def test_hopfield_memory_node_runs_inference_without_trainable_memory(rng_key):
    prototypes = jax.random.normal(rng_key, (4, 6))
    structure = _build_graph(prototypes=prototypes)
    params = initialize_params(structure, rng_key)
    batch_size = 2
    query = jax.random.normal(rng_key, (batch_size, 6))
    clamps = {"input": query}
    state = initialize_graph_state(
        structure,
        batch_size=batch_size,
        rng_key=rng_key,
        clamps=clamps,
        params=params,
    )

    inferred = run_inference(params, state, clamps, structure)
    assert inferred.nodes["memory"].z_mu.shape == (batch_size, 6)
    assert params.nodes["memory"].weights == {}
    assert params.nodes["memory"].biases == {}


def test_hopfield_memory_node_supports_multidimensional_shapes(rng_key):
    prototypes = jax.random.normal(rng_key, (3, 6))
    structure = _build_graph(shape=(2, 3), prototypes=prototypes)
    params = initialize_params(structure, rng_key)
    batch_size = 2
    query = jax.random.normal(rng_key, (batch_size, 2, 3))
    state = initialize_graph_state(
        structure,
        batch_size=batch_size,
        rng_key=rng_key,
        clamps={"input": query},
        params=params,
    )

    inferred = run_inference(params, state, {"input": query}, structure)

    assert inferred.nodes["memory"].z_mu.shape == (batch_size, 2, 3)


def test_hopfield_memory_node_supports_centered_memory(rng_key):
    prototypes = jax.random.normal(rng_key, (4, 6))
    center = jnp.mean(prototypes, axis=0)
    query = jax.random.normal(jax.random.PRNGKey(4), (2, 6))
    structure = _build_graph(prototypes=prototypes, center=center)
    params = initialize_params(structure, rng_key)
    info = structure.nodes["memory"].node_info
    edge_key = info.in_edges[0]

    actual, _ = HopfieldMemoryNode.predict(
        params.nodes["memory"],
        {edge_key: query},
        state=_node_state(query),
        node_info=info,
    )
    expected, _ = multi_step_retrieve(
        query=query - center,
        memory=prototypes - center,
        beta=1.5,
        num_updates=2,
        normalize=True,
    )

    assert jnp.allclose(actual, expected + center)


def test_hopfield_memory_node_rejects_mismatched_memory_dim(rng_key):
    with pytest.raises(ValueError, match="Prototype dimension"):
        HopfieldMemoryNode(
            shape=(6,),
            name="memory",
            prototypes=jax.random.normal(rng_key, (4, 5)),
        )


def test_frozen_memory_node_trains_in_a_fabricpc_model(rng_key):
    source = IdentityNode(shape=(6,), name="input")
    memory = HopfieldMemoryNode(
        shape=(6,),
        name="memory",
        prototypes=jax.random.normal(jax.random.PRNGKey(9), (4, 6)),
    )
    output = Linear(shape=(3,), name="output")
    structure = graph(
        nodes=[source, memory, output],
        edges=[
            Edge(source=source, target=memory.slot("in")),
            Edge(source=memory, target=output.slot("in")),
        ],
        task_map=TaskMap(x=source, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
    )
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(0.01)
    step = make_train_step(structure, optimizer, algorithm="backprop")
    x = jax.random.normal(rng_key, (4, 6))
    y = jax.random.normal(jax.random.PRNGKey(7), (4, 3))

    updated, _, _, _ = step(
        params,
        optimizer.init(params),
        {"x": x, "y": y},
        rng_key,
    )

    assert params.nodes["memory"].weights == {}
    assert params.nodes["memory"].biases == {}
    assert jnp.any(
        updated.nodes["output"].weights["memory->output:in"]
        != params.nodes["output"].weights["memory->output:in"]
    )
