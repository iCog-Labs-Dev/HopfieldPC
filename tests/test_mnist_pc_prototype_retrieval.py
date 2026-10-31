"""Focused tests for the FabricPC-trained MNIST Hopfield benchmark."""

import jax
import jax.numpy as jnp
import optax

from fabricpc.hopfield.benchmarks.mnist_pc_prototype_retrieval import (
    BRIDGE_NODE,
    build_pc_mnist_graph,
    graph_outputs,
    prepare_images,
    _assert_same_parameter_layout,
    _make_eval_batch,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import HopfieldMemoryNode
from fabricpc.training import make_train_step


def _small_graph(**kwargs):
    return build_pc_mnist_graph(
        c1=2,
        c2=4,
        hidden=8,
        embed_dim=4,
        infer_steps=1,
        **kwargs,
    )


def test_prepare_images_matches_baseline_normalization():
    images = jnp.full((2, 1, 28, 28), 0.1307, dtype=jnp.float32)

    prepared = prepare_images(images)

    assert prepared.shape == (2, 28, 28, 1)
    assert jnp.allclose(prepared, 0.0, atol=1e-6)


def test_identity_and_frozen_hopfield_graphs_share_parameter_layout(rng_key):
    prototypes = jax.random.normal(rng_key, (4, 4))
    baseline = _small_graph()
    hopfield = _small_graph(prototypes=prototypes, beta=2.0)
    baseline_params = initialize_params(baseline, rng_key)
    hopfield_params = initialize_params(hopfield, rng_key)

    _assert_same_parameter_layout(baseline_params, hopfield_params)

    assert isinstance(hopfield.nodes[BRIDGE_NODE], HopfieldMemoryNode)
    assert baseline_params.nodes[BRIDGE_NODE].weights == {}
    assert hopfield_params.nodes[BRIDGE_NODE].weights == {}
    assert baseline_params.nodes[BRIDGE_NODE].biases == {}
    assert hopfield_params.nodes[BRIDGE_NODE].biases == {}


def test_default_pc_cnn_matches_task13_trainable_parameter_count(rng_key):
    baseline = build_pc_mnist_graph(infer_steps=1)
    prototypes = jax.random.normal(rng_key, (80, 64))
    hopfield = build_pc_mnist_graph(
        prototypes=prototypes,
        infer_steps=1,
    )
    baseline_params = initialize_params(baseline, rng_key)
    hopfield_params = initialize_params(hopfield, rng_key)

    _assert_same_parameter_layout(baseline_params, hopfield_params)
    assert (
        sum(leaf.size for leaf in jax.tree_util.tree_leaves(baseline_params)) == 429_258
    )
    assert (
        sum(leaf.size for leaf in jax.tree_util.tree_leaves(hopfield_params)) == 429_258
    )


def test_hopfield_graph_inference_outputs_benchmark_metrics(rng_key):
    prototypes = jax.random.normal(rng_key, (4, 4))
    labels = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    structure = _small_graph(prototypes=prototypes, beta=2.0)
    params = initialize_params(structure, rng_key)
    images = jax.random.uniform(rng_key, (4, 1, 28, 28))
    evaluate_batch = _make_eval_batch(structure)

    outputs = graph_outputs(
        structure,
        params,
        evaluate_batch,
        images,
        batch_size=2,
        device=jax.devices("cpu")[0],
        prototypes=prototypes,
        memory_labels=labels,
        center=None,
        beta=2.0,
        num_updates=1,
    )

    assert outputs["head_pred"].shape == (4,)
    assert outputs["z"].shape == (4, 4)
    assert outputs["h"].shape == (4, 4)
    assert outputs["weights"].shape == (4, 4)
    assert outputs["class_mass"].shape == (4, 4)


def test_benchmark_graph_runs_a_pc_training_step(rng_key):
    structure = _small_graph()
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(1e-3)
    train_step = make_train_step(structure, optimizer, algorithm="pc")
    images = jax.random.uniform(rng_key, (2, 1, 28, 28))
    batch = {
        "x": prepare_images(images),
        "y": jnp.array([1, 3], dtype=jnp.int32),
    }

    updated, _, metrics, _ = train_step(
        params,
        optimizer.init(params),
        batch,
        rng_key,
    )

    assert jnp.isfinite(metrics["energy"])
    assert updated.nodes["bridge"].weights == {}
    assert updated.nodes["bridge"].biases == {}
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(params),
            jax.tree_util.tree_leaves(updated),
        )
    )
