"""Offline tests for the CIFAR-10 PC retrieval benchmark (no download needed)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.graph_initialization import initialize_params
from fabricpc.hopfield.benchmarks import cifar10_pc_prototype_retrieval as cb
from fabricpc.hopfield.benchmarks import mnist_baseline as mb
from fabricpc.hopfield.benchmarks import mnist_pc_prototype_retrieval as pcm


def test_prepare_images_shape_and_normalization():
    out = cb.prepare_images(jnp.zeros((2, 3, 32, 32)))
    assert out.shape == (2, 32, 32, 3)
    expected = -np.asarray(cb.CIFAR10_MEAN) / np.asarray(cb.CIFAR10_STD)
    np.testing.assert_allclose(np.asarray(out[0, 5, 5]), expected, rtol=1e-5)


def test_prepare_images_rejects_mnist_shape():
    with pytest.raises(ValueError):
        cb.prepare_images(jnp.zeros((2, 1, 28, 28)))


@pytest.mark.parametrize(
    "kind,level", [("gaussian", 0.1), ("salt_pepper", 0.1), ("occlusion", 8)]
)
def test_shared_corruptions_support_three_channels(kind, level):
    images = jnp.ones((4, 3, 32, 32))
    out = mb.make_corrupted_test_set(images, seed=0, kind=kind, level=level)
    assert out.shape == images.shape
    if kind == "occlusion":
        zeros = np.asarray((out == 0).sum(axis=(1, 2, 3)))
        assert (zeros == 3 * level * level).all()


def test_graphs_build_and_share_parameter_layout():
    kwargs = dict(c1=4, c2=4, hidden=8, embed_dim=8, infer_steps=1)
    identity = cb.build_pc_cifar10_graph(**kwargs)
    prototypes = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    hopfield = cb.build_pc_cifar10_graph(prototypes=prototypes, beta=2.0, **kwargs)
    params = initialize_params(identity, jax.random.PRNGKey(0))
    pcm._assert_same_parameter_layout(
        params, initialize_params(hopfield, jax.random.PRNGKey(0))
    )

    images = jax.random.uniform(jax.random.PRNGKey(2), (2, 3, 32, 32))
    for structure in (identity, hopfield):
        pred, z, h = cb.make_eval_batch(structure)(
            params, images, jax.random.PRNGKey(3)
        )
        assert pred.shape == (2,) and z.shape == (2, 8) and h.shape == (2, 8)


def test_config_loads_full_and_quick():
    full = mb.load_config(cb.DEFAULT_CONFIG_PATH)
    quick = mb.load_config(cb.DEFAULT_CONFIG_PATH, quick=True)
    assert full["reports_dir"].startswith("reports/cifar10")
    assert quick["reports_dir"].endswith("_quick")
    assert max(full["occlusion_sizes"]) <= 32
