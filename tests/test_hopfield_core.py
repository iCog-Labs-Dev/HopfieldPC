"""Independent checks for the modern Hopfield core."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.hopfield import hopfield_retrieve, validate_hopfield_inputs


@pytest.fixture(autouse=True)
def double_precision():
    """Enable float64 for the duration of each test."""
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        from jax.experimental import enable_x64

    with enable_x64():
        yield


def example():
    query = jnp.asarray([[0.8, 0.5], [-0.3, 0.7], [0.0, 0.0]], jnp.float64)
    memory = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], jnp.float64)
    return query, memory


def numpy_retrieve(query, memory, beta):
    """Compute the retrieval equations independently in NumPy."""
    query = np.asarray(query, dtype=np.float64)
    memory = np.asarray(memory, dtype=np.float64)
    logits = beta * (query @ memory.T)
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = exponentials / exponentials.sum(axis=-1, keepdims=True)
    return weights @ memory, weights


@pytest.mark.parametrize("beta", [0.25, 2.0, 12.0])
def test_retrieval_matches_independent_numpy_reference(beta):
    query, memory = example()
    expected_state, expected_weights = numpy_retrieve(query, memory, beta)
    state, weights = hopfield_retrieve(query, memory, beta=beta)
    np.testing.assert_allclose(state, expected_state, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weights, expected_weights, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(np.asarray(weights) >= 0.0)


def test_single_memory_is_returned_for_every_query():
    query, memory = example()
    memory = memory[:1]
    for beta in (0.1, 7.0):
        retrieved, weights = hopfield_retrieve(query, memory, beta=beta)
        np.testing.assert_array_equal(retrieved, np.repeat(memory, len(query), axis=0))
        np.testing.assert_array_equal(weights, np.ones((len(query), 1)))


def test_duplicate_memories_receive_equal_weights():
    query, _ = example()
    memory = jnp.asarray([[1.0, 0.0]] * 3, jnp.float64)
    for beta in (0.1, 1.0, 1000.0):
        retrieved, weights = hopfield_retrieve(query, memory, beta=beta)
        np.testing.assert_allclose(weights, 1.0 / 3.0, atol=1e-12)
        np.testing.assert_allclose(
            retrieved, np.repeat(memory[:1], len(query), axis=0), atol=1e-12
        )


def test_nonfinite_inactive_values_do_not_affect_retrieval():
    query, active_memory = example()
    mask = jnp.asarray([True, False, True, False, True])
    memory = jnp.asarray(
        [[1.0, 0.0], [np.nan, np.inf], [0.0, 1.0], [-np.inf, np.nan], [-1.0, 0.0]],
        jnp.float64,
    )
    validate_hopfield_inputs(query, memory, beta=2.0, active_mask=mask)
    retrieved, weights = hopfield_retrieve(query, memory, beta=2.0, active_mask=mask)
    expected, expected_weights = hopfield_retrieve(query, active_memory, beta=2.0)
    np.testing.assert_allclose(retrieved, expected, atol=1e-12)
    np.testing.assert_allclose(
        weights[:, np.asarray(mask)], expected_weights, atol=1e-12
    )
    np.testing.assert_array_equal(weights[:, ~np.asarray(mask)], 0.0)


def test_retrieval_supports_query_and_active_memory_gradients():
    query = jnp.asarray([[0.0]], jnp.float64)
    memory = jnp.asarray([[-1.0], [np.nan], [1.0]], jnp.float64)
    mask = jnp.asarray([True, False, True])

    def retrieve_sum(q, m):
        retrieved, _ = hopfield_retrieve(q, m, beta=2.0, active_mask=mask)
        return retrieved.sum()

    query_grad, memory_grad = jax.grad(retrieve_sum, argnums=(0, 1))(query, memory)
    np.testing.assert_allclose(query_grad, [[2.0]], atol=1e-12)
    np.testing.assert_allclose(memory_grad, [[0.5], [0.0], [0.5]], atol=1e-12)


def test_compiled_and_vmapped_retrieval_matches_eager_results():
    query, memory = example()
    mask = jnp.asarray([True, False, True])

    def kernel(q, m, a):
        return hopfield_retrieve(q, m, beta=2.0, active_mask=a)

    expected = kernel(query, memory, mask)
    compiled = jax.jit(kernel)(query, memory, mask)
    vmapped = jax.vmap(
        lambda q: tuple(value[0] for value in kernel(q[None, :], memory, mask))
    )(query)
    for want, jit_value, vmap_value in zip(expected, compiled, vmapped):
        np.testing.assert_allclose(jit_value, want, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(vmap_value, want, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_promotes_before_dot_products(dtype):
    query = jnp.asarray([[300.0, 300.0]], dtype)
    memory = jnp.asarray([[300.0, 300.0], [-300.0, -300.0]], dtype)
    retrieved, weights = hopfield_retrieve(query, memory)
    assert retrieved.dtype == weights.dtype == jnp.float32
    np.testing.assert_array_equal(retrieved, [[300.0, 300.0]])
    np.testing.assert_array_equal(weights, [[1.0, 0.0]])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_standard_precision_output_dtype_is_preserved(dtype):
    query, memory = example()
    retrieved, weights = hopfield_retrieve(query.astype(dtype), memory.astype(dtype))
    assert retrieved.dtype == weights.dtype == dtype


def test_centering_prevents_positive_logit_overflow_at_large_beta():
    query = jnp.asarray([[1.0]], jnp.float32)
    memory = jnp.asarray([[4.0], [3.0]], jnp.float32)
    retrieved, weights = hopfield_retrieve(query, memory, beta=1e38)
    np.testing.assert_array_equal(retrieved, [[4.0]])
    np.testing.assert_array_equal(weights, [[1.0, 0.0]])


@pytest.mark.parametrize("beta", [True, 0.0, -1.0, math.nan, math.inf, 1e300, 1e-300])
def test_beta_must_be_positive_finite_and_representable(beta):
    query, memory = example()
    query, memory = query.astype(jnp.float32), memory.astype(jnp.float32)
    for operation in (hopfield_retrieve, validate_hopfield_inputs):
        with pytest.raises(ValueError, match="beta"):
            operation(query, memory, beta=beta)


def test_validator_rejects_invalid_shapes_dtypes_and_masks():
    query, memory = example()
    cases = [
        (query[:, :, None], memory, None),
        (query, memory[0], None),
        (query[:, :1], memory, None),
        (query[:0], memory, None),
        (query, memory[:0], None),
        (query.astype(jnp.int32), memory, None),
        (query.astype(jnp.int32), memory.astype(jnp.int32), None),
        (query.astype(jnp.float32), memory, None),
        (query, memory, jnp.asarray([1, 0, 1])),
        (query, memory, jnp.asarray([True, False])),
    ]
    for invalid_query, invalid_memory, mask in cases:
        with pytest.raises((TypeError, ValueError)):
            validate_hopfield_inputs(invalid_query, invalid_memory, active_mask=mask)


def test_validator_rejects_empty_bank_and_nonfinite_active_data():
    query, memory = example()
    with pytest.raises(ValueError, match="at least one active memory"):
        validate_hopfield_inputs(query, memory, active_mask=jnp.zeros(3, dtype=bool))
    for invalid_query, invalid_memory in (
        (query.at[0, 0].set(jnp.nan), memory),
        (query.at[0, 0].set(jnp.inf), memory),
        (query, memory.at[0, 0].set(jnp.nan)),
        (query, memory.at[0, 0].set(jnp.inf)),
    ):
        with pytest.raises(ValueError, match="finite"):
            validate_hopfield_inputs(invalid_query, invalid_memory)


def test_validator_rejects_traced_inputs():
    query, memory = example()
    with pytest.raises(TypeError, match="outside JAX transformations"):
        jax.jit(validate_hopfield_inputs)(query, memory)
