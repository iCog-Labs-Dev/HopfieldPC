"""Independent checks for the modern Hopfield core."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.hopfield import (
    hopfield_energy,
    hopfield_retrieve,
    validate_hopfield_inputs,
)


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


def numpy_reference(query, memory, beta):
    """Compute retrieval and energy independently in NumPy."""
    query = np.asarray(query, dtype=np.float64)
    memory = np.asarray(memory, dtype=np.float64)
    logits = beta * (query @ memory.T)
    maximum = logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(logits - maximum)
    partition = exponentials.sum(axis=-1, keepdims=True)
    weights = exponentials / partition
    energy = 0.5 * (query**2).sum(axis=-1) - (maximum + np.log(partition))[:, 0] / beta
    return weights @ memory, weights, energy


@pytest.mark.parametrize("beta", [0.25, 2.0, 12.0])
def test_retrieval_and_energy_match_independent_numpy_reference(beta):
    query, memory = example()
    expected_state, expected_weights, expected_energy = numpy_reference(
        query, memory, beta
    )
    state, weights = hopfield_retrieve(query, memory, beta=beta)
    np.testing.assert_allclose(state, expected_state, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weights, expected_weights, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(np.asarray(weights) >= 0.0)
    energy = hopfield_energy(query, memory, beta)
    np.testing.assert_allclose(energy, expected_energy, atol=1e-12, rtol=1e-12)
    offset = (
        math.log(len(memory)) / beta
        + 0.5 * np.sum(np.asarray(memory) ** 2, axis=-1).max()
    )
    full_energy = hopfield_energy(query, memory, beta, include_constants=True)
    np.testing.assert_allclose(full_energy, expected_energy + offset, atol=1e-12)


def test_single_memory_is_returned_for_every_query():
    query, memory = example()
    memory = memory[:1]
    for beta in (0.1, 7.0):
        retrieved, weights = hopfield_retrieve(query, memory, beta=beta)
        np.testing.assert_array_equal(retrieved, np.repeat(memory, len(query), axis=0))
        np.testing.assert_array_equal(weights, np.ones((len(query), 1)))
        energy = hopfield_energy(query, memory, beta, include_constants=True)
        np.testing.assert_allclose(
            energy, 0.5 * np.sum((query - memory) ** 2, axis=-1), atol=1e-12
        )


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


@pytest.mark.parametrize("include_constants", [False, True])
def test_energy_mask_matches_compact_bank_in_value_and_memory_gradient(
    include_constants,
):
    query, _ = example()
    memory = jnp.asarray([[1.0, 0.0], [np.nan, np.inf], [0.0, 2.0]], jnp.float64)
    mask = jnp.asarray([True, False, True])
    active_memory = memory[mask]

    def energy(m, active_mask=None):
        return hopfield_energy(
            query, m, 2.0, active_mask=active_mask, include_constants=include_constants
        )

    np.testing.assert_allclose(energy(memory, mask), energy(active_memory), atol=1e-12)
    masked_grad = jax.grad(lambda m: energy(m, mask).sum())(memory)
    compact_grad = jax.grad(lambda m: energy(m).sum())(active_memory)
    np.testing.assert_allclose(masked_grad[mask], compact_grad, atol=1e-12)
    np.testing.assert_array_equal(masked_grad[~mask], 0.0)

    _, weights = hopfield_retrieve(query, active_memory, beta=2.0)
    expected_grad = -weights.T @ query
    if include_constants:
        expected_grad = expected_grad.at[1].add(len(query) * active_memory[1])
    np.testing.assert_allclose(compact_grad, expected_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("include_constants", [False, True])
def test_energy_query_gradient_is_query_minus_retrieved(include_constants):
    query, memory = example()
    mask = jnp.asarray([True, False, True])

    def energy(q):
        return hopfield_energy(
            q, memory, 3.0, active_mask=mask, include_constants=include_constants
        ).sum()

    # The zero query exercises tied similarities.
    retrieved, _ = hopfield_retrieve(query, memory, beta=3.0, active_mask=mask)
    expected = query - retrieved
    np.testing.assert_allclose(jax.grad(energy)(query), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(query), expected, atol=1e-12)


def test_energy_mean_reduction_scales_query_gradient():
    query, memory = example()
    per_query = hopfield_energy(query, memory, 2.0)
    mean_energy = hopfield_energy(query, memory, 2.0, reduce="mean")
    assert per_query.shape == (len(query),)
    assert mean_energy.shape == ()
    np.testing.assert_allclose(mean_energy, per_query.mean(), atol=1e-12)
    retrieved, _ = hopfield_retrieve(query, memory, beta=2.0)
    gradient = jax.grad(lambda q: hopfield_energy(q, memory, 2.0, reduce="mean"))(query)
    np.testing.assert_allclose(gradient, (query - retrieved) / len(query), atol=1e-12)
    with pytest.raises(ValueError, match="reduce"):
        hopfield_energy(query, memory, reduce="median")
    with pytest.raises(TypeError, match="include_constants"):
        hopfield_energy(query, memory, include_constants=1)


@pytest.mark.parametrize("beta", [0.1, 1.0, 8.0])
def test_fixed_bank_retrieval_satisfies_energy_descent_bound(beta):
    rng = np.random.default_rng(19)
    query = jnp.asarray(rng.normal(size=(4, 5)), jnp.float64)
    memory = jnp.asarray(rng.normal(size=(7, 5)) * 0.7, jnp.float64)
    mask = jnp.asarray([True, False, True, True, False, True, False])

    for _ in range(5):
        retrieved, _ = hopfield_retrieve(query, memory, beta=beta, active_mask=mask)
        bound = 0.5 * jnp.sum((query - retrieved) ** 2, axis=-1)
        for include_constants in (False, True):
            options = dict(active_mask=mask, include_constants=include_constants)
            before = hopfield_energy(query, memory, beta, **options)
            after = hopfield_energy(retrieved, memory, beta, **options)
            assert np.all(np.asarray(before - after >= bound - 1e-11))
        query = retrieved


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


def test_compiled_and_vmapped_core_matches_eager_results():
    query, memory = example()
    mask = jnp.asarray([True, False, True])

    def kernel(q, m, a):
        retrieved, weights = hopfield_retrieve(q, m, beta=2.0, active_mask=a)
        energy = hopfield_energy(q, m, 2.0, active_mask=a, include_constants=True)
        return retrieved, weights, energy

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
    energy = hopfield_energy(query, jnp.zeros_like(query))
    assert energy.dtype == jnp.float32
    np.testing.assert_array_equal(energy, [90000.0])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_standard_precision_output_dtype_is_preserved(dtype):
    query, memory = example()
    query, memory = query.astype(dtype), memory.astype(dtype)
    retrieved, weights = hopfield_retrieve(query, memory)
    assert retrieved.dtype == weights.dtype == hopfield_energy(query, memory).dtype
    assert retrieved.dtype == dtype


def test_centering_prevents_positive_logit_overflow_at_large_beta():
    query = jnp.asarray([[1.0]], jnp.float32)
    memory = jnp.asarray([[4.0], [3.0]], jnp.float32)
    retrieved, weights = hopfield_retrieve(query, memory, beta=1e38)
    np.testing.assert_array_equal(retrieved, [[4.0]])
    np.testing.assert_array_equal(weights, [[1.0, 0.0]])


def test_energy_gradient_remains_finite_at_large_beta():
    query = jnp.asarray([[1.0]], jnp.float32)
    memory = jnp.asarray([[4.0], [0.0]], jnp.float32)

    def energy(q):
        return hopfield_energy(q, memory, 1e38).sum()

    value, gradient = jax.jit(jax.value_and_grad(energy))(query)
    np.testing.assert_array_equal(value, -3.5)
    np.testing.assert_array_equal(gradient, [[-3.0]])


@pytest.mark.parametrize("beta", [1e-5, 1.2e-38])
def test_full_energy_remains_finite_and_accurate_at_small_beta(beta):
    query = jnp.asarray([[1.0]], jnp.float32)
    memory = jnp.repeat(jnp.asarray([[-4.0], [2.0]], jnp.float32), 64, axis=0)
    # The active mean and max squared norm give the beta-to-zero limit of 9.5.
    # For this symmetric bank, the leading finite-beta correction is -4.5*beta.
    energy = hopfield_energy(query, memory, beta, include_constants=True)
    np.testing.assert_allclose(energy, [9.5 - 4.5 * beta], atol=2e-6, rtol=0)


@pytest.mark.parametrize("beta", [True, 0.0, -1.0, math.nan, math.inf, 1e300, 1e-300])
def test_beta_must_be_positive_finite_and_representable(beta):
    query, memory = example()
    query, memory = query.astype(jnp.float32), memory.astype(jnp.float32)
    for operation in (hopfield_retrieve, hopfield_energy, validate_hopfield_inputs):
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
