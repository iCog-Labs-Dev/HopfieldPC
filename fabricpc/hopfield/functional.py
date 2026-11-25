"""Pure JAX modern Hopfield retrieval with raw dot-product similarities.

Memory values are inputs, never mutated. This module has no node, dataset,
classifier, or memory-learning dependency.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._validation import prepare_inputs, validate_beta


def _centered_scores(
    query: jax.Array, memory: jax.Array, mask: jax.Array, beta: float
) -> tuple[jax.Array, jax.Array]:
    """Subtract the maximum active similarity BEFORE inverse-temperature scaling."""
    similarities = jnp.matmul(query, memory.T, precision=jax.lax.Precision.HIGHEST)
    maximum = jax.lax.stop_gradient(
        jnp.max(jnp.where(mask[None, :], similarities, -jnp.inf), axis=-1)
    )
    scores = beta * (similarities - maximum[:, None])
    return jnp.where(mask[None, :], scores, -jnp.inf), maximum


def _retrieve_prepared(
    query: jax.Array, memory: jax.Array, mask: jax.Array, beta: float
) -> tuple[jax.Array, jax.Array]:
    scores, _ = _centered_scores(query, memory, mask, beta)
    weights = jax.nn.softmax(scores, axis=-1)
    retrieved = jnp.matmul(weights, memory, precision=jax.lax.Precision.HIGHEST)
    return retrieved, weights


def hopfield_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = 1.0,
    active_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return ``softmax(beta * query @ memory.T) @ memory`` and its weights.

    Inputs have shapes ``[B, D]`` and ``[N, D]`` and share a floating dtype.
    Outputs have shapes ``[B, D]`` and ``[B, N]``. Float16/bfloat16 inputs are
    promoted before arithmetic; both outputs remain float32. Float32/float64
    are retained. Enable JAX x64 in the application if float64 is needed.

    ``active_mask`` is a boolean ``[N]`` array. Inactive rows have zero weight.
    At least one memory must be active; queries and active memories must be
    finite. Call ``validate_hopfield_inputs`` outside JAX transformations to
    check those value preconditions. Shape/dtype checks are trace-safe.

    ``beta`` is a finite positive static Python real, representable in the
    working dtype. Close over it or mark it static with ``jax.jit``. Query
    and memory remain differentiable. No normalization or memory write occurs.
    """
    query_work, memory_work, mask = prepare_inputs(query, memory, active_mask)
    beta = validate_beta(beta, query_work.dtype)
    return _retrieve_prepared(query_work, memory_work, mask, beta)
