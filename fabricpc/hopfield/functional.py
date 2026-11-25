"""Pure JAX modern Hopfield retrieval with raw dot-product similarities.

Memory values are inputs, never mutated. This module has no node, dataset,
classifier, or memory-learning dependency.
"""

from __future__ import annotations

from typing import Literal, overload

import jax
import jax.numpy as jnp

from ._validation import prepare_inputs, validate_beta


def _similarities(query: jax.Array, memory: jax.Array) -> jax.Array:
    return jnp.matmul(query, memory.T, precision=jax.lax.Precision.HIGHEST)


def _centered_scores(
    similarities: jax.Array, mask: jax.Array, beta: float
) -> tuple[jax.Array, jax.Array]:
    """Subtract the maximum active similarity BEFORE inverse-temperature scaling."""
    maximum = jax.lax.stop_gradient(
        jnp.max(jnp.where(mask[None, :], similarities, -jnp.inf), axis=-1)
    )
    scores = beta * (similarities - maximum[:, None])
    return jnp.where(mask[None, :], scores, -jnp.inf), maximum


def _retrieve_prepared(
    query: jax.Array, memory: jax.Array, mask: jax.Array, beta: float
) -> tuple[jax.Array, jax.Array]:
    scores, _ = _centered_scores(_similarities(query, memory), mask, beta)
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


@overload
def multi_step_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = ...,
    num_updates: int = ...,
    active_mask: jax.Array | None = ...,
    return_states: Literal[False] = ...,
) -> tuple[jax.Array, jax.Array]: ...


@overload
def multi_step_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = ...,
    num_updates: int = ...,
    active_mask: jax.Array | None = ...,
    return_states: Literal[True],
) -> tuple[jax.Array, jax.Array, list[jax.Array]]: ...


@overload
def multi_step_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = ...,
    num_updates: int = ...,
    active_mask: jax.Array | None = ...,
    return_states: bool,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, list[jax.Array]]: ...


def multi_step_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = 1.0,
    num_updates: int = 1,
    active_mask: jax.Array | None = None,
    return_states: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, list[jax.Array]]:
    """Apply ``num_updates`` retrieval steps with a fixed bank and mask.

    Return the final state and the weights that produce it from the previous
    state. ``return_states=True`` also returns ``[q0, ..., qT]`` as a list in
    the working dtype. The default single update matches ``hopfield_retrieve``.

    The input, mask, and precision contracts match ``hopfield_retrieve``.
    Keep ``beta``, ``num_updates`` (a positive Python integer), and
    ``return_states`` static under ``jax.jit``.
    """
    if (
        isinstance(num_updates, bool)
        or not isinstance(num_updates, int)
        or num_updates <= 0
    ):
        raise ValueError("num_updates must be a positive static Python integer.")
    if not isinstance(return_states, bool):
        raise TypeError("return_states must be a boolean static argument.")
    query_work, memory_work, mask = prepare_inputs(query, memory, active_mask)
    beta = validate_beta(beta, query_work.dtype)
    initial_weights = jnp.zeros(
        (query_work.shape[0], memory_work.shape[0]), dtype=query_work.dtype
    )

    def step(carry, _unused):
        state, _ = carry
        retrieved, weights = _retrieve_prepared(state, memory_work, mask, beta)
        return (retrieved, weights), retrieved if return_states else None

    (retrieved, weights), history = jax.lax.scan(
        step, (query_work, initial_weights), xs=None, length=num_updates
    )
    if return_states:
        return retrieved, weights, [query_work, *history]
    return retrieved, weights
