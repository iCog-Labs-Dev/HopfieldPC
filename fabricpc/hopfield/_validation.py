"""Static kernel checks and explicit host validation for the Hopfield core."""

from __future__ import annotations

import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np


def validate_beta(beta: float, dtype: jnp.dtype | None = None) -> float:
    """Validate a static inverse temperature, optionally in the working dtype."""
    if (
        isinstance(beta, (bool, np.bool_))
        or not isinstance(beta, numbers.Real)
        or not math.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError(
            "beta must be a finite positive Python real number; "
            "keep it static when using jax.jit."
        )
    value = float(beta)
    if dtype is not None:
        limits = jnp.finfo(dtype)
        if value < float(limits.tiny) or value > float(limits.max):
            raise ValueError(
                f"beta must be representable as a positive normal value in {dtype}."
            )
    return value


def working_dtype(dtype: jnp.dtype) -> jnp.dtype:
    if dtype in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        return jnp.dtype(jnp.float32)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("Inputs must use float16, bfloat16, float32, or float64.")
    return dtype


def prepare_mask(active_mask: jax.Array | None, memory_size: int) -> jax.Array:
    if active_mask is None:
        return jnp.ones((memory_size,), dtype=jnp.bool_)
    mask = jnp.asarray(active_mask)
    if mask.shape != (memory_size,):
        raise ValueError(f"active_mask must have shape [{memory_size}].")
    if mask.dtype != jnp.bool_:
        raise TypeError("active_mask must have boolean dtype.")
    return mask


def prepare_inputs(
    query: jax.Array,
    memory: jax.Array,
    active_mask: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Check static metadata, promote arithmetic, and neutralize inactive rows.

    This helper does not transfer array values to the host. Runtime value
    preconditions are checked separately by ``validate_hopfield_inputs``.
    """
    query, memory = jnp.asarray(query), jnp.asarray(memory)
    if query.ndim != 2 or memory.ndim != 2:
        raise ValueError("query and memory must have shapes [batch, dim] and [N, dim].")
    if any(size == 0 for size in (*query.shape, *memory.shape)):
        raise ValueError(
            "batch size, memory size, and feature dimension must be positive."
        )
    if query.shape[1] != memory.shape[1]:
        raise ValueError("query and memory must have the same feature dimension.")
    if query.dtype != memory.dtype:
        raise TypeError("query and memory must have the same floating dtype.")
    dtype = working_dtype(query.dtype)
    mask = prepare_mask(active_mask, memory.shape[0])
    query_work = query.astype(dtype)
    # Mask BEFORE any dot product: unused storage may contain NaN or Inf.
    memory_work = jnp.where(mask[:, None], memory, 0).astype(dtype)
    return query_work, memory_work, mask


def validate_hopfield_inputs(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = 1.0,
    active_mask: jax.Array | None = None,
) -> None:
    """Validate concrete inputs before a compiled inference episode.

    Requires finite queries, finite active memories, and at least one active
    memory. Inactive memory values are ignored. This host-side function may
    synchronize device arrays and must not be called inside ``jit``/``grad``.
    The numerical kernels make these value assumptions without host callbacks.
    Static shape/dtype checks still run when those kernels are traced.
    """
    query_work, memory_work, mask = prepare_inputs(query, memory, active_mask)
    validate_beta(beta, query_work.dtype)
    if any(
        isinstance(value, jax.core.Tracer) for value in (query_work, memory_work, mask)
    ):
        raise TypeError(
            "validate_hopfield_inputs must run outside JAX transformations."
        )
    if not np.asarray(mask).any():
        raise ValueError("active_mask must contain at least one active memory.")
    if not np.isfinite(np.asarray(query_work)).all():
        raise ValueError("query must contain only finite values.")
    if not np.isfinite(np.asarray(memory_work)).all():
        raise ValueError("Active memory values must be finite.")
