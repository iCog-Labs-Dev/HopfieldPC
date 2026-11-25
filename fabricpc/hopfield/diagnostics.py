"""Diagnostics for modern Hopfield attention distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._validation import prepare_mask, working_dtype


def attention_entropy(
    weights: jax.Array,
    *,
    active_mask: jax.Array | None = None,
    normalize: bool = False,
) -> jax.Array:
    """Return per-query Shannon entropy in nats for ``weights[B, N]``.

    Active weights must be finite, nonnegative probabilities that sum to one
    in each row. At least one memory must be active. Inactive weights are
    ignored. Float16 and bfloat16 inputs are promoted to float32.

    With ``normalize=True``, divide by the log of the active memory count.
    A single active memory has normalized entropy zero. Keep ``normalize``
    static under ``jax.jit``.
    """
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean static argument.")
    weights = jnp.asarray(weights)
    if weights.ndim != 2 or any(size == 0 for size in weights.shape):
        raise ValueError("weights must have a nonempty shape [batch, N].")
    mask = prepare_mask(active_mask, weights.shape[1])
    probabilities = jnp.where(mask[None, :], weights, 0).astype(
        working_dtype(weights.dtype)
    )
    # Zero weights contribute zero without evaluating log(0).
    log_probabilities = jnp.log(jnp.where(probabilities > 0, probabilities, 1))
    entropy = -jnp.sum(probabilities * log_probabilities, axis=-1)
    if normalize:
        count = jnp.sum(mask, dtype=probabilities.dtype)
        max_entropy = jnp.log(jnp.maximum(count, 2))
        entropy = jnp.where(count > 1, entropy / max_entropy, 0)
    return entropy
