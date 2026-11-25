"""Modern Hopfield energy consistent with raw dot-product retrieval."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ._validation import prepare_inputs, validate_beta
from .functional import _centered_scores, _similarities


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _log_partition(
    similarities: jax.Array, mask: jax.Array, beta: float, normalize: bool
) -> jax.Array:
    """Return the log partition in similarity units, optionally normalized."""
    scores, maximum = _centered_scores(similarities, mask, beta)
    log_sum = logsumexp(scores, axis=-1)
    if not normalize:
        return maximum + log_sum / beta

    # Subtract log(count) before dividing by beta so the two terms stay finite
    # at small beta. Near uniform weights, use log1p to retain small differences.
    count = jnp.sum(mask, dtype=similarities.dtype)
    mean_delta = jnp.sum(jnp.where(mask, jnp.expm1(scores), 0), axis=-1) / count
    log_mean = jnp.where(
        mean_delta > -0.5, jnp.log1p(mean_delta), log_sum - jnp.log(count)
    )

    # The first-order limit handles scaled scores that round to zero before
    # division. The log-mean correction is quadratic in the score spread.
    near_uniform = jnp.all(
        jnp.where(mask, scores >= -jnp.finfo(scores.dtype).eps, True), axis=-1
    )
    mean_similarity = jnp.sum(jnp.where(mask, similarities / count, 0), axis=-1)
    return jnp.where(near_uniform, mean_similarity, maximum + log_mean / beta)


@_log_partition.defjvp
def _log_partition_jvp(beta, normalize, primals, tangents):
    similarities, mask = primals
    similarity_tangent, _ = tangents
    partition = _log_partition(similarities, mask, beta, normalize)
    scores, _ = _centered_scores(similarities, mask, beta)
    weights = jax.nn.softmax(scores, axis=-1)
    # Cancel beta analytically before differentiating to avoid underflow in 1/beta.
    return partition, jnp.sum(weights * similarity_tangent, axis=-1)


def hopfield_energy(
    query: jax.Array,
    memory: jax.Array,
    beta: float = 1.0,
    *,
    active_mask: jax.Array | None = None,
    include_constants: bool = False,
    reduce: str = "none",
) -> jax.Array:
    """Compute ``0.5 * ||q||^2 - logsumexp(beta * q @ M.T) / beta``.

    ``include_constants=True`` adds ``log(N_active)/beta + 0.5 * max_active
    ||m_i||^2``. These terms are constant with respect to the query for a
    fixed bank and beta.

    The per-query gradient is ``q - R(q)`` for one-step retrieval ``R``.
    With a fixed bank and beta, retrieval satisfies
    ``E(q) - E(R(q)) >= 0.5 * ||q - R(q)||^2`` in exact arithmetic.

    ``reduce='none'`` returns one energy per query. ``reduce='mean'``
    averages the energies and divides query gradients by the batch size.
    Keep beta and the configuration flags static under ``jax.jit``. The
    input, mask, and precision contracts match ``hopfield_retrieve``.
    """
    if not isinstance(include_constants, bool):
        raise TypeError("include_constants must be a boolean static argument.")
    if reduce not in ("none", "mean"):
        raise ValueError("reduce must be 'none' or 'mean'.")

    query_work, memory_work, mask = prepare_inputs(query, memory, active_mask)
    beta = validate_beta(beta, query_work.dtype)
    partition = _log_partition(
        _similarities(query_work, memory_work), mask, beta, include_constants
    )
    energy = 0.5 * jnp.sum(jnp.square(query_work), axis=-1) - partition
    if include_constants:
        maximum_memory_norm_sq = jnp.max(jnp.sum(jnp.square(memory_work), axis=-1))
        energy = energy + 0.5 * maximum_memory_norm_sq
    return jnp.mean(energy) if reduce == "mean" else energy
