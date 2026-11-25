"""Linear predictive-coding nodes with a fixed-bank Hopfield energy."""

from __future__ import annotations

import math
import numbers
from types import MappingProxyType
from typing import Any

import jax

from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.hopfield import hopfield_energy
from fabricpc.hopfield._validation import validate_beta
from fabricpc.nodes.linear import Linear


class HopfieldLinear(Linear):
    """Add a modern Hopfield energy to a vector-valued linear node.

    The energy of an internal node is its ordinary prediction-error energy
    plus ``hopfield_strength * hopfield_energy(z_latent, memory)`` per sample.
    Pass a bank prepared by ``prepare_hopfield_bank`` in the node's runtime
    buffers when initializing graph state. Its feature dimension and floating
    dtype must match the latent. The linear constructor options are available
    as keyword arguments.

    ``hopfield_strength`` is a fixed, nonnegative coupling (default 1).
    At zero coupling or on the source-node path, only the baseline energy is
    evaluated. ``beta`` is the positive inverse temperature of retrieval.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        name: str,
        *,
        hopfield_strength: float = 1.0,
        beta: float = 1.0,
        **linear_options: Any,
    ):
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError(
                "HopfieldLinear requires a vector shape (D,) with D positive."
            )
        if (
            isinstance(hopfield_strength, bool)
            or not isinstance(hopfield_strength, numbers.Real)
            or not math.isfinite(hopfield_strength)
            or hopfield_strength < 0
        ):
            raise ValueError(
                "hopfield_strength must be a finite nonnegative real number."
            )
        super().__init__(shape=shape, name=name, **linear_options)
        self._extra_config = MappingProxyType(
            {
                **self._extra_config,
                "hopfield_strength": float(hopfield_strength),
                "beta": validate_beta(beta),
            }
        )

    @staticmethod
    def energy(
        params: NodeParams,
        inputs: dict[str, jax.Array],
        state: NodeState,
        aux: Any,
        node_info: NodeInfo,
    ) -> jax.Array:
        baseline = Linear.energy(params, inputs, state, aux, node_info)
        strength = node_info.node_config["hopfield_strength"]
        if strength == 0 or node_info.in_degree == 0:
            return baseline

        buffers = state.buffers
        if (
            buffers is None
            or not {"hopfield_memory", "hopfield_active_mask"} <= buffers.keys()
        ):
            raise ValueError(
                f"Node '{node_info.name}' requires buffers from prepare_hopfield_bank "
                "when hopfield_strength is positive."
            )
        memory = jax.lax.stop_gradient(buffers["hopfield_memory"])
        memory_energy = hopfield_energy(
            state.z_latent,
            memory,
            beta=node_info.node_config["beta"],
            active_mask=buffers["hopfield_active_mask"],
        )
        return baseline + strength * memory_energy
