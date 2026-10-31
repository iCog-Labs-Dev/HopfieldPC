"""Modern Hopfield associative-memory node for FabricPC graphs."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import InitializerBase, NormalInitializer
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.hopfield import multi_step_retrieve, validate_beta
from fabricpc.nodes.base import NodeBase, SlotSpec


class HopfieldMemoryNode(NodeBase):
    """Associative retrieval node backed by a frozen prototype memory bank.

    The node accepts one query tensor, flattens it to the Hopfield core's
    ``[batch_size, dim]`` contract, retrieves from its fixed memory bank, and
    reshapes the result to ``shape``. Prototypes are held by the node rather
    than included in FabricPC graph parameters, so optimizers cannot update
    them.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        prototypes: jax.Array,
        *,
        beta: float = 1.0,
        num_updates: int = 1,
        normalize: bool = True,
        center: Optional[jax.Array] = None,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        **kwargs,
    ):
        memory = jnp.asarray(prototypes, dtype=jnp.float32)
        if memory.ndim != 2 or memory.shape[0] <= 0 or memory.shape[1] <= 0:
            raise ValueError(
                "prototypes must have shape [num_memories, dim] with both "
                f"dimensions positive, got {tuple(memory.shape)}."
            )
        if memory.shape[1] != math.prod(shape):
            raise ValueError(
                f"Prototype dimension {memory.shape[1]} does not match the "
                f"flattened node shape {math.prod(shape)}."
            )

        center_array = None
        if center is not None:
            center_array = jnp.asarray(center, dtype=jnp.float32)
            if center_array.shape != (memory.shape[1],):
                raise ValueError(
                    f"center must have shape ({memory.shape[1]},), "
                    f"got {tuple(center_array.shape)}."
                )

        beta = validate_beta(beta)
        if (
            isinstance(num_updates, bool)
            or not isinstance(num_updates, int)
            or num_updates <= 0
        ):
            raise ValueError(
                f"num_updates must be a positive integer, got {num_updates!r}."
            )

        prototype_values = tuple(
            tuple(float(value) for value in row)
            for row in jax.device_get(memory).tolist()
        )
        center_values = (
            tuple(float(value) for value in jax.device_get(center_array).tolist())
            if center_array is not None
            else None
        )
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,
            beta=beta,
            num_updates=num_updates,
            normalize=bool(normalize),
            prototypes=prototype_values,
            center=center_values,
            **kwargs,
        )
        self._prototypes = memory
        self._center = center_array

    @property
    def prototypes(self) -> jax.Array:
        """Frozen prototype bank with shape ``[num_memories, dim]``."""
        return self._prototypes

    @property
    def center(self) -> Optional[jax.Array]:
        """Optional mean vector removed before retrieval and restored after."""
        return self._center

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """One required, unscaled query input."""
        return {
            "in": SlotSpec(
                name="in",
                is_multi_input=False,
                is_variance_scalable=False,
                require_connected=True,
            )
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Dict[str, Any],
    ) -> NodeParams:
        for edge_key, input_shape in input_shapes.items():
            if tuple(input_shape) != tuple(node_shape):
                raise ValueError(
                    "HopfieldMemoryNode requires the query shape to match its "
                    f"output shape. Edge {edge_key!r}: input {tuple(input_shape)} "
                    f"vs. node {tuple(node_shape)}."
                )

        if len(input_shapes) != 1:
            raise ValueError(
                "HopfieldMemoryNode requires exactly one connected query input."
            )
        return NodeParams(weights={}, biases={})

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        _, query = next(iter(inputs.items()))
        query_flat = query.reshape(query.shape[0], -1)
        node_config = node_info.node_config
        memory = jnp.asarray(node_config["prototypes"], dtype=query_flat.dtype)

        center_values = node_config["center"]
        if center_values is not None:
            center = jnp.asarray(center_values, dtype=query_flat.dtype)
            query_flat = query_flat - center
            memory = memory - center

        retrieved, _ = multi_step_retrieve(
            query=query_flat,
            memory=memory,
            beta=node_info.node_config["beta"],
            num_updates=node_info.node_config["num_updates"],
            normalize=node_info.node_config["normalize"],
        )
        if center_values is not None:
            retrieved = retrieved + center

        retrieved = retrieved.reshape((query.shape[0],) + node_info.shape)
        activation = node_info.activation
        prediction = type(activation).forward(retrieved, activation.config)
        return prediction, None
