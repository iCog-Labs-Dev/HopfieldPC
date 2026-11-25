"""Standalone JAX functions for modern Hopfield retrieval, energy, and diagnostics."""

from ._validation import validate_hopfield_inputs
from .diagnostics import attention_entropy
from .energy import hopfield_energy
from .functional import hopfield_retrieve, multi_step_retrieve

__all__ = [
    "attention_entropy",
    "hopfield_energy",
    "hopfield_retrieve",
    "multi_step_retrieve",
    "validate_hopfield_inputs",
]
