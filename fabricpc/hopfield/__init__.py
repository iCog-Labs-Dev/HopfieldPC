"""Public FabricPC integration for the HopfieldPC memory core."""

from .src.hopfieldpc import (
    HopfieldMemory,
    HopfieldMemoryOutput,
    HopfieldShapeContract,
    HopfieldShapeError,
    argmax_memory_index,
    attention_entropy,
    build_retrieval_diagnostics,
    hopfield_energy,
    hopfield_retrieve,
    max_attention_weight,
    multi_step_retrieve,
    retrieval_distance,
    selected_memory_distance,
    topk_mass,
    topk_recall,
)
from .src.hopfieldpc.api import validate_beta

__all__ = [
    "HopfieldMemory",
    "HopfieldMemoryOutput",
    "HopfieldShapeContract",
    "HopfieldShapeError",
    "argmax_memory_index",
    "attention_entropy",
    "build_retrieval_diagnostics",
    "hopfield_energy",
    "hopfield_retrieve",
    "max_attention_weight",
    "multi_step_retrieve",
    "retrieval_distance",
    "selected_memory_distance",
    "topk_mass",
    "topk_recall",
    "validate_beta",
]
