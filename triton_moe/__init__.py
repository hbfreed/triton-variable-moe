"""Triton-based MoE kernels with variable-sized expert support."""

from .moe import TritonMoEMLP

__all__ = ["TritonMoEMLP"]
