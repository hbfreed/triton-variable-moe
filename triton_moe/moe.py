"""Triton-based MoE layer using custom kernels.

This module provides a drop-in replacement for the reference MoEMLP,
using Triton kernels instead of stk/megablocks ops.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .kernels import gather, scatter, grouped_gemm_up, grouped_gemm_down, relu_squared


@dataclass
class TritonMoEConfig:
    """Configuration for Triton MoE layer."""

    n_embd: int = 768
    expert_sizes: list = field(
        default_factory=lambda: [(64, 256)]
    )  # [(count, width), ...]
    num_active_experts: int = 8
    norm_topk_prob: bool = True
    block_size: int = 128


class TritonMoEMLP(nn.Module):
    """Triton-based Mixture of Experts MLP.

    This is a drop-in replacement for the reference MoEMLP using Triton kernels.
    """

    def __init__(self, config: TritonMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = sum(count for count, _ in config.expert_sizes)
        self.num_active_experts = config.num_active_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.block_size = config.block_size

        # Build expert width and offset arrays
        self.expert_widths = []
        self.expert_offsets = [0]
        for count, size in config.expert_sizes:
            assert size % self.block_size == 0, f"expert sizes must be divisible by {self.block_size}"
            for _ in range(count):
                self.expert_widths.append(size)
                self.expert_offsets.append(self.expert_offsets[-1] + size)
        self.total_expert_width = self.expert_offsets[-1]

        # Router
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        # Expert weights (concatenated)
        self.w1 = nn.Parameter(torch.empty(config.n_embd, self.total_expert_width))
        self.w2 = nn.Parameter(torch.empty(self.total_expert_width, config.n_embd))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)

        # Register metadata buffers
        self.register_buffer(
            "expert_widths_t",
            torch.tensor(self.expert_widths, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "expert_offsets_t",
            torch.tensor(self.expert_offsets, dtype=torch.int32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor | None]:
        """Forward pass through Triton MoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            output: Output tensor of shape [batch_size, seq_len, n_embd]
            aux_loss: Dictionary of auxiliary losses (or None)
            f_i: Token distribution per expert (or None)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass using Triton kernels. "
            "Steps: "
            "1. Route tokens (top-k selection) "
            "2. Sort tokens by expert (can use torch.sort initially) "
            "3. gather() - permute tokens to expert-sorted order "
            "4. grouped_gemm_up() - x @ W1 with activation "
            "5. grouped_gemm_down() - intermediate @ W2 "
            "6. scatter() - permute back and apply routing weights"
        )

    def forward_with_intermediates(self, x: torch.Tensor) -> dict[str, Any]:
        """Forward pass that returns all intermediate tensors for testing.

        This method mirrors the reference implementation for comparison.
        """
        raise NotImplementedError(
            "TODO: Implement forward_with_intermediates for Triton kernels"
        )
