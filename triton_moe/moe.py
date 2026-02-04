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

from .kernels import (
    fused_route,
    grouped_gemm_down,
    grouped_gemm_down_autograd,
    grouped_gemm_up,
    grouped_gemm_up_autograd,
    padded_gather,
    padded_gather_autograd,
    padded_scatter,
    padded_scatter_autograd,
)


@dataclass
class TritonMoEConfig:
    """Configuration for Triton MoE layer."""

    n_embd: int = 768
    expert_sizes: list = field(default_factory=lambda: [(64, 256)])
    num_active_experts: int = 8
    norm_topk_prob: bool = True
    block_size: int = 128


class TritonMoEMLP(nn.Module):
    """Triton-based Mixture of Experts MLP.

    Drop-in replacement for the reference MoEMLP using pure Triton kernels
    for gather, scatter, and grouped GEMMs. Eliminates topology construction
    overhead from stk/megablocks.

    Key differences from reference:
    - No sparse matrix topology construction
    - Direct indexed GEMMs instead of stk.ops.sdd/dsd
    - Uses our Triton padded_gather/padded_scatter
    - Uses fused_route for sorting + cumsum (torch.sort + fused cumsum kernel)
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
            assert size % self.block_size == 0, (
                f"expert sizes must be divisible by {self.block_size}"
            )
            for _ in range(count):
                self.expert_widths.append(size)
                self.expert_offsets.append(self.expert_offsets[-1] + size)
        self.total_expert_width = self.expert_offsets[-1]
        self.max_expert_width = max(self.expert_widths)

        # Router
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        # Expert weights (concatenated)
        self.w1 = nn.Parameter(torch.empty(config.n_embd, self.total_expert_width))
        self.w2 = nn.Parameter(torch.empty(self.total_expert_width, config.n_embd))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)

        # Register metadata buffers (moved to device with model)
        self.register_buffer(
            "expert_widths_t",
            torch.tensor(self.expert_widths, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "expert_weight_offsets_t",
            torch.tensor(self.expert_offsets, dtype=torch.int32),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor | None]:
        """Forward pass through Triton MoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            output: Output tensor of shape [batch_size, seq_len, n_embd]
            aux_loss: None (not implemented yet)
            router_probs: None (not implemented yet)
        """
        batch_size, seq_len, n_embd = x.shape
        device = x.device

        # 1. Routing
        x_flat = rearrange(x, "b s d -> (b s) d")
        router_logits = self.router(x_flat)
        router_probs = F.sigmoid(router_logits.float())

        top_k_weights, selected_experts = torch.topk(
            router_probs, self.num_active_experts, dim=-1
        )

        # Normalize weights
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / (
                top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        top_k_weights = top_k_weights.to(x.dtype)

        top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
        selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        # 2. Sort tokens by expert + compute bins with fused cumsum kernel
        bin_ids, indices, tokens_per_expert, bins, padded_bins = fused_route(
            selected_experts_flat,
            self.num_experts,
            block_size=self.block_size,
            rounding="up",  # dropless MoE
        )

        # 3. Compute expert token offsets for our kernels
        expert_token_offsets = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                padded_bins,
            ]
        )
        # padded_tokens_per_expert = diff of padded_bins
        padded_tokens_per_expert = torch.cat(
            [padded_bins[:1], padded_bins[1:] - padded_bins[:-1]]
        )

        # 4. Gather - permute tokens to expert-sorted order
        x_gathered = padded_gather_autograd(
            x_flat, indices, bin_ids, bins, padded_bins, self.num_active_experts
        )

        # 5. Up-projection with fused activation: y = relu_squared(x @ W1)
        x_up = grouped_gemm_up_autograd(
            x_gathered,
            self.w1,
            expert_token_offsets,
            self.expert_weight_offsets_t,
            self.expert_widths_t,
            padded_tokens_per_expert,
            self.max_expert_width,
            self.config.n_embd,
            activation="relu_squared",
        )

        # 6. Down-projection: y = x_up @ W2
        x_down = grouped_gemm_down_autograd(
            x_up,
            self.w2,
            expert_token_offsets,
            self.expert_weight_offsets_t,
            self.expert_widths_t,
            padded_tokens_per_expert,
            self.config.n_embd,
            self.max_expert_width,
        )

        # 7. Scatter - permute back and apply routing weights
        output_flat = padded_scatter_autograd(
            x_down,
            indices,
            bin_ids,
            top_k_weights_flat,
            bins,
            padded_bins,
            self.num_active_experts,
        )

        output = rearrange(output_flat, "(b s) d -> b s d", b=batch_size)

        return output, None, None

    def forward_with_intermediates(self, x: torch.Tensor) -> dict[str, Any]:
        """Forward pass that returns all intermediate tensors for testing.

        This method mirrors the reference implementation for comparison.
        """
        batch_size, seq_len, n_embd = x.shape
        device = x.device

        # 1. Routing
        x_flat = rearrange(x, "b s d -> (b s) d")
        router_logits = self.router(x_flat)
        router_probs = F.sigmoid(router_logits.float())

        top_k_weights, selected_experts = torch.topk(
            router_probs, self.num_active_experts, dim=-1
        )

        if self.norm_topk_prob:
            top_k_weights = top_k_weights / (
                top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        top_k_weights = top_k_weights.to(x.dtype)

        top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
        selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        # 2. Sort tokens by expert + compute bins with fused cumsum kernel
        bin_ids, indices, tokens_per_expert, bins, padded_bins = fused_route(
            selected_experts_flat,
            self.num_experts,
            block_size=self.block_size,
            rounding="up",
        )

        # 3. Compute expert token offsets
        expert_token_offsets = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                padded_bins,
            ]
        )
        padded_tokens_per_expert = torch.cat(
            [padded_bins[:1], padded_bins[1:] - padded_bins[:-1]]
        )

        # 4. Gather
        x_gathered = padded_gather(
            x_flat, indices, bin_ids, bins, padded_bins, self.num_active_experts
        )

        # 5. Up-projection
        x_up = grouped_gemm_up(
            x_gathered,
            self.w1,
            expert_token_offsets,
            self.expert_weight_offsets_t,
            self.expert_widths_t,
            padded_tokens_per_expert,
            self.max_expert_width,
            activation="relu_squared",
        )

        # 6. Down-projection
        x_down = grouped_gemm_down(
            x_up,
            self.w2,
            expert_token_offsets,
            self.expert_weight_offsets_t,
            self.expert_widths_t,
            padded_tokens_per_expert,
            self.config.n_embd,
        )

        # 7. Scatter
        output_flat = padded_scatter(
            x_down,
            indices,
            bin_ids,
            top_k_weights_flat,
            bins,
            padded_bins,
            self.num_active_experts,
        )

        output = rearrange(output_flat, "(b s) d -> b s d", b=batch_size)

        return {
            "output": output,
            "x_flat": x_flat,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "top_k_weights": top_k_weights,
            "top_k_weights_flat": top_k_weights_flat,
            "selected_experts": selected_experts,
            "selected_experts_flat": selected_experts_flat,
            "bin_ids": bin_ids,
            "indices": indices,
            "tokens_per_expert": tokens_per_expert,
            "bins": bins,
            "padded_bins": padded_bins,
            "x_gathered": x_gathered,
            "x_after_up": x_up,
            "x_after_down": x_down,
            "x_scattered": output_flat,
        }

    def forward_profiled(
        self, x: torch.Tensor, profiler
    ) -> tuple[torch.Tensor, None, None]:
        """Forward pass with per-step profiling.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]
            profiler: CUDAStepProfiler instance to record step times

        Returns:
            output: Output tensor of shape [batch_size, seq_len, n_embd]
            aux_loss: None (not implemented)
            router_probs: None (not implemented)
        """
        batch_size, seq_len, n_embd = x.shape
        device = x.device

        with profiler.step("flatten"):
            x_flat = rearrange(x, "b s d -> (b s) d")

        with profiler.step("routing"):
            router_logits = self.router(x_flat)
            router_probs = F.sigmoid(router_logits.float())
            top_k_weights, selected_experts = torch.topk(
                router_probs, self.num_active_experts, dim=-1
            )
            if self.norm_topk_prob:
                top_k_weights = top_k_weights / (
                    top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
                )
            top_k_weights = top_k_weights.to(x.dtype)
            top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
            selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        with profiler.step("fused_route"):
            bin_ids, indices, tokens_per_expert, bins, padded_bins = fused_route(
                selected_experts_flat,
                self.num_experts,
                block_size=self.block_size,
                rounding="up",
            )
            expert_token_offsets = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.int32),
                    padded_bins,
                ]
            )
            padded_tokens_per_expert = torch.cat(
                [padded_bins[:1], padded_bins[1:] - padded_bins[:-1]]
            )

        with profiler.step("gather"):
            x_gathered = padded_gather(
                x_flat, indices, bin_ids, bins, padded_bins, self.num_active_experts
            )

        with profiler.step("gemm_up"):
            x_up = grouped_gemm_up(
                x_gathered,
                self.w1,
                expert_token_offsets,
                self.expert_weight_offsets_t,
                self.expert_widths_t,
                padded_tokens_per_expert,
                self.max_expert_width,
                activation="relu_squared",
            )

        with profiler.step("gemm_down"):
            x_down = grouped_gemm_down(
                x_up,
                self.w2,
                expert_token_offsets,
                self.expert_weight_offsets_t,
                self.expert_widths_t,
                padded_tokens_per_expert,
                self.config.n_embd,
            )

        with profiler.step("scatter"):
            output_flat = padded_scatter(
                x_down,
                indices,
                bin_ids,
                top_k_weights_flat,
                bins,
                padded_bins,
                self.num_active_experts,
            )

        with profiler.step("reshape"):
            output = rearrange(output_flat, "(b s) d -> b s d", b=batch_size)

        return output, None, None
