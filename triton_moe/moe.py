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

        # For aux loss computation
        # Normalized expert widths for compute loss (proportional to FLOPs)
        mean_expert_width = sum(self.expert_widths) / self.num_experts
        self.register_buffer(
            "expert_widths_normalized",
            torch.tensor(
                [w / mean_expert_width for w in self.expert_widths], dtype=torch.float32
            ),
            persistent=False,
        )

        # Group membership for variable expert load balance loss
        expert_to_group = []
        group_sizes = []
        valid_group_idx = 0
        for count, size in config.expert_sizes:
            for _ in range(count):
                if count > 1:
                    expert_to_group.append(valid_group_idx)
                else:
                    expert_to_group.append(-1)
            if count > 1:
                group_sizes.append(count)
                valid_group_idx += 1

        self._num_valid_groups = len(group_sizes)
        self._all_experts_valid = all(g >= 0 for g in expert_to_group)

        if self._num_valid_groups > 0:
            group_membership = torch.zeros(self.num_experts, self._num_valid_groups)
            for i, g in enumerate(expert_to_group):
                if g >= 0:
                    group_membership[i, g] = 1.0

            self.register_buffer("group_membership", group_membership, persistent=False)
            self.register_buffer(
                "group_sizes", torch.tensor(group_sizes, dtype=torch.float32), persistent=False
            )
            valid_indices = [i for i, g in enumerate(expert_to_group) if g >= 0]
            self.register_buffer(
                "valid_expert_indices", torch.tensor(valid_indices, dtype=torch.long), persistent=False
            )
        else:
            self.register_buffer("group_membership", torch.empty(0), persistent=False)
            self.register_buffer("group_sizes", torch.tensor([1.0]), persistent=False)
            self.register_buffer(
                "valid_expert_indices", torch.empty(0, dtype=torch.long), persistent=False
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor | None]:
        """Forward pass through Triton MoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            output: Output tensor of shape [batch_size, seq_len, n_embd]
            aux_loss: Dictionary containing router_z_loss, load_balance_loss, compute_loss
            f_i: Expert usage fractions tensor
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

        # 8. Compute auxiliary losses
        # Router z-loss: keeps router logits small
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        # Expert usage fractions (f_i)
        f_i = (tokens_per_expert.float() / tokens_per_expert.sum()).to(x.dtype)

        # Load balance loss
        load_balance_loss = self._compute_load_balance_loss(router_probs, f_i)

        # Compute loss: weighted sum of normalized expert widths (proportional to FLOPs)
        compute_loss = (
            router_probs @ self.expert_widths_normalized.to(router_probs.dtype)
        ).mean()

        aux_loss = {
            "router_z_loss": router_z_loss,
            "load_balance_loss": load_balance_loss,
            "compute_loss": compute_loss,
        }

        return output, aux_loss, f_i

    def _compute_load_balance_loss(
        self, router_probs: torch.Tensor, f_i: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balance loss within expert groups (vectorized)."""
        p_i = router_probs.mean(dim=0)

        # Fast path for uniform expert sizes
        if len(set(self.expert_widths)) == 1:
            return self.num_experts * (f_i.float() @ p_i.float())

        # Fast path if no valid groups
        if self._num_valid_groups == 0:
            return f_i.float() @ p_i.float()

        # Compute f_i * p_i elementwise
        fi_pi = f_i.float() * p_i.float()

        # If all experts are in valid groups, skip masking
        membership = self.group_membership.to(fi_pi.dtype)
        if self._all_experts_valid:
            group_sums = fi_pi @ membership
        else:
            fi_pi_valid = fi_pi.index_select(0, self.valid_expert_indices)
            membership_valid = membership.index_select(0, self.valid_expert_indices)
            group_sums = fi_pi_valid @ membership_valid

        group_losses = self.group_sizes.to(fi_pi.dtype) * group_sums
        return group_losses.mean()

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor | None]:
        """Forward pass with per-step profiling.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]
            profiler: CUDAStepProfiler instance to record step times

        Returns:
            output: Output tensor of shape [batch_size, seq_len, n_embd]
            aux_loss: Dictionary containing router_z_loss, load_balance_loss, compute_loss
            f_i: Expert usage fractions tensor
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

        with profiler.step("aux_loss"):
            router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            f_i = (tokens_per_expert.float() / tokens_per_expert.sum()).to(x.dtype)
            load_balance_loss = self._compute_load_balance_loss(router_probs, f_i)
            compute_loss = (
                router_probs @ self.expert_widths_normalized.to(router_probs.dtype)
            ).mean()

            aux_loss = {
                "router_z_loss": router_z_loss,
                "load_balance_loss": load_balance_loss,
                "compute_loss": compute_loss,
            }

        return output, aux_loss, f_i
