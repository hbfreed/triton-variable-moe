"""
MoE MLP reference implementation.
Extracted and adapted from nanoMoEchat/nanochat/gpt.py.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import stk
import stk.ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megablocks import ops
from megablocks.layers.relu_squared import relu_squared

from .topology_var import topology_var


@dataclass
class MoEConfig:
    """Configuration for MoE layer."""

    n_embd: int = 768
    expert_sizes: list = field(
        default_factory=lambda: [(64, 256)]
    )  # [(count, width), ...]
    num_active_experts: int = 8
    norm_topk_prob: bool = True
    block_size: int = 128


class MoEMLP(nn.Module):
    """Mixture of Experts MLP with variable-sized expert support.

    This is the reference implementation used for testing Triton kernels.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = sum(count for count, _ in config.expert_sizes)
        self.num_active_experts = config.num_active_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.block_size = config.block_size

        # expert_widths: FFN width for each expert, expanded from config.expert_sizes tuples
        # e.g. config.expert_sizes=[(2, 1024), (1, 512)] -> expert_widths=[1024, 1024, 512]
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

        # compute normalized expert widths for aux losses
        mean_expert_width = sum(self.expert_widths) / self.num_experts
        self.register_buffer(
            "expert_widths_normalized",
            torch.tensor(
                [w / mean_expert_width for w in self.expert_widths], dtype=torch.float32
            ),
            persistent=False,
        )

        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        self.w1 = nn.Parameter(torch.empty(config.n_embd, self.total_expert_width))
        self.w2 = nn.Parameter(torch.empty(self.total_expert_width, config.n_embd))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)

        # need this for megablocks ops
        self.sort_end_bit = max(int(math.ceil(math.log2(self.num_experts))), 1)

        self.transpose_sort_end_bit = max(
            int(math.ceil(math.log2(self.num_experts))), 1
        )

        # Register buffers for efficient CUDA kernel access
        self.register_buffer(
            "expert_size_blocks",
            torch.tensor(
                [s // self.block_size for s in self.expert_widths], dtype=torch.int32
            ),
            persistent=False,
        )
        self.register_buffer(
            "expert_block_offsets",
            torch.tensor(
                [o // self.block_size for o in self.expert_offsets], dtype=torch.int32
            ),
            persistent=False,
        )

        # Precompute tensors for vectorized load balance loss computation
        # Use matrix multiplication instead of scatter_add for small expert counts
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
            # Build group membership matrix: [num_experts, num_groups]
            # group_membership[i, j] = 1 if expert i belongs to group j, else 0
            # This lets us use matmul instead of scatter_add
            group_membership = torch.zeros(self.num_experts, self._num_valid_groups)
            for i, g in enumerate(expert_to_group):
                if g >= 0:
                    group_membership[i, g] = 1.0

            self.register_buffer("group_membership", group_membership, persistent=False)
            self.register_buffer(
                "group_sizes",
                torch.tensor(group_sizes, dtype=torch.float32),
                persistent=False,
            )

            # Precompute indices of valid experts for masking (if needed)
            valid_indices = [i for i, g in enumerate(expert_to_group) if g >= 0]
            self.register_buffer(
                "valid_expert_indices",
                torch.tensor(valid_indices, dtype=torch.long),
                persistent=False,
            )
        else:
            # Dummy buffers for uniform case
            self.register_buffer("group_membership", torch.empty(0), persistent=False)
            self.register_buffer("group_sizes", torch.tensor([1.0]), persistent=False)
            self.register_buffer(
                "valid_expert_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )

    @torch.compiler.disable
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            output: Output tensor of shape [batch_size, seq_len, n_embd]
            aux_loss: Dictionary of auxiliary losses
            f_i: Token distribution per expert
        """
        batch_size, seq_len, n_embd = x.shape

        x_flat = rearrange(
            x, "batch_size seq_len n_embd -> (batch_size seq_len) n_embd "
        )

        router_logits = self.router(x_flat)

        router_probs = F.sigmoid(router_logits.to(torch.float32))

        top_k_weights, selected_experts = torch.topk(
            router_probs, self.num_active_experts, dim=-1
        )

        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
        )

        top_k_weights = top_k_weights.to(x.dtype)

        top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
        selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        bin_ids, indices, tokens_per_expert = self._sort_tokens_by_expert(
            selected_experts_flat
        )

        # Compute bins for gather/scatter
        bins = ops.inclusive_cumsum(tokens_per_expert, 0).contiguous()

        # Build topology dynamically each forward (like dMoE)
        padded_bins, topology = self._create_topology(x_flat, tokens_per_expert)

        x_permuted = ops.padded_gather(
            x_flat, indices, bin_ids, bins, padded_bins, self.num_active_experts
        )
        x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)
        x_permuted = relu_squared(x_permuted)
        x_permuted = stk.ops.dsd(x_permuted, self.w2)
        x_permuted = ops.padded_scatter(
            x_permuted,
            indices,
            bin_ids,
            top_k_weights_flat,
            bins,
            padded_bins,
            self.num_active_experts,
        )
        output = rearrange(
            x_permuted,
            "(batch_size seq_len) n_embd -> batch_size seq_len n_embd",
            batch_size=batch_size,
            seq_len=seq_len,
        )

        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        # Reuse histogram from routing instead of slow scatter_add_
        f_i = (tokens_per_expert.float() / tokens_per_expert.sum()).to(x.dtype)
        load_balance_loss = self._compute_load_balance_loss(
            router_probs, selected_experts_flat, f_i
        )
        router_probs_flat = rearrange(
            router_probs,
            "(batch_size seq_len) n_embd -> (batch_size seq_len) n_embd",
            batch_size=batch_size,
            seq_len=seq_len,
        )
        compute_loss = (
            router_probs_flat
            @ self.expert_widths_normalized.to(router_probs_flat.dtype)
        ).mean()

        aux_loss = {
            "router_z_loss": router_z_loss,
            "load_balance_loss": load_balance_loss,
            "compute_loss": compute_loss,
        }

        return output, aux_loss, f_i

    @torch.compiler.disable
    def forward_with_intermediates(self, x: torch.Tensor) -> dict[str, Any]:
        """Forward pass that returns all intermediate tensors for testing.

        This method is used to compare Triton kernel outputs against the reference.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Dictionary containing:
                - output: Final output tensor
                - x_flat: Flattened input
                - router_logits: Router output before activation
                - router_probs: Router probabilities (sigmoid output)
                - top_k_weights: Normalized routing weights
                - selected_experts: Expert indices for each token
                - bin_ids: Sorted expert assignments
                - indices: Permutation indices from sorting
                - tokens_per_expert: Token count per expert
                - bins: Cumulative token counts
                - padded_bins: Padded cumulative counts
                - topology: Sparse topology matrix
                - x_gathered: Tokens gathered by expert assignment
                - x_after_up: After up-projection (sdd)
                - x_after_activation: After activation function
                - x_after_down: After down-projection (dsd)
                - x_scattered: After scatter back to original order
        """
        batch_size, seq_len, n_embd = x.shape

        x_flat = rearrange(
            x, "batch_size seq_len n_embd -> (batch_size seq_len) n_embd "
        )

        router_logits = self.router(x_flat)
        router_probs = F.sigmoid(router_logits.to(torch.float32))

        top_k_weights, selected_experts = torch.topk(
            router_probs, self.num_active_experts, dim=-1
        )

        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
        top_k_weights = top_k_weights.to(x.dtype)

        top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
        selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        bin_ids, indices, tokens_per_expert = self._sort_tokens_by_expert(
            selected_experts_flat
        )

        bins = ops.inclusive_cumsum(tokens_per_expert, 0).contiguous()
        padded_bins, topology = self._create_topology(x_flat, tokens_per_expert)

        # Gather
        x_gathered = ops.padded_gather(
            x_flat, indices, bin_ids, bins, padded_bins, self.num_active_experts
        )

        # Up-projection
        x_after_up = stk.ops.sdd(x_gathered, self.w1, topology)

        # Activation
        x_after_activation = relu_squared(x_after_up)

        # Down-projection
        x_after_down = stk.ops.dsd(x_after_activation, self.w2)

        # Scatter
        x_scattered = ops.padded_scatter(
            x_after_down,
            indices,
            bin_ids,
            top_k_weights_flat,
            bins,
            padded_bins,
            self.num_active_experts,
        )

        output = rearrange(
            x_scattered,
            "(batch_size seq_len) n_embd -> batch_size seq_len n_embd",
            batch_size=batch_size,
            seq_len=seq_len,
        )

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
            "topology": topology,
            "x_gathered": x_gathered,
            "x_after_up": x_after_up,
            "x_after_activation": x_after_activation,
            "x_after_down": x_after_down,
            "x_scattered": x_scattered,
        }

    def _sort_tokens_by_expert(self, selected_experts_flat):
        """Group token assignments by expert id."""
        bin_ids, indices = ops.sort(selected_experts_flat, self.sort_end_bit)
        tokens_per_expert = ops.histogram(selected_experts_flat, self.num_experts)
        return bin_ids, indices, tokens_per_expert

    def _create_topology(self, x, tokens_per_expert):
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.block_size)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = padded_bins.contiguous()

        padded_tokens = padded_bins[-1].clamp_min(self.block_size)

        block_rows = padded_tokens // self.block_size

        # Use variable-size topology with per-expert block counts
        column_indices = topology_var(
            padded_bins,
            self.expert_size_blocks,
            self.expert_block_offsets,
            self.block_size,
            block_rows,
        )

        # Compute all expert token blocks at once
        expert_token_blocks = padded_tokens_per_expert // self.block_size

        # Repeat each expert's size by how many token blocks it handles
        repeated_sizes = torch.repeat_interleave(
            self.expert_size_blocks, expert_token_blocks
        )

        # Cumulative sum gives you offsets
        offsets = torch.cat([repeated_sizes.new_zeros(1), repeated_sizes.cumsum(0)])

        column_indices = column_indices.to(torch.int32)
        offsets = offsets.to(torch.int32)

        shape = (padded_tokens, self.total_expert_width)

        num_blocks = column_indices.numel()
        data_placeholder = torch.empty(
            num_blocks,
            self.block_size,
            self.block_size,
            dtype=x.dtype,
            device="meta",
        )

        row_indices = stk.ops.row_indices(
            shape, data_placeholder, offsets, column_indices
        )
        row_indices = row_indices.to(torch.int32)

        column_indices_t, offsets_t, block_offsets_t = self._sparse_transpose(
            shape, row_indices, column_indices
        )
        column_indices_t = column_indices_t.to(torch.int32)
        offsets_t = offsets_t.to(torch.int32)
        block_offsets_t = block_offsets_t.to(torch.int32)

        topology = stk.Matrix(
            shape,
            data_placeholder,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

        return padded_bins, topology

    def _sparse_transpose(self, size, row_indices, column_indices):
        # Use total_expert_width instead of d_ffn * num_experts
        block_columns = self.total_expert_width // self.block_size

        _, gather_indices = ops.sort(
            column_indices.int(),
            self.transpose_sort_end_bit,
        )

        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    @torch.compiler.disable
    def forward_profiled(
        self, x: torch.Tensor, profiler
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with per-step profiling.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]
            profiler: CUDAStepProfiler instance to record step times

        Returns:
            output: Output tensor of shape [batch_size, seq_len, n_embd]
            aux_loss: Dictionary of auxiliary losses
            f_i: Token distribution per expert
        """
        batch_size, seq_len, n_embd = x.shape

        with profiler.step("flatten"):
            x_flat = rearrange(
                x, "batch_size seq_len n_embd -> (batch_size seq_len) n_embd "
            )

        with profiler.step("routing"):
            router_logits = self.router(x_flat)
            router_probs = F.sigmoid(router_logits.to(torch.float32))
            top_k_weights, selected_experts = torch.topk(
                router_probs, self.num_active_experts, dim=-1
            )
            top_k_weights = top_k_weights / (
                top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
            top_k_weights = top_k_weights.to(x.dtype)
            top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
            selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        with profiler.step("sort"):
            bin_ids, indices, tokens_per_expert = self._sort_tokens_by_expert(
                selected_experts_flat
            )
            bins = ops.inclusive_cumsum(tokens_per_expert, 0).contiguous()

        with profiler.step("topo_setup"):
            padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.block_size)
            padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0).contiguous()
            padded_tokens = padded_bins[-1].clamp_min(self.block_size)
            block_rows = padded_tokens // self.block_size

        with profiler.step("topo_column_idx"):
            column_indices = topology_var(
                padded_bins,
                self.expert_size_blocks,
                self.expert_block_offsets,
                self.block_size,
                block_rows,
            )
            expert_token_blocks = padded_tokens_per_expert // self.block_size
            repeated_sizes = torch.repeat_interleave(
                self.expert_size_blocks, expert_token_blocks
            )
            offsets = torch.cat([repeated_sizes.new_zeros(1), repeated_sizes.cumsum(0)])
            column_indices = column_indices.to(torch.int32)
            offsets = offsets.to(torch.int32)

        with profiler.step("topo_row_idx"):
            shape = (padded_tokens, self.total_expert_width)
            num_blocks = column_indices.numel()
            data_placeholder = torch.empty(
                num_blocks,
                self.block_size,
                self.block_size,
                dtype=x_flat.dtype,
                device="meta",
            )
            row_indices = stk.ops.row_indices(
                shape, data_placeholder, offsets, column_indices
            ).to(torch.int32)

        with profiler.step("topo_transpose"):
            block_columns = self.total_expert_width // self.block_size
            _, gather_indices = ops.sort(
                column_indices.int(), self.transpose_sort_end_bit
            )
            column_indices_t = row_indices.gather(0, gather_indices.long())
            block_offsets_t = gather_indices.int()
            zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
            nnz_per_column = ops.histogram(column_indices, block_columns)
            nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
            if nnz_per_column.dim() == 0:
                nnz_per_column = nnz_per_column.unsqueeze(0)
            offsets_t = torch.cat([zero, nnz_per_column])

        with profiler.step("topo_matrix"):
            topology = stk.Matrix(
                shape,
                data_placeholder,
                row_indices,
                column_indices,
                offsets,
                column_indices_t.to(torch.int32),
                offsets_t.to(torch.int32),
                block_offsets_t.to(torch.int32),
            )

        with profiler.step("gather"):
            x_permuted = ops.padded_gather(
                x_flat, indices, bin_ids, bins, padded_bins, self.num_active_experts
            )

        with profiler.step("sdd_up"):
            x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)

        with profiler.step("activation"):
            x_permuted = relu_squared(x_permuted)

        with profiler.step("dsd_down"):
            x_permuted = stk.ops.dsd(x_permuted, self.w2)

        with profiler.step("scatter"):
            x_permuted = ops.padded_scatter(
                x_permuted,
                indices,
                bin_ids,
                top_k_weights_flat,
                bins,
                padded_bins,
                self.num_active_experts,
            )

        with profiler.step("reshape"):
            output = rearrange(
                x_permuted,
                "(batch_size seq_len) n_embd -> batch_size seq_len n_embd",
                batch_size=batch_size,
                seq_len=seq_len,
            )

        with profiler.step("aux_z_loss"):
            router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        with profiler.step("aux_load_balance"):
            f_i = (tokens_per_expert.float() / tokens_per_expert.sum()).to(x.dtype)
            load_balance_loss = self._compute_load_balance_loss(
                router_probs, selected_experts_flat, f_i
            )

        with profiler.step("aux_compute_loss"):
            router_probs_flat = rearrange(
                router_probs,
                "(batch_size seq_len) n_embd -> (batch_size seq_len) n_embd",
                batch_size=batch_size,
                seq_len=seq_len,
            )
            compute_loss = (
                router_probs_flat
                @ self.expert_widths_normalized.to(router_probs_flat.dtype)
            ).mean()
            aux_loss = {
                "router_z_loss": router_z_loss,
                "load_balance_loss": load_balance_loss,
                "compute_loss": compute_loss,
            }

        return output, aux_loss, f_i

    def _compute_load_balance_loss(self, router_probs, experts_flat, f_i):
        """Compute load balance loss within expert groups (vectorized).

        For variable-sized experts, the loss is computed per group and averaged.
        Uses precomputed group_membership matrix for efficient matmul-based aggregation.
        """
        p_i = router_probs.mean(dim=0)

        # Fast path for uniform expert sizes
        if len(set(self.expert_widths)) == 1:
            return self.num_experts * (f_i.float() @ p_i.float())

        # Fast path if no valid groups (all single-expert groups)
        if self._num_valid_groups == 0:
            return f_i.float() @ p_i.float()

        # Compute f_i * p_i elementwise
        fi_pi = f_i.float() * p_i.float()

        # If all experts are in valid groups, skip masking entirely
        membership = self.group_membership.to(fi_pi.dtype)
        if self._all_experts_valid:
            # Use matmul to sum within groups: [num_experts] @ [num_experts, num_groups] -> [num_groups]
            group_sums = fi_pi @ membership
        else:
            # Only consider valid experts using index_select (faster than boolean indexing)
            fi_pi_valid = fi_pi.index_select(0, self.valid_expert_indices)
            membership_valid = membership.index_select(0, self.valid_expert_indices)
            group_sums = fi_pi_valid @ membership_valid

        # Multiply each group sum by group size, then average
        group_losses = self.group_sizes.to(fi_pi.dtype) * group_sums
        return group_losses.mean()
