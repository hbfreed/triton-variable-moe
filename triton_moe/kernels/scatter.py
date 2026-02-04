"""Scatter kernel: un-permute tokens back to original order with weighted accumulation.

Reference: megablocks.ops.padded_scatter
"""

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd
from torch.autograd.function import once_differentiable


@triton.jit
def _padded_scatter_kernel(
    x_ptr,              # [total_padded, hidden_size] - padded gathered input
    indices_ptr,        # [num_expanded] - indices into expanded (num_tokens * top_k) array
    bin_ids_ptr,        # [num_expanded] - which expert each position belongs to
    weights_ptr,        # [num_expanded] - routing weights
    bins_ptr,           # [num_experts] - cumsum of tokens per expert (unpadded)
    padded_bins_ptr,    # [num_experts] - cumsum of tokens per expert (padded)
    output_ptr,         # [num_tokens, top_k, hidden_size] - intermediate output
    hidden_size,
    top_k,
    x_stride_row,
    x_stride_col,
    out_stride_token,
    out_stride_k,
    out_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter from padded layout to (tokens, top_k, hidden) intermediate.

    Each program handles one position in the sorted indices.
    """
    pid = tl.program_id(0)  # position in sorted indices

    # Which expert does this position belong to?
    expert_id = tl.load(bin_ids_ptr + pid)

    # Where does this expert's bin start in unpadded space?
    bin_start = tl.load(bins_ptr + expert_id - 1) if expert_id > 0 else 0

    # Where does this expert's bin start in padded input?
    padded_start = tl.load(padded_bins_ptr + expert_id - 1) if expert_id > 0 else 0

    # Offset within the bin
    offset_in_bin = pid - bin_start

    # Input row in padded tensor
    in_row = padded_start + offset_in_bin

    # Output token index (indices are into num_tokens * top_k expanded array)
    expanded_idx = tl.load(indices_ptr + pid)
    token_idx = expanded_idx // top_k
    k_idx = expanded_idx % top_k

    # Load weight
    weight = tl.load(weights_ptr + expanded_idx)

    # Copy with weighting
    for offset in range(0, hidden_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        vals = tl.load(x_ptr + in_row * x_stride_row + cols * x_stride_col, mask=mask)
        tl.store(
            output_ptr + token_idx * out_stride_token + k_idx * out_stride_k + cols * out_stride_col,
            vals * weight,
            mask=mask,
        )


def padded_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Scatter with padding - matches megablocks.ops.padded_scatter interface.

    Args:
        x: Padded gathered tensor [padded_bins[-1], hidden_size]
        indices: Sorted indices [num_expanded] - positions in expanded (num_tokens * top_k) array
        bin_ids: Expert ID for each sorted position [num_expanded]
        weights: Routing weights [num_tokens * top_k] (indexed by expanded_idx)
        bins: Cumsum of tokens per expert, unpadded [num_experts]
        padded_bins: Cumsum of tokens per expert, padded [num_experts]
        top_k: Number of experts per token

    Returns:
        Output tensor [num_tokens, hidden_size]
    """
    total_padded, hidden_size = x.shape
    num_expanded = len(indices)
    num_experts = len(bins)
    num_tokens = len(weights) // top_k

    if num_expanded == 0:
        return torch.zeros(num_tokens, hidden_size, device=x.device, dtype=x.dtype)

    # Intermediate: (num_tokens, top_k, hidden_size)
    # Each (token, k) slot written exactly once - no conflicts
    intermediate = torch.zeros(
        num_tokens, top_k, hidden_size,
        device=x.device, dtype=x.dtype
    )

    BLOCK_SIZE = 128
    grid = (num_expanded,)

    _padded_scatter_kernel[grid](
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        intermediate,
        hidden_size,
        top_k,
        x.stride(0),
        x.stride(1),
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Sum over top_k dimension
    return intermediate.sum(dim=1)


# =============================================================================
# Autograd wrapper
# =============================================================================


class PaddedScatterOp(torch.autograd.Function):
    """Autograd function for padded_scatter."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices, bin_ids, weights, bins, padded_bins)
        ctx.top_k = top_k
        return padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k)

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        indices, bin_ids, weights, bins, padded_bins = ctx.saved_tensors
        top_k = ctx.top_k

        # Import here to avoid circular import
        from .gather import _weighted_padded_gather

        # Backward of scatter is gather with weights applied
        # grad_x[padded_pos] = grad_output[token] * weight
        grad_x = _weighted_padded_gather(
            grad_output, indices, bin_ids, weights, bins, padded_bins, top_k
        )

        return grad_x, None, None, None, None, None, None


def padded_scatter_autograd(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Padded scatter with autograd support."""
    return PaddedScatterOp.apply(x, indices, bin_ids, weights, bins, padded_bins, top_k)
