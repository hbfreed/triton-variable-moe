"""Gather kernel: rearrange tokens so all tokens for each expert are contiguous.

Reference: megablocks.ops.padded_gather
"""

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd
from torch.autograd.function import once_differentiable


@triton.jit
def _padded_gather_kernel(
    x_ptr,           # [num_tokens, hidden_size]
    indices_ptr,     # [num_expanded] indices into expanded array (divide by top_k for token idx)
    bin_ids_ptr,     # [num_expanded] which expert each position belongs to
    bins_ptr,        # [num_experts] cumsum of tokens per expert (unpadded)
    padded_bins_ptr, # [num_experts] cumsum of tokens per expert (padded)
    output_ptr,      # [total_padded, hidden_size]
    hidden_size,
    top_k,
    x_stride_row,
    x_stride_col,
    out_stride_row,
    out_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather with padding between expert bins.

    Each program handles one position in the sorted (but unpadded) indices.
    We compute where it goes in the padded output.
    """
    pid = tl.program_id(0)  # position in sorted indices

    # Which expert does this position belong to?
    expert_id = tl.load(bin_ids_ptr + pid)

    # Where does this expert's bin start in unpadded space?
    bin_start = tl.load(bins_ptr + expert_id - 1) if expert_id > 0 else 0

    # Where does this expert's bin start in padded output?
    padded_start = tl.load(padded_bins_ptr + expert_id - 1) if expert_id > 0 else 0

    # Offset within the bin
    offset_in_bin = pid - bin_start

    # Output position
    out_row = padded_start + offset_in_bin

    # Source token index (indices are into num_tokens * top_k expanded array)
    expanded_idx = tl.load(indices_ptr + pid)
    src_token = expanded_idx // top_k

    # Copy hidden_size elements
    for offset in range(0, hidden_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        vals = tl.load(x_ptr + src_token * x_stride_row + cols * x_stride_col, mask=mask)
        tl.store(output_ptr + out_row * out_stride_row + cols * out_stride_col, vals, mask=mask)


def padded_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Gather with padding - matches megablocks.ops.padded_gather interface.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        indices: Sorted indices [num_expanded] - positions in expanded (num_tokens * top_k) array
        bin_ids: Expert ID for each sorted position [num_expanded]
        bins: Cumsum of tokens per expert, unpadded [num_experts]
        padded_bins: Cumsum of tokens per expert, padded [num_experts]
        top_k: Number of experts per token

    Returns:
        Gathered tensor [padded_bins[-1], hidden_size] with zeros in padding slots
    """
    num_tokens, hidden_size = x.shape
    num_expanded = len(indices)
    num_experts = len(bins)
    total_padded = padded_bins[-1].item()

    # Allocate output with zeros (padding slots stay zero)
    output = torch.zeros(total_padded, hidden_size, device=x.device, dtype=x.dtype)

    if num_expanded == 0:
        return output

    BLOCK_SIZE = 128
    grid = (num_expanded,)

    _padded_gather_kernel[grid](
        x,
        indices,
        bin_ids,
        bins,
        padded_bins,
        output,
        hidden_size,
        top_k,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# =============================================================================
# Weighted padded gather (used internally for scatter backward)
# =============================================================================


@triton.jit
def _weighted_padded_gather_kernel(
    x_ptr,           # [num_tokens, hidden_size]
    indices_ptr,     # [num_expanded] indices into expanded array
    bin_ids_ptr,     # [num_expanded] which expert each position belongs to
    weights_ptr,     # [num_tokens * top_k] weights indexed by expanded_idx
    bins_ptr,        # [num_experts] cumsum of tokens per expert (unpadded)
    padded_bins_ptr, # [num_experts] cumsum of tokens per expert (padded)
    output_ptr,      # [total_padded, hidden_size]
    hidden_size,
    top_k,
    x_stride_row,
    x_stride_col,
    out_stride_row,
    out_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather with padding and weight application.

    Same as padded_gather but multiplies by weight[expanded_idx].
    Used for scatter backward.
    """
    pid = tl.program_id(0)  # position in sorted indices

    # Which expert does this position belong to?
    expert_id = tl.load(bin_ids_ptr + pid)

    # Where does this expert's bin start in unpadded space?
    bin_start = tl.load(bins_ptr + expert_id - 1) if expert_id > 0 else 0

    # Where does this expert's bin start in padded output?
    padded_start = tl.load(padded_bins_ptr + expert_id - 1) if expert_id > 0 else 0

    # Offset within the bin
    offset_in_bin = pid - bin_start

    # Output position
    out_row = padded_start + offset_in_bin

    # Source token index (indices are into num_tokens * top_k expanded array)
    expanded_idx = tl.load(indices_ptr + pid)
    src_token = expanded_idx // top_k

    # Load weight for this position
    weight = tl.load(weights_ptr + expanded_idx)

    # Copy hidden_size elements with weight applied
    for offset in range(0, hidden_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        vals = tl.load(x_ptr + src_token * x_stride_row + cols * x_stride_col, mask=mask)
        tl.store(output_ptr + out_row * out_stride_row + cols * out_stride_col, vals * weight, mask=mask)


def _weighted_padded_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Gather with padding and weight application. Internal use for scatter backward."""
    num_tokens, hidden_size = x.shape
    num_expanded = len(indices)
    num_experts = len(bins)
    total_padded = padded_bins[-1].item()

    output = torch.zeros(total_padded, hidden_size, device=x.device, dtype=x.dtype)

    if num_expanded == 0:
        return output

    BLOCK_SIZE = 128
    grid = (num_expanded,)

    _weighted_padded_gather_kernel[grid](
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        output,
        hidden_size,
        top_k,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# =============================================================================
# Autograd wrapper
# =============================================================================


class PaddedGatherOp(torch.autograd.Function):
    """Autograd function for padded_gather."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices, bin_ids, bins, padded_bins)
        ctx.top_k = top_k
        ctx.num_tokens = x.shape[0]
        return padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        indices, bin_ids, bins, padded_bins = ctx.saved_tensors
        top_k = ctx.top_k
        num_tokens = ctx.num_tokens

        # Import here to avoid circular import
        from .scatter import padded_scatter

        # Backward of gather is scatter (with uniform weights = 1)
        weights = torch.ones(
            num_tokens * top_k, device=grad_output.device, dtype=grad_output.dtype
        )

        grad_x = padded_scatter(
            grad_output, indices, bin_ids, weights, bins, padded_bins, top_k
        )

        return grad_x, None, None, None, None, None


def padded_gather_autograd(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Padded gather with autograd support."""
    return PaddedGatherOp.apply(x, indices, bin_ids, bins, padded_bins, top_k)
