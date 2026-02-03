"""Scatter kernel: un-permute tokens back to original order with weighted accumulation.

Reference: megablocks.ops.padded_scatter
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    gathered_ptr,        # [total_expanded_tokens, hidden_size]
    indices_ptr,         # [total_expanded_tokens] original token indices
    weights_ptr,         # [total_expanded_tokens] routing weights
    output_ptr,          # [num_tokens, hidden_size] output
    hidden_size,
    top_k,
    gathered_stride_row,
    gathered_stride_col,
    out_stride_row,
    out_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter tokens back to original positions with weighted accumulation.

    Each program handles one gathered row, atomically adding to output.
    """
    pid = tl.program_id(0)  # which gathered row

    # Where does this go in the output?
    flat_idx = tl.load(indices_ptr + pid)
    token_idx = flat_idx // top_k

    weight = tl.load(weights_ptr + pid)

    for offset in range(0, hidden_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        val = tl.load(
            gathered_ptr + pid * gathered_stride_row + cols * gathered_stride_col,
            mask=mask
        )
        tl.atomic_add(
            output_ptr + token_idx * out_stride_row + cols * out_stride_col,
            val * weight,
            mask=mask
        )


def scatter(
    x_gathered: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Scatter tokens back to original order with weighted accumulation.

    Args:
        x_gathered: Gathered tensor of shape [total_expanded_tokens, hidden_size]
        indices: Permutation indices of shape [total_expanded_tokens]
        weights: Routing weights of shape [total_expanded_tokens]
        num_tokens: Number of output tokens
        top_k: Number of experts per token

    Returns:
        Output tensor of shape [num_tokens, hidden_size]

    Reference: megablocks.ops.padded_scatter
    """
    raise NotImplementedError(
        "TODO: Implement Triton scatter kernel. "
        "Reference: megablocks.ops.padded_scatter"
    )
