"""Gather kernel: rearrange tokens so all tokens for each expert are contiguous.

Reference: megablocks.ops.padded_gather
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kernel(
    x_ptr,           # [num_tokens, hidden_size] input activations
    indices_ptr,     # [total_expanded_tokens] permutation indices
    output_ptr,      # [total_expanded_tokens, hidden_size] output
    hidden_size,
    x_stride_row,
    x_stride_col,
    out_stride_row,
    out_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather tokens by expert assignment.

    Each program handles one output row (one gathered token).
    """
    pid = tl.program_id(0)  # which output row

    # Which original token does this output row come from?
    src_idx = tl.load(indices_ptr + pid)

    # Copy hidden_size elements
    for offset in range(0, hidden_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        vals = tl.load(x_ptr + src_idx * x_stride_row + cols * x_stride_col, mask=mask)
        tl.store(output_ptr + pid * out_stride_row + cols * out_stride_col, vals, mask=mask)


def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    num_output_tokens: int | None = None,
) -> torch.Tensor:
    """Gather tokens by expert assignment.

    Args:
        x: Input tensor of shape [num_tokens, hidden_size]
        indices: Permutation indices of shape [total_expanded_tokens]
        num_output_tokens: Number of output tokens (defaults to len(indices))

    Returns:
        Gathered tensor of shape [total_expanded_tokens, hidden_size]

    Reference: megablocks.ops.padded_gather
    """
    raise NotImplementedError(
        "TODO: Implement Triton gather kernel. "
        "Reference: megablocks.ops.padded_gather"
    )
