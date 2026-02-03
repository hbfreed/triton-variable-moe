"""Grouped GEMM kernels for MoE up/down projections.

These are the main computational kernels. They support variable-sized experts
using dynamic tile scheduling - each program computes its (expert, m_tile, n_tile)
on-the-fly from the program ID, eliminating CPU-side tile schedule computation.

Reference: stk.ops.sdd (up-projection), stk.ops.dsd (down-projection)

Weight layouts (matching reference moe.py):
    w1: [hidden_size, total_expert_width]  - experts concatenated along columns
    w2: [total_expert_width, hidden_size]  - experts concatenated along rows

Intermediate layout (v1 - padded for simplicity):
    [total_tokens, max_expert_width]  - each expert writes to cols 0:expert_width
    Only valid regions are written; smaller experts leave cols expert_width:max unused.
    This wastes some memory but simplifies indexing. Can optimize to concatenated later.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_tile_info(
    pid,
    tokens_per_expert_ptr,
    expert_widths_ptr,
    num_experts,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute (expert_idx, m_start, n_start) from program ID.

    Iterates through experts, accumulating tile counts until we find
    which expert this pid belongs to, then computes local tile indices.
    """
    # Find which expert this pid belongs to by accumulating tile counts
    expert_idx = 0
    tiles_before = 0
    found = False

    for i in range(num_experts):
        num_tokens = tl.load(tokens_per_expert_ptr + i)
        expert_width = tl.load(expert_widths_ptr + i)

        # Compute tiles for this expert
        m_tiles = tl.cdiv(num_tokens, BLOCK_M)
        n_tiles = tl.cdiv(expert_width, BLOCK_N)
        expert_tiles = m_tiles * n_tiles

        # Check if pid falls within this expert's tiles (only update if not found yet)
        if not found:
            if pid < tiles_before + expert_tiles:
                expert_idx = i
                found = True
            else:
                tiles_before += expert_tiles

    # Compute local tile index within this expert
    local_pid = pid - tiles_before

    # Load this expert's dimensions
    expert_width = tl.load(expert_widths_ptr + expert_idx)
    n_tiles = tl.cdiv(expert_width, BLOCK_N)

    # Convert local_pid to (m_tile, n_tile) using row-major order
    m_tile = local_pid // n_tiles
    n_tile = local_pid % n_tiles

    m_start = m_tile * BLOCK_M
    n_start = n_tile * BLOCK_N

    return expert_idx, m_start, n_start


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["hidden_size", "num_experts"],
)
@triton.jit
def _grouped_gemm_up_kernel(
    # Pointers
    x_ptr,  # [total_tokens, hidden_size]
    w1_ptr,  # [hidden_size, total_expert_width]
    out_ptr,  # [total_tokens, max_expert_width] (padded)
    # Strides
    stride_x_row,
    stride_x_col,
    stride_w1_row,
    stride_w1_col,
    stride_out_row,
    stride_out_col,
    # Expert metadata
    expert_token_offsets_ptr,  # [num_experts + 1] cumsum of tokens per expert
    expert_weight_offsets_ptr,  # [num_experts + 1] cumsum of expert widths
    tokens_per_expert_ptr,  # [num_experts] token counts (for dynamic tile scheduling)
    expert_widths_ptr,  # [num_experts] width of each expert
    # Dimensions
    hidden_size,
    num_experts,
    # Tile sizes (set by autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Activation: 0=none, 1=relu_squared, 2=swiglu
    ACTIVATION: tl.constexpr,
):
    """Grouped GEMM for up-projection: intermediate = x @ W1

    Each program computes its (expert_idx, m_start, n_start) dynamically
    from the program ID, enabling autotune to vary block sizes freely.
    """
    pid = tl.program_id(0)

    # Dynamically compute which tile this program handles
    expert_idx, m_start, n_start = _compute_tile_info(
        pid, tokens_per_expert_ptr, expert_widths_ptr, num_experts, BLOCK_M, BLOCK_N
    )

    # Load expert-specific metadata
    token_start = tl.load(expert_token_offsets_ptr + expert_idx)
    token_end = tl.load(expert_token_offsets_ptr + expert_idx + 1)
    weight_col_start = tl.load(expert_weight_offsets_ptr + expert_idx)
    expert_width = tl.load(expert_widths_ptr + expert_idx)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if ACTIVATION == 2:  # swiglu needs two accumulators
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Offset ranges
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    # Global indices
    m_indices = token_start + m_start + offs_m

    # Boundary masks
    m_mask = m_indices < token_end
    n_mask = (n_start + offs_n) < expert_width

    # Weight column indices
    w1_col_indices = weight_col_start + n_start + offs_n
    if ACTIVATION == 2:  # swiglu needs second set of columns
        up_col_indices = weight_col_start + expert_width + n_start + offs_n

    # Main matmul loop over K (hidden_size)
    for k in range(0, hidden_size, BLOCK_K):
        k_indices = k + offs_k
        k_mask = k_indices < hidden_size

        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + m_indices[:, None] * stride_x_row + k_indices[None, :] * stride_x_col
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w1 tile [BLOCK_K, BLOCK_N]
        w1_ptrs = w1_ptr + k_indices[:, None] * stride_w1_row + w1_col_indices[None, :] * stride_w1_col
        w_mask = k_mask[:, None] & n_mask[None, :]
        w1_tile = tl.load(w1_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, w1_tile)

        # For swiglu: also load up weight tile
        if ACTIVATION == 2:
            up_ptrs = w1_ptr + k_indices[:, None] * stride_w1_row + up_col_indices[None, :] * stride_w1_col
            up_tile = tl.load(up_ptrs, mask=w_mask, other=0.0)
            acc_up += tl.dot(x_tile, up_tile)

    # Apply fused activation
    if ACTIVATION == 1:  # relu_squared
        acc = tl.where(acc > 0, acc * acc, 0.0)
    elif ACTIVATION == 2:  # swiglu: silu(gate) * up
        acc = (acc * tl.sigmoid(acc)) * acc_up

    # Store output tile [BLOCK_M, BLOCK_N]
    out_col_indices = n_start + offs_n
    out_ptrs = out_ptr + m_indices[:, None] * stride_out_row + out_col_indices[None, :] * stride_out_col
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def _compute_total_tiles(tokens_per_expert, expert_widths, block_m, block_n):
    """Compute total number of tiles for grid launch."""
    total = 0
    for num_tokens, width in zip(tokens_per_expert.tolist(), expert_widths.tolist()):
        m_tiles = (num_tokens + block_m - 1) // block_m
        n_tiles = (width + block_n - 1) // block_n
        total += m_tiles * n_tiles
    return total


def grouped_gemm_up(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    expert_weight_offsets: torch.Tensor,
    expert_widths: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    max_expert_width: int,
    activation: str = "relu_squared",
) -> torch.Tensor:
    """Grouped GEMM for up-projection with fused activation.

    Args:
        x: Gathered input [total_tokens, hidden_size] - tokens sorted by expert
        w1: Up-projection weights [hidden_size, total_expert_width]
        expert_token_offsets: Cumulative token counts [num_experts + 1]
        expert_weight_offsets: Cumulative weight column offsets [num_experts + 1]
        expert_widths: Per-expert intermediate sizes [num_experts]
        tokens_per_expert: Token counts per expert [num_experts]
        max_expert_width: Max expert width (for output allocation)
        activation: Activation function ("none", "relu_squared", "swiglu")

    Returns:
        Intermediate activations [total_tokens, max_expert_width] (padded)
    """
    total_tokens, hidden_size = x.shape
    num_experts = len(expert_widths)
    device = x.device
    dtype = x.dtype

    # Allocate output
    output = torch.zeros(total_tokens, max_expert_width, device=device, dtype=dtype)

    if total_tokens == 0:
        return output

    activation_code = {"none": 0, "relu_squared": 1, "swiglu": 2}.get(activation, 0)

    # Grid function computes total tiles based on autotune-selected block sizes
    def grid(meta):
        return (_compute_total_tiles(tokens_per_expert, expert_widths, meta["BLOCK_M"], meta["BLOCK_N"]),)

    # Launch kernel
    _grouped_gemm_up_kernel[grid](
        x,
        w1,
        output,
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        output.stride(0),
        output.stride(1),
        expert_token_offsets,
        expert_weight_offsets,
        tokens_per_expert,
        expert_widths,
        hidden_size,
        num_experts,
        ACTIVATION=activation_code,
    )

    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["hidden_size", "num_experts"],
)
@triton.jit
def _grouped_gemm_down_kernel(
    # Pointers
    intermediate_ptr,  # [total_tokens, max_expert_width] (padded)
    w2_ptr,  # [total_expert_width, hidden_size]
    out_ptr,  # [total_tokens, hidden_size]
    # Strides
    stride_int_row,
    stride_int_col,
    stride_w2_row,
    stride_w2_col,
    stride_out_row,
    stride_out_col,
    # Expert metadata
    expert_token_offsets_ptr,  # [num_experts + 1]
    expert_weight_offsets_ptr,  # [num_experts + 1] - row offsets in w2
    tokens_per_expert_ptr,  # [num_experts] for dynamic tile scheduling
    expert_widths_ptr,  # [num_experts]
    # Dimensions
    hidden_size,
    num_experts,
    # Tile sizes (set by autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped GEMM for down-projection: output = intermediate @ W2

    For down projection, we tile over (M=tokens, N=hidden_size) and loop over K=expert_width.
    Note: N dimension is hidden_size (uniform), but we use expert_widths_ptr to store
    hidden_size repeated for compatibility with _compute_tile_info.
    """
    pid = tl.program_id(0)

    # For down kernel, expert_widths_ptr contains hidden_size repeated (for tile computation)
    # We compute tile info using hidden_size as the "width" dimension
    expert_idx, m_start, n_start = _compute_tile_info(
        pid, tokens_per_expert_ptr, expert_widths_ptr, num_experts, BLOCK_M, BLOCK_N
    )

    # Load expert-specific metadata
    token_start = tl.load(expert_token_offsets_ptr + expert_idx)
    token_end = tl.load(expert_token_offsets_ptr + expert_idx + 1)
    weight_row_start = tl.load(expert_weight_offsets_ptr + expert_idx)

    # For down kernel, we need the actual expert width (K dimension)
    # expert_widths_ptr here contains hidden_size, so load from a separate source
    # Actually, we'll pass the real expert widths separately - see wrapper
    # For now, compute from weight offsets
    weight_row_end = tl.load(expert_weight_offsets_ptr + expert_idx + 1)
    expert_width = weight_row_end - weight_row_start

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_indices = token_start + m_start + offs_m

    m_mask = m_indices < token_end
    n_mask = (n_start + offs_n) < hidden_size

    # W2 column indices: just into hidden_size (no expert offset)
    w2_col_indices = n_start + offs_n

    # Loop over K (expert_width, variable per expert)
    for k in range(0, expert_width, BLOCK_K):
        k_indices = k + offs_k
        k_mask = k_indices < expert_width

        # Intermediate: [total_tokens, max_expert_width] padded layout
        intermediate_ptrs = intermediate_ptr + m_indices[:, None] * stride_int_row + k_indices[None, :] * stride_int_col
        int_mask = m_mask[:, None] & k_mask[None, :]
        int_tile = tl.load(intermediate_ptrs, mask=int_mask, other=0.0)

        # W2: [total_expert_width, hidden_size]
        w2_row_indices = weight_row_start + k_indices
        w2_ptrs = w2_ptr + w2_row_indices[:, None] * stride_w2_row + w2_col_indices[None, :] * stride_w2_col
        w_mask = k_mask[:, None] & n_mask[None, :]
        w2_tile = tl.load(w2_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(int_tile, w2_tile)

    out_col_indices = n_start + offs_n
    out_ptrs = out_ptr + m_indices[:, None] * stride_out_row + out_col_indices[None, :] * stride_out_col
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def grouped_gemm_down(
    intermediate: torch.Tensor,
    w2: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    expert_weight_offsets: torch.Tensor,
    expert_widths: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    """Grouped GEMM for down-projection.

    Args:
        intermediate: Intermediate activations [total_tokens, max_expert_width] (padded)
        w2: Down-projection weights [total_expert_width, hidden_size]
        expert_token_offsets: Cumulative token counts [num_experts + 1]
        expert_weight_offsets: Cumulative weight row offsets [num_experts + 1]
        expert_widths: Per-expert intermediate sizes [num_experts]
        tokens_per_expert: Token counts per expert [num_experts]
        hidden_size: Output hidden dimension

    Returns:
        Output tensor [total_tokens, hidden_size]
    """
    total_tokens = intermediate.shape[0]
    num_experts = len(expert_widths)
    device = intermediate.device
    dtype = intermediate.dtype

    output = torch.zeros(total_tokens, hidden_size, device=device, dtype=dtype)

    if total_tokens == 0:
        return output

    # For down kernel, tile over (tokens, hidden_size), so "width" for tiling is hidden_size
    hidden_size_per_expert = torch.full_like(expert_widths, hidden_size)

    def grid(meta):
        return (_compute_total_tiles(tokens_per_expert, hidden_size_per_expert, meta["BLOCK_M"], meta["BLOCK_N"]),)

    _grouped_gemm_down_kernel[grid](
        intermediate,
        w2,
        output,
        intermediate.stride(0),
        intermediate.stride(1),
        w2.stride(0),
        w2.stride(1),
        output.stride(0),
        output.stride(1),
        expert_token_offsets,
        expert_weight_offsets,
        tokens_per_expert,
        hidden_size_per_expert,  # Used for tile computation (N = hidden_size)
        hidden_size,
        num_experts,
    )

    return output
