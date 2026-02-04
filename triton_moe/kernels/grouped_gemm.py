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

Backward pass math:
    Up-projection forward:  y = activation(x @ W1)
    Up-projection backward: grad_x = grad_act @ W1.T
                            grad_W1 = x.T @ grad_act
                            (where grad_act = grad_y * activation_derivative)

    Down-projection forward:  y = intermediate @ W2
    Down-projection backward: grad_intermediate = grad_y @ W2.T
                              grad_W2 = intermediate.T @ grad_y
"""

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd
from torch.autograd.function import once_differentiable


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
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
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
        x_ptrs = (
            x_ptr
            + m_indices[:, None] * stride_x_row
            + k_indices[None, :] * stride_x_col
        )
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w1 tile [BLOCK_K, BLOCK_N]
        w1_ptrs = (
            w1_ptr
            + k_indices[:, None] * stride_w1_row
            + w1_col_indices[None, :] * stride_w1_col
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        w1_tile = tl.load(w1_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, w1_tile)

        # For swiglu: also load up weight tile
        if ACTIVATION == 2:
            up_ptrs = (
                w1_ptr
                + k_indices[:, None] * stride_w1_row
                + up_col_indices[None, :] * stride_w1_col
            )
            up_tile = tl.load(up_ptrs, mask=w_mask, other=0.0)
            acc_up += tl.dot(x_tile, up_tile)

    # Apply fused activation
    if ACTIVATION == 1:  # relu_squared
        acc = tl.where(acc > 0, acc * acc, 0.0)
    elif ACTIVATION == 2:  # swiglu: silu(gate) * up
        acc = (acc * tl.sigmoid(acc)) * acc_up

    # Store output tile [BLOCK_M, BLOCK_N]
    out_col_indices = n_start + offs_n
    out_ptrs = (
        out_ptr
        + m_indices[:, None] * stride_out_row
        + out_col_indices[None, :] * stride_out_col
    )
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
        return (
            _compute_total_tiles(
                tokens_per_expert, expert_widths, meta["BLOCK_M"], meta["BLOCK_N"]
            ),
        )

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
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
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
        intermediate_ptrs = (
            intermediate_ptr
            + m_indices[:, None] * stride_int_row
            + k_indices[None, :] * stride_int_col
        )
        int_mask = m_mask[:, None] & k_mask[None, :]
        int_tile = tl.load(intermediate_ptrs, mask=int_mask, other=0.0)

        # W2: [total_expert_width, hidden_size]
        w2_row_indices = weight_row_start + k_indices
        w2_ptrs = (
            w2_ptr
            + w2_row_indices[:, None] * stride_w2_row
            + w2_col_indices[None, :] * stride_w2_col
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        w2_tile = tl.load(w2_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(int_tile, w2_tile)

    out_col_indices = n_start + offs_n
    out_ptrs = (
        out_ptr
        + m_indices[:, None] * stride_out_row
        + out_col_indices[None, :] * stride_out_col
    )
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
        return (
            _compute_total_tiles(
                tokens_per_expert,
                hidden_size_per_expert,
                meta["BLOCK_M"],
                meta["BLOCK_N"],
            ),
        )

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


@triton.autotune(
    configs=[
        # BLOCK_K=32 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        # BLOCK_K=64 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        # Higher warp counts for larger tiles
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8
        ),
    ],
    key=["hidden_size", "num_experts"],
)
@triton.jit
def _grouped_gemm_up_backward_dx_kernel(
    # Pointers
    grad_output_ptr,  # [total_tokens, max_expert_width] - gradient of activated output
    pre_act_ptr,  # [total_tokens, max_expert_width] - pre-activation values (z = x @ W1)
    w1_ptr,  # [hidden_size, total_expert_width]
    grad_x_ptr,  # [total_tokens, hidden_size] - output
    # Strides
    stride_grad_out_row,
    stride_grad_out_col,
    stride_pre_act_row,
    stride_pre_act_col,
    stride_w1_row,
    stride_w1_col,
    stride_grad_x_row,
    stride_grad_x_col,
    # Expert metadata
    expert_token_offsets_ptr,
    expert_weight_offsets_ptr,
    tokens_per_expert_ptr,
    expert_widths_ptr,
    # Dimensions
    hidden_size,
    num_experts,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Activation: 0=none, 1=relu_squared
    ACTIVATION: tl.constexpr,
):
    """Backward kernel for grad_x = grad_act @ W1.T

    For relu_squared: grad_act = grad_output * 2 * pre_act * (pre_act > 0)

    Tiles over (M=tokens, N=hidden_size), loops over K=expert_width.
    """
    pid = tl.program_id(0)

    # For this kernel, we tile over (tokens, hidden_size)
    # Use hidden_size as the "width" for tile computation
    hidden_sizes_ptr = (
        expert_widths_ptr  # Repurposed - caller passes hidden_size repeated
    )

    expert_idx, m_start, n_start = _compute_tile_info(
        pid, tokens_per_expert_ptr, hidden_sizes_ptr, num_experts, BLOCK_M, BLOCK_N
    )

    # Load expert metadata
    token_start = tl.load(expert_token_offsets_ptr + expert_idx)
    token_end = tl.load(expert_token_offsets_ptr + expert_idx + 1)
    weight_col_start = tl.load(expert_weight_offsets_ptr + expert_idx)
    weight_col_end = tl.load(expert_weight_offsets_ptr + expert_idx + 1)
    expert_width = weight_col_end - weight_col_start

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_indices = token_start + m_start + offs_m
    m_mask = m_indices < token_end
    n_mask = (n_start + offs_n) < hidden_size

    # Loop over K (expert_width)
    for k in range(0, expert_width, BLOCK_K):
        k_indices = k + offs_k
        k_mask = k_indices < expert_width

        # Load grad_output tile [BLOCK_M, BLOCK_K]
        grad_out_ptrs = (
            grad_output_ptr
            + m_indices[:, None] * stride_grad_out_row
            + k_indices[None, :] * stride_grad_out_col
        )
        grad_out_mask = m_mask[:, None] & k_mask[None, :]
        grad_out_tile = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        # Apply activation derivative
        if ACTIVATION == 1:  # relu_squared: d/dz[relu(z)^2] = 2*z if z > 0, else 0
            pre_act_ptrs = (
                pre_act_ptr
                + m_indices[:, None] * stride_pre_act_row
                + k_indices[None, :] * stride_pre_act_col
            )
            pre_act_tile = tl.load(pre_act_ptrs, mask=grad_out_mask, other=0.0)
            grad_act_tile = tl.where(
                pre_act_tile > 0, grad_out_tile * 2.0 * pre_act_tile, 0.0
            )
        else:
            grad_act_tile = grad_out_tile

        # Load W1.T tile: W1 is [hidden_size, total_expert_width], we need [expert_width, hidden_size]
        # W1.T[k, n] = W1[n, weight_col_start + k]
        w1_col_indices = weight_col_start + k_indices
        w1_ptrs = (
            w1_ptr
            + (n_start + offs_n)[:, None] * stride_w1_row
            + w1_col_indices[None, :] * stride_w1_col
        )
        w_mask = n_mask[:, None] & k_mask[None, :]
        w1_tile_t = tl.load(w1_ptrs, mask=w_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        # grad_x += grad_act @ W1.T
        acc += tl.dot(grad_act_tile, tl.trans(w1_tile_t))

    # Store grad_x
    grad_x_col_indices = n_start + offs_n
    grad_x_ptrs = (
        grad_x_ptr
        + m_indices[:, None] * stride_grad_x_row
        + grad_x_col_indices[None, :] * stride_grad_x_col
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(grad_x_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        # BLOCK_K=32 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        # BLOCK_K=64 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        # Higher warp counts for larger tiles
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8
        ),
    ],
    key=["hidden_size", "num_experts"],
)
@triton.jit
def _grouped_gemm_up_backward_dw_kernel(
    # Pointers
    x_ptr,  # [total_tokens, hidden_size]
    grad_output_ptr,  # [total_tokens, max_expert_width]
    pre_act_ptr,  # [total_tokens, max_expert_width]
    grad_w1_ptr,  # [hidden_size, total_expert_width] - output
    # Strides
    stride_x_row,
    stride_x_col,
    stride_grad_out_row,
    stride_grad_out_col,
    stride_pre_act_row,
    stride_pre_act_col,
    stride_grad_w1_row,
    stride_grad_w1_col,
    # Expert metadata
    expert_token_offsets_ptr,
    expert_weight_offsets_ptr,
    tokens_per_expert_ptr,
    expert_widths_ptr,
    hidden_sizes_ptr,  # [num_experts] - all same value (hidden_size), for tile scheduling
    # Dimensions
    hidden_size,
    num_experts,
    # Tile sizes
    BLOCK_M: tl.constexpr,  # tiles over hidden_size
    BLOCK_N: tl.constexpr,  # tiles over expert_width
    BLOCK_K: tl.constexpr,  # loops over tokens
    # Activation
    ACTIVATION: tl.constexpr,
):
    """Backward kernel for grad_W1 = x.T @ grad_act

    Tiles over (M=hidden_size, N=expert_width), loops over K=tokens.
    Each expert's weight gradient is independent.
    """
    pid = tl.program_id(0)

    # Tile over (hidden_size, expert_width) for each expert
    # M dimension = hidden_size (constant), N dimension = expert_width (varies)
    expert_idx, m_start, n_start = _compute_tile_info(
        pid, hidden_sizes_ptr, expert_widths_ptr, num_experts, BLOCK_M, BLOCK_N
    )

    # Actually for dW, we want to tile over (hidden_size, expert_width)
    # Let's compute expert_idx and local tile differently
    # For now, use a simpler grid: one tile per (hidden_size_tile, expert_width_tile, expert)

    # Load expert metadata
    token_start = tl.load(expert_token_offsets_ptr + expert_idx)
    token_end = tl.load(expert_token_offsets_ptr + expert_idx + 1)
    num_tokens = token_end - token_start
    weight_col_start = tl.load(expert_weight_offsets_ptr + expert_idx)
    expert_width = tl.load(expert_widths_ptr + expert_idx)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = tl.arange(0, BLOCK_M)  # hidden_size dimension
    offs_n = tl.arange(0, BLOCK_N)  # expert_width dimension
    offs_k = tl.arange(0, BLOCK_K)  # token dimension

    m_mask = (m_start + offs_m) < hidden_size
    n_mask = (n_start + offs_n) < expert_width

    # Loop over tokens
    for k in range(0, num_tokens, BLOCK_K):
        k_indices = token_start + k + offs_k
        k_mask = (k + offs_k) < num_tokens

        # Load x.T tile: x is [total_tokens, hidden_size], x.T is [hidden_size, total_tokens]
        # x.T[m, k] = x[k, m]
        x_ptrs = (
            x_ptr
            + k_indices[:, None] * stride_x_row
            + (m_start + offs_m)[None, :] * stride_x_col
        )
        x_mask = k_mask[:, None] & m_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_K, BLOCK_M]

        # Load grad_output tile [BLOCK_K, BLOCK_N]
        grad_col_indices = n_start + offs_n
        grad_out_ptrs = (
            grad_output_ptr
            + k_indices[:, None] * stride_grad_out_row
            + grad_col_indices[None, :] * stride_grad_out_col
        )
        grad_out_mask = k_mask[:, None] & n_mask[None, :]
        grad_out_tile = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        # Apply activation derivative
        if ACTIVATION == 1:  # relu_squared
            pre_act_ptrs = (
                pre_act_ptr
                + k_indices[:, None] * stride_pre_act_row
                + grad_col_indices[None, :] * stride_pre_act_col
            )
            pre_act_tile = tl.load(pre_act_ptrs, mask=grad_out_mask, other=0.0)
            grad_act_tile = tl.where(
                pre_act_tile > 0, grad_out_tile * 2.0 * pre_act_tile, 0.0
            )
        else:
            grad_act_tile = grad_out_tile

        # grad_W1 += x.T @ grad_act
        acc += tl.dot(tl.trans(x_tile), grad_act_tile)

    # Store grad_W1
    w1_col_indices = weight_col_start + n_start + offs_n
    grad_w1_ptrs = (
        grad_w1_ptr
        + (m_start + offs_m)[:, None] * stride_grad_w1_row
        + w1_col_indices[None, :] * stride_grad_w1_col
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(grad_w1_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        # BLOCK_K=32 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        # BLOCK_K=64 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        # Higher warp counts for larger tiles
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8
        ),
    ],
    key=["hidden_size", "num_experts"],
)
@triton.jit
def _grouped_gemm_down_backward_dintermediate_kernel(
    # Pointers
    grad_output_ptr,  # [total_tokens, hidden_size]
    w2_ptr,  # [total_expert_width, hidden_size]
    grad_int_ptr,  # [total_tokens, max_expert_width] - output
    # Strides
    stride_grad_out_row,
    stride_grad_out_col,
    stride_w2_row,
    stride_w2_col,
    stride_grad_int_row,
    stride_grad_int_col,
    # Expert metadata
    expert_token_offsets_ptr,
    expert_weight_offsets_ptr,
    tokens_per_expert_ptr,
    expert_widths_ptr,
    # Dimensions
    hidden_size,
    num_experts,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Backward kernel for grad_intermediate = grad_output @ W2.T

    Tiles over (M=tokens, N=expert_width), loops over K=hidden_size.
    """
    pid = tl.program_id(0)

    expert_idx, m_start, n_start = _compute_tile_info(
        pid, tokens_per_expert_ptr, expert_widths_ptr, num_experts, BLOCK_M, BLOCK_N
    )

    # Load expert metadata
    token_start = tl.load(expert_token_offsets_ptr + expert_idx)
    token_end = tl.load(expert_token_offsets_ptr + expert_idx + 1)
    weight_row_start = tl.load(expert_weight_offsets_ptr + expert_idx)
    expert_width = tl.load(expert_widths_ptr + expert_idx)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_indices = token_start + m_start + offs_m
    m_mask = m_indices < token_end
    n_mask = (n_start + offs_n) < expert_width

    # Loop over K (hidden_size)
    for k in range(0, hidden_size, BLOCK_K):
        k_indices = k + offs_k
        k_mask = k_indices < hidden_size

        # Load grad_output tile [BLOCK_M, BLOCK_K]
        grad_out_ptrs = (
            grad_output_ptr
            + m_indices[:, None] * stride_grad_out_row
            + k_indices[None, :] * stride_grad_out_col
        )
        grad_out_mask = m_mask[:, None] & k_mask[None, :]
        grad_out_tile = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        # Load W2.T tile: W2 is [total_expert_width, hidden_size], W2.T is [hidden_size, expert_width]
        # W2.T[k, n] = W2[weight_row_start + n, k]
        w2_row_indices = weight_row_start + n_start + offs_n
        w2_ptrs = (
            w2_ptr
            + w2_row_indices[:, None] * stride_w2_row
            + k_indices[None, :] * stride_w2_col
        )
        w_mask = n_mask[:, None] & k_mask[None, :]
        w2_tile = tl.load(w2_ptrs, mask=w_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        # grad_intermediate += grad_output @ W2.T
        acc += tl.dot(grad_out_tile, tl.trans(w2_tile))

    # Store grad_intermediate (into padded layout)
    grad_int_col_indices = n_start + offs_n
    grad_int_ptrs = (
        grad_int_ptr
        + m_indices[:, None] * stride_grad_int_row
        + grad_int_col_indices[None, :] * stride_grad_int_col
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(grad_int_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        # BLOCK_K=32 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        # BLOCK_K=64 variants
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        # Higher warp counts for larger tiles
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8
        ),
    ],
    key=["hidden_size", "num_experts"],
)
@triton.jit
def _grouped_gemm_down_backward_dw_kernel(
    # Pointers
    intermediate_ptr,  # [total_tokens, max_expert_width]
    grad_output_ptr,  # [total_tokens, hidden_size]
    grad_w2_ptr,  # [total_expert_width, hidden_size] - output
    # Strides
    stride_int_row,
    stride_int_col,
    stride_grad_out_row,
    stride_grad_out_col,
    stride_grad_w2_row,
    stride_grad_w2_col,
    # Expert metadata
    expert_token_offsets_ptr,
    expert_weight_offsets_ptr,
    tokens_per_expert_ptr,
    expert_widths_ptr,
    hidden_sizes_ptr,  # [num_experts] - all same value (hidden_size), for tile scheduling
    # Dimensions
    hidden_size,
    num_experts,
    # Tile sizes
    BLOCK_M: tl.constexpr,  # tiles over expert_width
    BLOCK_N: tl.constexpr,  # tiles over hidden_size
    BLOCK_K: tl.constexpr,  # loops over tokens
):
    """Backward kernel for grad_W2 = intermediate.T @ grad_output

    Tiles over (M=expert_width, N=hidden_size), loops over K=tokens.
    """
    pid = tl.program_id(0)

    # Tile over (expert_width, hidden_size) for each expert
    # M dimension = expert_width (varies), N dimension = hidden_size (constant)
    expert_idx, m_start, n_start = _compute_tile_info(
        pid, expert_widths_ptr, hidden_sizes_ptr, num_experts, BLOCK_M, BLOCK_N
    )

    # Load expert metadata
    token_start = tl.load(expert_token_offsets_ptr + expert_idx)
    token_end = tl.load(expert_token_offsets_ptr + expert_idx + 1)
    num_tokens = token_end - token_start
    weight_row_start = tl.load(expert_weight_offsets_ptr + expert_idx)
    expert_width = tl.load(expert_widths_ptr + expert_idx)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = tl.arange(0, BLOCK_M)  # expert_width dimension
    offs_n = tl.arange(0, BLOCK_N)  # hidden_size dimension
    offs_k = tl.arange(0, BLOCK_K)  # token dimension

    m_mask = (m_start + offs_m) < expert_width
    n_mask = (n_start + offs_n) < hidden_size

    # Loop over tokens
    for k in range(0, num_tokens, BLOCK_K):
        k_indices = token_start + k + offs_k
        k_mask = (k + offs_k) < num_tokens

        # Load intermediate.T tile: int is [total_tokens, max_expert_width]
        # int.T[m, k] = int[k, m]
        int_col_indices = m_start + offs_m
        int_ptrs = (
            intermediate_ptr
            + k_indices[:, None] * stride_int_row
            + int_col_indices[None, :] * stride_int_col
        )
        int_mask = k_mask[:, None] & m_mask[None, :]
        int_tile = tl.load(int_ptrs, mask=int_mask, other=0.0)  # [BLOCK_K, BLOCK_M]

        # Load grad_output tile [BLOCK_K, BLOCK_N]
        grad_out_ptrs = (
            grad_output_ptr
            + k_indices[:, None] * stride_grad_out_row
            + (n_start + offs_n)[None, :] * stride_grad_out_col
        )
        grad_out_mask = k_mask[:, None] & n_mask[None, :]
        grad_out_tile = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        # grad_W2 += intermediate.T @ grad_output
        acc += tl.dot(tl.trans(int_tile), grad_out_tile)

    # Store grad_W2
    w2_row_indices = weight_row_start + m_start + offs_m
    grad_w2_ptrs = (
        grad_w2_ptr
        + w2_row_indices[:, None] * stride_grad_w2_row
        + (n_start + offs_n)[None, :] * stride_grad_w2_col
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(grad_w2_ptrs, acc, mask=out_mask)


def grouped_gemm_up_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    pre_act: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    expert_weight_offsets: torch.Tensor,
    expert_widths: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    hidden_size: int,
    activation: str = "relu_squared",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for grouped_gemm_up.

    Args:
        grad_output: Gradient w.r.t. activated output [total_tokens, max_expert_width]
        x: Original input [total_tokens, hidden_size]
        w1: Up-projection weights [hidden_size, total_expert_width]
        pre_act: Pre-activation values (x @ W1) [total_tokens, max_expert_width]
        expert_token_offsets: Cumulative token counts [num_experts + 1]
        expert_weight_offsets: Cumulative weight offsets [num_experts + 1]
        expert_widths: Per-expert intermediate sizes [num_experts]
        tokens_per_expert: Token counts per expert [num_experts]
        hidden_size: Hidden dimension
        activation: Activation function used in forward

    Returns:
        grad_x: Gradient w.r.t. input [total_tokens, hidden_size]
        grad_w1: Gradient w.r.t. weights [hidden_size, total_expert_width]
    """
    total_tokens = x.shape[0]
    num_experts = len(expert_widths)
    device = x.device
    dtype = x.dtype

    activation_code = {"none": 0, "relu_squared": 1, "swiglu": 2}.get(activation, 0)

    # Allocate outputs
    grad_x = torch.zeros(total_tokens, hidden_size, device=device, dtype=dtype)
    grad_w1 = torch.zeros_like(w1)

    if total_tokens == 0:
        return grad_x, grad_w1

    # For grad_x kernel, we tile over (tokens, hidden_size)
    hidden_size_per_expert = torch.full_like(expert_widths, hidden_size)

    def grid_dx(meta):
        return (
            _compute_total_tiles(
                tokens_per_expert,
                hidden_size_per_expert,
                meta["BLOCK_M"],
                meta["BLOCK_N"],
            ),
        )

    _grouped_gemm_up_backward_dx_kernel[grid_dx](
        grad_output,
        pre_act,
        w1,
        grad_x,
        grad_output.stride(0),
        grad_output.stride(1),
        pre_act.stride(0),
        pre_act.stride(1),
        w1.stride(0),
        w1.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
        expert_token_offsets,
        expert_weight_offsets,
        tokens_per_expert,
        hidden_size_per_expert,
        hidden_size,
        num_experts,
        ACTIVATION=activation_code,
    )

    # For grad_w1 kernel, we tile over (hidden_size, expert_width) per expert
    # TODO: Implement proper grid for dW kernel
    def grid_dw(meta):
        # Simple grid: compute tiles needed for all experts
        total = 0
        for width in expert_widths.tolist():
            m_tiles = (hidden_size + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
            n_tiles = (width + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
            total += m_tiles * n_tiles
        return (total,)

    _grouped_gemm_up_backward_dw_kernel[grid_dw](
        x,
        grad_output,
        pre_act,
        grad_w1,
        x.stride(0),
        x.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        pre_act.stride(0),
        pre_act.stride(1),
        grad_w1.stride(0),
        grad_w1.stride(1),
        expert_token_offsets,
        expert_weight_offsets,
        tokens_per_expert,
        expert_widths,
        hidden_size_per_expert,  # For tile scheduling: M dimension = hidden_size
        hidden_size,
        num_experts,
        ACTIVATION=activation_code,
    )

    return grad_x, grad_w1


def grouped_gemm_down_backward(
    grad_output: torch.Tensor,
    intermediate: torch.Tensor,
    w2: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    expert_weight_offsets: torch.Tensor,
    expert_widths: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    hidden_size: int,
    max_expert_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for grouped_gemm_down.

    Args:
        grad_output: Gradient w.r.t. output [total_tokens, hidden_size]
        intermediate: Intermediate activations [total_tokens, max_expert_width]
        w2: Down-projection weights [total_expert_width, hidden_size]
        expert_token_offsets: Cumulative token counts [num_experts + 1]
        expert_weight_offsets: Cumulative weight offsets [num_experts + 1]
        expert_widths: Per-expert intermediate sizes [num_experts]
        tokens_per_expert: Token counts per expert [num_experts]
        hidden_size: Hidden dimension
        max_expert_width: Max expert width (for output allocation)

    Returns:
        grad_intermediate: Gradient w.r.t. intermediate [total_tokens, max_expert_width]
        grad_w2: Gradient w.r.t. weights [total_expert_width, hidden_size]
    """
    total_tokens = intermediate.shape[0]
    num_experts = len(expert_widths)
    device = intermediate.device
    dtype = intermediate.dtype

    # Allocate outputs
    grad_intermediate = torch.zeros(
        total_tokens, max_expert_width, device=device, dtype=dtype
    )
    grad_w2 = torch.zeros_like(w2)

    if total_tokens == 0:
        return grad_intermediate, grad_w2

    # For grad_intermediate kernel, tile over (tokens, expert_width)
    def grid_dint(meta):
        return (
            _compute_total_tiles(
                tokens_per_expert, expert_widths, meta["BLOCK_M"], meta["BLOCK_N"]
            ),
        )

    _grouped_gemm_down_backward_dintermediate_kernel[grid_dint](
        grad_output,
        w2,
        grad_intermediate,
        grad_output.stride(0),
        grad_output.stride(1),
        w2.stride(0),
        w2.stride(1),
        grad_intermediate.stride(0),
        grad_intermediate.stride(1),
        expert_token_offsets,
        expert_weight_offsets,
        tokens_per_expert,
        expert_widths,
        hidden_size,
        num_experts,
    )

    # For grad_w2 kernel, tile over (expert_width, hidden_size) per expert
    hidden_size_per_expert = torch.full_like(expert_widths, hidden_size)

    def grid_dw(meta):
        total = 0
        for width in expert_widths.tolist():
            m_tiles = (width + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
            n_tiles = (hidden_size + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
            total += m_tiles * n_tiles
        return (total,)

    _grouped_gemm_down_backward_dw_kernel[grid_dw](
        intermediate,
        grad_output,
        grad_w2,
        intermediate.stride(0),
        intermediate.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_w2.stride(0),
        grad_w2.stride(1),
        expert_token_offsets,
        expert_weight_offsets,
        tokens_per_expert,
        expert_widths,
        hidden_size_per_expert,  # For tile scheduling: N dimension = hidden_size
        hidden_size,
        num_experts,
    )

    return grad_intermediate, grad_w2


class GroupedGemmUp(torch.autograd.Function):
    """Autograd function for grouped GEMM up-projection with fused activation."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        expert_token_offsets: torch.Tensor,
        expert_weight_offsets: torch.Tensor,
        expert_widths: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        max_expert_width: int,
        hidden_size: int,
        activation: str = "relu_squared",
    ) -> torch.Tensor:
        """Forward pass - calls the Triton kernel and saves tensors for backward."""
        # Ensure x and w1 have same dtype for Triton kernel
        x = x.to(w1.dtype)

        # We need to save pre-activation values for backward
        # First compute x @ W1 without activation
        pre_act = grouped_gemm_up(
            x,
            w1,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
            max_expert_width,
            activation="none",
        )

        # Apply activation
        if activation == "relu_squared":
            output = torch.where(
                pre_act > 0, pre_act * pre_act, torch.zeros_like(pre_act)
            )
        else:
            output = pre_act

        # Save for backward
        ctx.save_for_backward(
            x,
            w1,
            pre_act,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
        )
        ctx.max_expert_width = max_expert_width
        ctx.hidden_size = hidden_size
        ctx.activation = activation

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass - computes gradients w.r.t. x and w1."""
        (
            x,
            w1,
            pre_act,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
        ) = ctx.saved_tensors

        grad_x, grad_w1 = grouped_gemm_up_backward(
            grad_output,
            x,
            w1,
            pre_act,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
            ctx.hidden_size,
            ctx.activation,
        )

        # Return gradients in same order as forward inputs
        # (x, w1, expert_token_offsets, expert_weight_offsets, expert_widths,
        #  tokens_per_expert, max_expert_width, hidden_size, activation)
        return grad_x, grad_w1, None, None, None, None, None, None, None


class GroupedGemmDown(torch.autograd.Function):
    """Autograd function for grouped GEMM down-projection."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        intermediate: torch.Tensor,
        w2: torch.Tensor,
        expert_token_offsets: torch.Tensor,
        expert_weight_offsets: torch.Tensor,
        expert_widths: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        hidden_size: int,
        max_expert_width: int,
    ) -> torch.Tensor:
        """Forward pass - calls the Triton kernel and saves tensors for backward."""
        # Ensure intermediate and w2 have same dtype for Triton kernel
        intermediate = intermediate.to(w2.dtype)

        output = grouped_gemm_down(
            intermediate,
            w2,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
            hidden_size,
        )

        # Save for backward
        ctx.save_for_backward(
            intermediate,
            w2,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
        )
        ctx.hidden_size = hidden_size
        ctx.max_expert_width = max_expert_width

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass - computes gradients w.r.t. intermediate and w2."""
        (
            intermediate,
            w2,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
        ) = ctx.saved_tensors

        grad_intermediate, grad_w2 = grouped_gemm_down_backward(
            grad_output,
            intermediate,
            w2,
            expert_token_offsets,
            expert_weight_offsets,
            expert_widths,
            tokens_per_expert,
            ctx.hidden_size,
            ctx.max_expert_width,
        )

        # Return gradients in same order as forward inputs
        # (intermediate, w2, expert_token_offsets, expert_weight_offsets,
        #  expert_widths, tokens_per_expert, hidden_size, max_expert_width)
        return grad_intermediate, grad_w2, None, None, None, None, None, None


def grouped_gemm_up_autograd(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    expert_weight_offsets: torch.Tensor,
    expert_widths: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    max_expert_width: int,
    hidden_size: int,
    activation: str = "relu_squared",
) -> torch.Tensor:
    """Grouped GEMM up-projection with autograd support.

    Use this instead of grouped_gemm_up() when you need gradients.
    """
    return GroupedGemmUp.apply(
        x,
        w1,
        expert_token_offsets,
        expert_weight_offsets,
        expert_widths,
        tokens_per_expert,
        max_expert_width,
        hidden_size,
        activation,
    )


def grouped_gemm_down_autograd(
    intermediate: torch.Tensor,
    w2: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    expert_weight_offsets: torch.Tensor,
    expert_widths: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    hidden_size: int,
    max_expert_width: int,
) -> torch.Tensor:
    """Grouped GEMM down-projection with autograd support.

    Use this instead of grouped_gemm_down() when you need gradients.
    """
    return GroupedGemmDown.apply(
        intermediate,
        w2,
        expert_token_offsets,
        expert_weight_offsets,
        expert_widths,
        tokens_per_expert,
        hidden_size,
        max_expert_width,
    )
