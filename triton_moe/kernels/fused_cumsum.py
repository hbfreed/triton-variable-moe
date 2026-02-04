"""Fused cumsum kernel for MoE routing.

Fuses the computation of `bins = cumsum(kept)` and `padded_bins = cumsum(rounded)`
into a single kernel with support for SonicMoE-style token rounding.

Supports three rounding modes:
    - "up" (dropless): Always round up to tile boundary. No tokens dropped, some wasted compute.
    - "down": Round down to tile boundary. Drops tokens, best tile utilization.
    - "nearest": Round to nearest tile boundary. Balance of both.

Sorting and histogram are handled by torch.sort and torch.bincount respectively,
which are well-optimized for these operations.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _cumsum_kernel(
    # Inputs
    tokens_per_expert_ptr,  # [num_experts] - histogram from torch.bincount
    # Outputs
    bins_ptr,  # [num_experts] - cumsum of kept tokens (inclusive)
    padded_bins_ptr,  # [num_experts] - cumsum of rounded tokens (inclusive)
    # Sizes
    num_experts,
    block_size,  # for padding/rounding calculation
    # Constants
    MAX_EXPERTS: tl.constexpr,
    ROUNDING: tl.constexpr,  # 0=up (dropless), 1=down, 2=nearest
):
    """
    Compute cumulative sums for bin boundaries with token rounding.

    Single block kernel - launches with grid=(1,).
    For num_experts <= 256, this all fits in registers.

    Rounding modes:
        ROUNDING=0 (up/dropless): round_up(tokens, block_size) - no tokens dropped
        ROUNDING=1 (down): round_down(tokens, block_size) - drops tokens for better utilization
        ROUNDING=2 (nearest): round_nearest(tokens, block_size) - balance

    Outputs:
        bins[i] = sum(kept_tokens[0:i+1])  -- inclusive cumsum of kept
        padded_bins[i] = sum(rounded_tokens[0:i+1])  -- inclusive cumsum of rounded
    """
    # Only one block runs this
    pid = tl.program_id(0)
    if pid > 0:
        return

    # Load histogram (all experts fit in registers for MAX_EXPERTS <= 256)
    expert_offs = tl.arange(0, MAX_EXPERTS)
    mask = expert_offs < num_experts

    tokens = tl.load(tokens_per_expert_ptr + expert_offs, mask=mask, other=0)

    # Apply rounding based on ROUNDING mode
    if ROUNDING == 0:  # up (dropless)
        rounded = ((tokens + block_size - 1) // block_size) * block_size
    elif ROUNDING == 1:  # down
        rounded = (tokens // block_size) * block_size
    else:  # nearest
        rounded = ((tokens + block_size // 2) // block_size) * block_size

    # Compute kept tokens (min of actual and rounded)
    # For "up": kept = tokens (all kept, rounded >= tokens)
    # For "down"/"nearest": kept = min(tokens, rounded)
    kept = tl.minimum(tokens, rounded)

    # Compute cumulative sums
    # bins = where each expert's kept tokens END in sorted array
    # padded_bins = where each expert's padded region ENDS in gathered buffer
    bins = tl.cumsum(kept, axis=0)
    padded_bins = tl.cumsum(rounded, axis=0)

    # Store results
    tl.store(bins_ptr + expert_offs, bins, mask=mask)
    tl.store(padded_bins_ptr + expert_offs, padded_bins, mask=mask)


@torch.compiler.disable
def fused_route(
    selected_experts: torch.Tensor,  # [num_assignments] int32/int64
    num_experts: int,
    block_size: int = 128,
    rounding: str = "up",  # "up" (dropless), "down", "nearest"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused routing: sort + bincount + fused cumsum kernel.

    Replaces:
        bin_ids, indices = torch.sort(selected_experts)
        tokens_per_expert = torch.bincount(selected_experts, minlength=num_experts)
        bins = torch.cumsum(tokens_per_expert, dim=0)
        padded_tokens = round_up(tokens_per_expert, block_size)
        padded_bins = torch.cumsum(padded_tokens, dim=0)

    With fused cumsum that computes bins + padded_bins + rounding in one kernel.

    Args:
        selected_experts: Expert assignments [num_tokens * top_k], int32
        num_experts: Number of experts
        block_size: Block size for padding/rounding
        rounding: Token rounding mode (SonicMoE-style):
            - "up" (dropless): Round up to tile boundary. No tokens dropped.
            - "down": Round down. Drops tokens for better tile utilization.
            - "nearest": Round to nearest. Balance of both.

    Returns:
        sorted_experts: Sorted expert IDs [num_kept] (bin_ids)
        indices: Permutation indices [num_kept]
        tokens_per_expert: Original histogram [num_experts] (before rounding)
        bins: Inclusive cumsum of kept tokens [num_experts]
        padded_bins: Inclusive cumsum of rounded tokens [num_experts]
    """
    assert selected_experts.is_cuda, "Input must be on CUDA"
    assert rounding in ("up", "down", "nearest"), f"Invalid rounding mode: {rounding}"
    selected_experts = selected_experts.to(torch.int32).contiguous()

    num_assignments = selected_experts.numel()
    device = selected_experts.device

    # Map rounding mode to integer for kernel
    ROUNDING_MAP = {"up": 0, "down": 1, "nearest": 2}
    rounding_mode = ROUNDING_MAP[rounding]

    sorted_experts, indices = torch.sort(selected_experts, stable=True)
    sorted_experts = sorted_experts.to(torch.int32)
    indices = indices.to(torch.int32)

    tokens_per_expert = torch.bincount(selected_experts, minlength=num_experts).to(
        torch.int32
    )

    # Computes bins (kept), padded_bins (rounded), and handles rounding in one kernel
    bins = torch.empty(num_experts, dtype=torch.int32, device=device)
    padded_bins = torch.empty(num_experts, dtype=torch.int32, device=device)

    MAX_EXPERTS = triton.next_power_of_2(num_experts)

    _cumsum_kernel[(1,)](
        tokens_per_expert,
        bins,
        padded_bins,
        num_experts,
        block_size,
        MAX_EXPERTS=MAX_EXPERTS,
        ROUNDING=rounding_mode,
    )

    if rounding != "up":
        # Build mask of which sorted positions to keep
        keep_mask = torch.zeros(num_assignments, dtype=torch.bool, device=device)

        # Compute kept per expert from bins (bins is cumsum of kept)
        kept_per_expert = torch.cat([bins[:1], bins[1:] - bins[:-1]])

        start = 0
        for e in range(num_experts):
            count = int(tokens_per_expert[e].item())
            keep = int(kept_per_expert[e].item())
            keep_mask[start : start + keep] = True
            start += count

        sorted_experts = sorted_experts[keep_mask]
        indices = indices[keep_mask]

    return sorted_experts, indices, tokens_per_expert, bins, padded_bins
