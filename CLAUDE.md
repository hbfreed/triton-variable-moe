# CLAUDE.md - Project Guide for AI Assistants

This file provides context for AI assistants working on this codebase.

## Project Overview

Triton-based MoE (Mixture of Experts) kernels with variable-sized expert support. The goal is to replace `stk.ops.sdd/dsd` and megablocks ops with custom Triton kernels that eliminate topology overhead.

**Key insight:** The reference implementation spends ~40-60% of forward pass time building sparse matrix topology. Our Triton kernels use direct indexing with precomputed offsets, eliminating this overhead entirely.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_grouped_gemm.py -v

# Run full MoE benchmarks (use CUDA_VISIBLE_DEVICES to select GPU)
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/bench_moe.py

# Run grouped GEMM forward benchmark (Triton vs stk)
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/bench_gemm_forward.py
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/bench_gemm_forward.py --config 8_uniform
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/bench_gemm_forward.py --warmup 100 --iterations 200
```

## Project Structure

- `reference/` - Reference implementation from nanoMoEchat (uses stk/megablocks)
- `triton_moe/kernels/` - Triton kernel implementations
  - `grouped_gemm.py` - Core GEMM kernels with dynamic tile scheduling + autotune
  - `gather.py`, `scatter.py` - Token routing kernels (not yet implemented)
  - `activation.py` - Fused activations (not yet implemented)
- `tests/` - Test suite comparing Triton vs reference
- `benchmarks/` - Performance comparison scripts

## Key Technical Concepts

### Memory Layout

**Reference (padded layout):** Tokens are gathered into blocks, padded to `block_size` alignment per expert.
- `padded_bins`: Cumulative sum of padded token counts
- `x_gathered`: Shape `[total_padded_tokens, hidden_size]`

**Our kernels expect:**
- `expert_token_offsets`: `[0, padded_bins[0], padded_bins[1], ...]` - where each expert's tokens start
- `expert_weight_offsets`: Cumsum of expert widths - where each expert's weights start in packed W matrix
- `expert_widths`: Width (intermediate dim) of each expert
- `padded_tokens_per_expert`: Number of padded tokens per expert

### Dynamic Tile Scheduling

The grouped GEMM kernels use on-the-fly tile computation from program ID, enabling Triton autotune to vary block sizes freely:

```python
@triton.jit
def _compute_tile_info(pid, tokens_per_expert_ptr, expert_widths_ptr, num_experts, BLOCK_M, BLOCK_N):
    """Compute (expert_idx, m_start, n_start) from program ID."""
    # Iterate through experts to find which one owns this tile
    # Returns the expert index and local tile offsets
```

### Weight Layout

Weights are packed contiguously: `W = [W_expert0, W_expert1, ...]`
- Up-projection W1: Shape `[hidden_size, total_expert_width]`
- Down-projection W2: Shape `[total_expert_width, hidden_size]`

### Activation

Uses ReLU-squared: `relu(x)^2` fused into the up-projection kernel.

## Testing Strategy

Tests compare full pipeline output (up+down projection) against reference `x_after_down`, since reference intermediates are in sparse `stk.Matrix` format:

```python
def test_grouped_gemm_up_matches_reference(reference_moe_small, test_input_small):
    ref = reference_moe_small.forward_with_intermediates(test_input_small)
    inputs = _get_kernel_inputs(reference_moe_small, ref)

    triton_up = grouped_gemm_up(...)
    triton_down = grouped_gemm_down(triton_up, ...)

    torch.testing.assert_close(triton_down, ref["x_after_down"], rtol=1e-2, atol=1e-2)
```

Tolerances are relaxed (1e-2) due to bfloat16 precision and different computation order.

## Current Implementation Status

### Implemented
- `grouped_gemm_up`: Up-projection with fused ReLU-squared, autotune, dynamic tile scheduling
- `grouped_gemm_down`: Down-projection, autotune, dynamic tile scheduling

### Not Yet Implemented
- `gather`: Token gathering with permutation
- `scatter`: Token scattering with weighted combine
- Backward passes
- Full MoE layer integration (`triton_moe/moe.py`)

## Performance Results

Benchmarks show 2-23x speedup over stk by eliminating topology overhead:

| Config | stk Total | Triton | Speedup |
|--------|-----------|--------|---------|
| 8_uniform | 1.42ms | 0.27ms | 5.3x |
| 8_variable | 6.16ms | 0.27ms | 22.8x |
| 64_uniform | 3.04ms | 0.77ms | 3.9x |
| 128_uniform | 1.79ms | 0.87ms | 2.1x |

## Code Conventions

- Use bfloat16 for activations, float32 accumulation in kernels
- All tensor offsets are int32
- Kernels use `@triton.autotune` with multiple block size configs
- Test fixtures are in `tests/conftest.py`
- Use `torch.manual_seed(42)` for reproducible model weights, `123` for inputs
