# triton-variable-moe

Triton-based MoE (Mixture of Experts) kernels with support for variable-sized experts. **No megablocks or stk dependency required.**

## Performance

**2-2.5x faster** than megablocks/stk by eliminating sparse matrix topology construction.

| Config | Reference (stk) | Triton | Speedup |
|--------|-----------------|--------|---------|
| 8 experts (uniform) | 2.16 ms | 0.86 ms | **2.5x** |
| 8 experts (variable) | 2.36 ms | 0.92 ms | **2.6x** |
| 64 experts, k=8 | 2.29 ms | 1.08 ms | **2.1x** |
| 128 experts, k=8 | 2.23 ms | 1.16 ms | **1.9x** |

*RTX 3090, batch=8, seq=512, hidden=256*

## Installation

```bash
# Clone and install
cd triton-variable-moe
uv sync

# Run tests
uv run pytest tests/ -v

# Run benchmarks (requires dev deps: megablocks, stk)
uv sync --group dev
uv run python benchmarks/bench_moe.py --profile
```

## Dependencies

**Core (for using triton_moe):**
- `torch`
- `triton`
- `einops`

**Dev only (for benchmarks/tests vs reference):**
- `megablocks`
- `stanford-stk`

## Usage

```python
from triton_moe import TritonMoEMLP, TritonMoEConfig

config = TritonMoEConfig(
    n_embd=768,
    expert_sizes=[(8, 512)],  # 8 experts, 512 intermediate dim each
    num_active_experts=2,     # top-k routing
)

moe = TritonMoEMLP(config).cuda().bfloat16()
output, aux_loss, _ = moe(x)  # x: [batch, seq, hidden]
```

### Variable-sized experts

```python
config = TritonMoEConfig(
    n_embd=768,
    expert_sizes=[
        (4, 512),   # 4 large experts
        (8, 256),   # 8 medium experts
        (4, 128),   # 4 small experts
    ],
    num_active_experts=2,
)
```

## Project Structure

```
triton_moe/
├── kernels/
│   ├── grouped_gemm.py   # Up/down projection with fused activation
│   ├── gather.py         # Token gathering with padding
│   ├── scatter.py        # Token scattering with weights
│   └── fused_cumsum.py   # Fused routing (sort + cumsum)
└── moe.py                # TritonMoEMLP layer

reference/                # Reference impl (megablocks/stk) for testing
tests/                    # Test suite
benchmarks/               # Performance benchmarks
```

## How It Works

The key insight is eliminating sparse matrix topology construction. The reference implementation (megablocks/stk) spends ~40% of forward pass time building sparse matrix indices:

```python
# Reference: build sparse topology every forward pass
column_indices = topology_var(...)     # ~20% of time
row_indices = ...                       # ~8% of time
output = stk.ops.sdd(x, W, topology)

# Triton: direct indexing with precomputed offsets
output = grouped_gemm(x, W, expert_offsets, expert_widths)
```

Our kernels use the "grouped dense GEMM" pattern (SonicMoE-style) where each expert's computation is a dense GEMM, and we tile across experts dynamically.

## Backward Pass

Full backward pass support for training:

```python
from triton_moe.kernels import grouped_gemm_up_autograd, grouped_gemm_down_autograd

# Gradients computed automatically
output = moe(x)
loss = output.sum()
loss.backward()  # works!
```

## License

MIT
