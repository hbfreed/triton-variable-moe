#!/usr/bin/env python3
"""Benchmark grouped GEMM forward pass: Triton vs stk.

Compares our Triton grouped_gemm_up + grouped_gemm_down kernels against
stk's topology creation + sdd + dsd operations.

Usage:
    uv run benchmarks/bench_gemm_forward.py
    uv run benchmarks/bench_gemm_forward.py --warmup 50 --iterations 100
    CUDA_VISIBLE_DEVICES=1 uv run benchmarks/bench_gemm_forward.py
"""

import argparse
import time
from dataclasses import dataclass

import torch

from reference import MoEConfig, MoEMLP
from triton_moe.kernels import grouped_gemm_up, grouped_gemm_down


@dataclass
class BenchConfig:
    """Configuration for a benchmark run."""
    name: str
    n_embd: int
    expert_sizes: list[tuple[int, int]]
    num_active_experts: int
    batch_size: int = 8
    seq_len: int = 512
    block_size: int = 128


def get_kernel_inputs(moe: MoEMLP, ref: dict) -> dict:
    """Extract inputs for Triton kernels from reference MoE and forward results."""
    device = ref["x_gathered"].device
    padded_bins = ref["padded_bins"]

    expert_token_offsets = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        padded_bins.to(torch.int32)
    ])
    expert_weight_offsets = torch.tensor(moe.expert_offsets, dtype=torch.int32, device=device)
    expert_widths = torch.tensor(moe.expert_widths, dtype=torch.int32, device=device)

    padded_tokens_per_expert = padded_bins.clone().to(torch.int32)
    padded_tokens_per_expert[1:] -= padded_bins[:-1].to(torch.int32)

    return {
        "expert_token_offsets": expert_token_offsets,
        "expert_weight_offsets": expert_weight_offsets,
        "expert_widths": expert_widths,
        "padded_tokens_per_expert": padded_tokens_per_expert,
        "max_expert_width": max(moe.expert_widths),
        "hidden_size": moe.config.n_embd,
    }


def benchmark_stk_gemms(
    moe: MoEMLP,
    x_gathered: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
) -> tuple[float, float]:
    """Benchmark stk topology creation + sdd + dsd.

    Returns:
        (topology_time_ms, gemm_time_ms) averaged over iterations
    """
    import stk.ops
    from megablocks import ops
    from megablocks.layers.relu_squared import relu_squared

    # Warmup
    for _ in range(num_warmup):
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, moe.block_size)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0).contiguous()
        padded_tokens = padded_bins[-1].clamp_min(moe.block_size)
        block_rows = padded_tokens // moe.block_size

        from reference.topology_var import topology_var
        column_indices = topology_var(
            padded_bins,
            moe.expert_size_blocks,
            moe.expert_block_offsets,
            moe.block_size,
            block_rows,
        )
        expert_token_blocks = padded_tokens_per_expert // moe.block_size
        repeated_sizes = torch.repeat_interleave(moe.expert_size_blocks, expert_token_blocks)
        offsets = torch.cat([repeated_sizes.new_zeros(1), repeated_sizes.cumsum(0)])
        column_indices = column_indices.to(torch.int32)
        offsets = offsets.to(torch.int32)

        shape = (padded_tokens, moe.total_expert_width)
        num_blocks = column_indices.numel()
        data_placeholder = torch.empty(
            num_blocks, moe.block_size, moe.block_size,
            dtype=x_gathered.dtype, device="meta"
        )
        row_indices = stk.ops.row_indices(shape, data_placeholder, offsets, column_indices).to(torch.int32)

        block_columns = moe.total_expert_width // moe.block_size
        _, gather_indices = ops.sort(column_indices.int(), moe.transpose_sort_end_bit)
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()
        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])

        topology = stk.Matrix(
            shape, data_placeholder, row_indices, column_indices, offsets,
            column_indices_t.to(torch.int32), offsets_t.to(torch.int32), block_offsets_t.to(torch.int32)
        )

        x_up = stk.ops.sdd(x_gathered, moe.w1, topology)
        x_act = relu_squared(x_up)
        _ = stk.ops.dsd(x_act, moe.w2)

    torch.cuda.synchronize()

    # Benchmark
    topo_times = []
    gemm_times = []

    for _ in range(num_iterations):
        # Time topology creation
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        padded_tokens_per_expert = ops.round_up(tokens_per_expert, moe.block_size)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0).contiguous()
        padded_tokens = padded_bins[-1].clamp_min(moe.block_size)
        block_rows = padded_tokens // moe.block_size

        from reference.topology_var import topology_var
        column_indices = topology_var(
            padded_bins,
            moe.expert_size_blocks,
            moe.expert_block_offsets,
            moe.block_size,
            block_rows,
        )
        expert_token_blocks = padded_tokens_per_expert // moe.block_size
        repeated_sizes = torch.repeat_interleave(moe.expert_size_blocks, expert_token_blocks)
        offsets = torch.cat([repeated_sizes.new_zeros(1), repeated_sizes.cumsum(0)])
        column_indices = column_indices.to(torch.int32)
        offsets = offsets.to(torch.int32)

        shape = (padded_tokens, moe.total_expert_width)
        num_blocks = column_indices.numel()
        data_placeholder = torch.empty(
            num_blocks, moe.block_size, moe.block_size,
            dtype=x_gathered.dtype, device="meta"
        )
        row_indices = stk.ops.row_indices(shape, data_placeholder, offsets, column_indices).to(torch.int32)

        block_columns = moe.total_expert_width // moe.block_size
        _, gather_indices = ops.sort(column_indices.int(), moe.transpose_sort_end_bit)
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()
        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])

        topology = stk.Matrix(
            shape, data_placeholder, row_indices, column_indices, offsets,
            column_indices_t.to(torch.int32), offsets_t.to(torch.int32), block_offsets_t.to(torch.int32)
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Time GEMMs
        x_up = stk.ops.sdd(x_gathered, moe.w1, topology)
        x_act = relu_squared(x_up)
        _ = stk.ops.dsd(x_act, moe.w2)

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        topo_times.append((t1 - t0) * 1000)
        gemm_times.append((t2 - t1) * 1000)

    return sum(topo_times) / len(topo_times), sum(gemm_times) / len(gemm_times)


def benchmark_triton_gemms(
    moe: MoEMLP,
    x_gathered: torch.Tensor,
    kernel_inputs: dict,
    num_warmup: int,
    num_iterations: int,
) -> float:
    """Benchmark Triton grouped_gemm_up + grouped_gemm_down.

    Returns:
        Average time in ms
    """
    # Warmup (important for autotune)
    for _ in range(num_warmup):
        triton_up = grouped_gemm_up(
            x_gathered,
            moe.w1,
            kernel_inputs["expert_token_offsets"],
            kernel_inputs["expert_weight_offsets"],
            kernel_inputs["expert_widths"],
            kernel_inputs["padded_tokens_per_expert"],
            kernel_inputs["max_expert_width"],
            activation="relu_squared",
        )
        _ = grouped_gemm_down(
            triton_up,
            moe.w2,
            kernel_inputs["expert_token_offsets"],
            kernel_inputs["expert_weight_offsets"],
            kernel_inputs["expert_widths"],
            kernel_inputs["padded_tokens_per_expert"],
            kernel_inputs["hidden_size"],
        )

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        triton_up = grouped_gemm_up(
            x_gathered,
            moe.w1,
            kernel_inputs["expert_token_offsets"],
            kernel_inputs["expert_weight_offsets"],
            kernel_inputs["expert_widths"],
            kernel_inputs["padded_tokens_per_expert"],
            kernel_inputs["max_expert_width"],
            activation="relu_squared",
        )
        _ = grouped_gemm_down(
            triton_up,
            moe.w2,
            kernel_inputs["expert_token_offsets"],
            kernel_inputs["expert_weight_offsets"],
            kernel_inputs["expert_widths"],
            kernel_inputs["padded_tokens_per_expert"],
            kernel_inputs["hidden_size"],
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return sum(times) / len(times)


def run_benchmark(config: BenchConfig, num_warmup: int, num_iterations: int) -> dict:
    """Run benchmark for a single configuration."""
    device = torch.device("cuda")

    # Create model
    moe_config = MoEConfig(
        n_embd=config.n_embd,
        expert_sizes=config.expert_sizes,
        num_active_experts=config.num_active_experts,
        block_size=config.block_size,
    )
    moe = MoEMLP(moe_config).to(device).to(torch.bfloat16)

    # Create input
    torch.manual_seed(42)
    x = torch.randn(config.batch_size, config.seq_len, config.n_embd, device=device, dtype=torch.bfloat16)

    # Run reference forward to get intermediates (suppressing print statements)
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        ref = moe.forward_with_intermediates(x)
    finally:
        sys.stdout = old_stdout

    # Extract kernel inputs
    kernel_inputs = get_kernel_inputs(moe, ref)

    # Benchmark stk
    topo_ms, stk_gemm_ms = benchmark_stk_gemms(
        moe, ref["x_gathered"], ref["tokens_per_expert"],
        num_warmup, num_iterations
    )

    # Benchmark Triton
    triton_ms = benchmark_triton_gemms(
        moe, ref["x_gathered"], kernel_inputs,
        num_warmup, num_iterations
    )

    stk_total_ms = topo_ms + stk_gemm_ms
    speedup = stk_total_ms / triton_ms if triton_ms > 0 else 0

    return {
        "name": config.name,
        "topo_ms": topo_ms,
        "stk_gemm_ms": stk_gemm_ms,
        "stk_total_ms": stk_total_ms,
        "triton_ms": triton_ms,
        "speedup": speedup,
    }


def print_results(results: list[dict]):
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print("Grouped GEMM Forward Benchmark: Triton vs stk")
    print("=" * 90)
    print(f"{'Config':<20} {'Topology':<12} {'stk GEMM':<12} {'stk Total':<12} {'Triton':<12} {'Speedup':<10}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['topo_ms']:>8.3f} ms  "
            f"{r['stk_gemm_ms']:>8.3f} ms  "
            f"{r['stk_total_ms']:>8.3f} ms  "
            f"{r['triton_ms']:>8.3f} ms  "
            f"{r['speedup']:>6.2f}x"
        )

    print("=" * 90)
    print("\nNotes:")
    print("  - Topology: stk sparse matrix construction (column indices, row indices, transpose)")
    print("  - stk GEMM: sdd (up-projection) + activation + dsd (down-projection)")
    print("  - Triton: grouped_gemm_up (fused activation) + grouped_gemm_down")
    print("  - Speedup = stk Total / Triton")


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped GEMM forward pass")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument(
        "--config",
        type=str,
        choices=["8_uniform", "8_variable", "64_uniform", "64_variable", "128_uniform", "128_variable", "all"],
        default="all",
        help="Which configuration to benchmark (default: all)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"Input: batch={args.batch_size}, seq_len={args.seq_len}, n_embd={args.n_embd}")

    # Define configurations
    configs = [
        # 8 experts
        BenchConfig(
            name="8_uniform",
            n_embd=args.n_embd,
            expert_sizes=[(8, 512)],
            num_active_experts=2,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        ),
        BenchConfig(
            name="8_variable",
            n_embd=args.n_embd,
            expert_sizes=[(4, 512), (4, 256)],
            num_active_experts=2,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        ),
        # 64 experts
        BenchConfig(
            name="64_uniform",
            n_embd=args.n_embd,
            expert_sizes=[(64, 256)],
            num_active_experts=8,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        ),
        BenchConfig(
            name="64_variable",
            n_embd=args.n_embd,
            expert_sizes=[(16, 512), (32, 256), (16, 128)],
            num_active_experts=8,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        ),
        # 128 experts
        BenchConfig(
            name="128_uniform",
            n_embd=args.n_embd,
            expert_sizes=[(128, 256)],
            num_active_experts=8,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        ),
        BenchConfig(
            name="128_variable",
            n_embd=args.n_embd,
            expert_sizes=[(32, 512), (64, 256), (32, 128)],
            num_active_experts=8,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        ),
    ]

    # Filter configs if specific one requested
    if args.config != "all":
        configs = [c for c in configs if c.name == args.config]

    results = []
    for config in configs:
        print(f"  Benchmarking {config.name}...")
        result = run_benchmark(config, args.warmup, args.iterations)
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
