#!/usr/bin/env python3
"""Benchmark full MoE forward pass: Reference vs Triton.

Compares the complete MoE forward pass:
- Reference: router → sort → padded_bins → **topology** → gather → sdd → activation → dsd → scatter
- Triton: router → sort → padded_bins → gather → **grouped_gemm_up** → **grouped_gemm_down** → scatter

The key insight: the only code difference is that Triton skips topology construction
and uses direct indexed GEMMs instead of stk sparse matrix operations.

Usage:
    uv run benchmarks/bench_forward.py
    uv run benchmarks/bench_forward.py --config 8_uniform
    uv run benchmarks/bench_forward.py --warmup 100 --iterations 200
    CUDA_VISIBLE_DEVICES=1 uv run benchmarks/bench_forward.py
"""

import argparse
import io
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from megablocks import ops

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


def triton_forward(moe: MoEMLP, x: torch.Tensor) -> torch.Tensor:
    """MoE forward using Triton grouped GEMMs instead of stk.

    This implements the same computation as MoEMLP.forward() but:
    1. Skips topology construction entirely
    2. Uses grouped_gemm_up/down instead of stk.ops.sdd/dsd

    All other operations (routing, gather, scatter) are identical.

    Args:
        moe: Reference MoEMLP module (we reuse its weights and routing)
        x: Input tensor [batch_size, seq_len, n_embd]

    Returns:
        Output tensor [batch_size, seq_len, n_embd]
    """
    batch_size, seq_len, n_embd = x.shape
    device = x.device

    # 1. Routing (identical to reference)
    x_flat = rearrange(x, "b s d -> (b s) d")
    router_logits = moe.router(x_flat)
    router_probs = F.sigmoid(router_logits.float())

    top_k_weights, selected_experts = torch.topk(
        router_probs, moe.num_active_experts, dim=-1
    )
    top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
    top_k_weights = top_k_weights.to(x.dtype)

    top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
    selected_experts_flat = rearrange(selected_experts, "... -> (...)")

    # 2. Sort tokens by expert (identical to reference)
    bin_ids, indices = ops.sort(selected_experts_flat, moe.sort_end_bit)
    tokens_per_expert = ops.histogram(selected_experts_flat, moe.num_experts)
    bins = ops.inclusive_cumsum(tokens_per_expert, 0).contiguous()

    # 3. Compute padded bins (needed for gather/scatter)
    padded_tokens_per_expert = ops.round_up(tokens_per_expert, moe.block_size)
    padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0).contiguous()

    # 4. NO TOPOLOGY - just compute our kernel inputs
    expert_token_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        padded_bins.to(torch.int32)
    ])
    expert_weight_offsets = torch.tensor(
        moe.expert_offsets, dtype=torch.int32, device=device
    )
    expert_widths = torch.tensor(
        moe.expert_widths, dtype=torch.int32, device=device
    )
    padded_tokens_per_expert_int = padded_tokens_per_expert.to(torch.int32)

    # 5. Gather (identical to reference)
    x_gathered = ops.padded_gather(
        x_flat, indices, bin_ids, bins, padded_bins, moe.num_active_experts
    )

    # 6. Triton GEMMs (replaces topology + sdd + activation + dsd)
    x_up = grouped_gemm_up(
        x_gathered,
        moe.w1,
        expert_token_offsets,
        expert_weight_offsets,
        expert_widths,
        padded_tokens_per_expert_int,
        max(moe.expert_widths),
        activation="relu_squared",
    )
    x_down = grouped_gemm_down(
        x_up,
        moe.w2,
        expert_token_offsets,
        expert_weight_offsets,
        expert_widths,
        padded_tokens_per_expert_int,
        moe.config.n_embd,
    )

    # 7. Scatter (identical to reference)
    output = ops.padded_scatter(
        x_down,
        indices,
        bin_ids,
        top_k_weights_flat,
        bins,
        padded_bins,
        moe.num_active_experts,
    )

    return rearrange(output, "(b s) d -> b s d", b=batch_size)


def reference_forward(moe: MoEMLP, x: torch.Tensor) -> torch.Tensor:
    """Reference MoE forward pass (wraps moe.forward, discards aux outputs)."""
    output, _, _ = moe(x)
    return output


def verify_correctness(moe: MoEMLP, x: torch.Tensor, rtol: float = 1.6e-2, atol: float = 2e-3) -> bool:
    """Verify that Triton forward matches reference forward.

    Note: Triton kernels apply relu_squared in float32 before converting to bfloat16,
    while reference converts to bfloat16 first. This is more accurate but produces
    slightly different results (max diff ~0.002 for full pipeline).
    """
    with torch.no_grad():
        # Suppress reference output print statements
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            ref_output = reference_forward(moe, x)
        finally:
            sys.stdout = old_stdout

        triton_output = triton_forward(moe, x)

    try:
        torch.testing.assert_close(triton_output, ref_output, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(f"Correctness check failed: {e}")
        return False


def benchmark_reference_forward(
    moe: MoEMLP,
    x: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
) -> float:
    """Benchmark reference MoE forward pass.

    Returns:
        Average time in ms
    """
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = reference_forward(moe, x)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = reference_forward(moe, x)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return sum(times) / len(times)


def benchmark_triton_forward(
    moe: MoEMLP,
    x: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
) -> float:
    """Benchmark Triton MoE forward pass.

    Returns:
        Average time in ms
    """
    # Warmup (important for Triton autotune)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = triton_forward(moe, x)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = triton_forward(moe, x)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return sum(times) / len(times)


def run_benchmark(config: BenchConfig, num_warmup: int, num_iterations: int, verify: bool = True) -> dict:
    """Run benchmark for a single configuration."""
    device = torch.device("cuda")

    # Create model
    moe_config = MoEConfig(
        n_embd=config.n_embd,
        expert_sizes=config.expert_sizes,
        num_active_experts=config.num_active_experts,
        block_size=config.block_size,
    )
    torch.manual_seed(42)
    moe = MoEMLP(moe_config).to(device).to(torch.bfloat16)

    # Create input
    torch.manual_seed(123)
    x = torch.randn(config.batch_size, config.seq_len, config.n_embd, device=device, dtype=torch.bfloat16)

    # Verify correctness first
    correct = None
    if verify:
        correct = verify_correctness(moe, x)
        if not correct:
            print(f"  WARNING: {config.name} - Triton output does not match reference!")

    # Suppress print statements during benchmarks
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        # Benchmark reference
        ref_ms = benchmark_reference_forward(moe, x, num_warmup, num_iterations)

        # Benchmark Triton
        triton_ms = benchmark_triton_forward(moe, x, num_warmup, num_iterations)
    finally:
        sys.stdout = old_stdout

    speedup = ref_ms / triton_ms if triton_ms > 0 else 0

    return {
        "name": config.name,
        "ref_ms": ref_ms,
        "triton_ms": triton_ms,
        "speedup": speedup,
        "correct": correct,
    }


def print_results(results: list[dict]):
    """Print results in a formatted table."""
    print("\n" + "=" * 75)
    print("Full MoE Forward Pass Benchmark: Reference vs Triton")
    print("=" * 75)
    print(f"{'Config':<20} {'Reference':<15} {'Triton':<15} {'Speedup':<12} {'Correct':<10}")
    print("-" * 75)

    for r in results:
        correct_str = "✓" if r['correct'] else "✗" if r['correct'] is not None else "-"
        print(
            f"{r['name']:<20} "
            f"{r['ref_ms']:>10.3f} ms   "
            f"{r['triton_ms']:>10.3f} ms   "
            f"{r['speedup']:>8.2f}x    "
            f"{correct_str:^10}"
        )

    print("=" * 75)
    print("\nNotes:")
    print("  - Reference: Full MoE forward with topology construction + stk.ops.sdd/dsd")
    print("  - Triton: MoE forward with grouped_gemm_up/down (no topology)")
    print("  - Speedup = Reference / Triton")
    print("  - Correctness verified with rtol=1.6e-2, atol=2e-3 (Triton uses f32 relu_squared)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark full MoE forward pass")
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
    parser.add_argument("--no-verify", action="store_true", help="Skip correctness verification")
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
        result = run_benchmark(config, args.warmup, args.iterations, verify=not args.no_verify)
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
