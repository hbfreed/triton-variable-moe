#!/usr/bin/env python3
"""Performance benchmarks comparing Triton MoE against reference implementation.

Usage:
    uv run benchmarks/bench_moe.py
    uv run benchmarks/bench_moe.py --batch-sizes 4 8 16
    uv run benchmarks/bench_moe.py --profile
"""

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.cuda

# Import reference implementation
from reference import MoEConfig, MoEMLP

# Import profiler
from benchmarks.profiler import CUDAStepProfiler, AggregateProfiler

# Import Triton implementation (will fail gracefully if not implemented)
try:
    from triton_moe import TritonMoEMLP
    from triton_moe.moe import TritonMoEConfig

    TRITON_AVAILABLE = True
except (ImportError, NotImplementedError):
    TRITON_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    batch_size: int
    seq_len: int
    n_embd: int
    expert_sizes: list[tuple[int, int]]
    num_active_experts: int
    num_warmup: int = 10
    num_runs: int = 100


def benchmark_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    num_warmup: int,
    num_runs: int,
) -> tuple[float, float]:
    """Benchmark forward pass.

    Returns:
        mean_time_ms: Mean time per forward pass in milliseconds
        std_time_ms: Standard deviation of time per forward pass
    """
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    return mean_time, std_time


def benchmark_forward_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    num_warmup: int,
    num_runs: int,
) -> tuple[float, float]:
    """Benchmark forward + backward pass.

    Returns:
        mean_time_ms: Mean time per forward+backward pass in milliseconds
        std_time_ms: Standard deviation
    """
    # Warmup
    for _ in range(num_warmup):
        model.zero_grad()
        x_clone = x.clone().requires_grad_(True)
        output, _, _ = model(x_clone)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        model.zero_grad()
        x_clone = x.clone().requires_grad_(True)

        torch.cuda.synchronize()
        start = time.perf_counter()
        output, _, _ = model(x_clone)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    return mean_time, std_time


def benchmark_forward_profiled(
    model: MoEMLP,
    x: torch.Tensor,
    num_warmup: int,
    num_runs: int,
) -> AggregateProfiler:
    """Benchmark forward pass with per-step profiling.

    Returns:
        AggregateProfiler with timing data for each step
    """
    # Warmup (without profiling)
    for _ in range(num_warmup):
        with torch.no_grad():
            profiler = CUDAStepProfiler()
            _ = model.forward_profiled(x, profiler)

    torch.cuda.synchronize()

    # Benchmark with profiling
    agg = AggregateProfiler()
    for _ in range(num_runs):
        profiler = CUDAStepProfiler()
        with torch.no_grad():
            _ = model.forward_profiled(x, profiler)
        agg.add_result(profiler.get_result())

    return agg


def _categorize_backward_op(op_name: str) -> str | None:
    """Categorize a backward operation into a meaningful group.

    Returns None for ops we want to skip (overhead, etc.)
    """
    name_lower = op_name.lower()

    # Scatter/Gather backward (triton kernels from megablocks)
    if "_padded_copy_wgrad" in name_lower:
        return "scatter_wgrad"
    if "_padded_copy" in name_lower and "wgrad" not in name_lower:
        return "gather_bwd"

    # GEMM backward (stk kernels)
    if "_dsd_kernel" in name_lower:
        return "dsd_bwd"
    if "_sdd_kernel" in name_lower:
        return "sdd_bwd"
    if "_dds_kernel" in name_lower:
        return "dds_bwd"

    # The massive fill operation (zeroing tensors in scatter backward)
    if "fillfunctor" in name_lower:
        return "zero_fill"

    # Activation backward (threshold = relu backward)
    if "threshold" in name_lower:
        return "activation_bwd"

    # Routing backward
    if "sigmoid_backward" in name_lower:
        return "routing_bwd"

    # GEMM kernels (cutlass/cublas for router backward)
    if "cutlass" in name_lower or "cublas" in name_lower:
        return "gemm_bwd"

    # Reduce operations (for sum backward)
    if "reduce_kernel" in name_lower:
        return "reduce"

    # Element-wise ops (mul, add, div, neg)
    if "mulfunctor" in name_lower or "addfunctor" in name_lower:
        return "elementwise"
    if "divfunctor" in name_lower or "neg_kernel" in name_lower:
        return "elementwise"

    # Memory operations
    if "memcpy" in name_lower or "memset" in name_lower:
        return "memory"

    # Skip other low-level ops
    return None


def benchmark_backward_profiled(
    model: MoEMLP,
    x: torch.Tensor,
    num_warmup: int,
    num_runs: int,
) -> dict[str, dict[str, float]]:
    """Benchmark backward pass with per-step profiling using torch.profiler.

    Returns:
        Dictionary with 'mean' and 'std' sub-dicts mapping step names to times in ms,
        plus 'wall_clock_mean' and 'wall_clock_std' for total wall-clock time.
    """
    import time
    from collections import defaultdict

    # Warmup
    for _ in range(num_warmup):
        model.zero_grad()
        x_clone = x.clone().requires_grad_(True)
        out, _, _ = model(x_clone)
        out.sum().backward()

    torch.cuda.synchronize()

    # Collect timing data
    all_step_times: dict[str, list[float]] = defaultdict(list)
    wall_clock_times: list[float] = []

    for _ in range(num_runs):
        model.zero_grad()
        x_clone = x.clone().requires_grad_(True)

        # Forward pass (not profiled here)
        out, _, _ = model(x_clone)

        torch.cuda.synchronize()

        # Profile backward with wall-clock timing
        start_time = time.perf_counter()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            out.sum().backward()
        torch.cuda.synchronize()
        wall_clock_times.append((time.perf_counter() - start_time) * 1000)

        # Categorize and aggregate CUDA times
        step_times: dict[str, float] = defaultdict(float)
        for event in prof.key_averages():
            category = _categorize_backward_op(event.key)
            if category and event.device_time_total > 0:
                step_times[category] += event.device_time_total / 1000  # Convert to ms

        for name, time_ms in step_times.items():
            all_step_times[name].append(time_ms)

    # Compute mean and std
    means = {name: sum(times) / len(times) for name, times in all_step_times.items()}
    stds = {}
    for name, times in all_step_times.items():
        mean = means[name]
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        stds[name] = variance ** 0.5

    wall_mean = sum(wall_clock_times) / len(wall_clock_times)
    wall_std = (sum((t - wall_mean) ** 2 for t in wall_clock_times) / len(wall_clock_times)) ** 0.5

    return {
        "mean": means,
        "std": stds,
        "num_runs": num_runs,
        "wall_clock_mean": wall_mean,
        "wall_clock_std": wall_std,
    }


def run_benchmark(config: BenchmarkConfig) -> dict:
    """Run benchmarks for a given configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available, benchmarks will be slow")

    results = {"config": config.name}

    # Create reference model
    ref_config = MoEConfig(
        n_embd=config.n_embd,
        expert_sizes=config.expert_sizes,
        num_active_experts=config.num_active_experts,
        block_size=128,
    )
    ref_model = MoEMLP(ref_config).to(device).to(torch.bfloat16)

    # Create input
    torch.manual_seed(42)
    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.n_embd,
        device=device,
        dtype=torch.bfloat16,
    )

    # Benchmark reference forward
    ref_fwd_mean, ref_fwd_std = benchmark_forward(
        ref_model, x, config.num_warmup, config.num_runs
    )
    results["ref_forward_ms"] = ref_fwd_mean
    results["ref_forward_std"] = ref_fwd_std

    # Benchmark reference forward+backward
    ref_fwdbwd_mean, ref_fwdbwd_std = benchmark_forward_backward(
        ref_model, x, config.num_warmup, config.num_runs
    )
    results["ref_fwdbwd_ms"] = ref_fwdbwd_mean
    results["ref_fwdbwd_std"] = ref_fwdbwd_std

    # Benchmark Triton implementation if available
    if TRITON_AVAILABLE:
        try:
            triton_config = TritonMoEConfig(
                n_embd=config.n_embd,
                expert_sizes=config.expert_sizes,
                num_active_experts=config.num_active_experts,
                block_size=128,
            )
            triton_model = TritonMoEMLP(triton_config).to(device).to(torch.bfloat16)

            # Copy weights for fair comparison
            triton_model.router.weight.data.copy_(ref_model.router.weight.data)
            triton_model.w1.data.copy_(ref_model.w1.data)
            triton_model.w2.data.copy_(ref_model.w2.data)

            triton_fwd_mean, triton_fwd_std = benchmark_forward(
                triton_model, x, config.num_warmup, config.num_runs
            )
            results["triton_forward_ms"] = triton_fwd_mean
            results["triton_forward_std"] = triton_fwd_std
            results["forward_speedup"] = ref_fwd_mean / triton_fwd_mean

            triton_fwdbwd_mean, triton_fwdbwd_std = benchmark_forward_backward(
                triton_model, x, config.num_warmup, config.num_runs
            )
            results["triton_fwdbwd_ms"] = triton_fwdbwd_mean
            results["triton_fwdbwd_std"] = triton_fwdbwd_std
            results["fwdbwd_speedup"] = ref_fwdbwd_mean / triton_fwdbwd_mean

        except NotImplementedError:
            results["triton_forward_ms"] = None
            results["triton_fwdbwd_ms"] = None
            results["note"] = "Triton kernels not implemented"
    else:
        results["triton_forward_ms"] = None
        results["triton_fwdbwd_ms"] = None
        results["note"] = "Triton MoE not available"

    return results


def run_profiled_benchmark(config: BenchmarkConfig) -> tuple[AggregateProfiler, dict]:
    """Run profiled benchmark for a given configuration.

    Returns:
        Tuple of (forward_profiler, backward_profile_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for profiled benchmarks")

    # Create reference model
    ref_config = MoEConfig(
        n_embd=config.n_embd,
        expert_sizes=config.expert_sizes,
        num_active_experts=config.num_active_experts,
        block_size=128,
    )
    ref_model = MoEMLP(ref_config).to(device).to(torch.bfloat16)

    # Create input
    torch.manual_seed(42)
    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.n_embd,
        device=device,
        dtype=torch.bfloat16,
    )

    # Run profiled benchmarks
    fwd_profiler = benchmark_forward_profiled(
        ref_model, x, config.num_warmup, config.num_runs
    )
    bwd_profile = benchmark_backward_profiled(
        ref_model, x, config.num_warmup, config.num_runs
    )

    return fwd_profiler, bwd_profile


def print_backward_profile(bwd_profile: dict, indent: str = "  "):
    """Print backward profiling results."""
    means = bwd_profile["mean"]
    stds = bwd_profile["std"]
    num_runs = bwd_profile["num_runs"]
    wall_mean = bwd_profile.get("wall_clock_mean", 0)
    wall_std = bwd_profile.get("wall_clock_std", 0)

    if not means:
        print(f"{indent}No backward profiling data")
        return

    cuda_total = sum(means.values())

    # Sort by time (descending)
    sorted_steps = sorted(means.items(), key=lambda x: x[1], reverse=True)

    print(f"{indent}Backward Profiling ({num_runs} runs):")
    print(f"{indent}{'-' * 55}")
    for name, mean_time in sorted_steps:
        std_time = stds.get(name, 0)
        pct = (mean_time / cuda_total * 100) if cuda_total > 0 else 0
        print(f"{indent}  {name:<22} {mean_time:>7.3f} +/- {std_time:>6.3f} ms ({pct:>5.1f}%)")
    print(f"{indent}{'-' * 55}")
    cuda_std = (sum(s ** 2 for s in stds.values())) ** 0.5
    print(f"{indent}  {'CUDA TOTAL':<22} {cuda_total:>7.3f} +/- {cuda_std:>6.3f} ms")
    print(f"{indent}  {'WALL CLOCK':<22} {wall_mean:>7.3f} +/- {wall_std:>6.3f} ms")
    if wall_mean > 0:
        overhead = wall_mean - cuda_total
        overhead_pct = (overhead / wall_mean) * 100
        print(f"{indent}  {'OVERHEAD':<22} {overhead:>7.3f} ms ({overhead_pct:.1f}% of wall clock)")


def print_results(results: list[dict]):
    """Print benchmark results in a table format."""
    print("\n" + "=" * 80)
    print("MoE Benchmark Results")
    print("=" * 80)

    # Header
    print(f"{'Config':<20} {'Ref Fwd (ms)':<15} {'Ref Fwd+Bwd (ms)':<18} ", end="")
    if results[0].get("triton_forward_ms") is not None:
        print(f"{'Triton Fwd (ms)':<18} {'Triton Fwd+Bwd (ms)':<20} {'Speedup':<10}")
    else:
        print()

    print("-" * 80)

    for r in results:
        print(f"{r['config']:<20} ", end="")
        print(
            f"{r['ref_forward_ms']:.3f} +/- {r['ref_forward_std']:.3f}".ljust(15),
            end=" ",
        )
        print(
            f"{r['ref_fwdbwd_ms']:.3f} +/- {r['ref_fwdbwd_std']:.3f}".ljust(18), end=" "
        )

        if r.get("triton_forward_ms") is not None:
            print(
                f"{r['triton_forward_ms']:.3f} +/- {r['triton_forward_std']:.3f}".ljust(
                    18
                ),
                end=" ",
            )
            print(
                f"{r['triton_fwdbwd_ms']:.3f} +/- {r['triton_fwdbwd_std']:.3f}".ljust(
                    20
                ),
                end=" ",
            )
            print(f"{r.get('forward_speedup', 0):.2f}x")
        else:
            print(f"  ({r.get('note', 'N/A')})")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE implementations")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        help="Batch sizes to benchmark",
    )
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument(
        "--num-warmup", type=int, default=10, help="Number of warmup runs"
    )
    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of benchmark runs"
    )
    parser.add_argument("--profile", action="store_true", help="Run with CUDA profiler")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required for meaningful benchmarks")
        return

    # Define benchmark configurations
    configs = []

    # 8 uniform experts (baseline)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"8exp_uniform_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(8, 512)],  # 8 uniform experts @ 512 width
                num_active_experts=2,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    # 8 variable experts (4 large + 4 small)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"8exp_variable_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(4, 512), (4, 256)],  # 4 @ 512, 4 @ 256
                num_active_experts=2,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    # 16 uniform experts (more experts)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"16exp_uniform_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(16, 256)],  # 16 uniform experts @ 256 width
                num_active_experts=2,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    # 16 variable experts (mixed sizes)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"16exp_variable_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(4, 512), (8, 256), (4, 128)],  # mixed: 4@512, 8@256, 4@128
                num_active_experts=2,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    # 64 uniform experts with top_k=8 (high sparsity regime)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"64exp_uniform_k8_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(64, 256)],  # 64 uniform experts @ 256 width
                num_active_experts=8,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    # 64 variable experts with top_k=8 (high sparsity, variable sizes)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"64exp_variable_k8_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(16, 512), (32, 256), (16, 128)],  # 16@512, 32@256, 16@128
                num_active_experts=8,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    # 128 uniform experts with top_k=8 (Arcee-style sparsity)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"128exp_uniform_k8_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(128, 256)],  # 128 uniform experts @ 256 width
                num_active_experts=8,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    # 128 variable experts with top_k=8 (Arcee-style sparsity, variable sizes)
    for bs in args.batch_sizes:
        configs.append(
            BenchmarkConfig(
                name=f"128exp_variable_k8_bs{bs}",
                batch_size=bs,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                expert_sizes=[(32, 512), (64, 256), (32, 128)],  # 32@512, 64@256, 32@128
                num_active_experts=8,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
        )

    print(f"Running benchmarks with {len(configs)} configurations...")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

    if args.profile:
        # Run profiled benchmarks showing per-step timing
        print("\n" + "=" * 60)
        print("Per-Step Profiling Results (Reference Implementation)")
        print("=" * 60)

        for config in configs:
            print(f"\n{config.name} (batch={config.batch_size}, seq={config.seq_len}, embd={config.n_embd}):")
            fwd_agg, bwd_profile = run_profiled_benchmark(config)
            fwd_agg.print_summary()
            print()
            print_backward_profile(bwd_profile)
    else:
        # Run standard benchmarks
        results = []
        for config in configs:
            print(f"  Benchmarking {config.name}...")
            result = run_benchmark(config)
            results.append(result)

        # Print results
        print_results(results)


if __name__ == "__main__":
    main()
