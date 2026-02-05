"""Benchmark training step with actual nanoMoEchat model configuration.

Matches pretrain_smol.sh config:
- depth=12, dim=768, num_heads=12
- expert_sizes=[(64, 256)], num_active_experts=8
- batch_size=9, seq_len=1024
"""

import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add nanoMoEchat to path for reference implementation
sys.path.insert(0, "/home/henry/Documents/PythonProjects/nanoMoEchat")


@dataclass
class BenchConfig:
    """Minimal config matching pretrain_smol.sh."""

    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    sequence_len: int = 1024
    vocab_size: int = 50304
    expert_sizes: list = field(default_factory=lambda: [(64, 256)])
    num_active_experts: int = 8
    norm_topk_prob: bool = True
    block_size: int = 128


def create_reference_moe(config: BenchConfig) -> nn.Module:
    """Create reference MoE layer using stk/megablocks."""
    from nanochat.gpt import GPTConfig, MoEMLP

    # GPTConfig is used directly by MoEMLP
    moe_config = GPTConfig(
        n_embd=config.n_embd,
        expert_sizes=config.expert_sizes,
        num_active_experts=config.num_active_experts,
        norm_topk_prob=config.norm_topk_prob,
        block_size=config.block_size,
    )
    return MoEMLP(moe_config)


def create_triton_moe(config: BenchConfig) -> nn.Module:
    """Create Triton MoE layer."""
    from triton_moe import TritonMoEConfig, TritonMoEMLP

    tri_config = TritonMoEConfig(
        n_embd=config.n_embd,
        expert_sizes=config.expert_sizes,
        num_active_experts=config.num_active_experts,
        norm_topk_prob=config.norm_topk_prob,
        block_size=config.block_size,
    )
    return TritonMoEMLP(tri_config)


def benchmark_moe_step(
    moe: nn.Module,
    x: torch.Tensor,
    warmup: int = 10,
    iterations: int = 50,
) -> dict:
    """Benchmark forward + backward step."""
    device = x.device

    # Warmup
    for _ in range(warmup):
        x_input = x.clone().requires_grad_(True)
        output, aux_loss, _ = moe(x_input)
        loss = output.sum() + aux_loss["router_z_loss"]
        loss.backward()
        torch.cuda.synchronize()

    # Benchmark
    fwd_times = []
    bwd_times = []
    total_times = []

    for _ in range(iterations):
        x_input = x.clone().requires_grad_(True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        output, aux_loss, _ = moe(x_input)
        loss = output.sum() + aux_loss["router_z_loss"]

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

    return {
        "fwd_mean": sum(fwd_times) / len(fwd_times),
        "bwd_mean": sum(bwd_times) / len(bwd_times),
        "total_mean": sum(total_times) / len(total_times),
        "fwd_std": (sum((t - sum(fwd_times) / len(fwd_times)) ** 2 for t in fwd_times) / len(fwd_times)) ** 0.5,
        "bwd_std": (sum((t - sum(bwd_times) / len(bwd_times)) ** 2 for t in bwd_times) / len(bwd_times)) ** 0.5,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=9, help="Batch size (default: 9 from pretrain_smol.sh)")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--triton_only", action="store_true", help="Only benchmark Triton")
    parser.add_argument("--reference_only", action="store_true", help="Only benchmark reference")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = BenchConfig()

    print("=" * 80)
    print("MoE Training Step Benchmark (pretrain_smol.sh config)")
    print("=" * 80)
    print(f"Config: depth={config.n_layer}, dim={config.n_embd}, heads={config.n_head}")
    print(f"Experts: {config.expert_sizes}, active={config.num_active_experts}")
    print(f"Batch: {args.batch_size} x {args.seq_len} = {args.batch_size * args.seq_len} tokens")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # Create input
    x = torch.randn(args.batch_size, args.seq_len, config.n_embd, device=device, dtype=dtype)

    results = {}

    if not args.triton_only:
        print("Creating reference MoE (stk/megablocks)...")
        ref_moe = create_reference_moe(config).to(device).to(dtype)

        print("Benchmarking reference MoE...")
        results["reference"] = benchmark_moe_step(ref_moe, x, args.warmup, args.iterations)
        print(f"  Forward:  {results['reference']['fwd_mean']:.2f}ms ± {results['reference']['fwd_std']:.2f}ms")
        print(f"  Backward: {results['reference']['bwd_mean']:.2f}ms ± {results['reference']['bwd_std']:.2f}ms")
        print(f"  Total:    {results['reference']['total_mean']:.2f}ms")
        print()

        del ref_moe
        torch.cuda.empty_cache()

    if not args.reference_only:
        print("Creating Triton MoE...")
        tri_moe = create_triton_moe(config).to(device).to(dtype)

        print("Benchmarking Triton MoE...")
        results["triton"] = benchmark_moe_step(tri_moe, x, args.warmup, args.iterations)
        print(f"  Forward:  {results['triton']['fwd_mean']:.2f}ms ± {results['triton']['fwd_std']:.2f}ms")
        print(f"  Backward: {results['triton']['bwd_mean']:.2f}ms ± {results['triton']['bwd_std']:.2f}ms")
        print(f"  Total:    {results['triton']['total_mean']:.2f}ms")
        print()

        del tri_moe
        torch.cuda.empty_cache()

    # Summary
    if "reference" in results and "triton" in results:
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        ref = results["reference"]
        tri = results["triton"]

        fwd_speedup = ref["fwd_mean"] / tri["fwd_mean"]
        bwd_speedup = ref["bwd_mean"] / tri["bwd_mean"]
        total_speedup = ref["total_mean"] / tri["total_mean"]

        print(f"Forward:  {ref['fwd_mean']:.2f}ms → {tri['fwd_mean']:.2f}ms ({fwd_speedup:.2f}x)")
        print(f"Backward: {ref['bwd_mean']:.2f}ms → {tri['bwd_mean']:.2f}ms ({bwd_speedup:.2f}x)")
        print(f"Total:    {ref['total_mean']:.2f}ms → {tri['total_mean']:.2f}ms ({total_speedup:.2f}x)")

        if total_speedup < 1.0:
            print()
            print("WARNING: Triton is SLOWER than reference!")
            print("Breakdown:")
            print(f"  Forward accounts for {ref['fwd_mean'] / ref['total_mean'] * 100:.1f}% of reference time")
            print(f"  Backward accounts for {ref['bwd_mean'] / ref['total_mean'] * 100:.1f}% of reference time")


if __name__ == "__main__":
    main()
