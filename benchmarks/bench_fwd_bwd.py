"""Benchmark forward and backward passes for Triton vs Reference.

Run with: CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/bench_fwd_bwd.py
"""

import argparse
import io
import sys
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from megablocks import ops

from reference import MoEConfig, MoEMLP
from triton_moe.kernels import grouped_gemm_up_autograd, grouped_gemm_down_autograd


class SuppressStdout:
    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self.old_stdout


def benchmark_reference_forward(moe, x, warmup=50, iterations=200):
    """Benchmark reference forward pass."""
    # Warmup
    for _ in range(warmup):
        with SuppressStdout():
            out, _, _ = moe(x)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        with SuppressStdout():
            out, _, _ = moe(x)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms


def benchmark_reference_backward(moe, x, warmup=50, iterations=200):
    """Benchmark reference forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        moe.zero_grad()
        with SuppressStdout():
            out, _, _ = moe(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        moe.zero_grad()
        with SuppressStdout():
            out, _, _ = moe(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms


def get_triton_routing(moe, x):
    """Pre-compute routing tensors for Triton."""
    x_flat = rearrange(x, "b s d -> (b s) d")

    with torch.no_grad():
        router_logits = moe.router(x_flat)
        router_probs = F.sigmoid(router_logits.float())
        top_k_weights, selected_experts = torch.topk(
            router_probs, moe.num_active_experts, dim=-1
        )
        top_k_weights = (
            top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        ).to(x.dtype)
        top_k_weights_flat = top_k_weights.flatten()
        selected_experts_flat = selected_experts.flatten()

        bin_ids, indices = ops.sort(selected_experts_flat, moe.sort_end_bit)
        tokens_per_expert = ops.histogram(selected_experts_flat, moe.num_experts)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0).contiguous()
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, moe.block_size)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0).contiguous()

        expert_token_offsets = torch.cat(
            [torch.zeros(1, device=x.device, dtype=torch.int32), padded_bins.int()]
        )
        expert_weight_offsets = torch.tensor(
            moe.expert_offsets, dtype=torch.int32, device=x.device
        )
        expert_widths = torch.tensor(
            moe.expert_widths, dtype=torch.int32, device=x.device
        )

    x_gathered = ops.padded_gather(
        x_flat, indices, bin_ids, bins, padded_bins, moe.num_active_experts
    )

    return {
        "x_flat": x_flat,
        "x_gathered": x_gathered,
        "indices": indices,
        "bin_ids": bin_ids,
        "bins": bins,
        "padded_bins": padded_bins,
        "top_k_weights_flat": top_k_weights_flat,
        "padded_tokens_per_expert": padded_tokens_per_expert.int(),
        "expert_token_offsets": expert_token_offsets,
        "expert_weight_offsets": expert_weight_offsets,
        "expert_widths": expert_widths,
    }


def triton_forward_only(moe, routing, w1, w2):
    """Triton forward pass (GEMM only, routing pre-computed)."""
    x_up = grouped_gemm_up_autograd(
        routing["x_gathered"],
        w1,
        routing["expert_token_offsets"],
        routing["expert_weight_offsets"],
        routing["expert_widths"],
        routing["padded_tokens_per_expert"],
        max(moe.expert_widths),
        moe.config.n_embd,
        "relu_squared",
    )
    x_down = grouped_gemm_down_autograd(
        x_up,
        w2,
        routing["expert_token_offsets"],
        routing["expert_weight_offsets"],
        routing["expert_widths"],
        routing["padded_tokens_per_expert"],
        moe.config.n_embd,
        max(moe.expert_widths),
    )

    output_flat = ops.padded_scatter(
        x_down,
        routing["indices"],
        routing["bin_ids"],
        routing["top_k_weights_flat"],
        routing["bins"],
        routing["padded_bins"],
        moe.num_active_experts,
    )

    return output_flat


def benchmark_triton_forward(moe, x, warmup=50, iterations=200):
    """Benchmark Triton forward pass."""
    routing = get_triton_routing(moe, x)
    w1 = moe.w1.detach().clone().requires_grad_(True)
    w2 = moe.w2.detach().clone().requires_grad_(True)

    # Warmup
    for _ in range(warmup):
        out = triton_forward_only(moe, routing, w1, w2)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out = triton_forward_only(moe, routing, w1, w2)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms


def benchmark_triton_backward(moe, x, warmup=50, iterations=200):
    """Benchmark Triton forward + backward pass."""
    routing = get_triton_routing(moe, x)
    w1 = moe.w1.detach().clone().requires_grad_(True)
    w2 = moe.w2.detach().clone().requires_grad_(True)

    # Warmup
    for _ in range(warmup):
        w1.grad = None
        w2.grad = None
        out = triton_forward_only(moe, routing, w1, w2)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        w1.grad = None
        w2.grad = None
        out = triton_forward_only(moe, routing, w1, w2)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms


# Expert configurations
CONFIGS = {
    "8_uniform": {
        "expert_sizes": [(8, 512)],
        "num_active_experts": 2,
    },
    "8_variable": {
        "expert_sizes": [(4, 512), (4, 256)],
        "num_active_experts": 2,
    },
    "64_uniform": {
        "expert_sizes": [(64, 256)],
        "num_active_experts": 8,
    },
    "64_variable": {
        "expert_sizes": [(16, 512), (32, 256), (16, 128)],
        "num_active_experts": 8,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark forward and backward")
    parser.add_argument("--config", type=str, default=None, help="Specific config to run")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Benchmark iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")

    configs = {args.config: CONFIGS[args.config]} if args.config else CONFIGS

    print("=" * 80)
    print("MoE Forward + Backward Benchmark")
    print("=" * 80)
    print(f"Batch: {args.batch_size}, Seq: {args.seq_len}, Hidden: {args.hidden_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    results = []

    for config_name, config_params in configs.items():
        print(f"Config: {config_name}")

        config = MoEConfig(
            n_embd=args.hidden_size,
            expert_sizes=config_params["expert_sizes"],
            num_active_experts=config_params["num_active_experts"],
            block_size=128,
        )

        torch.manual_seed(42)
        moe = MoEMLP(config).to(device).to(torch.float32)

        torch.manual_seed(123)
        x = torch.randn(
            args.batch_size, args.seq_len, args.hidden_size,
            device=device, dtype=torch.float32
        )

        # Benchmark forward
        ref_fwd = benchmark_reference_forward(moe, x, args.warmup, args.iterations)
        tri_fwd = benchmark_triton_forward(moe, x, args.warmup, args.iterations)

        # Benchmark forward + backward
        ref_fwd_bwd = benchmark_reference_backward(moe, x, args.warmup, args.iterations)
        tri_fwd_bwd = benchmark_triton_backward(moe, x, args.warmup, args.iterations)

        # Compute backward-only time
        ref_bwd = ref_fwd_bwd - ref_fwd
        tri_bwd = tri_fwd_bwd - tri_fwd

        results.append({
            "config": config_name,
            "ref_fwd": ref_fwd,
            "tri_fwd": tri_fwd,
            "ref_bwd": ref_bwd,
            "tri_bwd": tri_bwd,
            "ref_total": ref_fwd_bwd,
            "tri_total": tri_fwd_bwd,
        })

        print(f"  Reference: fwd={ref_fwd:.2f}ms, bwd={ref_bwd:.2f}ms, total={ref_fwd_bwd:.2f}ms")
        print(f"  Triton:    fwd={tri_fwd:.2f}ms, bwd={tri_bwd:.2f}ms, total={tri_fwd_bwd:.2f}ms")
        print(f"  Speedup:   fwd={ref_fwd/tri_fwd:.2f}x, bwd={ref_bwd/tri_bwd:.2f}x, total={ref_fwd_bwd/tri_fwd_bwd:.2f}x")
        print()

    # Summary table
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'Config':<15} {'Ref Fwd':>10} {'Tri Fwd':>10} {'Fwd Speedup':>12} {'Ref Bwd':>10} {'Tri Bwd':>10} {'Bwd Speedup':>12}")
    print("-" * 80)
    for r in results:
        fwd_speedup = r["ref_fwd"] / r["tri_fwd"]
        bwd_speedup = r["ref_bwd"] / r["tri_bwd"]
        print(f"{r['config']:<15} {r['ref_fwd']:>9.2f}ms {r['tri_fwd']:>9.2f}ms {fwd_speedup:>11.2f}x {r['ref_bwd']:>9.2f}ms {r['tri_bwd']:>9.2f}ms {bwd_speedup:>11.2f}x")

    print()
    print(f"{'Config':<15} {'Ref Total':>12} {'Tri Total':>12} {'Total Speedup':>14}")
    print("-" * 55)
    for r in results:
        total_speedup = r["ref_total"] / r["tri_total"]
        print(f"{r['config']:<15} {r['ref_total']:>11.2f}ms {r['tri_total']:>11.2f}ms {total_speedup:>13.2f}x")


if __name__ == "__main__":
    main()
