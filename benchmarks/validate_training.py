"""Validate that Triton and reference implementations train similarly.

This script trains a simple model with both implementations and compares
loss curves to verify that the gradient differences don't affect training.

Run with: CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/validate_training.py
"""

import argparse
import io
import sys

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


def triton_forward(moe, x, w1, w2):
    """Forward pass using Triton kernels."""
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

    x_up = grouped_gemm_up_autograd(
        x_gathered,
        w1,
        expert_token_offsets,
        expert_weight_offsets,
        expert_widths,
        padded_tokens_per_expert.int(),
        max(moe.expert_widths),
        moe.config.n_embd,
        "relu_squared",
    )
    x_down = grouped_gemm_down_autograd(
        x_up,
        w2,
        expert_token_offsets,
        expert_weight_offsets,
        expert_widths,
        padded_tokens_per_expert.int(),
        moe.config.n_embd,
        max(moe.expert_widths),
    )

    output_flat = ops.padded_scatter(
        x_down,
        indices,
        bin_ids,
        top_k_weights_flat,
        bins,
        padded_bins,
        moe.num_active_experts,
    )

    return rearrange(output_flat, "(b s) d -> b s d", b=x.shape[0])


def train_step_reference(moe, x, target, optimizer):
    """Single training step using reference implementation."""
    optimizer.zero_grad()
    with SuppressStdout():
        output, _, _ = moe(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_triton(moe, x, target, optimizer, w1, w2):
    """Single training step using Triton implementation."""
    optimizer.zero_grad()
    output = triton_forward(moe, x, w1, w2)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_training(
    n_steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 64,
    hidden_size: int = 128,
    seed: int = 42,
):
    """Train with both implementations and compare loss curves."""
    device = torch.device("cuda")

    # Create config
    config = MoEConfig(
        n_embd=hidden_size,
        expert_sizes=[(4, 256)],  # 4 experts, 256 width each
        num_active_experts=2,
        block_size=128,
    )

    # Initialize models with same weights
    torch.manual_seed(seed)
    ref_moe = MoEMLP(config).to(device).to(torch.float32)

    torch.manual_seed(seed)
    tri_moe = MoEMLP(config).to(device).to(torch.float32)

    # Verify weights are identical
    assert torch.equal(ref_moe.w1, tri_moe.w1)
    assert torch.equal(ref_moe.w2, tri_moe.w2)

    # For Triton, we need separate weight tensors with gradients
    w1_tri = tri_moe.w1.detach().clone().requires_grad_(True)
    w2_tri = tri_moe.w2.detach().clone().requires_grad_(True)

    # Create optimizers
    ref_optimizer = torch.optim.Adam(ref_moe.parameters(), lr=1e-3)
    tri_optimizer = torch.optim.Adam(
        [w1_tri, w2_tri, tri_moe.router.weight], lr=1e-3
    )

    # Training loop
    ref_losses = []
    tri_losses = []

    print(f"Training for {n_steps} steps...")
    print(f"Config: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")
    print()

    for step in range(n_steps):
        # Generate random input and target
        torch.manual_seed(seed + step)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        target = torch.randn(batch_size, seq_len, hidden_size, device=device)

        # Reference step
        ref_loss = train_step_reference(ref_moe, x, target, ref_optimizer)
        ref_losses.append(ref_loss)

        # Triton step (with same input)
        torch.manual_seed(seed + step)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        target = torch.randn(batch_size, seq_len, hidden_size, device=device)

        tri_loss = train_step_triton(tri_moe, x, target, tri_optimizer, w1_tri, w2_tri)
        tri_losses.append(tri_loss)

        if (step + 1) % 20 == 0:
            print(f"Step {step + 1:3d}: ref_loss={ref_loss:.4f}, tri_loss={tri_loss:.4f}")

    # Analyze results
    ref_losses = torch.tensor(ref_losses)
    tri_losses = torch.tensor(tri_losses)

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Reference - Initial: {ref_losses[0]:.4f}, Final: {ref_losses[-1]:.4f}")
    print(f"Triton    - Initial: {tri_losses[0]:.4f}, Final: {tri_losses[-1]:.4f}")
    print()

    # Check if both converged similarly
    ref_reduction = (ref_losses[0] - ref_losses[-1]) / ref_losses[0]
    tri_reduction = (tri_losses[0] - tri_losses[-1]) / tri_losses[0]

    print(f"Loss reduction - Reference: {ref_reduction * 100:.1f}%")
    print(f"Loss reduction - Triton:    {tri_reduction * 100:.1f}%")
    print()

    # Correlation between loss curves
    correlation = torch.corrcoef(torch.stack([ref_losses, tri_losses]))[0, 1]
    print(f"Loss curve correlation: {correlation:.4f}")

    if correlation > 0.9 and abs(ref_reduction - tri_reduction) < 0.1:
        print()
        print("✓ PASSED: Both implementations train similarly")
        return True
    else:
        print()
        print("✗ WARNING: Training dynamics differ significantly")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate training convergence")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    validate_training(
        n_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
