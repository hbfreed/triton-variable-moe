"""Profile backward pass in detail to find bottlenecks."""

import sys

import torch

sys.path.insert(0, "/home/henry/Documents/PythonProjects/nanoMoEchat")

from triton_moe import TritonMoEConfig, TritonMoEMLP


def profile_triton_backward():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = TritonMoEConfig(
        n_embd=768,
        expert_sizes=[(64, 256)],
        num_active_experts=8,
        norm_topk_prob=True,
        block_size=128,
    )

    moe = TritonMoEMLP(config).to(device).to(dtype)

    batch_size = 9
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, 768, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        x_in = x.clone().requires_grad_(True)
        out, aux, _ = moe(x_in)
        loss = out.sum() + aux["router_z_loss"]
        loss.backward()
        torch.cuda.synchronize()

    print("=" * 80)
    print("Triton MoE Profile")
    print("=" * 80)

    # Use torch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        x_in = x.clone().requires_grad_(True)
        out, aux, _ = moe(x_in)
        loss = out.sum() + aux["router_z_loss"]
        loss.backward()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print()
    print("=" * 80)
    print("Reference MoE Profile")
    print("=" * 80)

    # Now profile reference
    from nanochat.gpt import GPTConfig, MoEMLP

    ref_config = GPTConfig(
        n_embd=768,
        expert_sizes=[(64, 256)],
        num_active_experts=8,
        norm_topk_prob=True,
        block_size=128,
    )

    ref_moe = MoEMLP(ref_config).to(device).to(dtype)

    # Warmup
    for _ in range(5):
        x_in = x.clone().requires_grad_(True)
        out, aux, _ = ref_moe(x_in)
        loss = out.sum() + aux["router_z_loss"]
        loss.backward()
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        x_in = x.clone().requires_grad_(True)
        out, aux, _ = ref_moe(x_in)
        loss = out.sum() + aux["router_z_loss"]
        loss.backward()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    profile_triton_backward()
