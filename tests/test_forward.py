"""Tests for full MoE forward pass."""

import pytest
import torch

from triton_moe import TritonMoEMLP
from triton_moe.moe import TritonMoEConfig


class TestForwardPass:
    """Test complete forward pass against reference implementation."""

    def test_forward_matches_reference_small(self, reference_moe_small, test_input_small):
        """Test that Triton MoE forward matches reference (small config)."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device).to(torch.bfloat16)

            # Copy weights from reference
            triton_moe.router.weight.data.copy_(reference_moe_small.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_small.w1.data)
            triton_moe.w2.data.copy_(reference_moe_small.w2.data)

            triton_output, _, _ = triton_moe(test_input_small)
        except NotImplementedError:
            pytest.skip("Triton MoE forward not implemented yet")

        ref_output, _, _ = reference_moe_small(test_input_small)

        torch.testing.assert_close(
            triton_output,
            ref_output,
            rtol=1.6e-2,
            atol=2e-3,
        )

    def test_forward_matches_reference_uniform(self, reference_moe_uniform, test_input_medium):
        """Test that Triton MoE forward matches reference (uniform experts)."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_uniform.config.n_embd,
                expert_sizes=reference_moe_uniform.config.expert_sizes,
                num_active_experts=reference_moe_uniform.num_active_experts,
                block_size=reference_moe_uniform.block_size,
            )).to(test_input_medium.device).to(torch.bfloat16)

            triton_moe.router.weight.data.copy_(reference_moe_uniform.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_uniform.w1.data)
            triton_moe.w2.data.copy_(reference_moe_uniform.w2.data)

            triton_output, _, _ = triton_moe(test_input_medium)
        except NotImplementedError:
            pytest.skip("Triton MoE forward not implemented yet")

        ref_output, _, _ = reference_moe_uniform(test_input_medium)

        torch.testing.assert_close(
            triton_output,
            ref_output,
            rtol=1.6e-2,
            atol=2e-3,
        )

    def test_forward_matches_reference_variable(self, reference_moe_variable, test_input_medium):
        """Test that Triton MoE forward matches reference (variable experts)."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_variable.config.n_embd,
                expert_sizes=reference_moe_variable.config.expert_sizes,
                num_active_experts=reference_moe_variable.num_active_experts,
                block_size=reference_moe_variable.block_size,
            )).to(test_input_medium.device).to(torch.bfloat16)

            triton_moe.router.weight.data.copy_(reference_moe_variable.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_variable.w1.data)
            triton_moe.w2.data.copy_(reference_moe_variable.w2.data)

            triton_output, _, _ = triton_moe(test_input_medium)
        except NotImplementedError:
            pytest.skip("Triton MoE forward not implemented yet")

        ref_output, _, _ = reference_moe_variable(test_input_medium)

        torch.testing.assert_close(
            triton_output,
            ref_output,
            rtol=1.6e-2,
            atol=2e-3,
        )

    def test_forward_deterministic(self, reference_moe_small, test_input_small):
        """Test that forward pass is deterministic."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device).to(torch.bfloat16)

            triton_moe.router.weight.data.copy_(reference_moe_small.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_small.w1.data)
            triton_moe.w2.data.copy_(reference_moe_small.w2.data)

            output1, _, _ = triton_moe(test_input_small)
            output2, _, _ = triton_moe(test_input_small)
        except NotImplementedError:
            pytest.skip("Triton MoE forward not implemented yet")

        torch.testing.assert_close(output1, output2)

    def test_forward_output_shape(self, reference_moe_small, test_input_small):
        """Test that forward pass produces correct output shape."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device).to(torch.bfloat16)

            triton_output, _, _ = triton_moe(test_input_small)
        except NotImplementedError:
            pytest.skip("Triton MoE forward not implemented yet")

        assert triton_output.shape == test_input_small.shape

    def test_forward_preserves_dtype(self, reference_moe_small, test_input_small):
        """Test that forward pass preserves input dtype."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device).to(torch.bfloat16)

            triton_output, _, _ = triton_moe(test_input_small)
        except NotImplementedError:
            pytest.skip("Triton MoE forward not implemented yet")

        assert triton_output.dtype == test_input_small.dtype


class TestForwardWithIntermediates:
    """Test forward_with_intermediates for debugging and testing."""

    def test_intermediates_match_reference(self, reference_moe_small, test_input_small):
        """Test that all intermediate values match reference."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device).to(torch.bfloat16)

            triton_moe.router.weight.data.copy_(reference_moe_small.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_small.w1.data)
            triton_moe.w2.data.copy_(reference_moe_small.w2.data)

            triton_intermediates = triton_moe.forward_with_intermediates(test_input_small)
        except NotImplementedError:
            pytest.skip("Triton MoE forward_with_intermediates not implemented yet")

        # Compare intermediate tensors (skip x_after_up/activation which are stk.Matrix in reference)
        for key in ["x_gathered", "x_after_down", "output"]:
            if key in triton_intermediates and key in ref:
                ref_val = ref[key]
                triton_val = triton_intermediates[key]
                # Skip if reference is not a tensor (e.g., stk.Matrix)
                if not isinstance(ref_val, torch.Tensor):
                    continue
                torch.testing.assert_close(
                    triton_val,
                    ref_val,
                    rtol=1.6e-2,
                    atol=2e-3,
                    msg=f"Mismatch in {key}",
                )
