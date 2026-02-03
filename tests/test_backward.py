"""Tests for MoE backward pass (gradients)."""

import pytest
import torch

from triton_moe import TritonMoEMLP
from triton_moe.moe import TritonMoEConfig


class TestBackwardPass:
    """Test backward pass gradient computation."""

    def test_backward_runs(self, reference_moe_small, test_input_small):
        """Test that backward pass runs without errors."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device)

            x = test_input_small.clone().requires_grad_(True)
            output, _, _ = triton_moe(x)
            loss = output.sum()
            loss.backward()
        except NotImplementedError:
            pytest.skip("Triton MoE backward not implemented yet")

        # Check that gradients were computed
        assert x.grad is not None
        assert triton_moe.w1.grad is not None
        assert triton_moe.w2.grad is not None
        assert triton_moe.router.weight.grad is not None

    def test_input_gradient_matches_reference(self, reference_moe_small, test_input_small):
        """Test that input gradients match reference implementation."""
        # Reference backward
        x_ref = test_input_small.clone().requires_grad_(True)
        ref_output, _, _ = reference_moe_small(x_ref)
        ref_loss = ref_output.sum()
        ref_loss.backward()
        ref_grad = x_ref.grad.clone()

        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device)

            triton_moe.router.weight.data.copy_(reference_moe_small.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_small.w1.data)
            triton_moe.w2.data.copy_(reference_moe_small.w2.data)

            x_triton = test_input_small.clone().requires_grad_(True)
            triton_output, _, _ = triton_moe(x_triton)
            triton_loss = triton_output.sum()
            triton_loss.backward()
            triton_grad = x_triton.grad
        except NotImplementedError:
            pytest.skip("Triton MoE backward not implemented yet")

        torch.testing.assert_close(
            triton_grad,
            ref_grad,
            rtol=1.6e-2,
            atol=1e-5,
        )

    def test_weight_gradient_matches_reference(self, reference_moe_small, test_input_small):
        """Test that weight gradients match reference implementation."""
        # Reference backward
        x_ref = test_input_small.clone()
        reference_moe_small.zero_grad()
        ref_output, _, _ = reference_moe_small(x_ref)
        ref_loss = ref_output.sum()
        ref_loss.backward()
        ref_w1_grad = reference_moe_small.w1.grad.clone()
        ref_w2_grad = reference_moe_small.w2.grad.clone()

        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device)

            triton_moe.router.weight.data.copy_(reference_moe_small.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_small.w1.data)
            triton_moe.w2.data.copy_(reference_moe_small.w2.data)

            x_triton = test_input_small.clone()
            triton_moe.zero_grad()
            triton_output, _, _ = triton_moe(x_triton)
            triton_loss = triton_output.sum()
            triton_loss.backward()
        except NotImplementedError:
            pytest.skip("Triton MoE backward not implemented yet")

        torch.testing.assert_close(
            triton_moe.w1.grad,
            ref_w1_grad,
            rtol=1.6e-2,
            atol=1e-5,
            msg="W1 gradient mismatch",
        )
        torch.testing.assert_close(
            triton_moe.w2.grad,
            ref_w2_grad,
            rtol=1.6e-2,
            atol=1e-5,
            msg="W2 gradient mismatch",
        )

    def test_gradcheck(self, reference_moe_small, device):
        """Test gradient correctness using torch.autograd.gradcheck."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(device).double()  # gradcheck needs float64

            # Small input for gradcheck
            x = torch.randn(2, 16, reference_moe_small.config.n_embd, device=device, dtype=torch.float64, requires_grad=True)

            def forward_fn(x):
                out, _, _ = triton_moe(x)
                return out

            result = torch.autograd.gradcheck(forward_fn, x, eps=1e-4, atol=1e-3, rtol=1e-3)
        except NotImplementedError:
            pytest.skip("Triton MoE backward not implemented yet")

        assert result

    def test_backward_variable_experts(self, reference_moe_variable, test_input_medium):
        """Test backward pass with variable expert sizes."""
        # Reference backward
        x_ref = test_input_medium.clone().requires_grad_(True)
        ref_output, _, _ = reference_moe_variable(x_ref)
        ref_loss = ref_output.sum()
        ref_loss.backward()
        ref_grad = x_ref.grad.clone()

        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_variable.config.n_embd,
                expert_sizes=reference_moe_variable.config.expert_sizes,
                num_active_experts=reference_moe_variable.num_active_experts,
                block_size=reference_moe_variable.block_size,
            )).to(test_input_medium.device)

            triton_moe.router.weight.data.copy_(reference_moe_variable.router.weight.data)
            triton_moe.w1.data.copy_(reference_moe_variable.w1.data)
            triton_moe.w2.data.copy_(reference_moe_variable.w2.data)

            x_triton = test_input_medium.clone().requires_grad_(True)
            triton_output, _, _ = triton_moe(x_triton)
            triton_loss = triton_output.sum()
            triton_loss.backward()
        except NotImplementedError:
            pytest.skip("Triton MoE backward not implemented yet")

        torch.testing.assert_close(
            x_triton.grad,
            ref_grad,
            rtol=1.6e-2,
            atol=1e-5,
        )


class TestBackwardNumericalStability:
    """Test numerical stability of backward pass."""

    def test_no_nan_gradients(self, reference_moe_small, test_input_small):
        """Test that gradients don't contain NaN values."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device)

            x = test_input_small.clone().requires_grad_(True)
            output, _, _ = triton_moe(x)
            loss = output.sum()
            loss.backward()
        except NotImplementedError:
            pytest.skip("Triton MoE backward not implemented yet")

        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(triton_moe.w1.grad).any()
        assert not torch.isnan(triton_moe.w2.grad).any()

    def test_no_inf_gradients(self, reference_moe_small, test_input_small):
        """Test that gradients don't contain Inf values."""
        try:
            triton_moe = TritonMoEMLP(TritonMoEConfig(
                n_embd=reference_moe_small.config.n_embd,
                expert_sizes=reference_moe_small.config.expert_sizes,
                num_active_experts=reference_moe_small.num_active_experts,
                block_size=reference_moe_small.block_size,
            )).to(test_input_small.device)

            x = test_input_small.clone().requires_grad_(True)
            output, _, _ = triton_moe(x)
            loss = output.sum()
            loss.backward()
        except NotImplementedError:
            pytest.skip("Triton MoE backward not implemented yet")

        assert not torch.isinf(x.grad).any()
        assert not torch.isinf(triton_moe.w1.grad).any()
        assert not torch.isinf(triton_moe.w2.grad).any()
