"""Tests for the Triton gather kernel."""

import pytest
import torch

from triton_moe.kernels import padded_gather, padded_gather_autograd


class TestPaddedGatherKernel:
    """Test padded_gather kernel against reference implementation."""

    def test_padded_gather_matches_reference(self, reference_moe_small, test_input_small):
        """Test that Triton padded_gather matches megablocks.ops.padded_gather."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        triton_gathered = padded_gather(
            ref["x_flat"],
            ref["indices"],
            ref["bin_ids"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        torch.testing.assert_close(
            triton_gathered,
            ref["x_gathered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_padded_gather_uniform_experts(self, reference_moe_uniform, test_input_medium):
        """Test padded_gather with uniform expert sizes."""
        ref = reference_moe_uniform.forward_with_intermediates(test_input_medium)

        triton_gathered = padded_gather(
            ref["x_flat"],
            ref["indices"],
            ref["bin_ids"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_uniform.num_active_experts,
        )

        torch.testing.assert_close(
            triton_gathered,
            ref["x_gathered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_padded_gather_variable_experts(self, reference_moe_variable, test_input_medium):
        """Test padded_gather with variable expert sizes."""
        ref = reference_moe_variable.forward_with_intermediates(test_input_medium)

        triton_gathered = padded_gather(
            ref["x_flat"],
            ref["indices"],
            ref["bin_ids"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_variable.num_active_experts,
        )

        torch.testing.assert_close(
            triton_gathered,
            ref["x_gathered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_padded_gather_preserves_dtype(self, reference_moe_small, test_input_small):
        """Test that padded_gather preserves input dtype."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        triton_gathered = padded_gather(
            ref["x_flat"],
            ref["indices"],
            ref["bin_ids"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        assert triton_gathered.dtype == ref["x_flat"].dtype

    def test_padded_gather_correct_shape(self, reference_moe_small, test_input_small):
        """Test that padded_gather produces correct output shape."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        triton_gathered = padded_gather(
            ref["x_flat"],
            ref["indices"],
            ref["bin_ids"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        assert triton_gathered.shape == ref["x_gathered"].shape


class TestPaddedGatherAutograd:
    """Test padded_gather autograd functionality."""

    def test_autograd_forward_matches(self, reference_moe_small, test_input_small):
        """Test that autograd wrapper forward matches non-autograd version."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        triton_gathered = padded_gather_autograd(
            ref["x_flat"],
            ref["indices"],
            ref["bin_ids"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        torch.testing.assert_close(
            triton_gathered,
            ref["x_gathered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_autograd_backward_runs(self, reference_moe_small, test_input_small):
        """Test that backward pass runs without error."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        x = ref["x_flat"].detach().clone().requires_grad_(True)
        gathered = padded_gather_autograd(
            x,
            ref["indices"],
            ref["bin_ids"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        loss = gathered.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_autograd_backward_matches_megablocks(self, reference_moe_small, test_input_small):
        """Test that backward matches megablocks gradient."""
        from megablocks import ops

        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        # Detach all tensors to avoid graph conflicts
        x_data = ref["x_flat"].detach().clone()
        indices = ref["indices"].detach().clone()
        bin_ids = ref["bin_ids"].detach().clone()
        bins = ref["bins"].detach().clone()
        padded_bins = ref["padded_bins"].detach().clone()
        top_k = reference_moe_small.num_active_experts

        # Megablocks backward
        x_mega = x_data.clone().requires_grad_(True)
        gathered_mega = ops.padded_gather(
            x_mega, indices, bin_ids, bins, padded_bins, top_k
        )
        gathered_mega.sum().backward()

        # Triton backward
        x_triton = x_data.clone().requires_grad_(True)
        gathered_triton = padded_gather_autograd(
            x_triton, indices, bin_ids, bins, padded_bins, top_k
        )
        gathered_triton.sum().backward()

        torch.testing.assert_close(
            x_triton.grad,
            x_mega.grad,
            rtol=1e-3,
            atol=1e-3,
        )
