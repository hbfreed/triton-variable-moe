"""Tests for the Triton scatter kernel."""

import pytest
import torch

from triton_moe.kernels import padded_scatter, padded_scatter_autograd


class TestPaddedScatterKernel:
    """Test padded_scatter kernel against reference implementation."""

    def test_padded_scatter_matches_reference(self, reference_moe_small, test_input_small):
        """Test that Triton padded_scatter matches megablocks.ops.padded_scatter."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        triton_scattered = padded_scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        torch.testing.assert_close(
            triton_scattered,
            ref["x_scattered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_padded_scatter_uniform_experts(self, reference_moe_uniform, test_input_medium):
        """Test padded_scatter with uniform expert sizes."""
        ref = reference_moe_uniform.forward_with_intermediates(test_input_medium)

        triton_scattered = padded_scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_uniform.num_active_experts,
        )

        torch.testing.assert_close(
            triton_scattered,
            ref["x_scattered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_padded_scatter_variable_experts(self, reference_moe_variable, test_input_medium):
        """Test padded_scatter with variable expert sizes."""
        ref = reference_moe_variable.forward_with_intermediates(test_input_medium)

        triton_scattered = padded_scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_variable.num_active_experts,
        )

        torch.testing.assert_close(
            triton_scattered,
            ref["x_scattered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_padded_scatter_preserves_dtype(self, reference_moe_small, test_input_small):
        """Test that padded_scatter preserves input dtype."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        triton_scattered = padded_scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        assert triton_scattered.dtype == ref["x_after_down"].dtype

    def test_padded_scatter_correct_shape(self, reference_moe_small, test_input_small):
        """Test that padded_scatter produces correct output shape."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)
        batch_size, seq_len, n_embd = test_input_small.shape
        num_tokens = batch_size * seq_len

        triton_scattered = padded_scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        assert triton_scattered.shape == (num_tokens, n_embd)

    def test_padded_scatter_weights_applied(self, reference_moe_small, test_input_small):
        """Test that routing weights are correctly applied in scatter."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)
        batch_size, seq_len, _ = test_input_small.shape
        num_tokens = batch_size * seq_len

        # Test with uniform weights
        uniform_weights = torch.ones_like(ref["top_k_weights_flat"])

        triton_scattered_uniform = padded_scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            uniform_weights,
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        # With actual weights
        triton_scattered_weighted = padded_scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        # They should not be equal (unless weights happen to be all 1s)
        if not torch.allclose(ref["top_k_weights_flat"], uniform_weights):
            assert not torch.allclose(triton_scattered_uniform, triton_scattered_weighted)


class TestPaddedScatterAutograd:
    """Test padded_scatter autograd functionality."""

    def test_autograd_forward_matches(self, reference_moe_small, test_input_small):
        """Test that autograd wrapper forward matches non-autograd version."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        triton_scattered = padded_scatter_autograd(
            ref["x_after_down"],
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        torch.testing.assert_close(
            triton_scattered,
            ref["x_scattered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_autograd_backward_runs(self, reference_moe_small, test_input_small):
        """Test that backward pass runs without error."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        x = ref["x_after_down"].detach().clone().requires_grad_(True)
        scattered = padded_scatter_autograd(
            x,
            ref["indices"],
            ref["bin_ids"],
            ref["top_k_weights_flat"],
            ref["bins"],
            ref["padded_bins"],
            reference_moe_small.num_active_experts,
        )

        loss = scattered.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_autograd_backward_matches_megablocks(self, reference_moe_small, test_input_small):
        """Test that backward matches megablocks gradient."""
        from megablocks import ops

        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        # Detach all tensors to avoid graph conflicts
        x_data = ref["x_after_down"].detach().clone()
        indices = ref["indices"].detach().clone()
        bin_ids = ref["bin_ids"].detach().clone()
        weights = ref["top_k_weights_flat"].detach().clone()
        bins = ref["bins"].detach().clone()
        padded_bins = ref["padded_bins"].detach().clone()
        top_k = reference_moe_small.num_active_experts

        # Megablocks backward
        x_mega = x_data.clone().requires_grad_(True)
        scattered_mega = ops.padded_scatter(
            x_mega, indices, bin_ids, weights, bins, padded_bins, top_k
        )
        scattered_mega.sum().backward()

        # Triton backward
        x_triton = x_data.clone().requires_grad_(True)
        scattered_triton = padded_scatter_autograd(
            x_triton, indices, bin_ids, weights, bins, padded_bins, top_k
        )
        scattered_triton.sum().backward()

        # Only compare non-padding rows
        num_expanded = len(indices)
        torch.testing.assert_close(
            x_triton.grad[:num_expanded],
            x_mega.grad[:num_expanded],
            rtol=1e-3,
            atol=1e-3,
        )
