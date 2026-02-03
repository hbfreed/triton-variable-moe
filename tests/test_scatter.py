"""Tests for the Triton scatter kernel."""

import pytest
import torch

from triton_moe.kernels import scatter


class TestScatterKernel:
    """Test scatter kernel against reference implementation."""

    def test_scatter_matches_reference(self, reference_moe_small, test_input_small):
        """Test that Triton scatter matches megablocks.ops.padded_scatter."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        batch_size, seq_len, _ = test_input_small.shape
        num_tokens = batch_size * seq_len

        try:
            triton_scattered = scatter(
                ref["x_after_down"],
                ref["indices"],
                ref["top_k_weights_flat"],
                num_tokens,
                reference_moe_small.num_active_experts,
            )
        except NotImplementedError:
            pytest.skip("Scatter kernel not implemented yet")

        torch.testing.assert_close(
            triton_scattered,
            ref["x_scattered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_scatter_uniform_experts(self, reference_moe_uniform, test_input_medium):
        """Test scatter with uniform expert sizes."""
        ref = reference_moe_uniform.forward_with_intermediates(test_input_medium)

        batch_size, seq_len, _ = test_input_medium.shape
        num_tokens = batch_size * seq_len

        try:
            triton_scattered = scatter(
                ref["x_after_down"],
                ref["indices"],
                ref["top_k_weights_flat"],
                num_tokens,
                reference_moe_uniform.num_active_experts,
            )
        except NotImplementedError:
            pytest.skip("Scatter kernel not implemented yet")

        torch.testing.assert_close(
            triton_scattered,
            ref["x_scattered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_scatter_variable_experts(self, reference_moe_variable, test_input_medium):
        """Test scatter with variable expert sizes."""
        ref = reference_moe_variable.forward_with_intermediates(test_input_medium)

        batch_size, seq_len, _ = test_input_medium.shape
        num_tokens = batch_size * seq_len

        try:
            triton_scattered = scatter(
                ref["x_after_down"],
                ref["indices"],
                ref["top_k_weights_flat"],
                num_tokens,
                reference_moe_variable.num_active_experts,
            )
        except NotImplementedError:
            pytest.skip("Scatter kernel not implemented yet")

        torch.testing.assert_close(
            triton_scattered,
            ref["x_scattered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_scatter_preserves_dtype(self, reference_moe_small, test_input_small):
        """Test that scatter preserves input dtype."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        batch_size, seq_len, _ = test_input_small.shape
        num_tokens = batch_size * seq_len

        try:
            triton_scattered = scatter(
                ref["x_after_down"],
                ref["indices"],
                ref["top_k_weights_flat"],
                num_tokens,
                reference_moe_small.num_active_experts,
            )
        except NotImplementedError:
            pytest.skip("Scatter kernel not implemented yet")

        assert triton_scattered.dtype == ref["x_after_down"].dtype

    def test_scatter_correct_shape(self, reference_moe_small, test_input_small):
        """Test that scatter produces correct output shape."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        batch_size, seq_len, n_embd = test_input_small.shape
        num_tokens = batch_size * seq_len

        try:
            triton_scattered = scatter(
                ref["x_after_down"],
                ref["indices"],
                ref["top_k_weights_flat"],
                num_tokens,
                reference_moe_small.num_active_experts,
            )
        except NotImplementedError:
            pytest.skip("Scatter kernel not implemented yet")

        assert triton_scattered.shape == (num_tokens, n_embd)

    def test_scatter_weights_applied(self, reference_moe_small, test_input_small):
        """Test that routing weights are correctly applied in scatter."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        batch_size, seq_len, _ = test_input_small.shape
        num_tokens = batch_size * seq_len

        # Test with uniform weights (should just sum contributions)
        uniform_weights = torch.ones_like(ref["top_k_weights_flat"])

        try:
            triton_scattered_uniform = scatter(
                ref["x_after_down"],
                ref["indices"],
                uniform_weights,
                num_tokens,
                reference_moe_small.num_active_experts,
            )
        except NotImplementedError:
            pytest.skip("Scatter kernel not implemented yet")

        # With uniform weights, result should differ from weighted version
        triton_scattered_weighted = scatter(
            ref["x_after_down"],
            ref["indices"],
            ref["top_k_weights_flat"],
            num_tokens,
            reference_moe_small.num_active_experts,
        )

        # They should not be equal (unless weights happen to be all 1s)
        if not torch.allclose(ref["top_k_weights_flat"], uniform_weights):
            assert not torch.allclose(triton_scattered_uniform, triton_scattered_weighted)
