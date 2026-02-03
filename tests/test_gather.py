"""Tests for the Triton gather kernel."""

import pytest
import torch

from triton_moe.kernels import gather


class TestGatherKernel:
    """Test gather kernel against reference implementation."""

    def test_gather_matches_reference(self, reference_moe_small, test_input_small):
        """Test that Triton gather matches megablocks.ops.padded_gather."""
        # Get reference intermediates
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        try:
            # Run Triton kernel
            triton_gathered = gather(
                ref["x_flat"],
                ref["indices"],
            )
        except NotImplementedError:
            pytest.skip("Gather kernel not implemented yet")

        # Compare
        torch.testing.assert_close(
            triton_gathered,
            ref["x_gathered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_gather_uniform_experts(self, reference_moe_uniform, test_input_medium):
        """Test gather with uniform expert sizes."""
        ref = reference_moe_uniform.forward_with_intermediates(test_input_medium)

        try:
            triton_gathered = gather(
                ref["x_flat"],
                ref["indices"],
            )
        except NotImplementedError:
            pytest.skip("Gather kernel not implemented yet")

        torch.testing.assert_close(
            triton_gathered,
            ref["x_gathered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_gather_variable_experts(self, reference_moe_variable, test_input_medium):
        """Test gather with variable expert sizes."""
        ref = reference_moe_variable.forward_with_intermediates(test_input_medium)

        try:
            triton_gathered = gather(
                ref["x_flat"],
                ref["indices"],
            )
        except NotImplementedError:
            pytest.skip("Gather kernel not implemented yet")

        torch.testing.assert_close(
            triton_gathered,
            ref["x_gathered"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_gather_preserves_dtype(self, reference_moe_small, test_input_small):
        """Test that gather preserves input dtype."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        try:
            triton_gathered = gather(
                ref["x_flat"],
                ref["indices"],
            )
        except NotImplementedError:
            pytest.skip("Gather kernel not implemented yet")

        assert triton_gathered.dtype == ref["x_flat"].dtype

    def test_gather_correct_shape(self, reference_moe_small, test_input_small):
        """Test that gather produces correct output shape."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)

        try:
            triton_gathered = gather(
                ref["x_flat"],
                ref["indices"],
            )
        except NotImplementedError:
            pytest.skip("Gather kernel not implemented yet")

        assert triton_gathered.shape == ref["x_gathered"].shape
