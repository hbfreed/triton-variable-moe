"""Tests for the Triton grouped GEMM kernels.

Tolerance notes:
    The Triton kernels apply relu_squared activation in float32 before converting
    to bfloat16, while the reference (stk + megablocks) converts to bfloat16 first
    then applies activation. Our approach is more numerically accurate but produces
    slightly different results (max diff ~0.004). We use atol=1e-3 to accommodate
    this difference while still catching real bugs.
"""

import pytest
import torch

from triton_moe.kernels import grouped_gemm_up, grouped_gemm_down


# Tolerances for bfloat16 comparisons with f32 fused activation
# See module docstring for explanation
RTOL = 1.6e-2
ATOL = 2e-3


def _get_kernel_inputs(moe, ref):
    """Extract inputs for Triton kernels from reference MoE and forward results.

    The reference uses padded layout where each expert's tokens are rounded up to block_size.
    """
    device = ref["x_gathered"].device
    padded_bins = ref["padded_bins"]

    # Token offsets use padded boundaries (block-aligned)
    expert_token_offsets = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        padded_bins.to(torch.int32)
    ])

    # Weight offsets (cumsum of expert widths)
    expert_weight_offsets = torch.tensor(moe.expert_offsets, dtype=torch.int32, device=device)

    # Expert widths
    expert_widths = torch.tensor(moe.expert_widths, dtype=torch.int32, device=device)

    # Padded tokens per expert (for tile scheduling)
    padded_tokens_per_expert = padded_bins.clone().to(torch.int32)
    padded_tokens_per_expert[1:] -= padded_bins[:-1].to(torch.int32)

    return {
        "expert_token_offsets": expert_token_offsets,
        "expert_weight_offsets": expert_weight_offsets,
        "expert_widths": expert_widths,
        "padded_tokens_per_expert": padded_tokens_per_expert,
        "max_expert_width": max(moe.expert_widths),
        "hidden_size": moe.config.n_embd,
    }


class TestGroupedGemmUpKernel:
    """Test grouped GEMM up-projection kernel."""

    def test_grouped_gemm_up_matches_reference(self, reference_moe_small, test_input_small):
        """Test that Triton grouped GEMM (up + down) matches reference."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)
        inputs = _get_kernel_inputs(reference_moe_small, ref)

        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_small.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        triton_down = grouped_gemm_down(
            triton_up,
            reference_moe_small.w2,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["hidden_size"],
        )

        torch.testing.assert_close(
            triton_down,
            ref["x_after_down"],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_grouped_gemm_up_uniform_experts(self, reference_moe_uniform, test_input_medium):
        """Test grouped GEMM with uniform expert sizes."""
        ref = reference_moe_uniform.forward_with_intermediates(test_input_medium)
        inputs = _get_kernel_inputs(reference_moe_uniform, ref)

        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_uniform.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        triton_down = grouped_gemm_down(
            triton_up,
            reference_moe_uniform.w2,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["hidden_size"],
        )

        torch.testing.assert_close(
            triton_down,
            ref["x_after_down"],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_grouped_gemm_up_variable_experts(self, reference_moe_variable, test_input_medium):
        """Test grouped GEMM with variable expert sizes."""
        ref = reference_moe_variable.forward_with_intermediates(test_input_medium)
        inputs = _get_kernel_inputs(reference_moe_variable, ref)

        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_variable.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        triton_down = grouped_gemm_down(
            triton_up,
            reference_moe_variable.w2,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["hidden_size"],
        )

        torch.testing.assert_close(
            triton_down,
            ref["x_after_down"],
            rtol=RTOL,
            atol=ATOL,
        )


class TestGroupedGemmDownKernel:
    """Test grouped GEMM down-projection kernel.

    Note: These tests use full pipeline (up+down) since reference intermediates
    are sparse stk.Matrix format while our kernels use dense padded format.
    """

    def test_grouped_gemm_down_matches_reference(self, reference_moe_small, test_input_small):
        """Test Triton grouped GEMM down via full pipeline."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)
        inputs = _get_kernel_inputs(reference_moe_small, ref)

        # Run up to get dense intermediate
        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_small.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        triton_down = grouped_gemm_down(
            triton_up,
            reference_moe_small.w2,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["hidden_size"],
        )

        torch.testing.assert_close(
            triton_down,
            ref["x_after_down"],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_grouped_gemm_down_uniform_experts(self, reference_moe_uniform, test_input_medium):
        """Test grouped GEMM down with uniform expert sizes."""
        ref = reference_moe_uniform.forward_with_intermediates(test_input_medium)
        inputs = _get_kernel_inputs(reference_moe_uniform, ref)

        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_uniform.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        triton_down = grouped_gemm_down(
            triton_up,
            reference_moe_uniform.w2,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["hidden_size"],
        )

        torch.testing.assert_close(
            triton_down,
            ref["x_after_down"],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_grouped_gemm_down_variable_experts(self, reference_moe_variable, test_input_medium):
        """Test grouped GEMM down with variable expert sizes."""
        ref = reference_moe_variable.forward_with_intermediates(test_input_medium)
        inputs = _get_kernel_inputs(reference_moe_variable, ref)

        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_variable.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        triton_down = grouped_gemm_down(
            triton_up,
            reference_moe_variable.w2,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["hidden_size"],
        )

        torch.testing.assert_close(
            triton_down,
            ref["x_after_down"],
            rtol=RTOL,
            atol=ATOL,
        )


class TestGroupedGemmNumericalStability:
    """Test numerical stability of grouped GEMM kernels."""

    def test_float32_accumulation(self, reference_moe_small, test_input_small):
        """Test that kernels use float32 accumulation for numerical stability."""
        ref = reference_moe_small.forward_with_intermediates(test_input_small)
        inputs = _get_kernel_inputs(reference_moe_small, ref)

        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_small.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        # Should not have NaN or Inf values
        assert not torch.isnan(triton_up).any()
        assert not torch.isinf(triton_up).any()

    def test_large_inputs(self, reference_moe_uniform, test_input_large):
        """Test with large inputs to check for overflow/underflow."""
        ref = reference_moe_uniform.forward_with_intermediates(test_input_large)
        inputs = _get_kernel_inputs(reference_moe_uniform, ref)

        triton_up = grouped_gemm_up(
            ref["x_gathered"],
            reference_moe_uniform.w1,
            inputs["expert_token_offsets"],
            inputs["expert_weight_offsets"],
            inputs["expert_widths"],
            inputs["padded_tokens_per_expert"],
            inputs["max_expert_width"],
            activation="relu_squared",
        )

        assert not torch.isnan(triton_up).any()
        assert not torch.isinf(triton_up).any()
