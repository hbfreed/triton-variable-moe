"""Fused activation functions for MoE kernels.

These can be fused into the grouped GEMM kernels for better performance.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _relu_squared_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU squared: relu(x)^2"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    x = tl.maximum(x, 0.0)
    out = x * x
    tl.store(out_ptr + offset, out, mask=mask)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """ReLU squared activation: relu(x)^2

    Args:
        x: Input tensor

    Returns:
        relu(x)^2

    Reference: megablocks.layers.relu_squared.relu_squared
    """
    raise NotImplementedError(
        "TODO: Implement Triton relu_squared kernel. "
        "Reference: megablocks.layers.relu_squared.relu_squared"
    )


@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU (Swish): x * sigmoid(x)"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offset, out, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation: x * sigmoid(x)

    Args:
        x: Input tensor

    Returns:
        x * sigmoid(x)
    """
    raise NotImplementedError(
        "TODO: Implement Triton silu kernel"
    )


@triton.jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU: silu(gate) * up"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    gate = tl.load(gate_ptr + offset, mask=mask)
    up = tl.load(up_ptr + offset, mask=mask)

    # silu(gate) * up
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up
    tl.store(out_ptr + offset, out, mask=mask)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: silu(gate) * up

    For SwiGLU models, the up-projection produces both gate and up,
    then this combines them.

    Args:
        gate: Gate tensor (x @ W_gate)
        up: Up tensor (x @ W_up)

    Returns:
        silu(gate) * up
    """
    raise NotImplementedError(
        "TODO: Implement Triton swiglu kernel"
    )
