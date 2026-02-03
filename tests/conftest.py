"""Shared test fixtures for MoE kernel tests."""

import pytest
import torch

from reference import MoEConfig, MoEMLP


def _skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def device():
    """Return CUDA device if available, otherwise skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def uniform_moe_config():
    """MoE configuration with uniform expert sizes."""
    return MoEConfig(
        n_embd=256,
        expert_sizes=[(8, 512)],  # 8 experts, each with width 512
        num_active_experts=2,
        block_size=128,
    )


@pytest.fixture
def variable_moe_config():
    """MoE configuration with variable expert sizes."""
    return MoEConfig(
        n_embd=256,
        expert_sizes=[
            (4, 512),   # 4 experts with width 512
            (4, 256),   # 4 experts with width 256
        ],
        num_active_experts=2,
        block_size=128,
    )


@pytest.fixture
def small_moe_config():
    """Small MoE configuration for quick tests."""
    return MoEConfig(
        n_embd=128,
        expert_sizes=[(4, 256)],  # 4 experts, each with width 256
        num_active_experts=2,
        block_size=128,
    )


@pytest.fixture
def reference_moe_uniform(uniform_moe_config, device):
    """Reference MoEMLP with uniform experts and deterministic weights."""
    torch.manual_seed(42)
    moe = MoEMLP(uniform_moe_config).to(device).to(torch.bfloat16)
    return moe


@pytest.fixture
def reference_moe_variable(variable_moe_config, device):
    """Reference MoEMLP with variable experts and deterministic weights."""
    torch.manual_seed(42)
    moe = MoEMLP(variable_moe_config).to(device).to(torch.bfloat16)
    return moe


@pytest.fixture
def reference_moe_small(small_moe_config, device):
    """Small reference MoEMLP for quick tests."""
    torch.manual_seed(42)
    moe = MoEMLP(small_moe_config).to(device).to(torch.bfloat16)
    return moe


@pytest.fixture
def test_input_small(small_moe_config, device):
    """Small test input tensor."""
    torch.manual_seed(123)
    batch_size, seq_len = 2, 64
    return torch.randn(
        batch_size, seq_len, small_moe_config.n_embd,
        device=device, dtype=torch.bfloat16
    )


@pytest.fixture
def test_input_medium(uniform_moe_config, device):
    """Medium test input tensor."""
    torch.manual_seed(123)
    batch_size, seq_len = 4, 256
    return torch.randn(
        batch_size, seq_len, uniform_moe_config.n_embd,
        device=device, dtype=torch.bfloat16
    )


@pytest.fixture
def test_input_large(uniform_moe_config, device):
    """Large test input tensor for stress testing."""
    torch.manual_seed(123)
    batch_size, seq_len = 8, 1024
    return torch.randn(
        batch_size, seq_len, uniform_moe_config.n_embd,
        device=device, dtype=torch.bfloat16
    )


@pytest.fixture(params=["small", "medium", "large"])
def test_input_sizes(request, small_moe_config, uniform_moe_config, device):
    """Parametrized fixture for different input sizes."""
    torch.manual_seed(123)

    if request.param == "small":
        batch_size, seq_len = 2, 64
        n_embd = small_moe_config.n_embd
    elif request.param == "medium":
        batch_size, seq_len = 4, 256
        n_embd = uniform_moe_config.n_embd
    else:  # large
        batch_size, seq_len = 8, 1024
        n_embd = uniform_moe_config.n_embd

    return torch.randn(batch_size, seq_len, n_embd, device=device, dtype=torch.bfloat16)
