"""Pytest configuration and shared fixtures."""

import pytest
import torch
from unittest.mock import MagicMock

from vid_diffusion_bench import BenchmarkSuite
from vid_diffusion_bench.models import ModelAdapter


@pytest.fixture
def benchmark_suite():
    """Provide BenchmarkSuite instance for testing."""
    return BenchmarkSuite(device="cpu")


@pytest.fixture
def mock_model():
    """Provide mock model adapter for testing."""
    mock = MagicMock(spec=ModelAdapter)
    mock.generate.return_value = torch.randn(16, 3, 256, 256)
    mock.requirements = {
        "vram_gb": 8,
        "precision": "fp16", 
        "dependencies": ["torch>=2.0"]
    }
    mock.name = "mock_model"
    return mock


@pytest.fixture
def sample_prompts():
    """Provide sample prompts for testing."""
    return [
        "A cat playing with a ball",
        "Sunset over the ocean",
        "Rain falling on a window"
    ]


@pytest.fixture
def sample_video():
    """Provide sample video tensor for testing."""
    return torch.randn(16, 3, 256, 256)  # 16 frames, RGB, 256x256


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )