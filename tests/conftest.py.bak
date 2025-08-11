"""Pytest configuration and shared fixtures for vid_diffusion_bench tests."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
import tempfile
import os
from pathlib import Path

from vid_diffusion_bench import BenchmarkSuite
from vid_diffusion_bench.models import ModelAdapter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


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
def mock_gpu_available():
    """Mock GPU availability."""
    original_available = torch.cuda.is_available
    torch.cuda.is_available = Mock(return_value=True)
    yield True
    torch.cuda.is_available = original_available


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailability."""
    original_available = torch.cuda.is_available
    torch.cuda.is_available = Mock(return_value=False)
    yield False
    torch.cuda.is_available = original_available


@pytest.fixture
def benchmark_config():
    """Default benchmark configuration."""
    return {
        "models": ["test-model"],
        "prompts": ["test prompt"],
        "num_frames": 16,
        "fps": 8,
        "resolution": (256, 256),
        "batch_size": 1,
        "seed": 42
    }


@pytest.fixture(autouse=True)
def set_test_env():
    """Set environment variables for testing."""
    os.environ["TESTING"] = "1"
    os.environ["TORCH_HOME"] = "/tmp/torch_test"
    yield
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "TORCH_HOME" in os.environ:
        del os.environ["TORCH_HOME"]


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