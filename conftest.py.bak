"""
Global test configuration and fixtures for Video Diffusion Benchmark Suite.

This file provides shared test configuration, fixtures, and utilities
used across all test modules.
"""

import os
import pytest
import torch
from unittest.mock import Mock
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security-focused tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    
    for item in items:
        # Add markers based on test file location
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        # Auto-mark GPU tests
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def skip_if_no_gpu(gpu_available):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available for testing")


@pytest.fixture(scope="function")
def mock_torch_device():
    """Mock torch device for consistent testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="function")
def sample_video_tensor():
    """Generate sample video tensor for testing."""
    # Create sample video: batch=1, frames=8, channels=3, height=64, width=64
    return torch.randn(1, 8, 3, 64, 64)


@pytest.fixture(scope="function")
def sample_prompts():
    """Standard set of test prompts."""
    return [
        "A cat playing piano",
        "Ocean waves at sunset",
        "City traffic time-lapse",
        "Forest in autumn colors"
    ]


@pytest.fixture(scope="function")
def mock_model_registry():
    """Mock model registry with test models."""
    registry = {
        "test-model-small": {
            "class": "MockModel",
            "vram_gb": 2,
            "precision": "fp16",
            "max_frames": 16
        },
        "test-model-large": {
            "class": "MockModel", 
            "vram_gb": 12,
            "precision": "fp32",
            "max_frames": 64
        },
        "test-model-fast": {
            "class": "MockModel",
            "vram_gb": 4,
            "precision": "fp16", 
            "max_frames": 32,
            "speed": "fast"
        }
    }
    return registry


@pytest.fixture(scope="function")
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.generate.return_value = torch.randn(8, 3, 256, 256)  # 8 frames
    model.requirements = {
        "vram_gb": 8,
        "precision": "fp16",
        "dependencies": ["torch>=2.0"]
    }
    model.name = "mock-model"
    return model


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(scope="function")
def benchmark_config():
    """Default benchmark configuration for testing."""
    return {
        "num_frames": 16,
        "fps": 8,
        "resolution": (256, 256),
        "batch_size": 1,
        "seed": 42,
        "num_inference_steps": 20
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "tests" / "data"


@pytest.fixture(scope="function")
def mock_metrics():
    """Mock metrics results for testing."""
    return {
        "fvd": 45.2,
        "inception_score": 3.8,
        "clip_score": 0.82,
        "temporal_consistency": 0.94,
        "latency_ms": 12500,
        "peak_vram_gb": 8.4,
        "throughput_fps": 0.64
    }


@pytest.fixture(scope="function")
def environment_variables():
    """Mock environment variables for testing."""
    test_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_HOME": "/tmp/torch_cache",
        "HF_HOME": "/tmp/hf_cache",
        "WANDB_MODE": "disabled",
        "BENCHMARK_LOG_LEVEL": "INFO"
    }
    return test_env


@pytest.fixture(scope="function", autouse=True)
def cleanup_gpu_memory():
    """Automatically cleanup GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(scope="function")
def performance_profiler():
    """Simple performance profiler for tests."""
    import time
    from collections import defaultdict
    
    class Profiler:
        def __init__(self):
            self.times = defaultdict(list)
            self.start_times = {}
        
        def start(self, name):
            self.start_times[name] = time.time()
        
        def end(self, name):
            if name in self.start_times:
                duration = time.time() - self.start_times[name]
                self.times[name].append(duration)
                del self.start_times[name]
                return duration
            return 0
        
        def get_stats(self, name):
            times = self.times[name]
            if not times:
                return None
            return {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times)
            }
    
    return Profiler()


@pytest.fixture(scope="function")
def memory_tracker():
    """Track memory usage during tests."""
    import psutil
    
    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.initial_memory = self.get_memory_mb()
        
        def get_memory_mb(self):
            return self.process.memory_info().rss / 1024 / 1024
        
        def get_gpu_memory_gb(self):
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
            return 0
        
        def get_memory_increase(self):
            return self.get_memory_mb() - self.initial_memory
        
        def reset(self):
            self.initial_memory = self.get_memory_mb()
    
    return MemoryTracker()


# Pytest hooks for advanced test reporting
def pytest_runtest_makereport(item, call):
    """Generate custom test reports."""
    if call.when == "call":
        # Add custom information to test reports
        if hasattr(item, "rep_" + call.when):
            return
        
        rep = call.result
        setattr(item, "rep_" + call.when, rep)
        
        # Log performance information for slow tests
        if "slow" in [mark.name for mark in item.iter_markers()]:
            if hasattr(rep, "duration"):
                print(f"\n⏱️  Slow test {item.name} completed in {rep.duration:.2f}s")


# Custom assertions for video testing
def assert_valid_video_tensor(tensor, expected_frames=None, expected_shape=None):
    """Assert that tensor is a valid video tensor."""
    assert isinstance(tensor, torch.Tensor), "Expected torch.Tensor"
    assert tensor.dim() >= 4, "Video tensor should have at least 4 dimensions"
    
    if expected_frames:
        assert tensor.shape[-4] == expected_frames, f"Expected {expected_frames} frames"
    
    if expected_shape:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}"
    
    # Check for reasonable value range
    assert not torch.isnan(tensor).any(), "Video tensor contains NaN values"
    assert not torch.isinf(tensor).any(), "Video tensor contains infinite values"


def assert_valid_metrics(metrics, required_keys=None):
    """Assert that metrics dict contains expected keys and valid values."""
    if required_keys is None:
        required_keys = ["fvd", "latency", "peak_vram_gb"]
    
    for key in required_keys:
        assert key in metrics, f"Missing required metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} should be numeric"
        assert not math.isnan(metrics[key]), f"Metric {key} is NaN"
        assert not math.isinf(metrics[key]), f"Metric {key} is infinite"


# Make custom assertions available globally
pytest.assert_valid_video_tensor = assert_valid_video_tensor
pytest.assert_valid_metrics = assert_valid_metrics