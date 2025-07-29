# Testing Guide

## Overview

The Video Diffusion Benchmark Suite uses a comprehensive testing strategy with multiple test categories, fixtures, and automation.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── test_benchmark.py              # Core benchmark tests
├── test_metrics.py                # Metrics computation tests
├── test_profiler.py               # Performance profiling tests
├── test_prompts.py                # Prompt generation tests
├── models/                        # Model-specific tests
│   ├── __init__.py
│   └── test_registry.py           # Model registry tests
└── performance/                   # Performance and load tests
    ├── __init__.py
    └── test_benchmark_performance.py
```

## Test Categories

### Unit Tests
Fast, isolated tests that test individual components:

```bash
# Run all unit tests (excludes slow and integration tests)
pytest -m "not slow and not integration and not gpu"

# Run specific test file
pytest tests/test_metrics.py -v
```

### Integration Tests
Tests that verify component interactions:

```bash
# Run integration tests
pytest -m integration

# Run with coverage
pytest -m integration --cov=vid_diffusion_bench
```

### Performance Tests
Tests that verify performance characteristics:

```bash
# Run performance tests (marked as slow)
pytest -m "slow and performance"

# Run specific performance test
pytest tests/performance/test_benchmark_performance.py::TestBenchmarkPerformance::test_benchmark_latency
```

### GPU Tests
Tests that require GPU hardware:

```bash
# Run GPU tests (requires CUDA)
pytest -m gpu

# Skip GPU tests
pytest -m "not gpu"
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v
```

### Test Selection

```bash
# Run fast tests only
pytest -m "not slow"

# Run specific test patterns
pytest -k "test_fvd"

# Run tests by file
pytest tests/test_metrics.py

# Run single test
pytest tests/test_metrics.py::test_fvd_computation
```

### Environment-Specific Testing

```bash
# CPU-only testing
CUDA_VISIBLE_DEVICES="" pytest -m "not gpu"

# GPU testing (requires GPU)
pytest -m gpu

# Docker-based testing
docker-compose run vid-bench pytest
```

## Test Configuration

### Pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=vid_diffusion_bench",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "gpu: marks tests that require GPU",
    "performance: marks tests that measure performance",
]
```

### Test Markers

- `@pytest.mark.slow`: Tests that take >10 seconds
- `@pytest.mark.gpu`: Tests requiring GPU hardware
- `@pytest.mark.integration`: Tests involving multiple components
- `@pytest.mark.performance`: Performance/load tests

## Writing Tests

### Test Structure

```python
"""Test module docstring."""

import pytest
import torch
from unittest.mock import Mock, patch

from vid_diffusion_bench import BenchmarkSuite


class TestBenchmarkSuite:
    """Test suite for BenchmarkSuite class."""

    def test_initialization(self):
        """Test that BenchmarkSuite initializes correctly."""
        suite = BenchmarkSuite(device="cpu")
        assert suite.device == "cpu"

    @pytest.mark.slow
    def test_full_benchmark(self, benchmark_suite, mock_model):
        """Test complete benchmark execution."""
        # Test implementation
        pass

    @pytest.mark.gpu
    def test_gpu_benchmark(self):
        """Test GPU-specific functionality."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        # Test implementation
```

### Using Fixtures

```python
def test_with_fixtures(benchmark_suite, mock_model, sample_prompts):
    """Test using multiple fixtures."""
    results = benchmark_suite.evaluate_model(
        model_name="mock_model",
        prompts=sample_prompts[:2],  # Use subset of prompts
        num_frames=8
    )
    
    assert results is not None
    assert len(results) == 2
```

### Mocking External Dependencies

```python
@patch('vid_diffusion_bench.models.load_model')
def test_model_loading(mock_load_model, benchmark_suite):
    """Test model loading with mocked dependencies."""
    mock_model = Mock()
    mock_model.generate.return_value = torch.randn(16, 3, 256, 256)
    mock_load_model.return_value = mock_model
    
    result = benchmark_suite.evaluate_model("test_model", ["prompt"])
    
    mock_load_model.assert_called_once_with("test_model")
    assert result is not None
```

### Testing Error Conditions

```python
def test_invalid_model_name(benchmark_suite):
    """Test error handling for invalid model names."""
    with pytest.raises(ValueError, match="Unknown model"):
        benchmark_suite.evaluate_model("nonexistent_model", ["prompt"])

def test_cuda_out_of_memory_handling(benchmark_suite):
    """Test handling of CUDA out of memory errors."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.OutOfMemoryError'):
            # Test error handling
            pass
```

## Test Data and Fixtures

### Available Fixtures

#### Core Fixtures
- `benchmark_suite`: Configured BenchmarkSuite instance
- `mock_model`: Mock model adapter for testing
- `sample_prompts`: List of test prompts
- `sample_video`: Sample video tensor
- `temp_dir`: Temporary directory for file operations

#### Environment Fixtures
- `mock_gpu_available`: Mock GPU availability
- `mock_gpu_unavailable`: Mock GPU unavailability
- `set_test_env`: Automatically set test environment variables

#### Configuration Fixtures
- `benchmark_config`: Default benchmark configuration
- `metrics_config`: Default metrics configuration

### Creating Custom Fixtures

```python
@pytest.fixture
def custom_model():
    """Custom model fixture for specific tests."""
    model = Mock()
    model.generate.return_value = torch.zeros(8, 3, 128, 128)
    model.name = "custom_test_model"
    return model

@pytest.fixture(scope="session")
def large_dataset():
    """Session-scoped fixture for expensive setup."""
    # Expensive setup that runs once per test session
    dataset = create_large_test_dataset()
    yield dataset
    # Cleanup code here
```

## Performance Testing

### Benchmarking Test Performance

```python
def test_benchmark_performance(benchmark_suite, mock_model):
    """Test that benchmarks complete within time limits."""
    import time
    
    start_time = time.time()
    
    results = benchmark_suite.evaluate_model(
        model_name="mock_model",
        prompts=["test prompt"],
        num_frames=16
    )
    
    elapsed = time.time() - start_time
    
    # Should complete within reasonable time
    assert elapsed < 30.0, f"Benchmark took {elapsed:.2f}s"
    assert results is not None
```

### Memory Usage Testing

```python
@pytest.mark.gpu
def test_memory_usage(benchmark_suite, mock_model):
    """Test memory usage stays within bounds."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # Run benchmark
    benchmark_suite.evaluate_model("mock_model", ["prompt"])
    
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = peak_memory - initial_memory
    
    # Should not exceed reasonable memory usage
    assert memory_used < 2 * 1024**3  # 2GB limit
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests to main branch
- Pushes to main and develop branches
- Nightly scheduled runs

### Test Matrix

```yaml
strategy:
  matrix:
    python-version: [3.10, 3.11, 3.12]
    os: [ubuntu-latest, windows-latest, macos-latest]
    include:
      - python-version: 3.11
        os: ubuntu-latest
        gpu: true  # GPU tests on specific combination
```

### Test Commands in CI

```yaml
- name: Run tests
  run: |
    pytest --cov=vid_diffusion_bench --cov-report=xml
    
- name: Run GPU tests
  if: matrix.gpu
  run: |
    pytest -m gpu --cov=vid_diffusion_bench --cov-report=xml
    
- name: Run performance tests
  run: |
    pytest -m "slow and performance" --durations=10
```

## Coverage Requirements

### Coverage Targets
- Overall coverage: >85%
- Core modules: >90%
- Critical paths: 100%

### Coverage Reports

```bash
# Generate HTML coverage report
make test-cov

# View coverage report
open htmlcov/index.html

# Check coverage percentage
coverage report --show-missing
```

### Excluding Code from Coverage

```python
def debug_function():  # pragma: no cover
    """Debug function not included in coverage."""
    print("Debug information")

if TYPE_CHECKING:  # pragma: no cover
    # Type checking imports
    from typing import Optional
```

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with debugging information
pytest --tb=long -v

# Stop at first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Start pytest-xdist for parallel debugging
pytest -n auto --dist=each --tx=popen//python=python3.11
```

### Common Debugging Techniques

```python
def test_with_debugging(benchmark_suite):
    """Test with debugging information."""
    # Add print statements
    print(f"Device: {benchmark_suite.device}")
    
    # Use assertions for intermediate values
    result = benchmark_suite.some_method()
    assert result is not None, "Result should not be None"
    
    # Use pytest's capture for output inspection
    import logging
    logging.info("Debug information")
```

## Test Maintenance

### Regular Maintenance Tasks

1. **Update test dependencies**: Keep pytest and plugins up-to-date
2. **Review slow tests**: Optimize or mark appropriately
3. **Clean up obsolete tests**: Remove tests for removed features
4. **Update fixtures**: Keep test data current and relevant

### Best Practices

1. **Test naming**: Use descriptive names that explain what is being tested
2. **Test isolation**: Each test should be independent
3. **Test data**: Use realistic but minimal test data
4. **Assertions**: Use specific assertions with helpful messages
5. **Documentation**: Document complex test setups and expectations

### Test Review Checklist

- [ ] Tests cover new functionality
- [ ] Tests handle error conditions
- [ ] Performance impact is acceptable
- [ ] Tests are properly categorized with markers
- [ ] Fixtures are reused where appropriate
- [ ] Test names are descriptive
- [ ] Tests are deterministic and reproducible