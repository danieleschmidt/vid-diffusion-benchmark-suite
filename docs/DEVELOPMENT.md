# Development Guide

## Quick Start

### Prerequisites
- Python 3.10+
- Docker 20.10+
- NVIDIA Docker (for GPU support)
- CUDA 11.8+ (for local development)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make install-dev

# Verify installation
python -m vid_diffusion_bench.cli --help
```

## Development Workflow

### Code Style and Quality

We enforce strict code quality standards:

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security scanning
make security

# Run all quality checks
make format lint type-check security
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m "not gpu"            # Skip GPU-requiring tests
pytest tests/test_metrics.py   # Run specific test file
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks:

```bash
# Install hooks (done automatically with make install-dev)
pre-commit install

# Run hooks manually
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

## Project Structure

```
vid-diffusion-benchmark-suite/
├── src/vid_diffusion_bench/    # Main package
│   ├── __init__.py
│   ├── benchmark.py            # Core benchmark engine
│   ├── cli.py                  # Command-line interface
│   ├── metrics.py              # Evaluation metrics
│   ├── profiler.py             # Performance profiling
│   ├── prompts.py              # Test prompt generation
│   └── models/                 # Model integration
│       ├── __init__.py
│       ├── base.py             # Base model adapter
│       └── registry.py         # Model registry
├── tests/                      # Test suite
│   ├── conftest.py             # Pytest configuration
│   ├── test_*.py               # Test modules
│   └── models/                 # Model tests
├── docs/                       # Documentation
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # Container definition
├── Makefile                    # Development tasks
└── pyproject.toml              # Package configuration
```

## Adding New Models

### 1. Create Model Adapter

```python
# src/vid_diffusion_bench/models/my_model.py
from .base import ModelAdapter
from .registry import register_model

@register_model("my-awesome-model")
class MyAwesomeModel(ModelAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model here
    
    def generate(self, prompt: str, num_frames: int = 16, **kwargs):
        # Implementation here
        return video_tensor
    
    @property
    def requirements(self):
        return {
            "vram_gb": 24,
            "precision": "fp16",
            "dependencies": ["my-model-library>=1.0.0"]
        }
```

### 2. Create Tests

```python
# tests/models/test_my_model.py
import pytest
from vid_diffusion_bench.models import get_model

def test_my_model_generation():
    model = get_model("my-awesome-model")
    result = model.generate("A cat playing piano", num_frames=8)
    
    assert result.shape == (8, 3, 512, 512)  # frames, channels, height, width
    assert result.dtype == torch.float32
```

### 3. Create Docker Environment

```dockerfile
# docker/models/my-model/Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN pip install my-model-library>=1.0.0
COPY model_weights/ /app/weights/

# Model-specific setup
ENV MODEL_PATH=/app/weights
```

### 4. Update Documentation

Add your model to the supported models list in README.md and create model-specific documentation if needed.

## Testing Guidelines

### Test Categories

```python
@pytest.mark.slow
def test_full_benchmark():
    """Tests that take >10 seconds"""
    pass

@pytest.mark.gpu
def test_gpu_inference():
    """Tests requiring GPU"""
    pass

@pytest.mark.integration
def test_model_integration():
    """Integration tests"""
    pass
```

### Fixtures

```python
# tests/conftest.py
@pytest.fixture
def mock_model():
    """Mock model for unit tests"""
    return MockModelAdapter()

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available"""
    return torch.cuda.is_available()
```

## Performance Testing

### Benchmarking New Features

```python
# tests/performance/test_metrics_performance.py
import time
import pytest

def test_fvd_computation_speed():
    start = time.time()
    fvd_score = compute_fvd(sample_videos, reference_videos)
    duration = time.time() - start
    
    # Should complete within reasonable time
    assert duration < 60.0  # 1 minute max
```

### Memory Usage Testing

```python
def test_memory_usage():
    initial_memory = torch.cuda.memory_allocated()
    
    model = load_model("large-model")
    result = model.generate("test prompt")
    
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = peak_memory - initial_memory
    
    # Should not exceed expected memory usage
    assert memory_used < 24 * 1024**3  # 24GB
```

## Docker Development

### Building Images

```bash
# Build main image
docker build -t vid-bench:dev .

# Build model-specific images
docker build -f docker/models/svd/Dockerfile -t vid-bench/svd:latest .
```

### Testing with Docker Compose

```bash
# Start all services
docker-compose up -d

# Run tests in container
docker-compose exec vid-bench pytest

# View logs
docker-compose logs -f vid-bench
```

## Debugging

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use gradient checkpointing
   torch.cuda.empty_cache()
   ```

2. **Model Loading Errors**
   ```python
   # Check model path and permissions
   assert os.path.exists(model_path)
   assert os.access(model_path, os.R_OK)
   ```

3. **Dependency Conflicts**
   ```bash
   # Use fresh virtual environment
   pip freeze > current_deps.txt
   pip install -r requirements-dev.txt
   ```

### Debugging Tools

```python
# Add to code for debugging
import pdb; pdb.set_trace()

# For GPU debugging
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
```

## Documentation

### Building Docs

```bash
# Install doc dependencies
pip install -e ".[docs]"

# Build documentation
make docs

# Serve locally
make serve-docs
```

### Adding API Documentation

Use docstrings following Google style:

```python
def compute_fvd(generated_videos: torch.Tensor, reference_videos: torch.Tensor) -> float:
    """Compute Fréchet Video Distance between generated and reference videos.
    
    Args:
        generated_videos: Tensor of shape (N, T, C, H, W)
        reference_videos: Tensor of shape (M, T, C, H, W)
        
    Returns:
        FVD score as float (lower is better)
        
    Raises:
        ValueError: If video tensors have incompatible shapes
    """
    pass
```

## Release Process

### Version Bumping

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to 0.2.0"
git tag v0.2.0
```

### Building Release

```bash
# Build package
make build

# Test installation
pip install dist/*.whl

# Upload to PyPI (maintainers only)
twine upload dist/*
```

## Contributing Guidelines

1. **Fork and Branch**: Create feature branches from main
2. **Code Quality**: Ensure all quality checks pass
3. **Tests**: Add tests for new functionality
4. **Documentation**: Update docs for new features
5. **Pull Request**: Submit PR with clear description

### Commit Message Format

```
type(scope): description

- feat: new features
- fix: bug fixes
- docs: documentation changes
- style: formatting changes
- refactor: code restructuring
- test: test additions/modifications
- chore: maintenance tasks
```

Example:
```
feat(models): add support for CogVideo-v2

- Implement CogVideo adapter with temporal attention
- Add comprehensive test suite
- Update model registry and documentation
```