# Contributing to Video Diffusion Benchmark Suite

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 Quick Start

1. **Fork and clone** the repository
2. **Set up development environment**:
   ```bash
   make install-dev
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and ensure tests pass:
   ```bash
   make test lint type-check
   ```
5. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites
- Python 3.10+
- Docker 20.10+ (for model containers)
- NVIDIA Docker (for GPU support)
- Git

### Installation
```bash
# Clone your fork
git clone https://github.com/yourusername/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite

# Install in development mode
make install-dev

# Set up pre-commit hooks
pre-commit install
```

## 🎯 Ways to Contribute

### High Priority Areas
- **New model adapters** for video diffusion models
- **Additional evaluation metrics** (perceptual quality, motion smoothness)
- **Performance optimizations** (GPU memory, inference speed)
- **Documentation improvements** and tutorials

### Model Integration
To add a new video diffusion model:

1. **Create model adapter** in `src/vid_diffusion_bench/models/`
2. **Implement required interface**:
   ```python
   from vid_diffusion_bench import ModelAdapter, register_model
   
   @register_model("your-model-name")
   class YourModel(ModelAdapter):
       def generate(self, prompt, **kwargs):
           # Implementation here
           return video_tensor
   ```
3. **Add Docker configuration** in `docker/models/`
4. **Write tests** in `tests/models/test_your_model.py`
5. **Update documentation**

### Evaluation Metrics
For new metrics:

1. **Implement metric class** in `src/vid_diffusion_bench/metrics/`
2. **Follow the base interface**:
   ```python
   from vid_diffusion_bench.metrics import BaseMetric
   
   class YourMetric(BaseMetric):
       def compute(self, videos, references=None):
           # Implementation here
           return score
   ```
3. **Add comprehensive tests**
4. **Document the metric** with references

## 🧪 Testing

### Running Tests
```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_specific.py

# GPU tests (requires CUDA)
pytest -m gpu
```

### Test Categories
- **Unit tests**: Fast, isolated component tests
- **Integration tests**: Cross-component functionality
- **GPU tests**: Require CUDA-enabled environment
- **Slow tests**: Long-running benchmarks

### Writing Tests
- Use pytest fixtures for common setup
- Mock external dependencies (models, APIs)
- Test both success and failure cases
- Include GPU tests for model-related code

## 📝 Code Style

We use automated formatting and linting:

```bash
# Format code
make format

# Check linting
make lint

# Type checking
make type-check
```

### Standards
- **Black** for code formatting (88 character line limit)
- **isort** for import sorting
- **Ruff** for fast linting
- **mypy** for type checking
- **Type hints** required for public APIs

## 🔒 Security

### Guidelines
- **Never commit secrets** (API keys, tokens, passwords)
- **Use environment variables** for configuration
- **Validate inputs** especially from external sources
- **Follow least privilege** principles

### Security Checks
```bash
# Run security audit
make security

# Check for secrets
detect-secrets scan --baseline .secrets.baseline
```

## 📚 Documentation

### Building Docs
```bash
# Build documentation
make docs

# Serve locally
make serve-docs
```

### Documentation Standards
- **Docstrings** for all public functions/classes
- **Type hints** in function signatures
- **Examples** in docstrings where helpful
- **README updates** for new features

## 🐳 Docker Integration

### Model Containers
Each model should have its own Docker container:

```dockerfile
# docker/models/your-model/Dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install model-specific dependencies
RUN pip install your-model-requirements

# Copy model code
COPY src/vid_diffusion_bench/models/your_model.py /app/

# Set entrypoint
ENTRYPOINT ["python", "/app/your_model.py"]
```

### Testing Docker Builds
```bash
# Build all containers
docker compose build

# Test specific model
docker compose run your-model python evaluate.py --test
```

## 🚦 CI/CD Guidelines

### Pull Request Process
1. **Create feature branch** from main
2. **Make atomic commits** with clear messages
3. **Ensure all checks pass** (tests, linting, type checking)
4. **Update documentation** if needed
5. **Request review** from maintainers

### Commit Messages
Follow conventional commits:
```
feat: add support for DreamVideo-v3 model
fix: resolve memory leak in batch processing
docs: update installation instructions
test: add GPU tests for SVD model
```

### Branch Protection
- All PRs require review
- Status checks must pass
- Branch must be up-to-date

## 🎨 UI/UX (Streamlit Dashboard)

### Dashboard Contributions
- **Maintain consistency** with existing design
- **Test across different screen sizes**
- **Optimize performance** for large datasets
- **Add interactive features** thoughtfully

### Dashboard Testing
```bash
# Run dashboard locally
streamlit run dashboard/app.py

# Test with sample data
python scripts/generate_sample_data.py
```

## 📊 Performance Considerations

### Optimization Guidelines
- **Profile before optimizing** using py-spy or memory-profiler
- **Batch operations** when possible
- **Use appropriate data types** (fp16 for inference)
- **Memory-efficient implementations** for large datasets

### Benchmarking
```bash
# Profile memory usage
python -m memory_profiler your_script.py

# CPU profiling
py-spy record -o profile.svg -- python your_script.py
```

## 🤝 Community

### Getting Help
- **GitHub Discussions** for questions and ideas
- **Discord** for real-time chat
- **Issues** for bug reports and feature requests

### Code of Conduct
We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful and inclusive.

## 🏆 Recognition

Contributors are recognized through:
- **GitHub contributors page**
- **Release notes** acknowledgments
- **Community showcase** for significant contributions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Questions?** Open an issue or reach out on Discord. We're here to help! 🎉