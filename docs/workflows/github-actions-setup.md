# GitHub Actions CI/CD Setup Guide

This document outlines the recommended GitHub Actions workflows for the Video Diffusion Benchmark Suite.

## Required Workflows

### 1. Continuous Integration (CI)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests
      run: pytest --cov=vid_diffusion_bench --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Run security checks
      run: |
        bandit -r src/
        safety check
    
    - name: Run secret detection
      uses: Yelp/detect-secrets-action@v1.4.0
      with:
        args: '--baseline .secrets.baseline'
```

### 2. Docker Build and Push

Create `.github/workflows/docker.yml`:

```yaml
name: Docker

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 3. Nightly Benchmarks

Create `.github/workflows/nightly-benchmark.yml`:

```yaml
name: Nightly Benchmark

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:     # Allow manual trigger

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    if: github.repository_owner == 'yourusername'  # Only run on main repo
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
    
    - name: Free GPU memory
      run: |
        nvidia-smi
        docker system prune -f
    
    - name: Run benchmark suite
      run: |
        python -m vid_diffusion_bench.run_full \
          --models new,updated \
          --output results/nightly_$(date +%Y%m%d).json \
          --upload-wandb
      env:
        WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        CUDA_VISIBLE_DEVICES: 0
    
    - name: Update leaderboard
      run: |
        python scripts/update_leaderboard.py \
          --results results/nightly_$(date +%Y%m%d).json
    
    - name: Archive results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results-${{ github.run_id }}
        path: results/
        retention-days: 30
```

### 4. Release Automation

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
        draft: false
        prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}
```

### 5. Documentation Deployment

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
    
    - name: Build documentation
      run: mkdocs build
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site
```

## Required Secrets

Set up these secrets in your repository settings:

### Repository Secrets
- `PYPI_API_TOKEN`: PyPI token for package publishing
- `WANDB_API_KEY`: Weights & Biases API key for experiment tracking
- `CODECOV_TOKEN`: Codecov token for coverage reporting (optional)

### Environment Variables
```yaml
# In workflow files
env:
  PYTHONPATH: src
  CUDA_VISIBLE_DEVICES: 0  # For GPU workflows
  TORCH_HOME: ~/.cache/torch  # Cache model weights
```

## Workflow Features

### Branch Protection Rules
Set up branch protection for `main`:
- Require status checks to pass
- Require branches to be up to date
- Require review from code owners
- Restrict pushes to main branch

### Caching Strategy
- **Pip dependencies**: Cache based on `pyproject.toml` hash
- **Model weights**: Cache frequently used model weights
- **Docker layers**: Use GitHub Actions cache for Docker builds

### Matrix Testing
Test across multiple Python versions and platforms:

```yaml
strategy:
  matrix:
    python-version: [3.10, 3.11, 3.12]
    os: [ubuntu-latest, windows-latest, macos-latest]
```

### GPU Workflow Considerations
For GPU-intensive workflows:
- Use self-hosted runners with GPU support
- Implement proper GPU memory cleanup
- Set resource limits to prevent conflicts
- Use conditional execution based on repository ownership

### Security Best Practices
- Use minimal required permissions
- Pin action versions (e.g., `@v4` not `@main`)
- Validate secrets and environment variables
- Scan for vulnerabilities in dependencies
- Use signed commits where possible

## Monitoring and Notifications

### Slack Integration
```yaml
- name: Notify Slack on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#ci-alerts'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Performance Monitoring
Track workflow performance:
- Job duration metrics
- Resource utilization
- Success/failure rates
- Artifact sizes

## Troubleshooting

### Common Issues
1. **GPU Memory Errors**: Increase memory cleanup between jobs
2. **Timeout Errors**: Adjust timeout values for long-running benchmarks
3. **Docker Build Failures**: Check multi-platform compatibility
4. **Secret Access**: Verify repository permissions and secret names

### Debug Mode
Enable debug logging:
```yaml
- name: Debug information
  run: |
    echo "GitHub Context: ${{ toJson(github) }}"
    echo "Runner Context: ${{ toJson(runner) }}"
    nvidia-smi  # For GPU runners
```

This setup provides comprehensive CI/CD coverage while maintaining security and performance best practices.