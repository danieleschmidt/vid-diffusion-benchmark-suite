# Installation

This guide walks you through installing the Video Diffusion Benchmark Suite in different environments.

## Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 12.1+ (for GPU acceleration)
- **Docker**: 20.10+ (for containerized models)
- **NVIDIA Docker**: 2.0+ (for GPU containers)

## Quick Installation

### Using pip (Recommended)

```bash
# Install from PyPI
pip install vid-diffusion-benchmark-suite

# Or install from source
git clone https://github.com/yourusername/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite
pip install -e .
```

### Using conda

```bash
# Create environment
conda create -n vid-bench python=3.10
conda activate vid-bench

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install benchmark suite
pip install vid-diffusion-benchmark-suite
```

## Development Installation

For contributing or development:

```bash
# Clone repository
git clone https://github.com/yourusername/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite

# Install in development mode
make install-dev

# Set up pre-commit hooks
pre-commit install
```

## Docker Installation

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite

# Build and start services
docker-compose up -d

# Access dashboard at http://localhost:8501
```

### Using Docker directly

```bash
# Build image
docker build -t vid-bench .

# Run container
docker run -it --gpus all vid-bench
```

## Verification

Verify your installation:

```bash
# Check CLI
vid-bench --version

# Run basic test
python -c "import vid_diffusion_bench; print('Installation successful!')"

# Test GPU access (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Common Issues

**CUDA not detected**
```bash
# Verify CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Docker GPU access**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Performance Optimization

For optimal performance:

```bash
# Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Enable mixed precision training
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
```

## Next Steps

- **[Quick Start](quickstart.md)** - Run your first benchmark
- **[Docker Setup](docker.md)** - Set up containerized models
- **[Model Integration](../user-guide/model-integration.md)** - Add new models