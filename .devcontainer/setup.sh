#!/bin/bash
set -e

echo "🚀 Setting up Video Diffusion Benchmark Suite development environment..."

# Update system packages
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Install additional development tools
pip install \
    black \
    isort \
    pylint \
    mypy \
    pre-commit \
    jupyter \
    jupyterlab \
    streamlit

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
mkdir -p /workspace/{models,results,logs,data}

# Download sample models (if needed)
echo "📥 Setting up model cache directories..."
mkdir -p /workspace/models/{svd-xt,svd-base,mock-models}

# Setup git configuration (if not already set)
if [ -z "$(git config --global user.name)" ]; then
    git config --global user.name "Dev Container User"
    git config --global user.email "dev@vid-diffusion-bench.local"
fi

# Create .env file from template if it doesn't exist
if [ ! -f /workspace/.env ]; then
    echo "📝 Creating .env file from template..."
    cp /workspace/.env.example /workspace/.env 2>/dev/null || echo "# Environment variables" > /workspace/.env
fi

# Run initial tests to verify setup
echo "🧪 Running setup verification tests..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test import of our package
python -c "from vid_diffusion_bench import BenchmarkSuite; print('✅ Package imports successfully')"

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  make test          # Run tests"
echo "  make lint          # Run linting"
echo "  make format        # Format code"
echo "  make dev           # Start development server"
echo "  streamlit run dashboard/app.py --server.port 8501"
echo ""
echo "📚 Documentation: https://vid-diffusion-bench.readthedocs.io"
echo "🐛 Issues: https://github.com/yourusername/vid-diffusion-benchmark-suite/issues"