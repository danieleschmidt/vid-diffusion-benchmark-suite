#!/bin/bash
set -euo pipefail

# Video Diffusion Benchmark Suite Setup Script
# This script sets up the development environment and dependencies

echo "🚀 Setting up Video Diffusion Benchmark Suite..."

# Check system requirements
check_requirements() {
    echo "📋 Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$python_version >= 3.10" | bc -l) -eq 0 ]]; then
        echo "❌ Python 3.10+ is required, found $python_version"
        exit 1
    fi
    
    echo "✅ Python $python_version found"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "⚠️  Docker not found - some features will be unavailable"
    else
        echo "✅ Docker found"
    fi
    
    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "⚠️  No NVIDIA GPU detected - CPU-only mode"
    fi
}

# Setup Python environment
setup_python_env() {
    echo "🐍 Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install the package in development mode
    echo "Installing vid-diffusion-benchmark-suite..."
    pip install -e ".[dev]"
    
    echo "✅ Python environment setup complete"
}

# Setup pre-commit hooks
setup_precommit() {
    echo "🔧 Setting up pre-commit hooks..."
    
    source venv/bin/activate
    pre-commit install
    
    echo "✅ Pre-commit hooks installed"
}

# Download model weights (optional)
download_models() {
    if [ "${DOWNLOAD_MODELS:-false}" = "true" ]; then
        echo "📥 Downloading model weights..."
        
        mkdir -p models
        
        # Download small test models for development
        python -c "
import torch
from diffusers import StableVideoDiffusionPipeline

# Download SVD model for testing (smallest available)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    torch_dtype=torch.float16
)
pipe.save_pretrained('models/svd-xt')
print('✅ SVD model downloaded')
"
    else
        echo "⏭️  Skipping model download (set DOWNLOAD_MODELS=true to enable)"
    fi
}

# Setup Docker environment
setup_docker() {
    if command -v docker &> /dev/null; then
        echo "🐳 Setting up Docker environment..."
        
        # Build main image
        docker build -t vid-diffusion-benchmark:latest .
        
        # Pull supporting services
        docker-compose pull redis prometheus grafana
        
        echo "✅ Docker environment setup complete"
    else
        echo "⏭️  Docker not available, skipping Docker setup"
    fi
}

# Create necessary directories
create_directories() {
    echo "📁 Creating directory structure..."
    
    mkdir -p {results,models,logs,monitoring/rules,dashboard}
    
    echo "✅ Directory structure created"
}

# Run tests to verify installation
run_tests() {
    echo "🧪 Running tests to verify installation..."
    
    source venv/bin/activate
    
    # Run quick tests only
    pytest tests/ -v -m "not slow and not gpu" --tb=short
    
    if [ $? -eq 0 ]; then
        echo "✅ All tests passed"
    else
        echo "⚠️  Some tests failed - check output above"
    fi
}

# Main setup function
main() {
    echo "🎬 Video Diffusion Benchmark Suite Setup"
    echo "========================================"
    
    check_requirements
    create_directories
    setup_python_env
    setup_precommit
    download_models
    setup_docker
    run_tests
    
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run a quick test: vid-bench --help"
    echo "3. Start the dashboard: streamlit run dashboard/app.py"
    echo "4. Read the documentation: docs/README.md"
    echo ""
    echo "For development:"
    echo "- Run tests: make test"
    echo "- Format code: make format"
    echo "- Start services: docker-compose up -d"
    echo ""
}

# Handle command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "test")
        source venv/bin/activate
        run_tests
        ;;
    "docker")
        setup_docker
        ;;
    "models")
        DOWNLOAD_MODELS=true download_models
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        rm -rf venv/ .pytest_cache/ __pycache__/ *.egg-info/
        docker system prune -f
        echo "✅ Cleanup complete"
        ;;
    *)
        echo "Usage: $0 [setup|test|docker|models|clean]"
        echo ""
        echo "Commands:"
        echo "  setup  - Full setup (default)"
        echo "  test   - Run tests only"
        echo "  docker - Setup Docker only"
        echo "  models - Download models only"
        echo "  clean  - Clean up generated files"
        ;;
esac