# Template Dockerfile for Video Diffusion Models
# Copy this template and customize for your specific model

ARG PYTORCH_VERSION=2.3.0
ARG CUDA_VERSION=12.1
ARG PYTHON_VERSION=3.11

# Use official PyTorch base image with CUDA support
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn8-devel as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash model
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model-specific files
COPY model_server.py .
COPY config.yaml .
COPY healthcheck.py .
COPY utils/ ./utils/

# Create necessary directories
RUN mkdir -p /app/models /app/cache /app/results /app/logs

# Change ownership to model user
RUN chown -R model:model /app
USER model

# Expose port for model server
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_NAME=template-model
ENV MODEL_VERSION=1.0.0
ENV CACHE_DIR=/app/cache
ENV RESULTS_DIR=/app/results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py || exit 1

# Default command
CMD ["python", "model_server.py"]

# Multi-stage build for production
FROM base as production

# Copy only necessary files for production
COPY --chown=model:model production_config.yaml ./config.yaml

# Set production environment variables
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV ENABLE_DEBUG=false

# Production command
CMD ["python", "model_server.py", "--config", "config.yaml"]

# Development stage with additional tools
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install debugging tools
RUN pip install ipdb jupyter tensorboard wandb

# Copy development configuration
COPY dev_config.yaml ./config.yaml

# Set development environment variables
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG
ENV ENABLE_DEBUG=true

# Expose additional ports for development
EXPOSE 8888 6006

# Development command
CMD ["python", "model_server.py", "--config", "config.yaml", "--debug"]