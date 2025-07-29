# Multi-stage Docker build for Video Diffusion Benchmark Suite
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Production stage
FROM base as production

# Copy only necessary files
COPY pyproject.toml .
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Install production dependencies
RUN pip install -e .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
USER app

# Set entrypoint
ENTRYPOINT ["vid-bench"]
CMD ["--help"]