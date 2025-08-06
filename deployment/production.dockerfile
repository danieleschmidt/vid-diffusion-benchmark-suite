# Production-Optimized Multi-stage Dockerfile
# Optimized for scalable video diffusion benchmarking with research capabilities

FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    redis-server \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to use python3.11 as python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1

# Builder stage for dependencies
FROM base as builder

WORKDIR /build

# Copy requirements and install dependencies
COPY pyproject.toml requirements-dev.txt ./
COPY src/ ./src/

# Upgrade pip and install build dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the package and dependencies
RUN pip3 install --no-cache-dir -e . && \
    pip3 install --no-cache-dir -r requirements-dev.txt

# Production stage
FROM base as production

# Create application user
RUN groupadd -r benchmarkapp && useradd -r -g benchmarkapp -d /app -s /bin/bash benchmarkapp

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --from=builder /build/src ./src

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/cache /app/logs /app/models \
    && chown -R benchmarkapp:benchmarkapp /app

# Switch to application user
USER benchmarkapp

# Set Python path
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1

# Create configuration directory
RUN mkdir -p /app/config

# Copy configuration files
COPY deployment/config/ /app/config/

# Health check script
COPY deployment/healthcheck.py /app/healthcheck.py

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python3 /app/healthcheck.py || exit 1

# Expose ports
EXPOSE 8000 8080 9090

# Set default command
CMD ["python3", "-m", "vid_diffusion_bench.api.app"]

# Development stage (for debugging)
FROM production as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip3 install --no-cache-dir \
    jupyter \
    ipython \
    debugpy \
    pytest-xdist \
    pytest-cov

USER benchmarkapp

# Expose additional ports for development
EXPOSE 8888 5678

CMD ["bash"]

# Distributed worker stage
FROM production as worker

# Additional dependencies for distributed computing
USER root
RUN pip3 install --no-cache-dir \
    ray[default] \
    celery[redis] \
    dask[complete]

USER benchmarkapp

# Worker-specific configuration
ENV WORKER_TYPE=distributed_worker
ENV WORKER_CONCURRENCY=4

CMD ["python3", "-m", "vid_diffusion_bench.scaling.distributed", "--worker"]

# API server stage  
FROM production as api-server

# API-specific configuration
ENV SERVER_TYPE=api_server
ENV PORT=8000
ENV WORKERS=4

CMD ["python3", "-m", "uvicorn", "vid_diffusion_bench.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Monitor stage
FROM production as monitor

# Monitoring-specific dependencies
USER root
RUN pip3 install --no-cache-dir \
    grafana-api \
    prometheus-client

USER benchmarkapp

# Monitoring configuration
ENV SERVICE_TYPE=monitor
ENV GRAFANA_URL=http://grafana:3000
ENV PROMETHEUS_URL=http://prometheus:9090

CMD ["python3", "-m", "vid_diffusion_bench.robustness.monitoring"]