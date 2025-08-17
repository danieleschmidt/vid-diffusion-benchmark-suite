# Deployment Guide - Enhanced Research Framework

## Overview

This guide provides comprehensive instructions for deploying the enhanced Video Diffusion Benchmark Suite in various environments, from development to production research clusters.

## Quick Start

### Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Hardware**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM (recommended)
- **Software**: Docker 20.10+, Docker Compose 2.0+, Python 3.10+

### 1-Command Research Deployment

```bash
# Clone and deploy research environment
git clone https://github.com/terragon-labs/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite
docker-compose -f docker-compose.research.yml up -d
```

Access services:
- **Main API**: http://localhost:8000
- **Research Dashboard**: http://localhost:3000 (Grafana)
- **Jupyter Environment**: http://localhost:8888
- **Model Registry**: http://localhost:9001 (MinIO)

## Environment-Specific Deployments

### Development Environment

```bash
# Development setup with hot reloading
docker-compose -f docker-compose.dev.yml up -d

# Or run locally
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
python -m vid_diffusion_bench.api.app
```

### Production Environment

```bash
# Production deployment
export ENVIRONMENT=production
docker-compose -f docker-compose.production.yml up -d
```

## Monitoring and Troubleshooting

### Health Checks

```bash
# API health check
curl localhost:8000/health

# Comprehensive health check
python scripts/healthcheck.py
```

### Common Issues

**GPU Memory Issues**:
```bash
# Check GPU memory
nvidia-smi

# Clear CUDA cache
docker exec research-api python -c "import torch; torch.cuda.empty_cache()"
```

## Support

- **Documentation**: https://docs.vid-diffusion-bench.com
- **GitHub Issues**: https://github.com/terragon-labs/vid-diffusion-benchmark-suite/issues

---

This guide provides instructions for deployment scenarios. For detailed configurations, refer to the comprehensive documentation.