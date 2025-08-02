# Docker Configuration

This directory contains Docker configurations for the Video Diffusion Benchmark Suite.

## Directory Structure

```
docker/
├── README.md                    # This documentation
├── base/                        # Base images
│   ├── Dockerfile.pytorch       # PyTorch base image
│   ├── Dockerfile.cuda          # CUDA base image
│   └── Dockerfile.runtime       # Runtime base image
├── models/                      # Model-specific containers
│   ├── svd-xt/                  # Stable Video Diffusion XT
│   ├── pika-lumiere/            # Pika Labs Lumiere
│   ├── cogvideo/                # CogVideo model
│   └── template/                # Template for new models
├── services/                    # Service containers
│   ├── api/                     # REST API service
│   ├── dashboard/               # Streamlit dashboard
│   ├── worker/                  # Background worker
│   └── scheduler/               # Task scheduler
└── monitoring/                  # Monitoring stack
    ├── prometheus/              # Prometheus configuration
    ├── grafana/                 # Grafana dashboards
    └── alertmanager/            # Alert configuration
```

## Base Images

### PyTorch Base Image
- **Purpose**: Common base for PyTorch-based models
- **Size**: ~8GB (optimized)
- **Includes**: PyTorch, CUDA runtime, common ML libraries
- **Usage**: Extend for model-specific containers

### CUDA Base Image  
- **Purpose**: NVIDIA CUDA development environment
- **Size**: ~6GB
- **Includes**: CUDA toolkit, cuDNN, development tools
- **Usage**: For models requiring custom CUDA kernels

### Runtime Base Image
- **Purpose**: Lightweight runtime environment
- **Size**: ~2GB
- **Includes**: Python runtime, basic system tools
- **Usage**: For CPU-only services and utilities

## Model Containers

Each model has its own containerized environment to ensure:
- **Dependency Isolation**: No conflicts between model requirements
- **Reproducibility**: Consistent execution environment
- **Resource Management**: Controlled GPU and memory allocation
- **Security**: Sandboxed execution

### Creating Model Containers

1. **Copy Template**: Use `template/` directory as starting point
2. **Configure Dependencies**: Update requirements and system packages
3. **Add Model Code**: Include model-specific inference code
4. **Test Container**: Validate container builds and runs correctly
5. **Document**: Update README with model-specific information

### Example Model Container Structure

```
models/your-model/
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── model_server.py            # Model inference server
├── config.yaml                # Model configuration
├── healthcheck.py             # Health check script
└── README.md                  # Model documentation
```

## Service Containers

### API Service
- **Port**: 8000
- **Purpose**: REST API for programmatic access
- **Features**: Authentication, rate limiting, validation
- **Dependencies**: Redis for caching, PostgreSQL for persistence

### Dashboard Service
- **Port**: 8501 (Streamlit)
- **Purpose**: Interactive web dashboard
- **Features**: Model comparison, visualization, real-time updates
- **Dependencies**: API service, Redis for session state

### Worker Service
- **Purpose**: Background task processing
- **Features**: Model evaluation, metric calculation, result processing
- **Dependencies**: Redis for task queue, shared storage for results

### Scheduler Service
- **Purpose**: Cron-like task scheduling
- **Features**: Automated benchmarks, cleanup tasks, reporting
- **Dependencies**: Worker service, database for state management

## Build and Deployment

### Local Development

```bash
# Build all services
docker-compose -f docker-compose.dev.yml build

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Scale workers
docker-compose -f docker-compose.dev.yml up -d --scale worker=4
```

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Health check
docker-compose -f docker-compose.prod.yml ps
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n vid-benchmark

# Scale deployment
kubectl scale deployment benchmark-api --replicas=3
```

## Configuration

### Environment Variables

Key environment variables for containers:

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@db:5432/vid_benchmark
REDIS_URL=redis://redis:6379/0

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Model Configuration
MODEL_CACHE_DIR=/app/cache/models
RESULTS_DIR=/app/results
MAX_CONCURRENT_EVALUATIONS=4

# Security
API_KEY=your_secret_api_key
JWT_SECRET=your_jwt_secret
CORS_ORIGINS=https://yourdomain.com

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
```

### Resource Limits

Default resource limits for containers:

```yaml
# API Service
resources:
  limits:
    memory: 2Gi
    cpu: 1000m
  requests:
    memory: 1Gi
    cpu: 500m

# Model Containers
resources:
  limits:
    memory: 32Gi
    cpu: 4000m
    nvidia.com/gpu: 1
  requests:
    memory: 16Gi
    cpu: 2000m
    nvidia.com/gpu: 1

# Worker Service
resources:
  limits:
    memory: 16Gi
    cpu: 8000m
  requests:
    memory: 8Gi
    cpu: 4000m
```

## Health Checks

All containers include health checks for monitoring:

### API Service Health Check
```python
# /health endpoint
{
    "status": "healthy",
    "timestamp": "2025-08-02T12:00:00Z",
    "version": "1.0.0",
    "dependencies": {
        "database": "healthy",
        "redis": "healthy",
        "gpu": "available"
    }
}
```

### Model Container Health Check
```python
# Model inference health check
{
    "model_name": "svd-xt",
    "status": "ready",
    "gpu_memory_used": "12.5GB",
    "gpu_memory_total": "24GB",
    "last_inference": "2025-08-02T11:59:30Z"
}
```

## Security Best Practices

### Container Security
1. **Non-root User**: All containers run as non-root user
2. **Minimal Base Images**: Use slim/alpine variants when possible
3. **Secrets Management**: Use Docker secrets or Kubernetes secrets
4. **Network Security**: Isolate containers using networks
5. **Image Scanning**: Regular vulnerability scanning

### Access Control
```yaml
# Example security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL
```

## Monitoring and Logging

### Prometheus Metrics
- Container resource usage
- Model inference latency
- API request metrics
- Error rates and health status

### Centralized Logging
```yaml
logging:
  driver: "fluentd"
  options:
    fluentd-address: "logging.example.com:24224"
    tag: "vid-benchmark.{{.Name}}"
```

### Grafana Dashboards
- System overview dashboard
- Model performance dashboard  
- Error monitoring dashboard
- Resource utilization dashboard

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Check NVIDIA runtime
   docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi
   ```

2. **Out of Memory**
   ```bash
   # Check container memory usage
   docker stats
   
   # Adjust memory limits
   docker-compose up -d --scale worker=2
   ```

3. **Model Loading Fails**
   ```bash
   # Check model container logs
   docker-compose logs model-container
   
   # Test model manually
   docker exec -it model-container python test_model.py
   ```

4. **Network Connectivity**
   ```bash
   # Test service connectivity
   docker-compose exec api curl http://model-container:8000/health
   ```

### Performance Optimization

1. **Multi-stage Builds**: Reduce image size
2. **Layer Caching**: Optimize Dockerfile layer order
3. **Resource Tuning**: Adjust CPU/memory limits
4. **GPU Optimization**: Use appropriate CUDA/cuDNN versions

### Debugging

```bash
# Enter container for debugging
docker-compose exec benchmark bash

# Check container logs
docker-compose logs -f benchmark

# Monitor resource usage
docker-compose exec benchmark htop

# Profile GPU usage
docker-compose exec benchmark nvidia-smi -l 1
```

## CI/CD Integration

### Build Pipeline
```yaml
# GitHub Actions example
- name: Build Docker images
  run: |
    docker build -t vid-benchmark:${{ github.sha }} .
    docker tag vid-benchmark:${{ github.sha }} vid-benchmark:latest

- name: Run tests in container
  run: |
    docker run --rm vid-benchmark:${{ github.sha }} pytest

- name: Push to registry
  run: |
    docker push vid-benchmark:${{ github.sha }}
    docker push vid-benchmark:latest
```

### Automated Deployment
```bash
# Deploy using Docker Compose
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --remove-orphans

# Deploy using Kubernetes
kubectl set image deployment/benchmark-api api=vid-benchmark:${{ github.sha }}
kubectl rollout status deployment/benchmark-api
```

For more information, see the main [README.md](../README.md) and [deployment documentation](../docs/DEPLOYMENT.md).