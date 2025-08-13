# Deployment Guide - Video Diffusion Benchmark Suite

## Overview

This guide covers production deployment of the Video Diffusion Benchmark Suite using containerization and Kubernetes orchestration.

## Prerequisites

- Docker 24.0+
- Kubernetes 1.25+
- kubectl configured
- GPU-enabled nodes (NVIDIA GPUs recommended)
- 32GB+ RAM per node
- 1TB+ storage for models and results

## Quick Start

### 1. Build Container

```bash
# Build production image
docker build -t vid-diffusion-bench:latest .

# Build with specific tag
docker build -t vid-diffusion-bench:v1.0.0 .
```

### 2. Run Locally

```bash
# Run basic container
docker run -p 8000:8000 vid-diffusion-bench:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 vid-diffusion-bench:latest

# Run with environment variables
docker run -e VID_BENCH_API_KEY=your_key \
  -e VID_BENCH_ENV=production \
  -p 8000:8000 vid-diffusion-bench:latest
```

### 3. Docker Compose

```bash
# Start full stack
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=3
```

## Kubernetes Deployment

### 1. Deploy Base Infrastructure

```bash
# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Deploy monitoring
kubectl apply -f k8s/monitoring/
```

### 2. Deploy Application

```bash
# Deploy main application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Deploy workers
kubectl apply -f k8s/worker-deployment.yaml

# Setup auto-scaling
kubectl apply -f k8s/hpa.yaml
```

### 3. Configure Ingress

```bash
# Deploy ingress controller (if needed)
kubectl apply -f k8s/ingress-controller.yaml

# Configure ingress
kubectl apply -f k8s/ingress.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VID_BENCH_ENV` | Environment (dev/staging/prod) | dev | No |
| `VID_BENCH_API_KEY` | Master API key | None | Yes |
| `VID_BENCH_JWT_SECRET` | JWT signing secret | Random | No |
| `DATABASE_URL` | PostgreSQL connection string | None | Yes |
| `REDIS_URL` | Redis connection string | redis://localhost:6379 | No |
| `PROMETHEUS_URL` | Prometheus endpoint | None | No |
| `LOG_LEVEL` | Logging level | INFO | No |

### Resource Configuration

```yaml
# Recommended resource limits
resources:
  requests:
    cpu: "2000m"
    memory: "8Gi"
    nvidia.com/gpu: 1
  limits:
    cpu: "4000m" 
    memory: "16Gi"
    nvidia.com/gpu: 1
```

## Security

### TLS Configuration

```yaml
# TLS certificate
apiVersion: v1
kind: Secret
metadata:
  name: vid-bench-tls
  namespace: vid-diffusion-bench
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### Network Policies

```yaml
# Restrict network access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vid-bench-network-policy
spec:
  podSelector:
    matchLabels:
      app: vid-diffusion-bench
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

## Monitoring

### Prometheus Configuration

```yaml
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vid-bench-metrics
spec:
  selector:
    matchLabels:
      app: vid-diffusion-bench
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Grafana Dashboards

Import pre-built dashboards:
- System metrics (CPU, Memory, GPU)
- Application metrics (Request rate, Latency)
- Business metrics (Benchmark success rate)

### Alerting Rules

```yaml
# Critical alerts
groups:
- name: vid-diffusion-bench
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
  
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: High memory usage
```

## Scaling

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vid-bench-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vid-diffusion-bench
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaling

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vid-bench-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vid-diffusion-bench
  updatePolicy:
    updateMode: "Auto"
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
kubectl exec -n vid-diffusion-bench postgres-0 -- \
  pg_dump -U postgres vid_bench > backup_$(date +%Y%m%d_%H%M%S).sql

# Upload to cloud storage
aws s3 cp backup_*.sql s3://vid-bench-backups/
```

### Disaster Recovery

1. **RTO Target**: 30 minutes
2. **RPO Target**: 1 hour
3. **Backup Schedule**: Daily full, hourly incremental
4. **Multi-region replication** for critical data

## Performance Tuning

### GPU Optimization

```yaml
# GPU node selector
nodeSelector:
  accelerator: nvidia-tesla-v100

# GPU resource sharing
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1
```

### Memory Optimization

```yaml
# Memory-optimized settings
env:
- name: PYTORCH_CUDA_ALLOC_CONF
  value: "max_split_size_mb:512"
- name: OMP_NUM_THREADS
  value: "4"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Check memory usage
   kubectl top pods -n vid-diffusion-bench
   
   # Scale down workers
   kubectl scale deployment vid-bench-worker --replicas=1
   ```

2. **GPU Not Available**
   ```bash
   # Check GPU resources
   kubectl describe nodes | grep nvidia.com/gpu
   
   # Verify GPU driver
   kubectl exec -it <pod> -- nvidia-smi
   ```

3. **Database Connection Failed**
   ```bash
   # Check database pod
   kubectl logs -n vid-diffusion-bench postgres-0
   
   # Test connection
   kubectl exec -it <app-pod> -- pg_isready -h postgres
   ```

### Logs and Debugging

```bash
# Application logs
kubectl logs -f deployment/vid-diffusion-bench

# Worker logs
kubectl logs -f deployment/vid-bench-worker

# System logs
kubectl logs -f daemonset/gpu-driver-installer
```

## Health Checks

### Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Readiness Probe

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

## CI/CD Integration

### GitHub Actions Deployment

```yaml
name: Deploy to Production
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      run: |
        docker build -t ${{ secrets.REGISTRY }}/vid-diffusion-bench:${{ github.ref_name }} .
        docker push ${{ secrets.REGISTRY }}/vid-diffusion-bench:${{ github.ref_name }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/vid-diffusion-bench \
          app=${{ secrets.REGISTRY }}/vid-diffusion-bench:${{ github.ref_name }}
```

## Cost Optimization

### Resource Right-sizing

- Monitor actual resource usage
- Adjust requests/limits based on metrics
- Use spot instances for non-critical workloads

### Auto-scaling Policies

- Scale down during low-usage periods
- Use cluster autoscaler for node management
- Implement budget alerts

This deployment guide ensures robust, scalable, and secure production deployment of the Video Diffusion Benchmark Suite.