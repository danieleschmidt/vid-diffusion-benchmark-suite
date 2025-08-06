# Deployment Guide

## Overview

This guide covers comprehensive deployment strategies for the Video Diffusion Benchmark Suite, from development environments to production-grade distributed systems.

## Quick Start Deployment

### Docker Compose (Recommended)

```bash
# Clone and navigate to repository
git clone https://github.com/your-org/video-diffusion-benchmark
cd video-diffusion-benchmark

# Start complete stack
docker-compose -f deployment/docker-compose.prod.yml up -d

# Verify deployment
curl http://localhost:8000/api/v1/health
```

### Single Container

```bash
# Build production image
docker build -f deployment/production.dockerfile -t vid-diffusion-bench:latest .

# Run with GPU support
docker run -d \
  --name vid-bench \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  vid-diffusion-bench:latest
```

## Architecture Overview

### Core Components

```mermaid
graph TB
    LB[Load Balancer/Nginx] --> API[API Server]
    API --> DB[(PostgreSQL)]
    API --> CACHE[(Redis)]
    API --> WORKER[Benchmark Workers]
    API --> MONITOR[Monitoring]
    WORKER --> GPU[GPU Nodes]
    MONITOR --> PROM[Prometheus]
    MONITOR --> GRAF[Grafana]
```

### Service Dependencies

- **API Server**: FastAPI application serving REST endpoints
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for temporary data and job queues
- **Workers**: Compute nodes for benchmark execution
- **Monitoring**: Prometheus + Grafana for observability
- **Load Balancer**: Nginx for request routing and SSL termination

## Production Deployment

### Prerequisites

1. **Hardware Requirements**
   - CPU: 8+ cores, 16+ GB RAM recommended
   - GPU: NVIDIA GPU with 8+ GB VRAM (RTX 3080 or better)
   - Storage: 100+ GB SSD for models and results
   - Network: Stable internet for model downloads

2. **Software Requirements**
   - Docker 20.10+
   - Docker Compose 2.0+
   - NVIDIA Container Toolkit
   - Python 3.9+ (for development)

### Environment Configuration

Create production environment file:

```bash
# deployment/.env.prod
DATABASE_URL=postgresql://user:password@postgres:5432/benchmark_db
REDIS_URL=redis://redis:6379
SECRET_KEY=your-secure-secret-key-here
ENVIRONMENT=production
LOG_LEVEL=INFO
GPU_MEMORY_FRACTION=0.8
MAX_WORKERS=4
RATE_LIMIT_PER_HOUR=1000
```

### SSL Configuration

1. **Generate SSL certificates** (using Let's Encrypt):
```bash
# Install certbot
sudo apt install certbot

# Generate certificates
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates to deployment directory
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem deployment/ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem deployment/ssl/
```

2. **Update Nginx configuration**:
```nginx
# deployment/nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    
    # ... rest of configuration
}
```

### Production Deployment Steps

1. **Prepare the environment**:
```bash
# Create production directories
mkdir -p /opt/vid-diffusion-bench/{data,models,logs,ssl}

# Set proper permissions
sudo chown -R $USER:$USER /opt/vid-diffusion-bench

# Copy SSL certificates
cp deployment/ssl/* /opt/vid-diffusion-bench/ssl/
```

2. **Deploy with Docker Compose**:
```bash
# Deploy production stack
docker-compose -f deployment/docker-compose.prod.yml up -d

# Verify all services are running
docker-compose -f deployment/docker-compose.prod.yml ps

# Check logs
docker-compose -f deployment/docker-compose.prod.yml logs -f
```

3. **Initialize database**:
```bash
# Run database migrations
docker-compose -f deployment/docker-compose.prod.yml exec api_server \
  python -m vid_diffusion_bench.database.migrations upgrade

# Create initial user
docker-compose -f deployment/docker-compose.prod.yml exec api_server \
  python -m vid_diffusion_bench.cli create-user --admin
```

4. **Verify deployment**:
```bash
# Health check
curl https://your-domain.com/api/v1/health

# API documentation
curl https://your-domain.com/docs

# Metrics endpoint
curl https://your-domain.com/api/v1/monitoring/metrics
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster 1.20+
- kubectl configured
- NVIDIA GPU Operator installed
- Persistent storage provisioner

### Deploy to Kubernetes

1. **Create namespace and secrets**:
```bash
# Create namespace
kubectl create namespace vid-diffusion-bench

# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=DATABASE_URL="postgresql://user:password@postgres:5432/benchmark_db" \
  --from-literal=REDIS_URL="redis://redis:6379" \
  --from-literal=SECRET_KEY="your-secure-secret-key" \
  -n vid-diffusion-bench
```

2. **Deploy using Helm** (recommended):
```bash
# Add repository
helm repo add vid-diffusion-bench https://your-org.github.io/vid-diffusion-bench-helm

# Install
helm install benchmark vid-diffusion-bench/vid-diffusion-bench \
  --namespace vid-diffusion-bench \
  --values deployment/kubernetes/values.prod.yaml
```

3. **Deploy using kubectl**:
```bash
# Apply all manifests
kubectl apply -k deployment/kubernetes/overlays/production/
```

### Kubernetes Configuration

**values.prod.yaml**:
```yaml
# Production Helm values
replicaCount: 3

image:
  repository: vid-diffusion-bench
  tag: "latest"
  pullPolicy: Always

resources:
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: 1
  requests:
    memory: "4Gi"
    cpu: "2"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: 100m
  hosts:
    - host: benchmark.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: benchmark-tls
      hosts:
        - benchmark.your-domain.com

postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-password"
    database: "benchmark_db"
  persistence:
    size: 20Gi

redis:
  enabled: true
  auth:
    enabled: false
  persistence:
    size: 5Gi

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

## Distributed Deployment

### Multi-Node Setup

For large-scale benchmarking, deploy across multiple nodes:

1. **Cluster Manager Node**:
```yaml
# docker-compose.cluster-manager.yml
version: '3.8'
services:
  cluster-manager:
    image: vid-diffusion-bench:latest
    command: ["python", "-m", "vid_diffusion_bench.scaling.cluster_manager"]
    environment:
      - ROLE=cluster_manager
      - CLUSTER_PORT=8765
    ports:
      - "8765:8765"
    networks:
      - benchmark_cluster
```

2. **Worker Nodes**:
```yaml
# docker-compose.worker.yml
version: '3.8'
services:
  worker:
    image: vid-diffusion-bench:latest
    command: ["python", "-m", "vid_diffusion_bench.scaling.worker"]
    environment:
      - ROLE=worker
      - CLUSTER_MANAGER_HOST=cluster-manager
      - CLUSTER_MANAGER_PORT=8765
      - GPU_DEVICE=${GPU_DEVICE:-0}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU_DEVICE:-0}"]
              capabilities: [gpu]
    networks:
      - benchmark_cluster
```

3. **Start distributed cluster**:
```bash
# Start cluster manager
docker-compose -f docker-compose.cluster-manager.yml up -d

# Start workers on different nodes
GPU_DEVICE=0 docker-compose -f docker-compose.worker.yml up -d
GPU_DEVICE=1 docker-compose -f docker-compose.worker.yml up -d
```

### Ray Cluster Deployment

For advanced distributed computing with Ray:

```yaml
# docker-compose.ray.yml
version: '3.8'
services:
  ray-head:
    image: vid-diffusion-bench:ray
    command: ["ray", "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0"]
    ports:
      - "8265:8265"  # Dashboard
      - "10001:10001"
      - "6379:6379"
    environment:
      - RAY_DISABLE_IMPORT_WARNING=1

  ray-worker:
    image: vid-diffusion-bench:ray
    command: ["ray", "start", "--address=ray-head:6379"]
    depends_on:
      - ray-head
    deploy:
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Cloud Deployment

### AWS EKS Deployment

1. **Create EKS cluster**:
```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster with GPU nodes
eksctl create cluster \
  --name vid-diffusion-bench \
  --region us-west-2 \
  --nodes 2 \
  --node-type p3.2xlarge \
  --managed
```

2. **Install NVIDIA device plugin**:
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.12.0/nvidia-device-plugin.yml
```

3. **Deploy application**:
```bash
helm install benchmark ./deployment/helm/vid-diffusion-bench \
  --namespace vid-diffusion-bench \
  --create-namespace \
  --values deployment/helm/values.aws.yaml
```

### Google Cloud GKE Deployment

1. **Create GKE cluster**:
```bash
# Create cluster with GPU nodes
gcloud container clusters create vid-diffusion-bench \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --num-nodes 2 \
  --machine-type n1-standard-4 \
  --zone us-central1-a
```

2. **Install GPU drivers**:
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Azure AKS Deployment

1. **Create AKS cluster**:
```bash
# Create resource group
az group create --name vid-diffusion-bench --location eastus

# Create AKS cluster with GPU nodes
az aks create \
  --resource-group vid-diffusion-bench \
  --name vid-diffusion-bench \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --generate-ssh-keys
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'vid-diffusion-bench'
    static_configs:
      - targets: ['api_server:8000']
    metrics_path: '/api/v1/metrics/prometheus'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

Import pre-configured dashboards:

```bash
# Import dashboards
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboard.json
```

### Health Checks

The deployment includes comprehensive health checks:

```python
# deployment/healthcheck.py
#!/usr/bin/env python3
"""Comprehensive health check script for deployment verification."""

import sys
import requests
import subprocess
import psutil
import time
from pathlib import Path

def check_api_health():
    """Check API server health."""
    try:
        response = requests.get('http://localhost:8000/api/v1/health', timeout=10)
        return response.status_code == 200
    except:
        return False

def check_database_connection():
    """Check database connectivity."""
    try:
        result = subprocess.run(
            ['docker-compose', 'exec', '-T', 'postgres', 'pg_isready'],
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0
    except:
        return False

def check_gpu_availability():
    """Check GPU availability."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0 and len(result.stdout) > 0
    except:
        return False

def check_system_resources():
    """Check system resource availability."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'memory_available': memory.available > 4 * 1024**3,  # 4GB
        'disk_free': disk.free > 10 * 1024**3,  # 10GB
        'cpu_count': psutil.cpu_count() >= 2
    }

def main():
    """Run comprehensive health checks."""
    checks = {
        'API Health': check_api_health(),
        'Database Connection': check_database_connection(),
        'GPU Availability': check_gpu_availability(),
        'System Resources': all(check_system_resources().values())
    }
    
    all_passed = all(checks.values())
    
    print("Health Check Results:")
    print("-" * 40)
    for check, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check}: {status}")
    
    print("-" * 40)
    overall = "✅ HEALTHY" if all_passed else "❌ UNHEALTHY"
    print(f"Overall Status: {overall}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

## Scaling Strategies

### Horizontal Scaling

1. **API Server Scaling**:
```yaml
# Scale API servers
docker-compose -f deployment/docker-compose.prod.yml up -d --scale api_server=3
```

2. **Worker Scaling**:
```yaml
# Scale benchmark workers
docker-compose -f deployment/docker-compose.prod.yml up -d --scale benchmark_worker=5
```

### Vertical Scaling

Update resource limits in docker-compose or Kubernetes configurations:

```yaml
# Increase resources
services:
  api_server:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
```

### Auto-scaling

Configure auto-scaling based on metrics:

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vid-diffusion-bench-hpa
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

## Security Configuration

### Network Security

1. **Firewall Rules**:
```bash
# Allow only necessary ports
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 22/tcp    # SSH
sudo ufw deny 8000/tcp   # Block direct API access
```

2. **Network Segmentation**:
```yaml
# Docker networks
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access
```

### Authentication and Authorization

Configure OAuth2 or JWT authentication:

```yaml
# Environment variables
ENABLE_AUTHENTICATION=true
JWT_SECRET_KEY=your-jwt-secret
OAUTH_CLIENT_ID=your-oauth-client-id
OAUTH_CLIENT_SECRET=your-oauth-client-secret
```

### Data Encryption

1. **Database encryption**:
```yaml
postgres:
  environment:
    - POSTGRES_INITDB_ARGS=--auth-host=md5
  command: |
    -c ssl=on
    -c ssl_cert_file=/etc/ssl/certs/server.crt
    -c ssl_key_file=/etc/ssl/private/server.key
```

2. **TLS for all connections**:
```nginx
# Force HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## Backup and Disaster Recovery

### Database Backups

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create database backup
docker-compose -f deployment/docker-compose.prod.yml exec -T postgres \
  pg_dump -U postgres benchmark_db > "$BACKUP_DIR/db_backup_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/db_backup_$DATE.sql"

# Upload to cloud storage (optional)
aws s3 cp "$BACKUP_DIR/db_backup_$DATE.sql.gz" s3://your-backup-bucket/
```

### Model and Data Backups

```bash
# Backup models and results
rsync -av --delete /opt/vid-diffusion-bench/models/ backup-server:/backups/models/
rsync -av --delete /opt/vid-diffusion-bench/data/ backup-server:/backups/data/
```

### Recovery Procedures

1. **Database Recovery**:
```bash
# Stop services
docker-compose -f deployment/docker-compose.prod.yml down

# Restore database
gunzip -c backup_file.sql.gz | docker-compose -f deployment/docker-compose.prod.yml exec -T postgres psql -U postgres -d benchmark_db

# Start services
docker-compose -f deployment/docker-compose.prod.yml up -d
```

2. **Complete System Recovery**:
```bash
# Restore from backup
./scripts/restore_system.sh --backup-date 20240115 --include-models --include-data

# Verify restoration
./deployment/healthcheck.py
```

## Performance Optimization

### Database Optimization

```sql
-- PostgreSQL optimizations
-- /deployment/postgres/postgresql.conf
shared_buffers = '256MB'
effective_cache_size = '1GB'
maintenance_work_mem = '64MB'
checkpoint_completion_target = 0.9
wal_buffers = '16MB'
default_statistics_target = 100
```

### Redis Optimization

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Application Optimization

```python
# Performance configuration
WORKER_CONNECTIONS=1000
MAX_REQUESTS=10000
MAX_REQUESTS_JITTER=100
PRELOAD_APP=true
WORKER_CLASS=uvicorn.workers.UvicornWorker
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**:
```bash
# Check GPU usage
nvidia-smi

# Adjust memory fraction
echo "GPU_MEMORY_FRACTION=0.6" >> .env.prod
```

2. **Database Connection Issues**:
```bash
# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U postgres -d benchmark_db -c "SELECT 1;"
```

3. **Performance Issues**:
```bash
# Monitor system resources
htop
iotop
nvidia-smi -l 1

# Check application logs
docker-compose logs api_server
```

### Log Analysis

```bash
# Centralized logging with ELK stack
docker-compose -f deployment/docker-compose.logging.yml up -d

# View logs in Kibana
open http://localhost:5601
```

## Maintenance

### Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash

# Clean up old containers
docker system prune -f

# Update Docker images
docker-compose -f deployment/docker-compose.prod.yml pull

# Restart services with zero downtime
docker-compose -f deployment/docker-compose.prod.yml up -d --no-deps api_server

# Run health checks
./deployment/healthcheck.py

# Clean up old benchmark results (keep last 30 days)
find /opt/vid-diffusion-bench/data/results -mtime +30 -type f -delete

# Update SSL certificates
certbot renew --quiet
```

### Updates and Upgrades

```bash
# Rolling update procedure
./scripts/rolling_update.sh --version v2.0.0 --strategy blue-green
```

This deployment guide provides comprehensive instructions for deploying the Video Diffusion Benchmark Suite in various environments, from development to large-scale production deployments.