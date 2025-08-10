# Production Deployment Guide - Video Diffusion Benchmark Suite

## 🎯 Overview

This guide provides comprehensive instructions for deploying the Video Diffusion Benchmark Suite in production environments, from single-node setups to large-scale distributed clusters.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPUs with CUDA 11.8+ (optional but recommended)
- **Memory**: Minimum 16GB RAM, 32GB+ recommended
- **Storage**: 100GB+ available space
- **Network**: High-bandwidth internet connection for model downloads

### Hardware Recommendations

#### Single Node Configuration
```
CPU: 16+ cores (Intel Xeon or AMD EPYC)
GPU: NVIDIA A100 (40GB/80GB) or RTX 4090
RAM: 64GB DDR4/DDR5
Storage: 1TB NVMe SSD
Network: 10Gbps Ethernet
```

#### Multi-Node Cluster
```
Master Node:
- CPU: 32+ cores
- RAM: 128GB
- Storage: 2TB NVMe SSD
- Network: 25Gbps+ with InfiniBand preferred

Worker Nodes (each):
- CPU: 16+ cores  
- GPU: 2-8x NVIDIA A100/H100
- RAM: 256GB+
- Storage: 1TB NVMe SSD
- Network: 25Gbps+ with InfiniBand
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

1. **Clone Repository**
```bash
git clone https://github.com/danieleschmidt/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env file with your configuration
```

3. **Start Services**
```bash
# Development environment
docker-compose up -d

# Production environment
docker-compose -f docker-compose.prod.yml up -d
```

### Production Docker Configuration

**docker-compose.prod.yml**
```yaml
version: '3.8'

services:
  benchmark-coordinator:
    image: vid-bench/coordinator:${VERSION:-latest}
    ports:
      - "8080:8080"
    environment:
      - MODE=production
      - SECURITY_ENABLED=true
      - I18N_ENABLED=true
      - DATABASE_URL=postgresql://user:pass@db:5432/benchmarks
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - benchmark-data:/app/data
    depends_on:
      - database
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  benchmark-worker:
    image: vid-bench/worker:${VERSION:-latest}
    deploy:
      replicas: 4
    environment:
      - COORDINATOR_URL=http://coordinator:8080
      - GPU_ENABLED=true
      - WORKER_ID=${HOSTNAME}
    volumes:
      - ./models:/app/models:ro
      - ./cache:/app/cache
      - benchmark-results:/app/results
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    depends_on:
      - benchmark-coordinator
    restart: unless-stopped

  database:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=benchmarks
      - POSTGRES_USER=benchmarkuser
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d:ro
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

  monitoring:
    image: vid-bench/monitoring:${VERSION:-latest}
    ports:
      - "3000:3000"
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - GRAFANA_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

volumes:
  benchmark-data:
  benchmark-results:
  postgres-data:
  redis-data:
  grafana-data:
  prometheus-data:

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## ☸️ Kubernetes Deployment

### Helm Chart Installation

1. **Add Helm Repository**
```bash
helm repo add vid-bench https://charts.vid-diffusion-bench.org
helm repo update
```

2. **Create Namespace**
```bash
kubectl create namespace vid-benchmark
```

3. **Install Chart**
```bash
helm install vid-benchmark vid-bench/vid-diffusion-benchmark \
  --namespace vid-benchmark \
  --values values.prod.yaml
```

### Kubernetes Manifests

**namespace.yaml**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vid-benchmark
  labels:
    name: vid-benchmark
```

**configmap.yaml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: benchmark-config
  namespace: vid-benchmark
data:
  config.yaml: |
    benchmark:
      default_models: ["svd-xt", "cogvideo"]
      max_concurrent_jobs: 10
      timeout_minutes: 60
    
    security:
      enable_auth: true
      rate_limiting: true
      max_requests_per_minute: 100
    
    internationalization:
      default_locale: "en"
      supported_locales: ["en", "es", "fr", "de", "ja", "zh-CN"]
    
    performance:
      optimization_level: "balanced"
      enable_gpu_optimization: true
      memory_efficient: true
```

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: benchmark-coordinator
  namespace: vid-benchmark
spec:
  replicas: 2
  selector:
    matchLabels:
      app: benchmark-coordinator
  template:
    metadata:
      labels:
        app: benchmark-coordinator
    spec:
      containers:
      - name: coordinator
        image: vid-bench/coordinator:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODE
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: benchmark-secrets
              key: database-url
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: benchmark-config
      - name: data
        persistentVolumeClaim:
          claimName: benchmark-data
```

**worker-deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: benchmark-worker
  namespace: vid-benchmark
spec:
  replicas: 4
  selector:
    matchLabels:
      app: benchmark-worker
  template:
    metadata:
      labels:
        app: benchmark-worker
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: worker
        image: vid-bench/worker:latest
        env:
        - name: COORDINATOR_URL
          value: "http://benchmark-coordinator:8080"
        - name: GPU_ENABLED
          value: "true"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: results
          mountPath: /app/results
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
      - name: results
        persistentVolumeClaim:
          claimName: results-storage
```

## 🌐 Cloud Platform Deployment

### AWS Deployment

#### Using AWS EKS
```bash
# Create EKS cluster
eksctl create cluster \
  --name vid-benchmark-cluster \
  --version 1.24 \
  --region us-west-2 \
  --nodegroup-name gpu-workers \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml

# Deploy application
kubectl apply -k k8s/overlays/aws/
```

#### CloudFormation Template
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Video Diffusion Benchmark Suite Infrastructure'

Parameters:
  InstanceType:
    Type: String
    Default: p3.2xlarge
    AllowedValues: [p3.2xlarge, p3.8xlarge, p4d.xlarge]

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      Tags:
        - Key: Name
          Value: vid-benchmark-vpc

  BenchmarkCluster:
    Type: AWS::EKS::Cluster
    Properties:
      Name: vid-benchmark-cluster
      Version: '1.24'
      RoleArn: !GetAtt EKSClusterRole.Arn
      ResourcesVpcConfig:
        SubnetIds: 
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2

  NodeGroup:
    Type: AWS::EKS::Nodegroup
    Properties:
      ClusterName: !Ref BenchmarkCluster
      NodegroupName: gpu-workers
      NodeRole: !GetAtt NodeInstanceRole.Arn
      InstanceTypes: 
        - !Ref InstanceType
      ScalingConfig:
        MinSize: 1
        MaxSize: 10
        DesiredSize: 2
```

### Google Cloud Platform
```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create vid-benchmark-cluster \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 2 \
  --zone us-west1-b \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Deploy application
kubectl apply -k k8s/overlays/gcp/
```

### Microsoft Azure
```bash
# Create AKS cluster with GPU support
az aks create \
  --resource-group vid-benchmark-rg \
  --name vid-benchmark-cluster \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group vid-benchmark-rg --name vid-benchmark-cluster

# Deploy NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/Azure/aks-engine/master/examples/addons/nvidia-device-plugin/nvidia-device-plugin.yaml
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Core Configuration
export VID_BENCH_MODE=production
export VID_BENCH_LOG_LEVEL=INFO
export VID_BENCH_DATA_DIR=/opt/vid-bench/data
export VID_BENCH_CONFIG_PATH=/opt/vid-bench/config

# Database Configuration
export DATABASE_URL=postgresql://user:pass@localhost:5432/benchmarks
export REDIS_URL=redis://localhost:6379

# Security Configuration  
export SECRET_KEY=your-secret-key-here
export JWT_SECRET=your-jwt-secret-here
export API_RATE_LIMIT=1000

# Performance Configuration
export GPU_MEMORY_FRACTION=0.8
export MAX_CONCURRENT_JOBS=10
export OPTIMIZATION_LEVEL=balanced

# Internationalization
export DEFAULT_LOCALE=en
export SUPPORTED_LOCALES=en,es,fr,de,ja,zh-CN

# Monitoring
export PROMETHEUS_ENDPOINT=http://prometheus:9090
export GRAFANA_ENDPOINT=http://grafana:3000
export JAEGER_ENDPOINT=http://jaeger:14268
```

### Configuration Files

**config/production.yaml**
```yaml
benchmark:
  default_timeout: 3600
  max_retries: 3
  batch_size: 4
  save_intermediate: false
  
models:
  cache_dir: /opt/vid-bench/models
  download_parallel: 4
  max_model_size_gb: 50
  
storage:
  results_retention_days: 90
  cleanup_temp_files: true
  compress_results: true
  
security:
  enable_authentication: true
  require_https: true
  cors_origins: []
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
  
monitoring:
  enable_metrics: true
  metrics_port: 8090
  health_check_interval: 30
  
logging:
  level: INFO
  format: json
  file: /opt/vid-bench/logs/app.log
  max_size_mb: 100
  backup_count: 5
```

## 📊 Monitoring and Observability

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vid-benchmark'
    static_configs:
      - targets: ['benchmark-coordinator:8090', 'benchmark-worker:8090']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    
  - job_name: 'gpu-exporter'
    static_configs:
      - targets: ['gpu-exporter:9445']
```

### Grafana Dashboards
Key metrics to monitor:
- **Throughput**: Videos generated per hour
- **Latency**: Average generation time per video
- **Resource Utilization**: GPU/CPU/Memory usage
- **Error Rates**: Failed generations and system errors
- **Queue Depth**: Pending benchmark tasks
- **Model Performance**: FVD, IS, CLIP scores over time

### Alerting Rules
```yaml
groups:
- name: vid-benchmark-alerts
  rules:
  - alert: HighLatency
    expr: benchmark_generation_duration_seconds > 300
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High video generation latency detected
      
  - alert: GPUMemoryHigh
    expr: gpu_memory_utilization > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: GPU memory usage critically high
      
  - alert: WorkerDown
    expr: up{job="vid-benchmark"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Benchmark worker is down
```

## 🔐 Security Configuration

### TLS/SSL Setup
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -subj "/CN=vid-benchmark.example.com"

# Create Kubernetes secret
kubectl create secret tls benchmark-tls \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=vid-benchmark
```

### Authentication Setup
```yaml
# OAuth2 Proxy configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oauth2-proxy
spec:
  template:
    spec:
      containers:
      - name: oauth2-proxy
        image: quay.io/oauth2-proxy/oauth2-proxy:latest
        args:
        - --provider=oidc
        - --client-id=your-client-id
        - --client-secret=your-client-secret
        - --cookie-secret=your-cookie-secret
        - --upstream=http://benchmark-coordinator:8080
        - --http-address=0.0.0.0:4180
```

## 🚀 Scaling and Performance

### Horizontal Scaling
```bash
# Scale coordinator replicas
kubectl scale deployment benchmark-coordinator --replicas=5

# Scale worker nodes
kubectl scale deployment benchmark-worker --replicas=10

# Enable auto-scaling
kubectl autoscale deployment benchmark-worker \
  --min=2 --max=20 --cpu-percent=70
```

### Vertical Scaling
```yaml
# Resource configuration for high-performance workloads
resources:
  requests:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "2"
  limits:
    memory: "64Gi"
    cpu: "16"
    nvidia.com/gpu: "2"
```

### Performance Tuning
```yaml
# GPU optimization settings
env:
- name: CUDA_VISIBLE_DEVICES
  value: "0,1"
- name: NVIDIA_DRIVER_CAPABILITIES
  value: "compute,utility"
- name: GPU_MEMORY_FRACTION
  value: "0.9"
- name: MIXED_PRECISION
  value: "true"
- name: COMPILE_MODELS
  value: "true"
```

## 🔄 Backup and Recovery

### Database Backup
```bash
#!/bin/bash
# backup-database.sh
BACKUP_DIR="/opt/backups/vid-benchmark"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL database
pg_dump -h postgres -U benchmarkuser -d benchmarks \
  --no-password --clean --if-exists \
  --file="$BACKUP_DIR/benchmark_db_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/benchmark_db_$DATE.sql"

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "benchmark_db_*.sql.gz" -mtime +30 -delete
```

### Model and Data Backup
```bash
#!/bin/bash
# backup-data.sh
BACKUP_DIR="/opt/backups/vid-benchmark"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup model cache
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" /opt/vid-bench/models/

# Backup results (last 30 days only)
find /opt/vid-bench/results -mtime -30 -type f \
  -exec tar -czf "$BACKUP_DIR/results_$DATE.tar.gz" {} +

# Sync to S3 (if using AWS)
aws s3 sync $BACKUP_DIR s3://vid-benchmark-backups/
```

## 🧪 Testing Deployment

### Health Checks
```bash
# Test coordinator health
curl -f http://localhost:8080/health

# Test worker connectivity
curl -f http://worker:8090/metrics

# Test database connectivity
psql -h postgres -U benchmarkuser -d benchmarks -c "SELECT 1;"

# Test Redis connectivity
redis-cli -h redis ping
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load test
locust -f tests/load/benchmark_load_test.py \
  --host http://localhost:8080 \
  --users 50 \
  --spawn-rate 5 \
  --run-time 300s
```

### Integration Testing
```bash
# Run full integration test suite
python -m pytest tests/integration/ -v

# Test specific deployment scenario
python -m pytest tests/deployment/test_kubernetes.py -v

# Test with actual models (longer running)
python -m pytest tests/e2e/ -v --model=svd-xt
```

## 📈 Monitoring Deployment Health

### Key Performance Indicators (KPIs)
1. **System Availability**: 99.9% uptime target
2. **Response Time**: <500ms API response time
3. **Throughput**: >100 videos/hour per GPU
4. **Error Rate**: <1% benchmark failure rate
5. **Resource Efficiency**: >80% GPU utilization

### Dashboard URLs
- **Grafana**: http://monitoring.vid-benchmark.example.com:3000
- **Prometheus**: http://monitoring.vid-benchmark.example.com:9090
- **Jaeger**: http://monitoring.vid-benchmark.example.com:16686
- **Application**: http://vid-benchmark.example.com

## 🆘 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi
```

#### Out of Memory Errors
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size in configuration
export BENCHMARK_BATCH_SIZE=1

# Enable memory optimization
export GPU_MEMORY_FRACTION=0.8
export ENABLE_MEMORY_OPTIMIZATION=true
```

#### Model Download Failures
```bash
# Check internet connectivity
curl -I https://huggingface.co

# Clear model cache
rm -rf /opt/vid-bench/models/cache/

# Manual model download
python scripts/download_models.py --model svd-xt --force
```

### Log Analysis
```bash
# View coordinator logs
kubectl logs -f deployment/benchmark-coordinator

# View worker logs with grep
kubectl logs -f deployment/benchmark-worker | grep ERROR

# Centralized logging with ELK/EFK stack
curl -X GET "elasticsearch:9200/_search?q=level:ERROR&sort=@timestamp:desc"
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**: Review monitoring dashboards and alerts
2. **Monthly**: Update security patches and dependencies  
3. **Quarterly**: Performance optimization review
4. **Annually**: Disaster recovery testing

### Professional Support
- **Community Support**: GitHub Issues and Discord
- **Enterprise Support**: Commercial support agreements available
- **Training**: Deployment workshops and certification programs
- **Consulting**: Custom deployment and optimization services

---

*This deployment guide ensures production-ready deployment of the Video Diffusion Benchmark Suite across various environments and scales.*