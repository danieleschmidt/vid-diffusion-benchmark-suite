# Deployment Guide

## Overview

The Video Diffusion Benchmark Suite supports multiple deployment scenarios, from local development to production-scale cloud deployments.

## Deployment Options

### 1. Local Development

#### Docker Compose (Recommended)
```bash
# Clone and setup
git clone https://github.com/yourusername/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite

# Start all services
docker-compose up -d

# Access dashboard
open http://localhost:8501
```

#### Native Installation
```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Run benchmark
vid-bench evaluate --model svd-xt --prompt "A cat playing piano"
```

### 2. Cloud Deployment

#### AWS ECS with GPU Support
```yaml
# aws-ecs-task-definition.json
{
  "family": "vid-diffusion-benchmark",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/VidBenchTaskRole",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/VidBenchExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "32768",
  "containerDefinitions": [
    {
      "name": "vid-bench",
      "image": "your-registry/vid-diffusion-benchmark:latest",
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "environment": [
        {
          "name": "CUDA_VISIBLE_DEVICES",
          "value": "0"
        }
      ],
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

#### Google Cloud Run with GPU
```yaml
# gcp-cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: vid-diffusion-benchmark
  annotations:
    run.googleapis.com/gpu-type: nvidia-tesla-t4
    run.googleapis.com/gpu-count: "1"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "5"
        run.googleapis.com/cpu: "4"
        run.googleapis.com/memory: "16Gi"
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/vid-diffusion-benchmark:latest
        ports:
        - containerPort: 8501
        env:
        - name: PORT
          value: "8501"
        resources:
          limits:
            nvidia.com/gpu: 1
```

#### Azure Container Instances
```yaml
# azure-aci.yaml
apiVersion: 2021-03-01
location: eastus
name: vid-diffusion-benchmark
properties:
  containers:
  - name: vid-bench
    properties:
      image: your-registry.azurecr.io/vid-diffusion-benchmark:latest
      resources:
        requests:
          cpu: 4
          memoryInGb: 32
          gpu:
            count: 1
            sku: K80
      ports:
      - protocol: TCP
        port: 8501
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8501
```

### 3. Kubernetes Deployment

#### Namespace and RBAC
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vid-diffusion-benchmark

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vid-bench-sa
  namespace: vid-diffusion-benchmark
```

#### ConfigMap for Configuration
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vid-bench-config
  namespace: vid-diffusion-benchmark
data:
  config.yaml: |
    models:
      cache_dir: /app/models
      default_precision: fp16
    benchmark:
      batch_size: 1
      num_frames: 16
      output_dir: /app/results
    dashboard:
      host: 0.0.0.0
      port: 8501
```

#### GPU Node Pool Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vid-diffusion-benchmark
  namespace: vid-diffusion-benchmark
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vid-diffusion-benchmark
  template:
    metadata:
      labels:
        app: vid-diffusion-benchmark
    spec:
      serviceAccountName: vid-bench-sa
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: vid-bench
        image: your-registry/vid-diffusion-benchmark:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            cpu: 4
            memory: 16Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 8
            memory: 32Gi
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: results
          mountPath: /app/results
        - name: models
          mountPath: /app/models
      volumes:
      - name: config
        configMap:
          name: vid-bench-config
      - name: results
        persistentVolumeClaim:
          claimName: vid-bench-results
      - name: models
        persistentVolumeClaim:
          claimName: vid-bench-models
```

#### Services and Ingress
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vid-diffusion-benchmark
  namespace: vid-diffusion-benchmark
spec:
  selector:
    app: vid-diffusion-benchmark
  ports:
  - port: 80
    targetPort: 8501
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vid-diffusion-benchmark
  namespace: vid-diffusion-benchmark
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - vid-bench.yourdomain.com
    secretName: vid-bench-tls
  rules:
  - host: vid-bench.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vid-diffusion-benchmark
            port:
              number: 80
```

### 4. Production Setup

#### Environment Variables
```bash
# Production environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_HOME=/app/models/.torch
export WANDB_API_KEY=your_wandb_key
export REDIS_URL=redis://redis-cluster:6379
export DATABASE_URL=postgresql://user:pass@db:5432/vidbench
export SECRET_KEY=your-secret-key
export ENVIRONMENT=production
```

#### Resource Requirements

| Component | CPU | Memory | GPU | Storage |
|-----------|-----|--------|-----|---------|
| Benchmark Engine | 8 cores | 32GB | 1x V100 | 100GB SSD |
| Dashboard | 2 cores | 4GB | - | 10GB |
| Redis | 2 cores | 8GB | - | 50GB |
| Prometheus | 4 cores | 16GB | - | 200GB |
| Grafana | 2 cores | 4GB | - | 10GB |

#### High Availability Setup
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vid-diffusion-benchmark-hpa
  namespace: vid-diffusion-benchmark
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vid-diffusion-benchmark
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

## Monitoring and Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
- job_name: 'vid-diffusion-benchmark'
  static_configs:
  - targets: ['vid-bench:8501']
  metrics_path: '/metrics'
  scrape_interval: 30s

- job_name: 'nvidia-gpu'
  static_configs:
  - targets: ['nvidia-dcgm-exporter:9400']
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Video Diffusion Benchmark",
    "panels": [
      {
        "title": "Benchmark Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(benchmark_completions_total[5m])",
            "legendFormat": "Benchmarks/sec"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration
```yaml
# logging/fluentd.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<match vid-bench.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name vid-benchmark-logs
  type_name _doc
  
  <buffer>
    @type file
    path /var/log/fluentd-buffers/vid-bench.buffer
    flush_mode interval
    flush_interval 10s
  </buffer>
</match>
```

## Security Considerations

### Network Security
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vid-bench-network-policy
  namespace: vid-diffusion-benchmark
spec:
  podSelector:
    matchLabels:
      app: vid-diffusion-benchmark
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
      port: 8501
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### Secret Management
```bash
# Create secrets
kubectl create secret generic vid-bench-secrets \
  --from-literal=wandb-key=your_wandb_key \
  --from-literal=db-password=secure_password \
  --namespace=vid-diffusion-benchmark

# Reference in deployment
env:
- name: WANDB_API_KEY
  valueFrom:
    secretKeyRef:
      name: vid-bench-secrets
      key: wandb-key
```

## Backup and Disaster Recovery

### Data Backup Strategy
```bash
#!/bin/bash
# backup-script.sh

# Backup results database
pg_dump -h $DB_HOST -U $DB_USER vid_benchmark > /backups/db_$(date +%Y%m%d).sql

# Backup model weights
rsync -av /app/models/ /backups/models/

# Backup configuration
kubectl get configmap vid-bench-config -o yaml > /backups/config_$(date +%Y%m%d).yaml

# Upload to cloud storage
aws s3 sync /backups/ s3://vid-bench-backups/$(date +%Y%m%d)/
```

### Disaster Recovery Plan
1. **Infrastructure**: Use Infrastructure as Code (Terraform/CloudFormation)
2. **Data**: Automated daily backups to cloud storage
3. **Monitoring**: Health checks and automated failover
4. **Documentation**: Runbooks for common failure scenarios

## Performance Optimization

### GPU Memory Optimization
```python
# config/optimization.py
OPTIMIZATION_SETTINGS = {
    'batch_size': 1,  # Reduce for memory constraints
    'precision': 'fp16',  # Use mixed precision
    'gradient_checkpointing': True,
    'model_parallel': True,  # For multi-GPU setups
    'memory_efficient_attention': True
}
```

### Caching Strategy
```yaml
# Redis configuration for caching
redis:
  maxmemory: 8gb
  maxmemory-policy: allkeys-lru
  save: "900 1"  # Save after 900 sec if at least 1 key changed
```

### Load Balancing
```nginx
# nginx.conf
upstream vid_bench_backend {
    least_conn;
    server vid-bench-1:8501 max_fails=3 fail_timeout=30s;
    server vid-bench-2:8501 max_fails=3 fail_timeout=30s;
    server vid-bench-3:8501 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name vid-bench.yourdomain.com;
    
    location / {
        proxy_pass http://vid_bench_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

This deployment guide covers various scenarios from development to production, ensuring scalable and robust deployment of the Video Diffusion Benchmark Suite.