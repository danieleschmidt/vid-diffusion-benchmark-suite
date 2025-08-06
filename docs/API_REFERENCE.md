# API Reference

## Overview

The Video Diffusion Benchmark Suite provides comprehensive APIs for model evaluation, metrics computation, and system monitoring. This reference covers all available endpoints, data models, and usage examples.

## Core API Endpoints

### Benchmark Management

#### POST /api/v1/benchmarks
Create and execute a benchmark run.

**Request Body:**
```json
{
  "name": "model_comparison_benchmark",
  "models": ["svd_xt_1_1", "stable_video", "pika_labs"],
  "prompts": [
    "A cat playing with a ball",
    "Ocean waves at sunset"
  ],
  "metrics": ["fvd", "is", "clip_similarity"],
  "config": {
    "batch_size": 4,
    "num_inference_steps": 25,
    "guidance_scale": 7.5
  }
}
```

**Response:**
```json
{
  "benchmark_id": "bench_12345",
  "status": "running",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T11:15:00Z"
}
```

#### GET /api/v1/benchmarks/{benchmark_id}
Get benchmark status and results.

**Response:**
```json
{
  "benchmark_id": "bench_12345",
  "status": "completed",
  "results": {
    "svd_xt_1_1": {
      "fvd": 85.2,
      "is": 32.1,
      "clip_similarity": 0.78
    }
  },
  "execution_time": 2580.5,
  "completion_time": "2024-01-15T11:13:22Z"
}
```

### Model Management

#### GET /api/v1/models
List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "svd_xt_1_1",
      "type": "diffusion",
      "status": "available",
      "capabilities": ["text_to_video", "image_to_video"],
      "max_resolution": "1024x576",
      "max_frames": 25
    }
  ]
}
```

#### POST /api/v1/models/{model_name}/evaluate
Evaluate a specific model.

**Request Body:**
```json
{
  "prompts": ["A dog running in a park"],
  "config": {
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

### Metrics API

#### POST /api/v1/metrics/compute
Compute metrics for video pairs.

**Request Body:**
```json
{
  "real_videos": ["path/to/real1.mp4", "path/to/real2.mp4"],
  "generated_videos": ["path/to/gen1.mp4", "path/to/gen2.mp4"],
  "metrics": ["fvd", "is", "clip_similarity"]
}
```

#### GET /api/v1/metrics/history
Get historical metrics data.

**Query Parameters:**
- `start_date`: Start date filter (ISO 8601)
- `end_date`: End date filter (ISO 8601)
- `model`: Filter by model name
- `metric`: Filter by metric type

### Health and Monitoring

#### GET /api/v1/health
System health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "checks": {
    "database": "healthy",
    "gpu": "healthy",
    "storage": "healthy"
  },
  "version": "1.0.0"
}
```

#### GET /api/v1/monitoring/metrics
Real-time system metrics.

**Response:**
```json
{
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 68.1,
    "gpu_utilization": 85.5,
    "disk_usage": 42.3
  },
  "benchmark": {
    "active_evaluations": 3,
    "completed_today": 127,
    "error_rate": 2.1
  }
}
```

## Python SDK Usage

### Basic Usage

```python
from vid_diffusion_bench import BenchmarkSuite, BenchmarkConfig

# Initialize benchmark suite
benchmark = BenchmarkSuite()

# Configure benchmark
config = BenchmarkConfig(
    models=["svd_xt_1_1", "stable_video"],
    prompts=[
        "A cat playing with a ball",
        "Ocean waves at sunset"
    ],
    metrics=["fvd", "is", "clip_similarity"],
    batch_size=4,
    num_inference_steps=25
)

# Run benchmark
results = benchmark.run(config)

# Print results
for model, metrics in results.items():
    print(f"{model}: FVD={metrics['fvd']:.1f}, IS={metrics['is']:.1f}")
```

### Advanced Usage with Custom Metrics

```python
from vid_diffusion_bench.research import NovelVideoMetrics
from vid_diffusion_bench.metrics import MetricRegistry

# Register custom metric
def custom_temporal_consistency(real_videos, generated_videos):
    # Your custom metric implementation
    return {"temporal_consistency": 0.85}

MetricRegistry.register("temporal_consistency", custom_temporal_consistency)

# Use novel metrics
novel_metrics = NovelVideoMetrics()
results = novel_metrics.compute_comprehensive_metrics(
    real_videos=["real1.mp4"],
    generated_videos=["gen1.mp4"]
)
```

### Distributed Benchmarking

```python
from vid_diffusion_bench.scaling import DistributedBenchmarkRunner

# Initialize distributed runner
runner = DistributedBenchmarkRunner()
runner.start_cluster()

# Add compute nodes
runner.start_local_node(port=8766)

# Run distributed benchmark
results = runner.run_distributed_benchmark(
    models=["svd_xt_1_1", "stable_video", "pika_labs"],
    prompts=prompts,
    metrics=["fvd", "is", "clip_similarity"]
)
```

### Research Framework

```python
from vid_diffusion_bench.research import ExperimentalFramework

# Initialize experimental framework
framework = ExperimentalFramework()

# Design experiment
experiment = framework.design_experiment(
    research_question="Does context compression improve generation quality?",
    variables={
        "compression_ratio": [0.1, 0.3, 0.5, 0.7],
        "model": ["svd_xt_1_1", "stable_video"]
    },
    metrics=["fvd", "is", "perceptual_quality"],
    sample_size=100
)

# Execute experiment
results = framework.execute_experiment(experiment)

# Analyze results
analysis = framework.analyze_results(results, significance_level=0.05)
```

## Data Models

### BenchmarkConfig
```python
@dataclass
class BenchmarkConfig:
    models: List[str]
    prompts: List[str]
    metrics: List[str]
    batch_size: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    num_frames: int = 16
    fps: int = 8
    seed: Optional[int] = None
    output_dir: str = "./outputs"
```

### BenchmarkResult
```python
@dataclass
class BenchmarkResult:
    benchmark_id: str
    model_results: Dict[str, Dict[str, float]]
    execution_time: float
    status: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### MetricResult
```python
@dataclass
class MetricResult:
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Error Handling

### Common Error Codes

- `BENCHMARK_001`: Invalid model configuration
- `BENCHMARK_002`: Insufficient GPU memory
- `BENCHMARK_003`: Model loading failed
- `METRIC_001`: Invalid video format
- `METRIC_002`: Metric computation failed
- `API_001`: Authentication failed
- `API_002`: Rate limit exceeded

### Error Response Format

```json
{
  "error": {
    "code": "BENCHMARK_002",
    "message": "Insufficient GPU memory for model loading",
    "details": {
      "required_memory_gb": 16.5,
      "available_memory_gb": 12.3
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Authentication

### API Key Authentication

Include the API key in the request header:
```
Authorization: Bearer your_api_key_here
```

### Rate Limiting

- Standard tier: 100 requests/hour
- Premium tier: 1000 requests/hour
- Enterprise tier: Unlimited

## WebSocket API

### Real-time Updates

Connect to WebSocket endpoint for real-time benchmark updates:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/benchmarks/{benchmark_id}');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Benchmark update:', update);
};
```

### Event Types

- `benchmark.started`: Benchmark execution started
- `benchmark.progress`: Progress update
- `benchmark.completed`: Benchmark completed
- `benchmark.failed`: Benchmark failed

## CLI Reference

### Basic Commands

```bash
# Run benchmark
vid-diffusion-bench run --models svd_xt_1_1 stable_video --prompts prompts.txt

# List available models
vid-diffusion-bench models list

# Compute metrics
vid-diffusion-bench metrics compute --real real_videos/ --generated generated_videos/

# Export results
vid-diffusion-bench export --format csv --output results.csv
```

### Configuration

Create `benchmark_config.yaml`:
```yaml
models:
  - svd_xt_1_1
  - stable_video
prompts:
  - "A cat playing with a ball"
  - "Ocean waves at sunset"
metrics:
  - fvd
  - is
  - clip_similarity
config:
  batch_size: 4
  num_inference_steps: 25
```

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Use larger batch sizes when GPU memory allows
2. **Model Caching**: Enable model caching to avoid repeated loading
3. **Distributed Processing**: Use multiple GPUs/nodes for large benchmarks
4. **Memory Management**: Monitor GPU memory usage and adjust batch sizes

### Monitoring

Monitor system performance using the built-in monitoring endpoints:
- `/api/v1/monitoring/metrics`: Real-time metrics
- `/api/v1/monitoring/alerts`: Active alerts
- `/api/v1/monitoring/health`: System health

## Integration Examples

### Docker Compose

```yaml
version: '3.8'
services:
  benchmark:
    image: vid-diffusion-bench:latest
    ports:
      - "8000:8000"
    environment:
      - GPU_MEMORY_FRACTION=0.8
      - CACHE_DIR=/app/cache
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vid-diffusion-bench
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vid-diffusion-bench
  template:
    metadata:
      labels:
        app: vid-diffusion-bench
    spec:
      containers:
      - name: benchmark
        image: vid-diffusion-bench:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "8Gi"
            cpu: "2000m"
```

## Support and Contact

- Documentation: https://vid-diffusion-bench.readthedocs.io
- Issues: https://github.com/your-org/vid-diffusion-bench/issues
- Discussions: https://github.com/your-org/vid-diffusion-bench/discussions
- Email: support@vid-diffusion-bench.com