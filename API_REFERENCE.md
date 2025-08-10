# API Reference - Video Diffusion Benchmark Suite

## 🎯 Overview

This document provides comprehensive API documentation for all components of the Video Diffusion Benchmark Suite, including REST APIs, Python SDK, CLI commands, and WebSocket interfaces.

## 📚 Table of Contents

- [REST API](#rest-api)
- [Python SDK](#python-sdk)
- [CLI Commands](#cli-commands)
- [WebSocket API](#websocket-api)
- [Federated API](#federated-api)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## 🌐 REST API

### Base URL
```
Production: https://api.vid-diffusion-bench.org/v1
Development: http://localhost:8080/api/v1
```

### Authentication
All API requests require authentication via API key or JWT token:
```http
Authorization: Bearer <token>
# or
X-API-Key: <api-key>
```

---

### Benchmarks API

#### Start Benchmark
**POST** `/benchmarks`

Start a new benchmark evaluation.

**Request Body:**
```json
{
  "model_name": "svd-xt",
  "prompts": [
    "A cat playing piano",
    "Ocean waves at sunset"
  ],
  "parameters": {
    "num_frames": 16,
    "fps": 8,
    "guidance_scale": 7.5,
    "num_inference_steps": 20
  },
  "evaluation_settings": {
    "compute_metrics": true,
    "save_videos": false,
    "optimization_level": "balanced"
  },
  "metadata": {
    "experiment_name": "cat_piano_test",
    "description": "Testing SVD-XT performance"
  }
}
```

**Response:**
```json
{
  "benchmark_id": "bench_abc123",
  "status": "submitted",
  "estimated_duration": 300,
  "created_at": "2025-01-01T10:00:00Z",
  "message": "Benchmark submitted successfully"
}
```

#### Get Benchmark Status
**GET** `/benchmarks/{benchmark_id}`

**Response:**
```json
{
  "benchmark_id": "bench_abc123",
  "status": "running",
  "progress": {
    "completed_prompts": 5,
    "total_prompts": 10,
    "percentage": 50.0,
    "estimated_remaining": 150
  },
  "current_task": "Generating video for prompt 6/10",
  "started_at": "2025-01-01T10:00:00Z",
  "updated_at": "2025-01-01T10:02:30Z"
}
```

#### Get Benchmark Results
**GET** `/benchmarks/{benchmark_id}/results`

**Response:**
```json
{
  "benchmark_id": "bench_abc123",
  "model_name": "svd-xt",
  "status": "completed",
  "results": {
    "metrics": {
      "fvd": 87.3,
      "inception_score": 42.1,
      "clip_similarity": 0.312,
      "temporal_consistency": 0.89,
      "overall_score": 94.2
    },
    "performance": {
      "avg_latency_ms": 4200,
      "throughput_fps": 14.3,
      "peak_vram_gb": 16.2,
      "avg_power_watts": 245.7,
      "efficiency_score": 88.3
    },
    "success_rate": 1.0,
    "completed_at": "2025-01-01T10:05:15Z"
  }
}
```

#### List Benchmarks
**GET** `/benchmarks`

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Results per page (default: 20, max: 100)
- `status` (string): Filter by status (`submitted`, `running`, `completed`, `failed`)
- `model_name` (string): Filter by model name
- `created_after` (datetime): Filter by creation date
- `sort` (string): Sort order (`created_at`, `-created_at`, `status`)

**Response:**
```json
{
  "benchmarks": [
    {
      "benchmark_id": "bench_abc123",
      "model_name": "svd-xt", 
      "status": "completed",
      "created_at": "2025-01-01T10:00:00Z",
      "metrics_summary": {
        "overall_score": 94.2,
        "avg_latency_ms": 4200
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

#### Cancel Benchmark  
**DELETE** `/benchmarks/{benchmark_id}`

**Response:**
```json
{
  "benchmark_id": "bench_abc123",
  "status": "cancelled",
  "message": "Benchmark cancelled successfully"
}
```

---

### Models API

#### List Available Models
**GET** `/models`

**Response:**
```json
{
  "models": [
    {
      "name": "svd-xt",
      "display_name": "Stable Video Diffusion XT",
      "version": "1.1",
      "description": "High-quality video generation model",
      "requirements": {
        "vram_gb": 16,
        "precision": "fp16",
        "dependencies": ["diffusers>=0.27.0"]
      },
      "supported_resolutions": ["576x576", "768x768", "1024x576"],
      "max_frames": 25,
      "status": "available"
    },
    {
      "name": "cogvideo",
      "display_name": "CogVideo",
      "version": "2.0",
      "description": "Text-to-video generation model",
      "requirements": {
        "vram_gb": 24,
        "precision": "fp32",
        "dependencies": ["transformers>=4.40.0"]
      },
      "supported_resolutions": ["480x480", "720x480"],
      "max_frames": 32,
      "status": "available"
    }
  ]
}
```

#### Get Model Details
**GET** `/models/{model_name}`

**Response:**
```json
{
  "name": "svd-xt",
  "display_name": "Stable Video Diffusion XT",
  "version": "1.1",
  "description": "High-quality video generation model optimized for temporal consistency",
  "architecture": "Latent Diffusion Model",
  "paper_url": "https://arxiv.org/abs/2311.15127",
  "huggingface_url": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt",
  "requirements": {
    "vram_gb": 16,
    "precision": "fp16",
    "cuda_capability": "7.0",
    "dependencies": [
      "diffusers>=0.27.0",
      "transformers>=4.40.0",
      "torch>=2.3.0"
    ]
  },
  "parameters": {
    "supported": {
      "num_frames": [16, 24, 25],
      "fps": [6, 7, 8, 10, 12],
      "guidance_scale": {"min": 1.0, "max": 15.0, "default": 7.5},
      "num_inference_steps": [10, 20, 25, 50],
      "resolution": ["576x576", "768x768", "1024x576"]
    }
  },
  "benchmarks": {
    "average_scores": {
      "fvd": 89.7,
      "inception_score": 40.2,
      "clip_similarity": 0.298,
      "overall_score": 91.5
    },
    "performance": {
      "avg_latency_ms": 3850,
      "throughput_fps": 15.6,
      "typical_vram_gb": 14.2
    }
  },
  "status": "available",
  "last_updated": "2025-01-01T09:00:00Z"
}
```

#### Model Health Check
**GET** `/models/{model_name}/health`

**Response:**
```json
{
  "model_name": "svd-xt",
  "status": "healthy",
  "last_check": "2025-01-01T10:15:00Z",
  "load_time_ms": 15420,
  "memory_usage_gb": 14.2,
  "test_generation_ms": 3850,
  "details": {
    "model_loaded": true,
    "gpu_available": true,
    "dependencies_ok": true
  }
}
```

---

### Streaming API

#### Start Streaming Benchmark
**POST** `/streaming/benchmarks`

**Request Body:**
```json
{
  "model_name": "svd-xt",
  "stream_config": {
    "duration_seconds": 60,
    "target_fps": 15.0,
    "adaptive_quality": true,
    "buffer_size": 30
  },
  "prompts": [
    "A cat playing piano",
    "Ocean waves at sunset"
  ],
  "quality_settings": {
    "initial_resolution": [512, 512],
    "initial_frames": 16,
    "quality_levels": 4
  }
}
```

**Response:**
```json
{
  "stream_id": "stream_xyz789",
  "status": "started",
  "websocket_url": "ws://localhost:8080/streaming/ws/stream_xyz789",
  "estimated_duration": 60,
  "created_at": "2025-01-01T10:20:00Z"
}
```

#### Get Streaming Results
**GET** `/streaming/benchmarks/{stream_id}/results`

**Response:**
```json
{
  "stream_id": "stream_xyz789",
  "status": "completed",
  "metrics": {
    "avg_frame_latency_ms": 85.3,
    "frame_drop_rate": 0.02,
    "quality_consistency": 0.94,
    "adaptive_quality_score": 0.87,
    "buffer_utilization": 0.65,
    "throughput_stability": 0.91
  },
  "performance": {
    "total_frames_generated": 900,
    "successful_frames": 882,
    "quality_adaptations": 15,
    "average_resolution": [486, 486],
    "average_frame_count": 14.2
  },
  "completed_at": "2025-01-01T10:21:00Z"
}
```

---

### Federated API

#### Create Federated Session
**POST** `/federated/sessions`

**Request Body:**
```json
{
  "session_id": "fed_session_001",
  "participants": ["university_a", "lab_b", "company_c"],
  "benchmark_spec": {
    "models": ["svd-xt", "cogvideo"],
    "prompts": ["A cat playing piano", "Ocean waves"],
    "parameters": {
      "num_frames": [16, 24],
      "guidance_scale": [5.0, 7.5]
    }
  },
  "privacy_config": {
    "privacy_level": "differential",
    "epsilon": 1.0,
    "delta": 1e-5
  },
  "consensus_threshold": 0.8,
  "timeout_minutes": 120
}
```

**Response:**
```json
{
  "session_id": "fed_session_001",
  "status": "created",
  "coordinator_endpoint": "https://coordinator.example.com:8080",
  "participant_count": 3,
  "estimated_duration": 120,
  "created_at": "2025-01-01T11:00:00Z"
}
```

#### Join Federated Session
**POST** `/federated/sessions/{session_id}/join`

**Request Body:**
```json
{
  "participant_id": "university_a",
  "public_key": "-----BEGIN PUBLIC KEY-----\n...",
  "capabilities": {
    "gpu_memory": "32GB",
    "compute_capability": "8.6",
    "models_supported": ["svd-xt", "cogvideo"]
  }
}
```

**Response:**
```json
{
  "session_id": "fed_session_001",
  "participant_id": "university_a",
  "status": "joined",
  "secure_channel_established": true,
  "benchmark_spec_received": true,
  "next_action": "wait_for_start_signal"
}
```

#### Get Federated Session Status
**GET** `/federated/sessions/{session_id}`

**Response:**
```json
{
  "session_id": "fed_session_001",
  "status": "running",
  "phase": "execution",
  "participants": {
    "total": 3,
    "active": 3,
    "completed": 1
  },
  "progress": {
    "tasks_completed": 25,
    "tasks_total": 75,
    "percentage": 33.3
  },
  "privacy_budget": {
    "epsilon_total": 1.0,
    "epsilon_spent": 0.3,
    "epsilon_remaining": 0.7
  },
  "estimated_completion": "2025-01-01T12:45:00Z"
}
```

---

### Analytics API

#### Get Benchmark Analytics
**GET** `/analytics/benchmarks`

**Query Parameters:**
- `period` (string): Time period (`1h`, `24h`, `7d`, `30d`)
- `model_name` (string): Filter by model
- `metric` (string): Specific metric to analyze
- `aggregation` (string): Aggregation method (`avg`, `min`, `max`, `p95`)

**Response:**
```json
{
  "period": "24h",
  "total_benchmarks": 1247,
  "success_rate": 0.987,
  "metrics": {
    "avg_quality_score": 89.3,
    "avg_latency_ms": 4150,
    "avg_throughput_fps": 14.7
  },
  "trends": {
    "quality_trend": 0.02,
    "performance_trend": -0.05,
    "usage_trend": 0.15
  },
  "top_models": [
    {
      "model_name": "svd-xt",
      "benchmark_count": 456,
      "avg_score": 91.2
    },
    {
      "model_name": "cogvideo", 
      "benchmark_count": 398,
      "avg_score": 87.8
    }
  ]
}
```

#### Get Model Comparison
**GET** `/analytics/models/compare`

**Query Parameters:**
- `models` (string): Comma-separated model names
- `metric` (string): Comparison metric
- `period` (string): Time period

**Response:**
```json
{
  "comparison_metric": "overall_score",
  "period": "7d",
  "models": [
    {
      "model_name": "svd-xt",
      "metrics": {
        "overall_score": 91.2,
        "fvd": 87.3,
        "latency_ms": 3850
      },
      "benchmark_count": 156,
      "rank": 1
    },
    {
      "model_name": "cogvideo",
      "metrics": {
        "overall_score": 87.8,
        "fvd": 95.1,
        "latency_ms": 5200
      },
      "benchmark_count": 134,
      "rank": 2
    }
  ],
  "statistical_significance": {
    "p_value": 0.003,
    "confidence_interval": 0.95,
    "significant": true
  }
}
```

---

## 🐍 Python SDK

### Installation
```bash
pip install vid-diffusion-bench
```

### Basic Usage

```python
from vid_diffusion_bench import BenchmarkSuite, StandardPrompts
from vid_diffusion_bench.streaming import StreamingBenchmark
from vid_diffusion_bench.federated import create_federated_session

# Initialize benchmark suite
suite = BenchmarkSuite(device="cuda", output_dir="./results")

# Run basic benchmark
result = suite.evaluate_model(
    model_name="svd-xt",
    prompts=StandardPrompts.DIVERSE_SET_V2[:5],
    num_frames=16,
    fps=8
)

print(f"Overall score: {result.metrics['overall_score']:.2f}")
print(f"Average latency: {result.performance['avg_latency_ms']:.0f}ms")
```

### Advanced Configuration

```python
from vid_diffusion_bench import BenchmarkSuite
from vid_diffusion_bench.optimization import OptimizationProfile
from vid_diffusion_bench.internationalization import set_locale

# Configure optimization
profile = OptimizationProfile(
    precision="fp16",
    compile_model=True,
    memory_efficient=True,
    tensorrt_optimization=True
)

# Set locale for internationalized output
set_locale("es")  # Spanish

# Initialize with custom configuration
suite = BenchmarkSuite(
    device="cuda",
    optimization_profile=profile,
    output_dir="./resultados",
    enable_monitoring=True
)

# Run comprehensive evaluation
models = ["svd-xt", "cogvideo", "pika-lumiere"]
comparison = suite.evaluate_multiple_models(
    model_names=models,
    prompts=StandardPrompts.CINEMATIC_SET,
    max_workers=2,
    save_videos=True
)

# Generate comparison report
report = suite.compare_models(comparison)
print(f"Pareto frontier models: {report['pareto_frontier']}")
```

### Streaming Benchmarks

```python
import asyncio
from vid_diffusion_bench.streaming import benchmark_live_streaming

async def run_streaming_test():
    prompts = ["A cat playing piano", "Ocean waves at sunset"]
    
    metrics = await benchmark_live_streaming(
        model_name="svd-xt",
        prompts=prompts,
        duration_seconds=60,
        target_fps=15.0,
        adaptive_quality=True
    )
    
    print(f"Average latency: {metrics.avg_frame_latency_ms:.1f}ms")
    print(f"Frame drop rate: {metrics.frame_drop_rate:.2%}")
    print(f"Quality consistency: {metrics.quality_consistency:.3f}")

# Run the streaming test
asyncio.run(run_streaming_test())
```

### Federated Benchmarking

```python
import asyncio
from vid_diffusion_bench.federated import create_federated_session, join_federated_session

async def coordinator_example():
    # Create federated session (coordinator)
    success = await create_federated_session(
        session_id="research_collaboration_2025",
        participants=["stanford", "mit", "openai"],
        benchmark_spec={
            "models": ["svd-xt", "cogvideo"],
            "prompts": ["A cat playing piano", "Ocean waves"],
            "parameters": {
                "num_frames": [16, 24],
                "guidance_scale": [5.0, 7.5, 10.0]
            }
        },
        privacy_level="differential"
    )
    print(f"Federated session created: {success}")

async def participant_example():
    # Join federated session (participant)
    success = await join_federated_session(
        participant_id="stanford",
        session_id="research_collaboration_2025",
        coordinator_endpoint="https://coordinator.example.com:8080"
    )
    print(f"Joined federated session: {success}")

# Run coordinator or participant
asyncio.run(coordinator_example())
```

### AI-Driven Optimization

```python
from vid_diffusion_bench.ai_optimization import (
    AIOptimizationEngine,
    OptimizationObjective,
    SearchSpace
)

# Initialize optimization engine
suite = BenchmarkSuite()
optimizer = AIOptimizationEngine(suite)

# Define custom search space
search_space = SearchSpace(
    continuous_params={
        "guidance_scale": (1.0, 15.0),
        "eta": (0.0, 1.0)
    },
    discrete_params={
        "num_inference_steps": [10, 20, 30, 50],
        "num_frames": [8, 16, 24, 32]
    },
    categorical_params={
        "scheduler": ["ddim", "dpm", "euler", "lms"],
        "precision": ["fp16", "fp32"]
    }
)

# Run optimization
result = optimizer.optimize_benchmark_parameters(
    model_name="svd-xt",
    objective=OptimizationObjective.MAXIMIZE_QUALITY,
    strategy="bayesian",
    search_space=search_space,
    max_evaluations=50
)

print(f"Best parameters: {result.best_parameters}")
print(f"Best score: {result.best_score:.3f}")
print(f"Optimization time: {result.elapsed_time:.1f}s")
```

---

## 💻 CLI Commands

### Installation
```bash
pip install vid-diffusion-bench[cli]
```

### Basic Commands

#### Run Benchmark
```bash
# Basic benchmark
vid-bench benchmark svd-xt \
  --prompts "A cat playing piano" "Ocean waves" \
  --num-frames 16 \
  --fps 8 \
  --output results/

# Batch benchmark with file
vid-bench benchmark svd-xt \
  --prompts-file prompts.txt \
  --batch-size 4 \
  --save-videos \
  --output results/batch_run/

# Multiple models comparison
vid-bench compare svd-xt cogvideo pika-lumiere \
  --prompts-file prompts.txt \
  --metrics fvd is clip_score \
  --output comparison_report.json
```

#### Model Management
```bash
# List available models
vid-bench models list

# Get model info
vid-bench models info svd-xt

# Download model
vid-bench models download cogvideo --cache-dir ./models/

# Health check
vid-bench models health-check svd-xt
```

#### Streaming Benchmarks
```bash
# Live streaming test
vid-bench streaming live svd-xt \
  --prompts "A cat playing piano" \
  --duration 60 \
  --target-fps 15 \
  --adaptive-quality

# Interactive generation test
vid-bench streaming interactive cogvideo \
  --prompts-file interactive_prompts.txt \
  --response-target 500ms
```

#### Optimization
```bash
# Auto-optimize model settings
vid-bench optimize svd-xt \
  --objective quality \
  --strategy bayesian \
  --max-evaluations 30 \
  --output optimized_params.json

# Performance profiling
vid-bench profile svd-xt \
  --test-prompts 5 \
  --optimization-levels baseline balanced performance \
  --detailed \
  --output profile_report.json
```

#### Configuration
```bash
# Set global configuration
vid-bench config set optimization_level balanced
vid-bench config set default_locale es
vid-bench config set gpu_memory_fraction 0.8

# View configuration
vid-bench config show

# Reset to defaults
vid-bench config reset
```

#### Internationalization
```bash
# Set locale
vid-bench locale set es

# List supported locales
vid-bench locale list

# Get locale info
vid-bench locale info zh-CN
```

### Advanced Usage

#### Distributed Deployment
```bash
# Start coordinator
vid-bench distributed coordinator \
  --port 8080 \
  --max-workers 10 \
  --auto-scaling

# Join as worker
vid-bench distributed worker \
  --coordinator http://master:8080 \
  --gpu-count 2 \
  --worker-id node-001

# Submit distributed job
vid-bench distributed submit \
  --coordinator http://master:8080 \
  --models svd-xt cogvideo \
  --prompts-file large_dataset.txt \
  --priority high
```

#### Federated Benchmarking
```bash
# Create federated session (coordinator)
vid-bench federated create \
  --session-id collaboration_2025 \
  --participants university_a lab_b company_c \
  --spec benchmark_spec.json \
  --privacy-level differential

# Join federated session (participant)
vid-bench federated join \
  --session-id collaboration_2025 \
  --participant-id university_a \
  --coordinator https://coord.example.com:8080
```

#### Monitoring and Analytics
```bash
# Real-time monitoring
vid-bench monitor dashboard \
  --port 3000 \
  --refresh-rate 5s

# Generate analytics report
vid-bench analytics report \
  --period 30d \
  --models svd-xt cogvideo \
  --output analytics_report.pdf

# Export metrics
vid-bench analytics export \
  --format prometheus \
  --output metrics.txt
```

---

## 🔌 WebSocket API

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

// Authentication
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'authenticate',
        token: 'your-jwt-token'
    }));
};
```

### Real-time Benchmark Updates
```javascript
// Subscribe to benchmark updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'benchmark_updates',
    benchmark_id: 'bench_abc123'
}));

// Handle updates
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
        case 'benchmark_progress':
            console.log(`Progress: ${message.data.percentage}%`);
            break;
            
        case 'benchmark_completed':
            console.log('Benchmark completed:', message.data.results);
            break;
            
        case 'benchmark_error':
            console.error('Benchmark error:', message.data.error);
            break;
    }
};
```

### Streaming Benchmark WebSocket
```javascript
// Connect to streaming benchmark
const streamWs = new WebSocket('ws://localhost:8080/streaming/ws/stream_xyz789');

streamWs.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
        case 'frame_generated':
            console.log('Frame generated:', {
                frame_id: message.data.frame_id,
                latency_ms: message.data.latency_ms,
                quality_level: message.data.quality_level
            });
            break;
            
        case 'quality_adapted':
            console.log('Quality adapted:', {
                new_resolution: message.data.resolution,
                new_frame_count: message.data.frames,
                reason: message.data.reason
            });
            break;
            
        case 'buffer_status':
            console.log('Buffer utilization:', message.data.utilization);
            break;
            
        case 'stream_completed':
            console.log('Stream completed:', message.data.final_metrics);
            break;
    }
};
```

---

## 🔒 Authentication

### API Key Authentication
```bash
# Generate API key
curl -X POST http://localhost:8080/api/v1/auth/api-keys \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "production-key", "permissions": ["benchmark.create", "models.read"]}'

# Use API key
curl -X GET http://localhost:8080/api/v1/models \
  -H "X-API-Key: vdb_abc123def456"
```

### JWT Token Authentication
```bash
# Login to get JWT token
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password"}'

# Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Use JWT token
curl -X POST http://localhost:8080/api/v1/benchmarks \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{"model_name": "svd-xt", "prompts": ["test"]}'
```

### OAuth2 Integration
```python
from vid_diffusion_bench.auth import OAuth2Client

# Configure OAuth2
oauth_client = OAuth2Client(
    client_id="your-client-id",
    client_secret="your-client-secret",
    authorization_url="https://auth.example.com/oauth/authorize",
    token_url="https://auth.example.com/oauth/token"
)

# Get authorization URL
auth_url = oauth_client.get_authorization_url(
    redirect_uri="http://localhost:8080/callback",
    scopes=["benchmark.create", "models.read"]
)

# Exchange authorization code for token
token = oauth_client.exchange_code_for_token(
    code="auth_code_from_callback",
    redirect_uri="http://localhost:8080/callback"
)
```

---

## ⚠️ Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'invalid-model' not found",
    "details": {
      "model_name": "invalid-model",
      "available_models": ["svd-xt", "cogvideo", "pika-lumiere"]
    },
    "request_id": "req_abc123",
    "timestamp": "2025-01-01T10:00:00Z"
  }
}
```

### HTTP Status Codes
- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **202 Accepted**: Request accepted for processing
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

### Error Codes
| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_PARAMETERS` | Request parameters are invalid | 400 |
| `MODEL_NOT_FOUND` | Specified model doesn't exist | 404 |
| `INSUFFICIENT_MEMORY` | Not enough GPU memory | 422 |
| `GENERATION_TIMEOUT` | Video generation timed out | 408 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `AUTHENTICATION_FAILED` | Invalid credentials | 401 |
| `PERMISSION_DENIED` | Insufficient permissions | 403 |
| `BENCHMARK_NOT_FOUND` | Benchmark ID doesn't exist | 404 |
| `INVALID_PROMPT` | Prompt validation failed | 422 |
| `SERVER_OVERLOADED` | Server at capacity | 503 |

### Python SDK Error Handling
```python
from vid_diffusion_bench import BenchmarkSuite
from vid_diffusion_bench.exceptions import (
    ModelNotFoundError,
    InsufficientMemoryError,
    GenerationTimeoutError,
    ValidationError
)

suite = BenchmarkSuite()

try:
    result = suite.evaluate_model(
        model_name="invalid-model",
        prompts=["test prompt"]
    )
except ModelNotFoundError as e:
    print(f"Model not found: {e.model_name}")
    print(f"Available models: {e.available_models}")
except InsufficientMemoryError as e:
    print(f"Need {e.required_gb}GB, have {e.available_gb}GB")
except GenerationTimeoutError as e:
    print(f"Generation timed out after {e.timeout_seconds}s")
except ValidationError as e:
    print(f"Validation failed: {e.details}")
```

---

## 🚦 Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 3600
```

### Rate Limit Configuration
| Endpoint Pattern | Limit | Window | Burst |
|------------------|-------|---------|-------|
| `/api/v1/benchmarks` | 100 req | 1 hour | 20 |
| `/api/v1/models` | 1000 req | 1 hour | 100 |
| `/api/v1/analytics` | 500 req | 1 hour | 50 |
| `/api/v1/streaming` | 50 req | 1 hour | 10 |
| `/api/v1/federated` | 10 req | 1 hour | 2 |

### Rate Limit Exceeded Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 45 seconds.",
    "details": {
      "limit": 100,
      "remaining": 0,
      "reset_at": "2025-01-01T11:00:00Z",
      "retry_after": 45
    }
  }
}
```

---

## 📝 Examples

### Complete Benchmark Workflow
```python
import asyncio
from vid_diffusion_bench import BenchmarkSuite, StandardPrompts
from vid_diffusion_bench.internationalization import set_locale, translate

async def complete_workflow():
    # Set Spanish locale
    set_locale("es")
    
    # Initialize suite
    suite = BenchmarkSuite(device="cuda")
    
    print(translate("benchmark.model_evaluation"))  # "Evaluación del Modelo"
    
    # Single model evaluation
    result = suite.evaluate_model(
        model_name="svd-xt",
        prompts=StandardPrompts.DIVERSE_SET_V2[:3],
        num_frames=16,
        fps=8,
        save_videos=True
    )
    
    print(f"Puntuación general: {result.metrics['overall_score']:.2f}")
    
    # Multi-model comparison
    models = ["svd-xt", "cogvideo"]
    comparison = suite.evaluate_multiple_models(
        model_names=models,
        prompts=StandardPrompts.CINEMATIC_SET[:2],
        max_workers=2
    )
    
    # Generate comparison report
    report = suite.compare_models(comparison)
    print(f"Modelos Pareto: {report['pareto_frontier']}")

# Run the workflow
asyncio.run(complete_workflow())
```

### REST API Integration
```python
import requests
import time

class VidBenchClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def start_benchmark(self, model_name, prompts, **params):
        response = requests.post(
            f"{self.base_url}/benchmarks",
            headers=self.headers,
            json={
                "model_name": model_name,
                "prompts": prompts,
                "parameters": params
            }
        )
        return response.json()
    
    def wait_for_completion(self, benchmark_id, timeout=3600):
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.base_url}/benchmarks/{benchmark_id}",
                headers=self.headers
            )
            status = response.json()
            
            if status["status"] == "completed":
                return self.get_results(benchmark_id)
            elif status["status"] == "failed":
                raise Exception(f"Benchmark failed: {status}")
            
            time.sleep(10)
        
        raise TimeoutError("Benchmark timed out")
    
    def get_results(self, benchmark_id):
        response = requests.get(
            f"{self.base_url}/benchmarks/{benchmark_id}/results",
            headers=self.headers
        )
        return response.json()

# Usage
client = VidBenchClient("http://localhost:8080/api/v1", "your-api-key")

# Start benchmark
benchmark = client.start_benchmark(
    model_name="svd-xt",
    prompts=["A cat playing piano", "Ocean waves"],
    num_frames=16,
    fps=8
)

# Wait for completion and get results
results = client.wait_for_completion(benchmark["benchmark_id"])
print(f"Quality score: {results['results']['metrics']['overall_score']}")
```

---

*This API reference provides comprehensive documentation for all interfaces and integration patterns available in the Video Diffusion Benchmark Suite.*