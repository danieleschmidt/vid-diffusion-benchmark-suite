# Video Diffusion Benchmark Suite - API Reference

## Overview

The Video Diffusion Benchmark Suite provides a comprehensive REST API for benchmarking video generation models. This API enables programmatic access to all framework capabilities including model evaluation, research experiments, and result analysis.

## Authentication

All API endpoints require authentication using API keys or JWT tokens.

### API Key Authentication

Include your API key in the Authorization header:

```http
Authorization: Bearer your_api_key_here
```

## Base URL

```
https://api.vid-diffusion-bench.ai/v1
```

## Core Endpoints

### Benchmark Management

#### POST /benchmarks

Create a new benchmark evaluation.

**Request Body:**
```json
{
  "model_name": "stable-video-diffusion",
  "prompts": [
    "A cat playing with a ball",
    "Ocean waves crashing on shore"
  ],
  "config": {
    "num_frames": 16,
    "fps": 8,
    "resolution": [512, 512],
    "batch_size": 4
  }
}
```

**Response:**
```json
{
  "benchmark_id": "bench_abc123",
  "status": "queued",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### GET /benchmarks/{benchmark_id}

Retrieve benchmark results.

#### GET /models

List available models.

### Research Framework

#### POST /experiments

Create a research experiment.

#### GET /experiments/{experiment_id}

Retrieve experiment results and analysis.

## Python SDK Example

```python
import vid_diffusion_bench as vdb

client = vdb.Client(api_key="your_api_key")

benchmark = client.benchmarks.create(
    model_name="stable-video-diffusion",
    prompts=["A cat playing with a ball"],
    config={"num_frames": 16, "fps": 8}
)

result = client.benchmarks.wait_for_completion(benchmark.id)
print(f"Overall score: {result.overall_score}")
```

## Error Handling

All endpoints return standard HTTP status codes with detailed error messages.

## Rate Limits

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1000 requests/hour  
- **Enterprise**: Custom limits