# API Documentation - Enhanced Research Framework

## Overview

The Enhanced Video Diffusion Benchmark Suite provides comprehensive REST APIs for accessing advanced research capabilities including adaptive algorithms, novel metrics, validation frameworks, error handling, scaling systems, and quantum acceleration.

## Base Configuration

**Base URL**: `http://localhost:8000`  
**API Version**: `v1`  
**Authentication**: Bearer token (production deployments)  
**Content-Type**: `application/json`

## API Endpoints

### 1. Research Framework APIs

#### 1.1 Adaptive Algorithms

**Endpoint**: `/api/v1/research/adaptive`

##### POST /api/v1/research/adaptive/optimize

Optimize diffusion parameters for specific video content using adaptive algorithms.

**Request Body**:
```json
{
  "video_data": {
    "tensor_shape": [16, 3, 512, 512],
    "content_type": "video/tensor",
    "encoding": "base64"
  },
  "model_name": "svd-xt-1.1",
  "base_config": {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "height": 512,
    "width": 512,
    "batch_size": 1
  },
  "optimization_mode": "balanced",
  "quality_target": 0.85,
  "performance_weight": 0.3
}
```

**Response**:
```json
{
  "status": "success",
  "optimization_id": "opt_abc123def456",
  "optimized_config": {
    "num_inference_steps": 42,
    "guidance_scale": 8.2,
    "height": 512,
    "width": 512,
    "batch_size": 1,
    "precision": "fp16"
  },
  "content_analysis": {
    "complexity_score": 0.73,
    "motion_intensity": 0.62,
    "texture_density": 0.45,
    "temporal_coherence": 0.88,
    "semantic_complexity": 0.56
  },
  "performance_prediction": {
    "expected_latency": 4.2,
    "expected_memory_gb": 12.5,
    "expected_quality": 0.87,
    "confidence": 0.92
  },
  "optimization_time": 1.8
}
```

##### GET /api/v1/research/adaptive/stats

Get adaptation statistics and performance metrics.

**Response**:
```json
{
  "status": "active",
  "cached_configs": 127,
  "history_size": 450,
  "avg_latency": 3.2,
  "avg_quality": 0.84,
  "adaptation_coverage": 0.73,
  "optimization_success_rate": 0.89
}
```

#### 1.2 Novel Metrics Evaluation

**Endpoint**: `/api/v1/research/metrics`

##### POST /api/v1/research/metrics/evaluate

Comprehensive video quality evaluation using novel metrics.

**Request Body**:
```json
{
  "generated_video": {
    "tensor_shape": [16, 3, 256, 256],
    "encoding": "base64",
    "format": "torch_tensor"
  },
  "reference_video": {
    "tensor_shape": [16, 3, 256, 256],
    "encoding": "base64",
    "format": "torch_tensor"
  },
  "text_prompt": "A cat playing with a ball in slow motion",
  "metrics": [
    "perceptual_quality",
    "temporal_coherence", 
    "semantic_consistency",
    "adaptive_quality"
  ],
  "evaluation_mode": "comprehensive"
}
```

**Response**:
```json
{
  "status": "success",
  "evaluation_id": "eval_xyz789abc123",
  "results": {
    "perceptual_quality": {
      "overall_score": 0.87,
      "confidence": 0.94,
      "frame_scores": [0.85, 0.88, 0.89, 0.86],
      "metadata": {
        "method": "neural_network",
        "model_version": "v2.1"
      }
    },
    "temporal_coherence": {
      "overall_score": 0.82,
      "confidence": 0.89,
      "frame_scores": [0.84, 0.81, 0.83, 0.80],
      "metadata": {
        "optical_flow_scores": [0.85, 0.79, 0.82],
        "feature_consistency_scores": [0.83, 0.83, 0.84]
      }
    },
    "semantic_consistency": {
      "overall_score": 0.79,
      "confidence": 0.86,
      "metadata": {
        "text_consistency": 0.82,
        "caption_consistency": 0.76,
        "cross_modal_alignment": 0.79
      }
    },
    "adaptive_quality": {
      "overall_score": 0.85,
      "confidence": 0.91,
      "metadata": {
        "content_features": {
          "complexity": 0.68,
          "motion": 0.45,
          "texture": 0.52,
          "color_diversity": 0.73
        },
        "adaptive_weights": {
          "sharpness": 1.2,
          "color_quality": 1.0,
          "temporal_smoothness": 1.1
        }
      }
    }
  },
  "overall_score": 0.83,
  "evaluation_time": 5.7
}
```

##### GET /api/v1/research/metrics/supported

List supported novel metrics and their descriptions.

**Response**:
```json
{
  "metrics": {
    "perceptual_quality": {
      "description": "Neural network-based perceptual quality assessment",
      "range": [0, 1],
      "higher_better": true,
      "computational_cost": "high"
    },
    "temporal_coherence": {
      "description": "Motion-aware temporal consistency analysis",
      "range": [0, 1], 
      "higher_better": true,
      "computational_cost": "medium"
    },
    "semantic_consistency": {
      "description": "Cross-modal semantic alignment scoring",
      "range": [0, 1],
      "higher_better": true,
      "computational_cost": "high"
    },
    "adaptive_quality": {
      "description": "Content-adaptive quality assessment",
      "range": [0, 1],
      "higher_better": true,
      "computational_cost": "medium"
    }
  }
}
```

#### 1.3 Validation Framework

**Endpoint**: `/api/v1/research/validation`

##### POST /api/v1/research/validation/validate

Comprehensive validation of research pipeline components.

**Request Body**:
```json
{
  "validation_type": "comprehensive",
  "video_data": {
    "tensor_shape": [10, 3, 64, 64],
    "encoding": "base64"
  },
  "prompts": [
    "A test video prompt",
    "Another validation prompt"
  ],
  "config": {
    "models": ["test_model"],
    "metrics": ["test_metric"],
    "seeds": [42, 123, 456],
    "num_samples_per_seed": 20
  },
  "validation_level": "strict"
}
```

**Response**:
```json
{
  "status": "success",
  "validation_id": "val_def456ghi789",
  "results": {
    "video_data": {
      "is_valid": true,
      "confidence": 1.0,
      "issues": [],
      "warnings": [],
      "metadata": {
        "shape": [10, 3, 64, 64],
        "dtype": "float32",
        "value_range": [0.0, 1.0],
        "std": 0.289
      }
    },
    "prompts": {
      "is_valid": true,
      "confidence": 0.95,
      "issues": [],
      "warnings": ["Prompt 2 has low complexity score"]
    },
    "config": {
      "is_valid": true,
      "confidence": 1.0,
      "issues": [],
      "warnings": []
    },
    "experiment_design": {
      "is_valid": true,
      "confidence": 0.88,
      "issues": [],
      "warnings": ["Consider increasing sample size to 30+"]
    },
    "overall": {
      "is_valid": true,
      "confidence": 0.95,
      "total_validations": 4,
      "failed_validations": 0
    }
  },
  "report": "=== VALIDATION REPORT ===\nOverall Status: PASSED (95% confidence)\n...",
  "validation_time": 2.3
}
```

#### 1.4 Error Handling and Recovery

**Endpoint**: `/api/v1/system/error-handling`

##### GET /api/v1/system/error-handling/stats

Get error handling statistics and recovery performance.

**Response**:
```json
{
  "total_errors": 127,
  "category_breakdown": {
    "memory": 45,
    "computation": 32,
    "io": 28,
    "network": 15,
    "data": 7
  },
  "severity_breakdown": {
    "critical": 3,
    "high": 28,
    "medium": 67,
    "low": 29
  },
  "recovery_attempted": 112,
  "recovery_successful": 95,
  "recovery_rate": 0.848,
  "recent_errors": 8,
  "error_patterns": {
    "memory_RuntimeError": {
      "count": 23,
      "recovery_success_rate": 0.87
    },
    "io_FileNotFoundError": {
      "count": 15,
      "recovery_success_rate": 0.93
    }
  },
  "recovery_strategy_stats": {
    "memory_recovery": {
      "attempts": 45,
      "successes": 39,
      "success_rate": 0.867
    },
    "computation_recovery": {
      "attempts": 32,
      "successes": 28,
      "success_rate": 0.875
    }
  }
}
```

##### POST /api/v1/system/error-handling/simulate

Simulate error conditions for testing recovery mechanisms.

**Request Body**:
```json
{
  "error_type": "memory",
  "severity": "high",
  "context": {
    "operation": "model_inference",
    "batch_size": 8,
    "model_size": "large"
  },
  "test_recovery": true
}
```

**Response**:
```json
{
  "status": "success",
  "simulation_id": "sim_jkl012mno345",
  "error_simulated": {
    "error_type": "RuntimeError",
    "message": "CUDA out of memory",
    "category": "memory",
    "severity": "high"
  },
  "recovery_attempted": true,
  "recovery_successful": true,
  "recovery_strategy": "memory_recovery",
  "recovery_actions": [
    "CUDA cache cleared",
    "Batch size reduced from 8 to 4",
    "Precision switched to FP16"
  ],
  "simulation_time": 0.8
}
```

### 2. Scaling and Optimization APIs

#### 2.1 Intelligent Scaling

**Endpoint**: `/api/v1/scaling`

##### GET /api/v1/scaling/status

Get current scaling system status and metrics.

**Response**:
```json
{
  "status": "active",
  "scaling_mode": "balanced",
  "current_allocation": {
    "cpu_cores": 8,
    "memory_gb": 32,
    "gpu_count": 2,
    "workers": 4
  },
  "resource_metrics": {
    "cpu_usage": 0.65,
    "memory_usage": 0.72,
    "gpu_usage": 0.58,
    "gpu_memory_usage": 0.81
  },
  "recent_actions": [
    {
      "timestamp": "2024-01-15T14:30:22Z",
      "action": "scale_up",
      "resource": "memory",
      "from": 24,
      "to": 32,
      "reason": "Memory usage exceeded 85% threshold"
    }
  ],
  "scaling_statistics": {
    "total_scaling_actions": 23,
    "successful_actions": 21,
    "success_rate": 0.913
  }
}
```

##### POST /api/v1/scaling/predict

Predict resource requirements for a given workload.

**Request Body**:
```json
{
  "workload": {
    "model_count": 3,
    "avg_model_size": 8.5,
    "avg_inference_time": 4.2,
    "batch_size": 2,
    "video_resolution": [512, 512],
    "video_length": 24,
    "complexity_score": 0.7,
    "memory_intensity": 0.8,
    "compute_intensity": 0.9
  }
}
```

**Response**:
```json
{
  "status": "success",
  "prediction_id": "pred_pqr678stu901",
  "predicted_requirements": {
    "cpu_cores": 12,
    "memory_gb": 48,
    "gpu_count": 3,
    "gpu_memory_gb": 24,
    "storage_gb": 150,
    "network_mbps": 500,
    "workers": 6,
    "confidence": 0.89
  },
  "cost_estimate": {
    "hourly_cost": 12.45,
    "daily_cost": 298.80,
    "monthly_cost": 8964.00,
    "currency": "USD"
  },
  "prediction_time": 0.3
}
```

#### 2.2 Quantum Acceleration

**Endpoint**: `/api/v1/optimization/quantum`

##### POST /api/v1/optimization/quantum/compress

Compress model tensors using quantum-inspired techniques.

**Request Body**:
```json
{
  "model_config": {
    "model_type": "diffusion_unet",
    "parameter_count": 860000000,
    "layer_info": {
      "conv_layers": 45,
      "attention_layers": 16,
      "linear_layers": 8
    }
  },
  "compression_method": "tensor_network",
  "compression_ratio_target": 5.0,
  "quality_threshold": 0.95
}
```

**Response**:
```json
{
  "status": "success",
  "compression_id": "comp_vwx234yza567",
  "results": {
    "original_size_mb": 3440,
    "compressed_size_mb": 688,
    "compression_ratio": 5.0,
    "quality_retention": 0.967,
    "memory_savings": 0.80,
    "decomposition_method": "svd",
    "reconstruction_error": 0.033
  },
  "performance_impact": {
    "inference_speedup": 1.23,
    "memory_reduction": 0.80,
    "quality_loss": 0.033
  },
  "compression_time": 45.7
}
```

##### POST /api/v1/optimization/quantum/sample

Generate quantum-enhanced samples for diffusion processes.

**Request Body**:
```json
{
  "distribution_config": {
    "distribution_type": "gaussian",
    "dimensions": 64,
    "parameters": {
      "mean": 0.0,
      "std": 1.0
    }
  },
  "num_samples": 1000,
  "quantum_enhancement": {
    "num_qubits": 8,
    "enhancement_mode": "amplitude_encoding",
    "entanglement_depth": 3
  }
}
```

**Response**:
```json
{
  "status": "success",
  "sampling_id": "samp_bcd890efg123",
  "samples": {
    "shape": [1000, 64],
    "encoding": "base64",
    "format": "numpy_array"
  },
  "quantum_metrics": {
    "entanglement_measure": 0.73,
    "coherence_score": 0.87,
    "fidelity": 0.92
  },
  "classical_comparison": {
    "distribution_match": 0.94,
    "statistical_improvement": 0.18,
    "quality_enhancement": 0.12
  },
  "generation_time": 2.1
}
```

### 3. System Management APIs

#### 3.1 Health and Monitoring

**Endpoint**: `/api/v1/system`

##### GET /api/v1/system/health

Comprehensive system health check.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:35:00Z",
  "version": "1.0.0",
  "components": {
    "api": {
      "status": "healthy",
      "response_time": 0.023,
      "uptime": 86400
    },
    "database": {
      "status": "healthy",
      "connection_pool": "8/10",
      "query_latency": 0.012
    },
    "redis": {
      "status": "healthy",
      "memory_usage": "2.1GB/8GB",
      "connected_clients": 15
    },
    "gpu": {
      "status": "healthy",
      "devices": [
        {
          "id": 0,
          "name": "NVIDIA A100",
          "memory_used": "24GB/80GB",
          "temperature": "62C"
        }
      ]
    }
  },
  "resource_usage": {
    "cpu_percent": 45.2,
    "memory_percent": 62.8,
    "disk_usage": "450GB/1TB",
    "network_io": {
      "bytes_sent": 1234567890,
      "bytes_recv": 987654321
    }
  }
}
```

##### GET /api/v1/system/metrics

Get detailed system metrics and performance statistics.

**Response**:
```json
{
  "timestamp": "2024-01-15T14:35:00Z",
  "metrics": {
    "request_metrics": {
      "total_requests": 12450,
      "requests_per_second": 15.2,
      "avg_response_time": 0.234,
      "error_rate": 0.012
    },
    "resource_metrics": {
      "cpu_usage_history": [0.45, 0.52, 0.48, 0.51],
      "memory_usage_history": [0.62, 0.64, 0.63, 0.65],
      "gpu_utilization": [0.78, 0.82, 0.75, 0.80]
    },
    "experiment_metrics": {
      "active_experiments": 8,
      "completed_experiments": 127,
      "failed_experiments": 3,
      "avg_experiment_duration": 1847.5
    },
    "model_metrics": {
      "models_loaded": 5,
      "cache_hit_rate": 0.847,
      "avg_inference_time": 3.2
    }
  }
}
```

#### 3.2 Configuration Management

##### GET /api/v1/system/config

Get current system configuration.

**Response**:
```json
{
  "environment": "production",
  "version": "1.0.0",
  "configuration": {
    "research_framework": {
      "adaptive_algorithms_enabled": true,
      "novel_metrics_enabled": true,
      "validation_framework_enabled": true,
      "error_recovery_enabled": true
    },
    "scaling": {
      "auto_scaling_enabled": true,
      "scaling_mode": "balanced",
      "max_workers": 16,
      "resource_limits": {
        "max_cpu_cores": 64,
        "max_memory_gb": 512,
        "max_gpu_count": 8
      }
    },
    "security": {
      "authentication_required": true,
      "encryption_enabled": true,
      "audit_logging": true
    },
    "monitoring": {
      "metrics_collection": true,
      "log_level": "INFO",
      "alert_thresholds": {
        "cpu_usage": 0.85,
        "memory_usage": 0.90,
        "error_rate": 0.05
      }
    }
  }
}
```

##### PUT /api/v1/system/config

Update system configuration (admin only).

**Request Body**:
```json
{
  "scaling": {
    "scaling_mode": "aggressive",
    "max_workers": 20
  },
  "monitoring": {
    "log_level": "DEBUG"
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Configuration updated successfully",
  "updated_fields": [
    "scaling.scaling_mode",
    "scaling.max_workers", 
    "monitoring.log_level"
  ],
  "restart_required": false
}
```

## Error Responses

All APIs follow consistent error response format:

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid video tensor format",
  "details": {
    "field": "video_data.tensor_shape",
    "expected": "4D tensor [T, C, H, W]",
    "received": "3D tensor [C, H, W]"
  },
  "request_id": "req_abc123def456",
  "timestamp": "2024-01-15T14:35:00Z"
}
```

### Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `RESOURCE_ERROR`: Insufficient resources
- `MODEL_ERROR`: Model loading or inference error  
- `COMPUTATION_ERROR`: Mathematical computation error
- `NETWORK_ERROR`: Network connectivity issue
- `AUTHENTICATION_ERROR`: Authentication failure
- `PERMISSION_ERROR`: Insufficient permissions
- `RATE_LIMIT_ERROR`: Rate limit exceeded
- `INTERNAL_ERROR`: Internal server error

## Rate Limiting

All APIs are subject to rate limiting:

- **Standard endpoints**: 100 requests/minute
- **Compute-intensive endpoints**: 10 requests/minute  
- **Admin endpoints**: 50 requests/minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642259400
```

## Authentication

Production deployments require Bearer token authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
     -H "Content-Type: application/json" \
     https://api.vid-diffusion-bench.com/api/v1/system/health
```

## SDKs and Client Libraries

### Python SDK

```python
from vid_diffusion_bench_client import VidDiffusionClient

# Initialize client
client = VidDiffusionClient(
    base_url="https://api.vid-diffusion-bench.com",
    api_token="your_api_token"
)

# Use adaptive optimization
result = client.adaptive.optimize(
    video_data=video_tensor,
    model_name="svd-xt-1.1",
    base_config=config
)

# Evaluate with novel metrics
metrics = client.metrics.evaluate(
    generated_video=video,
    text_prompt="A cat playing",
    metrics=["perceptual_quality", "temporal_coherence"]
)
```

### JavaScript SDK

```javascript
import { VidDiffusionClient } from '@vid-diffusion-bench/client';

const client = new VidDiffusionClient({
  baseUrl: 'https://api.vid-diffusion-bench.com',
  apiToken: 'your_api_token'
});

// Predict resource requirements
const prediction = await client.scaling.predict({
  workload: {
    model_count: 3,
    batch_size: 2,
    video_resolution: [512, 512]
  }
});
```

## Examples and Tutorials

Complete examples and tutorials are available in the `/examples` directory of the repository, including:

- Getting started with adaptive optimization
- Implementing custom novel metrics
- Setting up production deployments  
- Building research pipelines
- Quantum acceleration integration

For support and questions, please refer to the documentation at `https://docs.vid-diffusion-bench.com` or open an issue on the GitHub repository.