# ADR-001: Docker Containerization Strategy for Model Isolation

## Status
Accepted

## Context
The Video Diffusion Benchmark Suite needs to evaluate 300+ models with vastly different dependencies, Python versions, and system requirements. Many models have conflicting dependencies or require specific CUDA versions, making it impossible to run them in a single environment.

## Decision
We will use Docker containerization with a standardized adapter interface to isolate each model in its own environment while maintaining consistent evaluation protocols.

### Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model A       │    │   Model B       │    │   Model C       │
│   (Container)   │    │   (Container)   │    │   (Container)   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ Python 3.8      │    │ Python 3.11     │    │ Python 3.10     │
│ PyTorch 1.12    │    │ PyTorch 2.0     │    │ JAX 0.4         │
│ CUDA 11.6       │    │ CUDA 12.0       │    │ CUDA 11.8       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │           Evaluation Orchestrator               │
         │     (Standardized Input/Output Interface)       │
         └─────────────────────────────────────────────────┘
```

### Container Standards
1. **Base Images**: NVIDIA CUDA images for GPU support
2. **Interface**: Standardized REST API for model interaction
3. **Resource Limits**: Configurable CPU/memory/GPU allocation
4. **Isolation**: No shared state between model containers
5. **Versioning**: Immutable container tags for reproducibility

## Alternatives Considered

### 1. Conda/Virtual Environments
- **Pros**: Lighter weight, faster startup
- **Cons**: Dependency conflicts, system-level requirements, CUDA version conflicts
- **Verdict**: Rejected due to unsolvable dependency conflicts

### 2. Kubernetes Pods
- **Pros**: Enterprise-grade orchestration, auto-scaling
- **Cons**: Increased complexity, resource overhead for single-node deployments
- **Verdict**: Deferred to future phases, Docker Compose sufficient for MVP

### 3. Serverless Functions
- **Pros**: Pay-per-use, infinite scaling
- **Cons**: Cold start latency, size limits, GPU support limitations
- **Verdict**: Rejected due to model size and GPU requirements

## Consequences

### Positive
- **Dependency Isolation**: Each model can use its optimal environment
- **Reproducibility**: Immutable containers ensure consistent evaluation
- **Parallel Execution**: Multiple models can run simultaneously
- **Security**: Container isolation prevents model interference
- **Scalability**: Easy to distribute across multiple machines

### Negative
- **Storage Overhead**: Each container includes full dependencies (~5-10GB each)
- **Build Time**: Initial container builds are time-consuming
- **Resource Usage**: Higher memory overhead compared to shared environments
- **Complexity**: Requires Docker expertise for contributors

### Mitigation Strategies
- **Layer Caching**: Use multi-stage builds and shared base layers
- **Registry Optimization**: Pre-built images to reduce build times
- **Resource Monitoring**: Implement container resource tracking
- **Documentation**: Comprehensive Docker guides for contributors

## Implementation Details

### Container Structure
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install model-specific dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model adapter
COPY adapter/ /app/adapter/

# Standardized interface
EXPOSE 8080
CMD ["python", "/app/adapter/server.py"]
```

### Adapter Interface
```python
class ModelAdapter:
    def generate(self, prompt: str, **kwargs) -> torch.Tensor:
        """Generate video from text prompt."""
        pass
    
    def get_requirements(self) -> Dict[str, Any]:
        """Return hardware requirements."""
        pass
    
    def health_check(self) -> bool:
        """Verify model is ready."""
        pass
```

### Orchestration
- **Docker Compose**: Local development and single-node deployment
- **Future**: Kubernetes for multi-node scaling
- **Registry**: Private Docker registry for sharing images

## Validation
- [x] Prototype with 3 different models (SVD, Pika, CogVideo)
- [x] Measure resource overhead (<20% memory increase acceptable)
- [x] Verify GPU sharing works correctly between containers
- [x] Test parallel execution without interference
- [x] Validate reproducibility across different host systems

## References
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Multi-stage Build Optimization](https://docs.docker.com/build/building/multi-stage/)

## Decision Date
2025-07-01

## Decision Makers
- Daniel Schmidt (Technical Lead)
- [To be added: DevOps Engineer]
- [To be added: Research Scientist]