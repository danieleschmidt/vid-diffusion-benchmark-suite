# ADR-0001: Containerized Model Isolation

## Status
Accepted

## Context

Video diffusion models have complex and often conflicting dependency requirements. Different models may require:
- Different versions of PyTorch, CUDA, or other ML frameworks
- Specific Python versions or system libraries
- Conflicting package versions that cannot coexist
- Large model weights that need isolated storage

Running multiple models in the same environment leads to:
- Dependency conflicts and version mismatches
- Difficult environment management and reproducibility issues
- Potential memory leaks or interference between models
- Challenges in resource allocation and monitoring

## Decision

We will isolate each video diffusion model in its own Docker container with:

1. **Individual Dockerfiles** for each model with pinned dependencies
2. **Standardized interface** for model containers (REST API)
3. **Resource constraints** defined per container (GPU memory, CPU cores)
4. **Shared evaluation framework** that communicates with containers
5. **Container registry** for pre-built model images

Key implementation details:
- Each model container exposes a REST API on port 8000
- Standard request/response format for generation requests
- Automatic resource cleanup after evaluation
- Container health checks and monitoring

## Consequences

### Positive
- **Reproducibility**: Each model runs in identical environment every time
- **Isolation**: No dependency conflicts between models
- **Scalability**: Easy to add new models without affecting existing ones
- **Resource management**: Clear resource allocation and monitoring per model
- **Maintenance**: Individual containers can be updated independently

### Negative
- **Storage overhead**: Each container includes full dependency stack
- **Startup time**: Container initialization adds latency to benchmarks
- **Complexity**: Additional orchestration layer required
- **Development overhead**: Need to maintain Dockerfiles for each model

### Mitigation Strategies
- Use multi-stage builds to minimize image size
- Implement container warming and caching strategies
- Provide development tools for easy container management
- Create automated testing for container builds