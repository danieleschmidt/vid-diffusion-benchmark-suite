# ADR-0003: Standardized Evaluation Protocol

## Status
Accepted

## Context

Video diffusion model evaluation lacks standardization across the research community:
- Different papers use different datasets and metrics
- Inconsistent prompt engineering and generation parameters
- Varied hardware configurations affecting reproducibility
- No standard for temporal aspects (frame rates, duration)
- Quality metrics often incomparable between studies

This fragmentation makes it impossible to:
- Compare models fairly across different papers
- Track progress in the field objectively
- Make informed deployment decisions
- Reproduce research results reliably

## Decision

We will establish a comprehensive standardized evaluation protocol:

### 1. Standard Prompt Sets
- **Diverse Set V2**: 100 carefully curated prompts covering various scenarios
- **Motion Dynamics**: 25 prompts specifically testing motion quality
- **Scene Transitions**: 25 prompts testing temporal consistency
- **Camera Movements**: 25 prompts testing viewpoint changes
- **Edge Cases**: 25 challenging prompts (complex scenes, unusual requests)

### 2. Fixed Generation Parameters
- **Resolution**: 576x1024 (mobile-optimized aspect ratio)
- **Frame Count**: 25 frames (standard duration)
- **Frame Rate**: 7 FPS (balance of quality and speed)
- **Guidance Scale**: 7.5 (optimal for most models)
- **Random Seed**: Fixed per prompt for reproducibility

### 3. Standardized Metrics Suite
- **Quality Metrics**: FVD, IS, CLIP Similarity, LPIPS
- **Temporal Metrics**: Temporal Consistency, Motion Quality, Frame Interpolation Error
- **Efficiency Metrics**: Inference Latency, Peak VRAM, Throughput, Power Consumption
- **Composite Score**: Weighted combination balancing quality and efficiency

### 4. Hardware Configuration
- **Primary Tier**: RTX 4090 (24GB VRAM) - consumer high-end
- **Secondary Tier**: RTX 3080 (10GB VRAM) - consumer mid-range
- **Enterprise Tier**: A100 (40GB VRAM) - research/production
- **Power Measurement**: NVIDIA-ML for accurate power consumption

### 5. Evaluation Environment
- **Container Isolation**: Each model in separate Docker container
- **Resource Limits**: GPU memory allocation, CPU cores, wall-clock timeout
- **Multiple Runs**: 3 runs per prompt with statistical aggregation
- **Warmup**: 5 warmup generations before measurement

## Consequences

### Positive
- **Fair Comparison**: All models evaluated under identical conditions
- **Reproducibility**: Fixed parameters ensure consistent results
- **Community Adoption**: Standard that other researchers can adopt
- **Progress Tracking**: Clear metrics for field advancement
- **Practical Relevance**: Hardware tiers match real deployment scenarios

### Negative
- **Rigidity**: May not capture model-specific optimal settings
- **Hardware Dependence**: Results tied to specific GPU architectures
- **Prompt Bias**: Fixed prompt set may favor certain model types
- **Metric Limitations**: No single metric captures all aspects of video quality

### Mitigation Strategies
- Provide multiple evaluation modes (standard, custom, research)
- Regular review and update of prompt sets based on community feedback
- Clear documentation of protocol limitations and intended use cases
- Allow optional custom evaluation alongside standard protocol