"""Video Diffusion Benchmark Suite.

A unified test-bed for next-gen open-source video diffusion models (VDMs).
Provides standardized evaluation framework for comparing latency, quality,
and VRAM trade-offs across 300+ video generation models.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "vid-bench@yourdomain.com"

from .benchmark import BenchmarkSuite
from .models import ModelAdapter, register_model
from .metrics import VideoQualityMetrics
from .prompts import StandardPrompts
from .profiler import EfficiencyProfiler

__all__ = [
    "BenchmarkSuite",
    "ModelAdapter", 
    "register_model",
    "VideoQualityMetrics",
    "StandardPrompts",
    "EfficiencyProfiler",
]