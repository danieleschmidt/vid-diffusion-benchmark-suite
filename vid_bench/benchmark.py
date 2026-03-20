"""
Abstract base class and registry for video diffusion model benchmarks.
"""
from __future__ import annotations
import abc
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class BenchmarkReport:
    model_name: str
    fvd_proxy: float               # Frame-level FID proxy via pixel statistics
    temporal_consistency: float    # Mean cosine sim between consecutive frames [0,1]
    vram_estimate_mb: float        # Estimated VRAM usage in MB
    latency_ms: float              # Inference latency in milliseconds
    num_frames: int
    frame_height: int
    frame_width: int
    extra_metrics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkReport":
        extra = d.pop("extra_metrics", {})
        return cls(**d, extra_metrics=extra)


class VideoModelBenchmark(abc.ABC):
    """Abstract base for video diffusion model benchmarks."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        ...

    @abc.abstractmethod
    def generate_frames(self, prompt: str, num_frames: int,
                        height: int, width: int) -> np.ndarray:
        """
        Generate video frames.
        Returns ndarray of shape (num_frames, height, width, 3), dtype uint8.
        """
        ...

    @abc.abstractmethod
    def get_param_count(self) -> int:
        """Return number of model parameters."""
        ...

    def run(self, prompt: str = "a dog running", num_frames: int = 8,
            height: int = 64, width: int = 64) -> BenchmarkReport:
        """Run the full benchmark suite."""
        from vid_bench.metrics import (
            compute_fvd_proxy, compute_temporal_consistency,
            estimate_vram_mb
        )
        start = time.perf_counter()
        frames = self.generate_frames(prompt, num_frames, height, width)
        latency_ms = (time.perf_counter() - start) * 1000

        fvd = compute_fvd_proxy(frames)
        tc = compute_temporal_consistency(frames)
        vram = estimate_vram_mb(self.get_param_count())

        return BenchmarkReport(
            model_name=self.model_name,
            fvd_proxy=fvd,
            temporal_consistency=tc,
            vram_estimate_mb=vram,
            latency_ms=latency_ms,
            num_frames=num_frames,
            frame_height=height,
            frame_width=width,
        )
