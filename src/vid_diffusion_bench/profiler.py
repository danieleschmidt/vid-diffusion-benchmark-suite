"""Performance and efficiency profiling tools."""

import time
import psutil
import torch
from typing import Dict, Optional
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ProfileStats:
    """Container for profiling statistics."""
    latency_ms: float
    throughput_fps: float
    vram_peak_gb: float
    power_watts: float
    cpu_percent: float


class EfficiencyProfiler:
    """Hardware efficiency and performance profiler."""
    
    def __init__(self):
        """Initialize profiler."""
        self._stats = {}
        self._start_time = None
        self._start_memory = None
        
    @contextmanager
    def track(self, model_name: str):
        """Context manager for tracking model performance.
        
        Args:
            model_name: Name of model being profiled
        """
        # Start tracking
        self._start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._start_memory = torch.cuda.memory_allocated()
            
        try:
            yield
        finally:
            # End tracking and compute stats
            end_time = time.perf_counter()
            latency_ms = (end_time - self._start_time) * 1000
            
            vram_peak_gb = 0.0
            if torch.cuda.is_available():
                vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                
            self._stats[model_name] = ProfileStats(
                latency_ms=latency_ms,
                throughput_fps=1000.0 / latency_ms if latency_ms > 0 else 0.0,
                vram_peak_gb=vram_peak_gb,
                power_watts=0.0,  # Placeholder
                cpu_percent=psutil.cpu_percent()
            )
            
    def get_stats(self, model_name: Optional[str] = None) -> ProfileStats:
        """Get profiling statistics.
        
        Args:
            model_name: Specific model name, or latest if None
            
        Returns:
            Profiling statistics
        """
        if model_name:
            return self._stats[model_name]
        return list(self._stats.values())[-1] if self._stats else ProfileStats(0,0,0,0,0)