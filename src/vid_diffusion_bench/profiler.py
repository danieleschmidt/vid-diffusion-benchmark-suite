"""Performance and efficiency profiling tools."""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any, Tuple

import psutil
import torch
import numpy as np

# Try to import NVIDIA monitoring tools
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    
logger = logging.getLogger(__name__)


@dataclass
class ProfileStats:
    """Container for profiling statistics."""
    latency_ms: float
    throughput_fps: float
    vram_peak_gb: float
    vram_allocated_gb: float
    power_watts: float
    gpu_utilization_percent: float
    cpu_percent: float
    memory_gb: float
    temperature_celsius: float
    model_name: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class SystemSpecs:
    """System hardware specifications."""
    gpu_name: str
    gpu_memory_gb: float
    cpu_name: str
    cpu_cores: int
    memory_gb: float
    python_version: str
    torch_version: str
    cuda_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class GPUMonitor:
    """GPU monitoring utility using NVIDIA Management Library."""
    
    def __init__(self):
        self.nvml_available = NVML_AVAILABLE
        self.device_count = 0
        self.handles = []
        
        if self.nvml_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
                logger.info(f"NVML initialized with {self.device_count} GPU(s)")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_available = False
                
    def get_gpu_stats(self, device_id: int = 0) -> Dict[str, float]:
        """Get current GPU statistics."""
        stats = {
            "utilization_percent": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
            "temperature_celsius": 0.0,
            "power_watts": 0.0,
            "power_limit_watts": 0.0
        }
        
        if not self.nvml_available or device_id >= len(self.handles):
            # Fallback to PyTorch metrics
            if torch.cuda.is_available():
                stats["memory_used_gb"] = torch.cuda.memory_allocated(device_id) / (1024**3)
                stats["memory_total_gb"] = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            return stats
            
        try:
            handle = self.handles[device_id]
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["utilization_percent"] = util.gpu
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats["memory_used_gb"] = mem_info.used / (1024**3)
            stats["memory_total_gb"] = mem_info.total / (1024**3)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats["temperature_celsius"] = temp
            
            # Power consumption
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                stats["power_watts"] = power
                
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                stats["power_limit_watts"] = power_limit
            except:
                pass  # Power monitoring not available on all GPUs
                
        except Exception as e:
            logger.debug(f"Error getting GPU stats: {e}")
            
        return stats
        
    def get_gpu_name(self, device_id: int = 0) -> str:
        """Get GPU device name."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device_id)
        return "Unknown GPU"


class ResourceMonitor:
    """Continuous resource monitoring in background thread."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.gpu_monitor = GPUMonitor()
        self.monitoring = False
        self.samples = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 samples
        self.monitor_thread = None
        
    def start_monitoring(self, model_name: str):
        """Start monitoring resources for a model."""
        if self.monitoring:
            self.stop_monitoring()
            
        self.monitoring = True
        self.current_model = model_name
        self.samples[model_name].clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring resources."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_gb": psutil.virtual_memory().used / (1024**3),
                    **self.gpu_monitor.get_gpu_stats()
                }
                
                self.samples[self.current_model].append(sample)
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.debug(f"Monitor loop error: {e}")
                
    def get_peak_stats(self, model_name: str) -> Dict[str, float]:
        """Get peak resource usage for a model."""
        if model_name not in self.samples or not self.samples[model_name]:
            return {
                "peak_gpu_utilization": 0.0,
                "peak_memory_gb": 0.0,
                "peak_power_watts": 0.0,
                "avg_temperature": 0.0,
                "avg_cpu_percent": 0.0
            }
            
        samples = list(self.samples[model_name])
        
        return {
            "peak_gpu_utilization": max(s["utilization_percent"] for s in samples),
            "peak_memory_gb": max(s["memory_used_gb"] for s in samples),
            "peak_power_watts": max(s["power_watts"] for s in samples),
            "avg_temperature": np.mean([s["temperature_celsius"] for s in samples]),
            "avg_cpu_percent": np.mean([s["cpu_percent"] for s in samples])
        }


class EfficiencyProfiler:
    """Hardware efficiency and performance profiler."""
    
    def __init__(self, enable_continuous_monitoring: bool = True):
        """Initialize profiler.
        
        Args:
            enable_continuous_monitoring: Whether to enable background resource monitoring
        """
        self._stats = {}
        self._current_session = None
        self.gpu_monitor = GPUMonitor()
        
        if enable_continuous_monitoring:
            self.resource_monitor = ResourceMonitor()
        else:
            self.resource_monitor = None
            
        # Cache system specs
        self.system_specs = self._get_system_specs()
        
        logger.info("EfficiencyProfiler initialized")
        
    def _get_system_specs(self) -> SystemSpecs:
        """Get system hardware specifications."""
        # GPU info
        gpu_name = self.gpu_monitor.get_gpu_name() if torch.cuda.is_available() else "CPU Only"
        gpu_memory = 0.0
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_memory = props.total_memory / (1024**3)
            
        # CPU info
        cpu_name = "Unknown CPU"
        try:
            if sys.platform == "linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
        except:
            pass
            
        return SystemSpecs(
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory,
            cpu_name=cpu_name,
            cpu_cores=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or "Not Available"
        )
        
    @contextmanager
    def track(self, model_name: str):
        """Context manager for tracking model performance.
        
        Args:
            model_name: Name of model being profiled
        """
        logger.debug(f"Starting profiling session for: {model_name}")
        
        # Initialize session
        session = {
            "model_name": model_name,
            "start_time": time.perf_counter(),
            "start_timestamp": time.time(),
            "start_memory": {},
            "start_gpu_stats": {}
        }
        
        # Reset GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            session["start_memory"]["gpu_allocated"] = torch.cuda.memory_allocated()
            session["start_memory"]["gpu_reserved"] = torch.cuda.memory_reserved()
            
        # Get initial GPU stats
        session["start_gpu_stats"] = self.gpu_monitor.get_gpu_stats()
        
        # Start continuous monitoring if available
        if self.resource_monitor:
            self.resource_monitor.start_monitoring(model_name)
            
        self._current_session = session
        
        try:
            yield
        finally:
            # End profiling session
            self._finalize_session()
            
    def _finalize_session(self):
        """Finalize current profiling session."""
        if not self._current_session:
            return
            
        session = self._current_session
        model_name = session["model_name"]
        
        # Calculate latency
        end_time = time.perf_counter()
        latency_ms = (end_time - session["start_time"]) * 1000
        
        # GPU memory stats
        vram_peak_gb = 0.0
        vram_allocated_gb = 0.0
        if torch.cuda.is_available():
            vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            vram_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            
        # Get final GPU stats
        end_gpu_stats = self.gpu_monitor.get_gpu_stats()
        
        # Calculate power and temperature (average during execution)
        power_watts = end_gpu_stats.get("power_watts", 0.0)
        temperature = end_gpu_stats.get("temperature_celsius", 0.0)
        gpu_utilization = end_gpu_stats.get("utilization_percent", 0.0)
        
        # Stop continuous monitoring and get peak stats
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
            peak_stats = self.resource_monitor.get_peak_stats(model_name)
            
            # Use peak values from continuous monitoring
            power_watts = max(power_watts, peak_stats.get("peak_power_watts", 0.0))
            temperature = peak_stats.get("avg_temperature", temperature)
            gpu_utilization = max(gpu_utilization, peak_stats.get("peak_gpu_utilization", 0.0))
            
        # Calculate throughput
        throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        
        # Store final stats
        stats = ProfileStats(
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            vram_peak_gb=vram_peak_gb,
            vram_allocated_gb=vram_allocated_gb,
            power_watts=power_watts,
            gpu_utilization_percent=gpu_utilization,
            cpu_percent=psutil.cpu_percent(),
            memory_gb=psutil.virtual_memory().used / (1024**3),
            temperature_celsius=temperature,
            model_name=model_name,
            timestamp=time.time()
        )
        
        self._stats[model_name] = stats
        self._current_session = None
        
        logger.debug(f"Profiling completed for {model_name}: {latency_ms:.1f}ms, {vram_peak_gb:.1f}GB VRAM")
        
    def get_stats(self, model_name: Optional[str] = None) -> ProfileStats:
        """Get profiling statistics.
        
        Args:
            model_name: Specific model name, or latest if None
            
        Returns:
            Profiling statistics
        """
        if model_name and model_name in self._stats:
            return self._stats[model_name]
        elif self._stats:
            return list(self._stats.values())[-1]
        else:
            # Return empty stats
            return ProfileStats(
                latency_ms=0.0, throughput_fps=0.0, vram_peak_gb=0.0,
                vram_allocated_gb=0.0, power_watts=0.0, gpu_utilization_percent=0.0,
                cpu_percent=0.0, memory_gb=0.0, temperature_celsius=0.0,
                model_name="unknown", timestamp=time.time()
            )
            
    def get_all_stats(self) -> Dict[str, ProfileStats]:
        """Get all profiling statistics."""
        return self._stats.copy()
        
    def profile_model(
        self, 
        model_name: str, 
        batch_size: int = 1, 
        num_frames: int = 16,
        resolution: Tuple[int, int] = (512, 512)
    ) -> ProfileStats:
        """Profile a model with specific parameters.
        
        Args:
            model_name: Name of model to profile
            batch_size: Batch size for profiling
            num_frames: Number of frames to generate
            resolution: Video resolution
            
        Returns:
            Profiling statistics
        """
        # This would be implemented to actually run the model
        # For now, return mock stats based on parameters
        
        mock_latency = batch_size * num_frames * 0.1  # 100ms per frame
        mock_vram = batch_size * (resolution[0] * resolution[1] / 512**2) * 2.0  # 2GB base
        
        stats = ProfileStats(
            latency_ms=mock_latency * 1000,
            throughput_fps=1000.0 / (mock_latency * 1000),
            vram_peak_gb=mock_vram,
            vram_allocated_gb=mock_vram * 0.8,
            power_watts=250.0,
            gpu_utilization_percent=85.0,
            cpu_percent=25.0,
            memory_gb=8.0,
            temperature_celsius=72.0,
            model_name=model_name,
            timestamp=time.time()
        )
        
        self._stats[f"{model_name}_profile"] = stats
        return stats
        
    def save_stats(self, filepath: str):
        """Save profiling statistics to file.
        
        Args:
            filepath: Path to save statistics JSON file
        """
        data = {
            "system_specs": self.system_specs.to_dict(),
            "profiling_stats": {name: stats.to_dict() for name, stats in self._stats.items()},
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Profiling statistics saved to: {filepath}")
        
    def load_stats(self, filepath: str):
        """Load profiling statistics from file.
        
        Args:
            filepath: Path to load statistics JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Load stats
        for name, stats_dict in data.get("profiling_stats", {}).items():
            self._stats[name] = ProfileStats(**stats_dict)
            
        logger.info(f"Loaded {len(self._stats)} profiling records from: {filepath}")
        
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare profiling statistics across models.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Comparison analysis
        """
        available_models = [name for name in model_names if name in self._stats]
        
        if not available_models:
            return {"error": "No profiling data available for specified models"}
            
        comparison = {
            "models": available_models,
            "metrics": {},
            "rankings": {},
            "efficiency_analysis": {}
        }
        
        # Extract metrics for comparison
        metrics_data = {}
        for model in available_models:
            stats = self._stats[model]
            metrics_data[model] = {
                "latency_ms": stats.latency_ms,
                "throughput_fps": stats.throughput_fps,
                "vram_peak_gb": stats.vram_peak_gb,
                "power_watts": stats.power_watts,
                "efficiency_score": self._calculate_efficiency_score(stats)
            }
            
        comparison["metrics"] = metrics_data
        
        # Create rankings
        comparison["rankings"] = {
            "fastest": sorted(available_models, key=lambda x: metrics_data[x]["latency_ms"]),
            "most_efficient": sorted(available_models, key=lambda x: metrics_data[x]["efficiency_score"], reverse=True),
            "lowest_memory": sorted(available_models, key=lambda x: metrics_data[x]["vram_peak_gb"]),
            "lowest_power": sorted(available_models, key=lambda x: metrics_data[x]["power_watts"])
        }
        
        return comparison
        
    def _calculate_efficiency_score(self, stats: ProfileStats) -> float:
        """Calculate efficiency score for a model."""
        # Weighted score: speed (40%), memory efficiency (30%), power efficiency (30%)
        speed_score = max(0, (5000 - stats.latency_ms) / 5000)  # Normalize to 0-1
        memory_score = max(0, (16 - stats.vram_peak_gb) / 16)  # Assume 16GB max
        power_score = max(0, (400 - stats.power_watts) / 400)  # Assume 400W max
        
        return (speed_score * 0.4 + memory_score * 0.3 + power_score * 0.3) * 100


# Utility functions
def benchmark_system() -> Dict[str, Any]:
    """Run system benchmark to establish baseline performance."""
    profiler = EfficiencyProfiler()
    
    # Simple GPU benchmark
    if torch.cuda.is_available():
        with profiler.track("system_benchmark"):
            # Simulate some GPU work
            x = torch.randn(1000, 1000, device="cuda")
            for _ in range(100):
                x = torch.mm(x, x.t())
                
    return {
        "system_specs": profiler.system_specs.to_dict(),
        "benchmark_stats": profiler.get_stats().to_dict() if profiler._stats else None
    }


def profile_gpu_memory_usage(func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """Profile GPU memory usage of a function.
    
    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (function_result, memory_stats)
    """
    if not torch.cuda.is_available():
        result = func(*args, **kwargs)
        return result, {"memory_used_gb": 0.0}
        
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    result = func(*args, **kwargs)
    
    peak_memory = torch.cuda.max_memory_allocated()
    final_memory = torch.cuda.memory_allocated()
    
    memory_stats = {
        "initial_memory_gb": initial_memory / (1024**3),
        "peak_memory_gb": peak_memory / (1024**3),
        "final_memory_gb": final_memory / (1024**3),
        "memory_used_gb": (peak_memory - initial_memory) / (1024**3)
    }
    
    return result, memory_stats