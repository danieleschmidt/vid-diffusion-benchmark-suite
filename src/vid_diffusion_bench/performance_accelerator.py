"""Advanced performance acceleration and optimization for video diffusion benchmarks."""

import time
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
import json
from pathlib import Path
import weakref

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for intelligent caching system."""
    max_memory_mb: int = 2048
    max_disk_mb: int = 10240
    ttl_seconds: int = 3600
    compression_enabled: bool = True
    cache_dir: str = "cache/benchmarks"


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_model_compilation: bool = True
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_memory_efficient_attention: bool = True
    batch_size_optimization: bool = True
    tensor_parallelism: bool = False
    pipeline_parallelism: bool = False
    max_batch_size: int = 8
    memory_fraction: float = 0.9


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    operation: str
    duration: float
    memory_used: float
    gpu_memory_used: float
    throughput: float
    cache_hit_rate: float
    optimization_applied: List[str]
    timestamp: float


class IntelligentCache:
    """Multi-level intelligent caching system."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.memory_cache = {}
        self.cache_metadata = {}
        self.access_counts = {}
        self.access_times = {}
        self.cache_size = 0
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # Setup disk cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Intelligent cache initialized: {self.config.max_memory_mb}MB memory, {self.config.max_disk_mb}MB disk")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        # Create deterministic hash from arguments
        content = pickle.dumps((args, sorted(kwargs.items())), protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(content).hexdigest()
    
    def _get_memory_usage(self, obj: Any) -> int:
        """Estimate memory usage of object in bytes."""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 1024  # Default estimate
    
    def _evict_lru_memory(self, required_space: int):
        """Evict least recently used items from memory cache."""
        if not self.memory_cache:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(
            self.memory_cache.keys(),
            key=lambda k: self.access_times.get(k, 0)
        )
        
        freed_space = 0
        for key in sorted_keys:
            if freed_space >= required_space:
                break
            
            if key in self.memory_cache:
                obj_size = self.cache_metadata.get(key, {}).get('size', 0)
                del self.memory_cache[key]
                del self.cache_metadata[key]
                self.access_counts.pop(key, None)
                self.access_times.pop(key, None)
                self.cache_size -= obj_size
                freed_space += obj_size
                
                logger.debug(f"Evicted cache entry {key[:8]}... ({obj_size} bytes)")
    
    def _store_to_disk(self, key: str, value: Any) -> bool:
        """Store value to disk cache."""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            
            # Check disk space
            disk_usage = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            max_disk_bytes = self.config.max_disk_mb * 1024 * 1024
            
            if disk_usage > max_disk_bytes:
                # Clean up old disk cache files
                self._cleanup_disk_cache()
            
            with open(file_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to store to disk cache: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Any:
        """Load value from disk cache."""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            
            if not file_path.exists():
                return None
            
            # Check TTL
            age = time.time() - file_path.stat().st_mtime
            if age > self.config.ttl_seconds:
                file_path.unlink()
                return None
            
            with open(file_path, 'rb') as f:
        # SECURITY: pickle.loads() can execute arbitrary code. Only use with trusted data.
                return pickle.load(f)
                
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def _cleanup_disk_cache(self):
        """Clean up expired or excess disk cache files."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        current_time = time.time()
        
        # Remove expired files
        for file_path in cache_files:
            age = current_time - file_path.stat().st_mtime
            if age > self.config.ttl_seconds:
                file_path.unlink()
                logger.debug(f"Removed expired cache file: {file_path.name}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            current_time = time.time()
            
            # Check memory cache first
            if key in self.memory_cache:
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.access_times[key] = current_time
                self.hit_count += 1
                return self.memory_cache[key]
            
            # Check disk cache
            value = self._load_from_disk(key)
            if value is not None:
                # Promote to memory cache if space allows
                obj_size = self._get_memory_usage(value)
                max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
                
                if self.cache_size + obj_size <= max_memory_bytes:
                    self.memory_cache[key] = value
                    self.cache_metadata[key] = {
                        'size': obj_size,
                        'created': current_time
                    }
                    self.cache_size += obj_size
                    self.access_counts[key] = 1
                    self.access_times[key] = current_time
                
                self.hit_count += 1
                return value
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put value into cache."""
        with self.lock:
            current_time = time.time()
            obj_size = self._get_memory_usage(value)
            max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
            
            # Check if we need to evict from memory cache
            if self.cache_size + obj_size > max_memory_bytes:
                required_space = (self.cache_size + obj_size) - max_memory_bytes
                self._evict_lru_memory(required_space)
            
            # Store in memory cache
            if obj_size <= max_memory_bytes:
                self.memory_cache[key] = value
                self.cache_metadata[key] = {
                    'size': obj_size,
                    'created': current_time
                }
                self.cache_size += obj_size
                self.access_counts[key] = 1
                self.access_times[key] = current_time
            
            # Also store to disk for persistence
            self._store_to_disk(key, value)
    
    def cache_function(self, func: Callable) -> Callable:
        """Decorator to cache function results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = self.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}, executing...")
            result = func(*args, **kwargs)
            self.put(cache_key, result)
            
            return result
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total_requests)
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "memory_entries": len(self.memory_cache),
            "memory_size_mb": self.cache_size / (1024 * 1024),
            "max_memory_mb": self.config.max_memory_mb
        }


class ModelOptimizer:
    """Advanced model optimization for video diffusion models."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimized_models = weakref.WeakValueDictionary()
        self.optimization_cache = IntelligentCache()
        
        logger.info("Model optimizer initialized")
    
    def optimize_model(self, model: Any, model_name: str = "unknown") -> Any:
        """Apply comprehensive optimizations to a model."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping model optimization")
            return model
        
        start_time = time.time()
        optimizations_applied = []
        
        try:
            # Check if already optimized
            if id(model) in self.optimized_models:
                return self.optimized_models[id(model)]
            
            # Apply mixed precision if enabled
            if self.config.enable_mixed_precision and hasattr(model, 'half'):
                model = model.half()
                optimizations_applied.append("mixed_precision")
                logger.debug(f"Applied mixed precision to {model_name}")
            
            # Apply model compilation if available (PyTorch 2.0+)
            if self.config.enable_model_compilation and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    optimizations_applied.append("torch_compile")
                    logger.debug(f"Applied torch.compile to {model_name}")
                except Exception as e:
                    logger.warning(f"torch.compile failed for {model_name}: {e}")
            
            # Apply memory efficient attention if available
            if self.config.enable_memory_efficient_attention:
                try:
                    # Try to enable memory efficient attention
                    if hasattr(model, 'enable_attention_slicing'):
                        model.enable_attention_slicing()
                        optimizations_applied.append("attention_slicing")
                    
                    if hasattr(model, 'enable_cpu_offload'):
                        model.enable_cpu_offload()
                        optimizations_applied.append("cpu_offload")
                        
                except Exception as e:
                    logger.warning(f"Memory efficient attention setup failed: {e}")
            
            # Apply gradient checkpointing if enabled
            if self.config.enable_gradient_checkpointing:
                try:
                    if hasattr(model, 'gradient_checkpointing_enable'):
                        model.gradient_checkpointing_enable()
                        optimizations_applied.append("gradient_checkpointing")
                except Exception as e:
                    logger.warning(f"Gradient checkpointing failed: {e}")
            
            # Cache optimized model
            self.optimized_models[id(model)] = model
            
            duration = time.time() - start_time
            logger.info(f"Model {model_name} optimized in {duration:.2f}s: {optimizations_applied}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed for {model_name}: {e}")
            return model
    
    def optimize_batch_size(self, model: Any, input_shape: tuple, device: str = "cuda") -> int:
        """Automatically determine optimal batch size."""
        if not TORCH_AVAILABLE or device == "cpu":
            return 1
        
        if not torch.cuda.is_available():
            return 1
        
        # Start with batch size 1 and increase until OOM
        batch_size = 1
        max_batch_size = min(self.config.max_batch_size, 32)
        
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape[1:]).to(device)
            
            # Test increasing batch sizes
            while batch_size < max_batch_size:
                try:
                    test_batch_size = batch_size * 2
                    test_input = torch.randn(test_batch_size, *input_shape[1:]).to(device)
                    
                    # Test forward pass
                    with torch.no_grad():
                        _ = model(test_input)
                    
                    # Check memory usage
                    memory_used = torch.cuda.memory_allocated(device) / torch.cuda.max_memory_allocated(device)
                    if memory_used > self.config.memory_fraction:
                        break
                    
                    batch_size = test_batch_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise
            
            logger.info(f"Optimal batch size determined: {batch_size}")
            return batch_size
            
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}")
            return 1
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class PerformanceAccelerator:
    """Main performance acceleration coordinator."""
    
    def __init__(
        self, 
        cache_config: CacheConfig = None,
        optimization_config: OptimizationConfig = None
    ):
        self.cache = IntelligentCache(cache_config)
        self.optimizer = ModelOptimizer(optimization_config)
        self.metrics_history: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
        
        logger.info("Performance accelerator initialized")
    
    def accelerate_function(self, enable_cache: bool = True):
        """Decorator to accelerate function execution."""
        def decorator(func: Callable) -> Callable:
            if enable_cache:
                func = self.cache.cache_function(func)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Get initial memory state
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                else:
                    initial_gpu_memory = 0.0
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record performance metrics
                    duration = time.time() - start_time
                    
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        final_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                        gpu_memory_used = final_gpu_memory - initial_gpu_memory
                    else:
                        gpu_memory_used = 0.0
                    
                    cache_stats = self.cache.get_stats()
                    
                    metrics = PerformanceMetrics(
                        operation=func.__name__,
                        duration=duration,
                        memory_used=0.0,  # Would need psutil for accurate measurement
                        gpu_memory_used=gpu_memory_used,
                        throughput=1.0 / duration if duration > 0 else 0.0,
                        cache_hit_rate=cache_stats["hit_rate"],
                        optimization_applied=["caching"] if enable_cache else [],
                        timestamp=time.time()
                    )
                    
                    with self.lock:
                        self.metrics_history.append(metrics)
                        # Keep only recent metrics
                        if len(self.metrics_history) > 1000:
                            self.metrics_history = self.metrics_history[-1000:]
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Accelerated function {func.__name__} failed: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def optimize_model_pipeline(self, model: Any, model_name: str = "unknown") -> Any:
        """Optimize entire model pipeline."""
        return self.optimizer.optimize_model(model, model_name)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            if not self.metrics_history:
                return {"error": "No performance data available"}
            
            recent_metrics = self.metrics_history[-100:]  # Last 100 operations
            
            avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            
            # Group by operation
            by_operation = {}
            for metric in recent_metrics:
                op = metric.operation
                if op not in by_operation:
                    by_operation[op] = []
                by_operation[op].append(metric)
            
            operation_stats = {}
            for op, metrics in by_operation.items():
                operation_stats[op] = {
                    "count": len(metrics),
                    "avg_duration": sum(m.duration for m in metrics) / len(metrics),
                    "avg_throughput": sum(m.throughput for m in metrics) / len(metrics),
                    "total_gpu_memory": sum(m.gpu_memory_used for m in metrics)
                }
            
            return {
                "timestamp": time.time(),
                "total_operations": len(self.metrics_history),
                "recent_operations": len(recent_metrics),
                "averages": {
                    "duration": avg_duration,
                    "throughput": avg_throughput,
                    "cache_hit_rate": avg_cache_hit_rate
                },
                "by_operation": operation_stats,
                "cache_stats": self.cache.get_stats(),
                "optimization_config": asdict(self.optimizer.config)
            }
    
    def export_performance_data(self, file_path: str):
        """Export performance data to file."""
        data = {
            "export_time": time.time(),
            "performance_summary": self.get_performance_summary(),
            "full_metrics_history": [asdict(m) for m in self.metrics_history]
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Performance data exported to {file_path}")


# Global accelerator instance
_accelerator = PerformanceAccelerator()


def get_performance_accelerator() -> PerformanceAccelerator:
    """Get global performance accelerator."""
    return _accelerator


def accelerate(enable_cache: bool = True):
    """Decorator for function acceleration."""
    return _accelerator.accelerate_function(enable_cache)


def optimize_model(model: Any, model_name: str = "unknown") -> Any:
    """Optimize model with global accelerator."""
    return _accelerator.optimize_model_pipeline(model, model_name)


# Example usage
if __name__ == "__main__":
    # Example: Accelerated benchmark function
    @accelerate(enable_cache=True)
    def accelerated_benchmark(model_name: str, prompt: str) -> Dict[str, Any]:
        """Example accelerated benchmark function."""
        import secrets
        
        # Simulate expensive computation
        time.sleep(secrets.SystemRandom().uniform(0.5, 2.0))
        
        return {
            "model": model_name,
            "prompt": prompt,
            "fvd": secrets.SystemRandom().uniform(80, 120),
            "latency": secrets.SystemRandom().uniform(1, 5),
            "timestamp": time.time()
        }
    
    # Run accelerated benchmarks
    accelerator = get_performance_accelerator()
    
    # Test caching by running same function multiple times
    for i in range(5):
        result1 = accelerated_benchmark("test-model", "A cat playing piano")
        result2 = accelerated_benchmark("test-model", "A cat playing piano")  # Should be cached
        print(f"Run {i+1}: Duration difference shows caching effect")
    
    # Get performance summary
    summary = accelerator.get_performance_summary()
    print("Performance Summary:")
    print(json.dumps(summary, indent=2))
    
    # Export performance data
    accelerator.export_performance_data("performance_metrics.json")