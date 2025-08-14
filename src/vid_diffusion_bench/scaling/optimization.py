"""Performance optimization and caching for scalable benchmarking.

This module provides advanced performance optimization techniques including
intelligent caching, memory optimization, GPU optimization, and performance
profiling to maximize benchmarking throughput and efficiency.
"""

import time
import threading
import logging
import hashlib
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict, defaultdict
from functools import wraps, lru_cache
import weakref
import gc

try:
    import torch
    import torch.nn.utils.prune as prune
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    operation: str
    execution_time: float
    memory_used: float
    gpu_memory_used: float
    throughput: float
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size_mb: int = 1024  # 1GB default
    enable_memory_optimization: bool = True
    enable_gpu_optimization: bool = True
    enable_model_pruning: bool = False
    profiling_enabled: bool = True
    cache_persistence: bool = True
    cache_directory: str = "./cache"
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive


class LRUCache:
    """Thread-safe LRU cache with size limits and persistence."""
    
    def __init__(
        self,
        max_size: int = 128,
        max_memory_mb: int = 512,
        persistent: bool = True,
        cache_dir: Optional[str] = None
    ):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
            persistent: Whether to persist cache to disk
            cache_dir: Directory for persistent cache
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.persistent = persistent
        
        self.cache = OrderedDict()
        self.memory_usage = 0
        self._lock = threading.RLock()
        self._access_times = {}
        self._item_sizes = {}
        
        # Persistent storage
        if persistent and cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.index_file = self.cache_dir / "cache_index.json"
            self._load_persistent_cache()
        else:
            self.cache_dir = None
            self.index_file = None
        
        logger.info(f"LRUCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self._access_times[key] = datetime.now()
                return value
            
            # Try loading from persistent storage
            if self.persistent and self.cache_dir:
                return self._load_from_disk(key)
            
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate item size
            try:
                item_size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
                item_size_mb = item_size / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Could not calculate size for cache item: {e}")
                return False
            
            # Check if item is too large
            if item_size_mb > self.max_memory_mb * 0.5:  # Don't cache items > 50% of limit
                logger.warning(f"Item too large for cache: {item_size_mb:.1f}MB")
                return False
            
            # Make space if needed
            self._make_space(item_size_mb)
            
            # Add item
            if key in self.cache:
                # Update existing item
                old_size = self._item_sizes.get(key, 0)
                self.memory_usage -= old_size
            
            self.cache[key] = value
            self._item_sizes[key] = item_size_mb
            self.memory_usage += item_size_mb
            self._access_times[key] = datetime.now()
            
            # Save to persistent storage
            if self.persistent and self.cache_dir:
                self._save_to_disk(key, value)
            
            return True
    
    def _make_space(self, needed_mb: float):
        """Make space in cache for new item.
        
        Args:
            needed_mb: Space needed in MB
        """
        # Remove items until we have enough space
        while (len(self.cache) >= self.max_size or 
               self.memory_usage + needed_mb > self.max_memory_mb):
            
            if not self.cache:
                break
            
            # Remove least recently used item
            oldest_key = next(iter(self.cache))
            self._remove_item(oldest_key)
    
    def _remove_item(self, key: str):
        """Remove item from cache.
        
        Args:
            key: Key to remove
        """
        if key in self.cache:
            del self.cache[key]
            
            if key in self._item_sizes:
                self.memory_usage -= self._item_sizes[key]
                del self._item_sizes[key]
            
            if key in self._access_times:
                del self._access_times[key]
            
            # Remove from persistent storage
            if self.persistent and self.cache_dir:
                cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key.
        
        Args:
            key: Original key
            
        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def _save_to_disk(self, key: str, value: Any):
        """Save item to persistent storage.
        
        Args:
            key: Cache key
            value: Value to save
        """
        try:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Failed to save cache item to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load item from persistent storage.
        
        Args:
            key: Cache key
            
        Returns:
            Loaded value or None
        """
        try:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
        # SECURITY: pickle.loads() can execute arbitrary code. Only use with trusted data.
                    value = pickle.load(f)
                
                # Add to memory cache
                self.put(key, value)
                return value
        except Exception as e:
            logger.error(f"Failed to load cache item from disk: {e}")
        
        return None
    
    def _load_persistent_cache(self):
        """Load cache index from disk."""
        if not self.index_file.exists():
            return
        
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # Load frequently accessed items into memory
            for key_hash, metadata in index.items():
                # Load based on access frequency or recency
                if metadata.get('access_count', 0) > 5:
                    cache_file = self.cache_dir / f"{key_hash}.pkl"
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'rb') as f:
        # SECURITY: pickle.loads() can execute arbitrary code. Only use with trusted data.
                                value = pickle.load(f)
                            
                            # Reconstruct original key (this is simplified)
                            key = metadata.get('key', key_hash)
                            self.cache[key] = value
                            self._item_sizes[key] = metadata.get('size_mb', 0)
                            self.memory_usage += self._item_sizes[key]
                            
                        except Exception:
                            continue
                            
        except Exception as e:
            logger.error(f"Failed to load persistent cache: {e}")
    
    def clear(self):
        """Clear all cache contents."""
        with self._lock:
            self.cache.clear()
            self._item_sizes.clear()
            self._access_times.clear()
            self.memory_usage = 0
            
            # Clear persistent storage
            if self.persistent and self.cache_dir:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                if self.index_file.exists():
                    self.index_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.memory_usage,
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1),
                'items': list(self.cache.keys())
            }


class CacheManager:
    """Manages multiple caches with different strategies."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize cache manager.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.caches = {}
        
        if config.enable_caching:
            # Model cache for loaded models
            self.caches['models'] = LRUCache(
                max_size=16,  # Max 16 models
                max_memory_mb=config.cache_size_mb // 2,
                persistent=config.cache_persistence,
                cache_dir=f"{config.cache_directory}/models"
            )
            
            # Results cache for benchmark results
            self.caches['results'] = LRUCache(
                max_size=1000,  # Max 1000 results
                max_memory_mb=config.cache_size_mb // 4,
                persistent=config.cache_persistence,
                cache_dir=f"{config.cache_directory}/results"
            )
            
            # Metrics cache for computed metrics
            self.caches['metrics'] = LRUCache(
                max_size=500,  # Max 500 metric computations
                max_memory_mb=config.cache_size_mb // 4,
                persistent=config.cache_persistence,
                cache_dir=f"{config.cache_directory}/metrics"
            )
        
        self._hit_counts = defaultdict(int)
        self._miss_counts = defaultdict(int)
        
        logger.info(f"CacheManager initialized with {len(self.caches)} caches")
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """Get item from specific cache.
        
        Args:
            cache_name: Name of cache
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if cache_name not in self.caches:
            return None
        
        value = self.caches[cache_name].get(key)
        
        if value is not None:
            self._hit_counts[cache_name] += 1
        else:
            self._miss_counts[cache_name] += 1
        
        return value
    
    def put(self, cache_name: str, key: str, value: Any) -> bool:
        """Put item in specific cache.
        
        Args:
            cache_name: Name of cache
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successfully cached
        """
        if cache_name not in self.caches:
            return False
        
        return self.caches[cache_name].put(key, value)
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create deterministic key from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def cached_model_load(self, model_name: str, load_func: Callable) -> Any:
        """Cache model loading.
        
        Args:
            model_name: Model name/identifier
            load_func: Function to load model
            
        Returns:
            Loaded model (cached or fresh)
        """
        cache_key = f"model_{model_name}"
        
        # Try to get from cache
        model = self.get('models', cache_key)
        if model is not None:
            logger.debug(f"Model {model_name} loaded from cache")
            return model
        
        # Load model and cache
        logger.debug(f"Loading model {model_name} fresh")
        model = load_func()
        
        if model is not None:
            self.put('models', cache_key, model)
        
        return model
    
    def cached_result(self, cache_name: str, key_args: Tuple, compute_func: Callable) -> Any:
        """Cache computation result.
        
        Args:
            cache_name: Cache to use
            key_args: Arguments for key generation
            compute_func: Function to compute result
            
        Returns:
            Cached or computed result
        """
        cache_key = self.generate_key(*key_args)
        
        # Try to get from cache
        result = self.get(cache_name, cache_key)
        if result is not None:
            logger.debug(f"Result loaded from {cache_name} cache")
            return result
        
        # Compute result and cache
        logger.debug(f"Computing fresh result for {cache_name}")
        result = compute_func()
        
        if result is not None:
            self.put(cache_name, cache_key, result)
        
        return result
    
    def get_cache_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches.
        
        Returns:
            Cache statistics by cache name
        """
        stats = {}
        
        for cache_name, cache in self.caches.items():
            cache_stats = cache.get_stats()
            cache_stats.update({
                'hits': self._hit_counts[cache_name],
                'misses': self._miss_counts[cache_name],
                'hit_rate': (self._hit_counts[cache_name] / 
                           max(self._hit_counts[cache_name] + self._miss_counts[cache_name], 1))
            })
            stats[cache_name] = cache_stats
        
        return stats
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        
        self._hit_counts.clear()
        self._miss_counts.clear()
        
        logger.info("All caches cleared")


class MemoryOptimizer:
    """Optimizes memory usage for better performance."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize memory optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.enabled = config.enable_memory_optimization
        
        # Memory monitoring
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cleanup_callbacks = []
        self._monitoring_thread = None
        self._monitoring_active = False
        
        logger.info(f"MemoryOptimizer initialized (enabled={self.enabled})")
    
    def start_monitoring(self):
        """Start memory monitoring."""
        if not self.enabled or self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._memory_monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join()
        
        logger.info("Memory monitoring stopped")
    
    def _memory_monitoring_loop(self):
        """Memory monitoring loop."""
        while self._monitoring_active:
            try:
                if PSUTIL_AVAILABLE:
                    memory = psutil.virtual_memory()
                    memory_usage = memory.percent / 100
                    
                    if memory_usage > self.memory_threshold:
                        logger.warning(f"High memory usage detected: {memory_usage:.1%}")
                        self._trigger_cleanup()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(10)
    
    def _trigger_cleanup(self):
        """Trigger memory cleanup."""
        logger.info("Triggering memory cleanup")
        
        # Run registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        self.force_garbage_collection()
        
        # GPU memory cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.cleanup_gpu_memory()
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register cleanup callback.
        
        Args:
            callback: Function to call during cleanup
        """
        self.cleanup_callbacks.append(callback)
    
    def force_garbage_collection(self):
        """Force garbage collection."""
        import gc
        
        # Run multiple GC cycles
        collected = 0
        for _ in range(3):
            collected += gc.collect()
        
        logger.debug(f"Garbage collection freed {collected} objects")
    
    def cleanup_gpu_memory(self):
        """Cleanup GPU memory."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            logger.debug("GPU memory cleanup completed")
            
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics.
        
        Returns:
            Memory statistics
        """
        stats = {}
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            stats.update({
                'system_memory_percent': memory.percent,
                'system_memory_available_gb': memory.available / (1024**3),
                'system_memory_total_gb': memory.total / (1024**3)
            })
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                cached = torch.cuda.memory_reserved(i)
                
                stats[f'gpu_{i}_memory_allocated_gb'] = allocated / (1024**3)
                stats[f'gpu_{i}_memory_cached_gb'] = cached / (1024**3)
        
        return stats


class GPUOptimizer:
    """Optimizes GPU usage for better performance."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize GPU optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.enabled = config.enable_gpu_optimization and TORCH_AVAILABLE
        
        if self.enabled:
            self._setup_gpu_optimizations()
        
        logger.info(f"GPUOptimizer initialized (enabled={self.enabled})")
    
    def _setup_gpu_optimizations(self):
        """Setup GPU optimizations."""
        if not torch.cuda.is_available():
            self.enabled = False
            return
        
        # Enable mixed precision training if supported
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable optimized attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
        
        # Set memory fraction if configured
        if self.config.optimization_level >= 2:
            # Reserve 90% of GPU memory for caching
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(0.9, i)
    
    def optimize_model(self, model: Any) -> Any:
        """Optimize model for GPU inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        if not self.enabled or not TORCH_AVAILABLE:
            return model
        
        try:
            # Convert to half precision for memory efficiency
            if hasattr(model, 'half') and self.config.optimization_level >= 1:
                model = model.half()
            
            # Apply model compilation if available (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.config.optimization_level >= 2:
                model = torch.compile(model, mode='reduce-overhead')
            
            # Apply pruning if enabled
            if self.config.enable_model_pruning and hasattr(model, 'named_modules'):
                model = self._apply_pruning(model)
            
            logger.debug("Model GPU optimization applied")
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def _apply_pruning(self, model: Any, pruning_ratio: float = 0.2) -> Any:
        """Apply pruning to model for efficiency.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of weights to prune
            
        Returns:
            Pruned model
        """
        try:
            # Apply structured pruning to linear layers
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                elif isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            
            logger.debug(f"Applied {pruning_ratio:.1%} pruning to model")
            return model
            
        except Exception as e:
            logger.error(f"Model pruning failed: {e}")
            return model
    
    def setup_optimal_batch_size(self, model: Any, input_shape: Tuple[int, ...]) -> int:
        """Determine optimal batch size for model.
        
        Args:
            model: Model to test
            input_shape: Input tensor shape
            
        Returns:
            Optimal batch size
        """
        if not self.enabled:
            return 1
        
        try:
            device = next(model.parameters()).device
            
            # Test different batch sizes
            batch_sizes = [1, 2, 4, 8, 16, 32]
            optimal_batch_size = 1
            
            for batch_size in batch_sizes:
                try:
                    # Create dummy input
                    dummy_input = torch.randn(batch_size, *input_shape[1:]).to(device)
                    
                    # Test forward pass
                    with torch.no_grad():
                        _ = model(dummy_input)
                    
                    optimal_batch_size = batch_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise
            
            logger.debug(f"Optimal batch size determined: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Batch size optimization failed: {e}")
            return 1
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics.
        
        Returns:
            GPU statistics
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
        
        stats = {}
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            stats[f'gpu_{i}'] = {
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'allocated_memory_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'cached_memory_gb': torch.cuda.memory_reserved(i) / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            }
        
        return stats


class PerformanceProfiler:
    """Profiles performance and tracks optimization metrics."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize performance profiler.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.enabled = config.profiling_enabled
        
        self.metrics_history = []
        self.active_profiles = {}
        self._lock = threading.Lock()
        
        logger.info(f"PerformanceProfiler initialized (enabled={self.enabled})")
    
    def profile(self, operation: str):
        """Context manager for profiling operations.
        
        Args:
            operation: Operation name
        """
        return ProfileContext(self, operation) if self.enabled else DummyContext()
    
    def start_profile(self, operation: str) -> str:
        """Start profiling an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Profile ID
        """
        if not self.enabled:
            return ""
        
        profile_id = f"{operation}_{int(time.time() * 1000)}"
        
        with self._lock:
            self.active_profiles[profile_id] = {
                'operation': operation,
                'start_time': time.time(),
                'start_memory': self._get_memory_usage(),
                'start_gpu_memory': self._get_gpu_memory_usage()
            }
        
        return profile_id
    
    def end_profile(self, profile_id: str) -> Optional[PerformanceMetrics]:
        """End profiling an operation.
        
        Args:
            profile_id: Profile ID from start_profile
            
        Returns:
            Performance metrics or None
        """
        if not self.enabled or profile_id not in self.active_profiles:
            return None
        
        with self._lock:
            profile_data = self.active_profiles.pop(profile_id)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            execution_time = end_time - profile_data['start_time']
            memory_used = end_memory - profile_data['start_memory']
            gpu_memory_used = end_gpu_memory - profile_data['start_gpu_memory']
            
            # Calculate throughput (operations per second)
            throughput = 1.0 / execution_time if execution_time > 0 else 0.0
            
            metrics = PerformanceMetrics(
                operation=profile_data['operation'],
                execution_time=execution_time,
                memory_used=memory_used,
                gpu_memory_used=gpu_memory_used,
                throughput=throughput,
                cache_hit_rate=0.0  # Will be updated separately
            )
            
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns:
            Performance summary
        """
        if not self.metrics_history:
            return {}
        
        # Group metrics by operation
        operation_metrics = defaultdict(list)
        for metric in self.metrics_history:
            operation_metrics[metric.operation].append(metric)
        
        summary = {}
        for operation, metrics in operation_metrics.items():
            execution_times = [m.execution_time for m in metrics]
            memory_usage = [m.memory_used for m in metrics]
            throughputs = [m.throughput for m in metrics]
            
            summary[operation] = {
                'count': len(metrics),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'avg_memory_used': sum(memory_usage) / len(memory_usage),
                'avg_throughput': sum(throughputs) / len(throughputs),
                'total_time': sum(execution_times)
            }
        
        return summary


class ProfileContext:
    """Context manager for profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation: str):
        self.profiler = profiler
        self.operation = operation
        self.profile_id = None
    
    def __enter__(self):
        self.profile_id = self.profiler.start_profile(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_id:
            metrics = self.profiler.end_profile(self.profile_id)
            if metrics:
                logger.debug(f"Profile {self.operation}: {metrics.execution_time:.3f}s")


class DummyContext:
    """Dummy context manager when profiling is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class PerformanceOptimizer:
    """Main performance optimizer coordinating all optimizations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize performance optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.cache_manager = CacheManager(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.gpu_optimizer = GPUOptimizer(self.config)
        self.profiler = PerformanceProfiler(self.config)
        
        # Start monitoring
        if self.config.enable_memory_optimization:
            self.memory_optimizer.start_monitoring()
        
        logger.info("PerformanceOptimizer initialized")
    
    def optimize_model_loading(self, model_name: str, load_func: Callable) -> Any:
        """Optimize model loading with caching and GPU optimization.
        
        Args:
            model_name: Model name
            load_func: Function to load model
            
        Returns:
            Optimized model
        """
        with self.profiler.profile(f"model_load_{model_name}"):
            # Try to load from cache
            model = self.cache_manager.cached_model_load(model_name, load_func)
            
            # Apply GPU optimizations
            if model is not None:
                model = self.gpu_optimizer.optimize_model(model)
            
            return model
    
    def optimize_batch_processing(
        self,
        process_func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Optimize batch processing.
        
        Args:
            process_func: Function to process items
            items: Items to process
            batch_size: Batch size (auto-determined if None)
            
        Returns:
            Processed items
        """
        with self.profiler.profile("batch_processing"):
            # Determine optimal batch size
            if batch_size is None:
                batch_size = self._determine_optimal_batch_size(len(items))
            
            results = []
            
            # Process in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                # Process batch
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                # Cleanup between batches
                if i % (batch_size * 10) == 0:  # Every 10 batches
                    self.memory_optimizer.force_garbage_collection()
            
            return results
    
    def _determine_optimal_batch_size(self, num_items: int) -> int:
        """Determine optimal batch size based on available resources.
        
        Args:
            num_items: Total number of items
            
        Returns:
            Optimal batch size
        """
        # Start with reasonable default
        batch_size = 8
        
        # Adjust based on memory availability
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb > 16:
                batch_size = 32
            elif available_gb > 8:
                batch_size = 16
            elif available_gb < 4:
                batch_size = 4
        
        # Don't exceed number of items
        return min(batch_size, num_items)
    
    def cached_computation(
        self,
        cache_name: str,
        compute_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Perform cached computation.
        
        Args:
            cache_name: Cache to use
            compute_func: Function to compute result
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Computed or cached result
        """
        return self.cache_manager.cached_result(
            cache_name, 
            (args, tuple(sorted(kwargs.items()))),
            lambda: compute_func(*args, **kwargs)
        )
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report.
        
        Returns:
            Optimization report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'caching_enabled': self.config.enable_caching,
                'memory_optimization_enabled': self.config.enable_memory_optimization,
                'gpu_optimization_enabled': self.config.enable_gpu_optimization,
                'optimization_level': self.config.optimization_level
            },
            'cache_statistics': self.cache_manager.get_cache_statistics(),
            'memory_statistics': self.memory_optimizer.get_memory_stats(),
            'gpu_statistics': self.gpu_optimizer.get_gpu_stats(),
            'performance_summary': self.profiler.get_performance_summary()
        }
        
        return report
    
    def cleanup(self):
        """Cleanup optimizer resources."""
        # Stop monitoring
        self.memory_optimizer.stop_monitoring()
        
        # Clear caches
        self.cache_manager.clear_all_caches()
        
        # Force garbage collection
        self.memory_optimizer.force_garbage_collection()
        
        logger.info("PerformanceOptimizer cleanup completed")


# Decorator for automatic caching
def cached(cache_name: str = 'results', cache_manager: Optional[CacheManager] = None):
    """Decorator for automatic result caching.
    
    Args:
        cache_name: Cache to use
        cache_manager: Cache manager instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache_manager
            
            if cache_manager is None:
                # Use global optimizer if available
                global _global_optimizer
                if _global_optimizer is not None:
                    cache_manager = _global_optimizer.cache_manager
                else:
                    # Call function directly if no cache manager
                    return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = cache_manager.generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache_manager.get(cache_name, cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_manager.put(cache_name, cache_key, result)
            
            return result
        
        return wrapper
    return decorator


# Global optimizer instance
_global_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer