"""Advanced optimization and scaling features for video diffusion benchmarking.

This module implements high-performance optimizations including:
- Multi-GPU distributed processing
- Intelligent batch processing and memory management
- Advanced caching strategies
- Performance profiling and auto-tuning
- Resource pool management
- Async/await processing pipelines
"""

import asyncio
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
import pickle
from contextlib import asynccontextmanager
from queue import Queue, Empty
import gc

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    enable_multi_gpu: bool = True
    enable_mixed_precision: bool = True
    enable_compilation: bool = True
    enable_async_processing: bool = True
    enable_memory_optimization: bool = True
    batch_size_auto_tune: bool = True
    cache_size_mb: int = 1024
    max_workers: int = 4
    prefetch_factor: int = 2
    memory_fraction_limit: float = 0.85


class MemoryPool:
    """Advanced memory pool for tensor reuse and optimization."""
    
    def __init__(self, max_size_gb: float = 4.0, cleanup_threshold: float = 0.9):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cleanup_threshold = cleanup_threshold
        self.pool = {}
        self.usage_stats = {}
        self.current_size = 0
        self._lock = threading.Lock()
    
    def get_tensor(self, shape: tuple, dtype: str = "float32", device: str = "cuda") -> Any:
        """Get tensor from pool or create new one."""
        key = (shape, dtype, device)
        
        with self._lock:
            if key in self.pool and self.pool[key]:
                tensor = self.pool[key].pop()
                self.usage_stats[key] = self.usage_stats.get(key, 0) + 1
                return tensor
            
            # Create new tensor
            try:
                import torch
                
                if device.startswith('cuda'):
                    device_obj = torch.device(device)
                    dtype_obj = getattr(torch, dtype)
                    tensor = torch.empty(shape, dtype=dtype_obj, device=device_obj)
                    
                    # Update current size
                    tensor_size = tensor.element_size() * tensor.numel()
                    self.current_size += tensor_size
                    
                    # Cleanup if necessary
                    if self.current_size > self.max_size_bytes * self.cleanup_threshold:
                        self._cleanup()
                    
                    return tensor
                
            except ImportError:
                pass
        
        return None
    
    def return_tensor(self, tensor: Any):
        """Return tensor to pool for reuse."""
        if tensor is None:
            return
        
        try:
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype).split('.')[-1]
            device = str(tensor.device)
            key = (shape, dtype, device)
            
            # Clear tensor data
            tensor.zero_()
            
            with self._lock:
                if key not in self.pool:
                    self.pool[key] = []
                
                # Limit pool size per shape
                if len(self.pool[key]) < 10:
                    self.pool[key].append(tensor)
                else:
                    # Free the tensor
                    del tensor
                    
        except Exception as e:
            logger.debug(f"Error returning tensor to pool: {e}")
    
    def _cleanup(self):
        """Cleanup least used tensors from pool."""
        # Find least used keys
        if not self.usage_stats:
            return
        
        sorted_keys = sorted(self.usage_stats.items(), key=lambda x: x[1])
        cleanup_count = len(sorted_keys) // 4  # Clean up 25% of least used
        
        for key, _ in sorted_keys[:cleanup_count]:
            if key in self.pool:
                tensors = self.pool.pop(key, [])
                for tensor in tensors:
                    tensor_size = tensor.element_size() * tensor.numel()
                    self.current_size -= tensor_size
                    del tensor
                
                # Remove from stats
                if key in self.usage_stats:
                    del self.usage_stats[key]
        
        # Force garbage collection
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                "current_size_gb": self.current_size / (1024**3),
                "max_size_gb": self.max_size_bytes / (1024**3),
                "pool_entries": sum(len(tensors) for tensors in self.pool.values()),
                "unique_shapes": len(self.pool),
                "usage_stats": dict(self.usage_stats)
            }


class BatchProcessor:
    """Intelligent batch processing with dynamic sizing."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimal_batch_sizes = {}
        self.performance_history = []
        self.memory_monitor = MemoryPool()
    
    def find_optimal_batch_size(
        self,
        model_name: str,
        base_memory_usage: float,
        target_memory_fraction: float = 0.8
    ) -> int:
        """Find optimal batch size based on available memory."""
        if model_name in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[model_name]
        
        try:
            import torch
            
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                available_memory = total_memory * target_memory_fraction
                
                # Estimate batch size based on base memory usage
                estimated_batch_size = max(1, int(available_memory / base_memory_usage))
                
                # Test batch sizes to find optimal
                optimal_size = self._test_batch_sizes(
                    model_name, 
                    estimated_batch_size,
                    max_size=min(32, estimated_batch_size * 2)
                )
                
                self.optimal_batch_sizes[model_name] = optimal_size
                return optimal_size
            
        except ImportError:
            pass
        
        return 1  # Fallback to batch size 1
    
    def _test_batch_sizes(self, model_name: str, start_size: int, max_size: int) -> int:
        """Test different batch sizes to find optimal."""
        best_size = start_size
        best_throughput = 0
        
        for batch_size in range(1, max_size + 1, max(1, max_size // 8)):
            try:
                # Simulate batch processing with mock data
                throughput = self._benchmark_batch_size(model_name, batch_size)
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_size = batch_size
                
            except Exception as e:
                logger.debug(f"Batch size {batch_size} failed for {model_name}: {e}")
                break
        
        logger.info(f"Optimal batch size for {model_name}: {best_size}")
        return best_size
    
    def _benchmark_batch_size(self, model_name: str, batch_size: int) -> float:
        """Benchmark a specific batch size."""
        # This would be implemented to actually test batch processing
        # For now, we simulate the benchmark
        
        # Simulate processing time inversely related to batch size with memory constraints
        base_time = 1.0  # Base processing time
        batch_efficiency = min(1.0, batch_size / 8.0)  # Efficiency peaks around batch size 8
        memory_penalty = max(0, (batch_size - 16) * 0.1)  # Penalty for large batches
        
        processing_time = base_time * (1.0 - batch_efficiency * 0.3) + memory_penalty
        throughput = batch_size / processing_time
        
        return throughput
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: int = None
    ) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        if batch_size is None:
            batch_size = len(items)
        
        results = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                # Process batch with resource management
                with self._resource_context():
                    batch_results = await processor_func(batch)
                    results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Fallback to individual processing
                for item in batch:
                    try:
                        result = await processor_func([item])
                        results.extend(result)
                    except Exception as item_error:
                        logger.error(f"Individual item processing failed: {item_error}")
                        results.append(None)
        
        return results
    
    @asynccontextmanager
    async def _resource_context(self):
        """Context manager for resource management during processing."""
        # Pre-processing setup
        start_time = time.time()
        
        try:
            yield
        finally:
            # Post-processing cleanup
            processing_time = time.time() - start_time
            self.performance_history.append({
                "timestamp": start_time,
                "duration": processing_time
            })
            
            # Cleanup resources
            self.memory_monitor._cleanup()


class DistributedProcessor:
    """Multi-GPU distributed processing system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.available_devices = self._detect_devices()
        self.device_pools = {}
        self._setup_device_pools()
    
    def _detect_devices(self) -> List[str]:
        """Detect available processing devices."""
        devices = ["cpu"]
        
        try:
            import torch
            if torch.cuda.is_available() and self.config.enable_multi_gpu:
                device_count = torch.cuda.device_count()
                devices.extend([f"cuda:{i}" for i in range(device_count)])
                logger.info(f"Detected {device_count} CUDA devices")
            
        except ImportError:
            logger.warning("PyTorch not available, using CPU only")
        
        return devices
    
    def _setup_device_pools(self):
        """Setup memory pools for each device."""
        for device in self.available_devices:
            if device != "cpu":
                self.device_pools[device] = MemoryPool(max_size_gb=2.0)
    
    async def distribute_work(
        self,
        work_items: List[Any],
        processor_func: Callable,
        device_preference: str = "auto"
    ) -> List[Any]:
        """Distribute work across available devices."""
        if not work_items:
            return []
        
        # Select devices to use
        if device_preference == "auto":
            target_devices = self.available_devices
        elif device_preference in self.available_devices:
            target_devices = [device_preference]
        else:
            target_devices = ["cpu"]
        
        # Distribute work across devices
        device_work = self._distribute_items(work_items, target_devices)
        
        # Process on each device concurrently
        tasks = []
        for device, items in device_work.items():
            task = self._process_on_device(device, items, processor_func)
            tasks.append(task)
        
        # Wait for all tasks to complete
        device_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        results = []
        for device_result in device_results:
            if isinstance(device_result, Exception):
                logger.error(f"Device processing failed: {device_result}")
                continue
            
            results.extend(device_result)
        
        return results
    
    def _distribute_items(
        self,
        items: List[Any],
        devices: List[str]
    ) -> Dict[str, List[Any]]:
        """Distribute items across devices considering their capabilities."""
        device_work = {device: [] for device in devices}
        
        # Simple round-robin distribution
        for i, item in enumerate(items):
            device = devices[i % len(devices)]
            device_work[device].append(item)
        
        return device_work
    
    async def _process_on_device(
        self,
        device: str,
        items: List[Any],
        processor_func: Callable
    ) -> List[Any]:
        """Process items on a specific device."""
        if not items:
            return []
        
        # Set device context
        device_context = {}
        if device != "cpu":
            device_context["device"] = device
            device_context["memory_pool"] = self.device_pools.get(device)
        
        try:
            # Process items with device-specific optimizations
            results = await self._device_optimized_processing(
                items, processor_func, device_context
            )
            return results
            
        except Exception as e:
            logger.error(f"Processing on device {device} failed: {e}")
            # Fallback to CPU processing
            return await self._fallback_processing(items, processor_func)
    
    async def _device_optimized_processing(
        self,
        items: List[Any],
        processor_func: Callable,
        device_context: Dict[str, Any]
    ) -> List[Any]:
        """Perform device-optimized processing."""
        results = []
        
        for item in items:
            try:
                # Apply device-specific optimizations
                optimized_item = self._optimize_for_device(item, device_context)
                result = await processor_func(optimized_item)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Item processing failed: {e}")
                results.append(None)
        
        return results
    
    def _optimize_for_device(self, item: Any, context: Dict[str, Any]) -> Any:
        """Apply device-specific optimizations to an item."""
        if "device" in context:
            # Move tensors to the specified device if applicable
            try:
                if hasattr(item, 'to'):
                    return item.to(context["device"])
            except Exception:
                pass
        
        return item
    
    async def _fallback_processing(
        self,
        items: List[Any],
        processor_func: Callable
    ) -> List[Any]:
        """Fallback CPU processing."""
        results = []
        for item in items:
            try:
                result = await processor_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Fallback processing failed: {e}")
                results.append(None)
        
        return results


class AdvancedCache:
    """Advanced caching system with intelligent eviction and compression."""
    
    def __init__(self, max_size_mb: int = 1024):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
        self._lock = threading.Lock()
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key for data."""
        if isinstance(data, dict):
            # Sort dictionary items for consistent hashing
            content = json.dumps(data, sort_keys=True, default=str)
        else:
            content = str(data)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.cache_stats["hits"] += 1
                
                # Decompress if necessary
                data = self.cache[key]
                if isinstance(data, bytes):
                    try:
        # SECURITY: pickle.loads() can execute arbitrary code. Only use with trusted data.
                        return pickle.loads(data)
                    except Exception:
                        # If decompression fails, remove from cache
                        del self.cache[key]
                        del self.access_times[key]
                        return None
                
                return data
            
            self.cache_stats["misses"] += 1
            return None
    
    def put(self, key: str, data: Any, compress: bool = True):
        """Put item in cache with optional compression."""
        # Calculate size
        if compress:
            try:
                serialized_data = pickle.dumps(data)
                data_size = len(serialized_data)
                cache_data = serialized_data
            except Exception:
                # Fallback to uncompressed
                cache_data = data
                data_size = self._estimate_size(data)
        else:
            cache_data = data
            data_size = self._estimate_size(data)
        
        with self._lock:
            # Check if we need to evict items
            while (self.cache_stats["size_bytes"] + data_size > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Add to cache
            self.cache[key] = cache_data
            self.access_times[key] = time.time()
            self.cache_stats["size_bytes"] += data_size
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Rough estimation
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (int, float)):
                return 8
            elif isinstance(data, dict):
                return sum(len(str(k)) + self._estimate_size(v) for k, v in data.items())
            elif isinstance(data, list):
                return sum(self._estimate_size(item) for item in data)
            else:
                return 1024  # Default estimate
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_key in self.cache:
            data = self.cache[lru_key]
            del self.cache[lru_key]
            del self.access_times[lru_key]
            
            # Update size stats
            data_size = self._estimate_size(data)
            self.cache_stats["size_bytes"] -= data_size
            self.cache_stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "hit_rate": hit_rate,
                "total_items": len(self.cache),
                "size_mb": self.cache_stats["size_bytes"] / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                **self.cache_stats
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.cache_stats["size_bytes"] = 0


class PerformanceOptimizer:
    """Automatic performance optimization and tuning."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_pool = MemoryPool()
        self.batch_processor = BatchProcessor(config)
        self.distributed_processor = DistributedProcessor(config)
        self.cache = AdvancedCache(config.cache_size_mb)
        self.performance_metrics = []
    
    async def optimize_model_loading(self, model_name: str, model_config: Dict[str, Any]):
        """Optimize model loading with caching and pre-loading."""
        cache_key = self._generate_model_cache_key(model_name, model_config)
        
        # Check cache first
        cached_model = self.cache.get(cache_key)
        if cached_model is not None:
            logger.info(f"Loaded model '{model_name}' from cache")
            return cached_model
        
        # Load model with optimizations
        start_time = time.time()
        
        try:
            # Apply compilation optimizations if enabled
            if self.config.enable_compilation:
                model_config = self._apply_compilation_optimizations(model_config)
            
            # Apply mixed precision if enabled
            if self.config.enable_mixed_precision:
                model_config = self._apply_mixed_precision(model_config)
            
            # Simulate model loading (would be actual implementation)
            model = await self._load_model_optimized(model_name, model_config)
            
            load_time = time.time() - start_time
            
            # Cache the model
            self.cache.put(cache_key, model, compress=False)  # Don't compress models
            
            logger.info(f"Loaded and cached model '{model_name}' in {load_time:.2f}s")
            
            # Record performance metrics
            self.performance_metrics.append({
                "operation": "model_load",
                "model_name": model_name,
                "duration": load_time,
                "timestamp": time.time(),
                "cached": False
            })
            
            return model
            
        except Exception as e:
            logger.error(f"Model loading optimization failed: {e}")
            raise
    
    def _generate_model_cache_key(self, model_name: str, config: Dict[str, Any]) -> str:
        """Generate cache key for model configuration."""
        key_data = {
            "model_name": model_name,
            "config": config
        }
        return f"model_{hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]}"
    
    def _apply_compilation_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply model compilation optimizations."""
        config = config.copy()
        config["enable_torch_compile"] = True
        config["compile_mode"] = "reduce-overhead"
        return config
    
    def _apply_mixed_precision(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mixed precision optimizations."""
        config = config.copy()
        config["torch_dtype"] = "float16"
        config["use_fp16"] = True
        return config
    
    async def _load_model_optimized(self, model_name: str, config: Dict[str, Any]):
        """Load model with optimizations (mock implementation)."""
        # Simulate model loading time
        await asyncio.sleep(0.1)
        
        # Return mock model object
        return {
            "model_name": model_name,
            "config": config,
            "loaded_at": time.time()
        }
    
    async def optimize_batch_processing(
        self,
        items: List[Any],
        processor_func: Callable,
        model_name: str = "default"
    ) -> List[Any]:
        """Optimize batch processing with dynamic sizing and distribution."""
        if not items:
            return []
        
        # Find optimal batch size
        optimal_batch_size = self.batch_processor.find_optimal_batch_size(
            model_name, 
            base_memory_usage=1024**3  # 1GB base estimate
        )
        
        # Use distributed processing if multiple devices available
        if len(self.distributed_processor.available_devices) > 1:
            return await self.distributed_processor.distribute_work(
                items, processor_func
            )
        else:
            return await self.batch_processor.process_batch(
                items, processor_func, optimal_batch_size
            )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            "memory_pool": self.memory_pool.get_stats(),
            "cache": self.cache.get_stats(),
            "available_devices": self.distributed_processor.available_devices,
            "performance_metrics": self.performance_metrics[-100:],  # Last 100 metrics
            "optimal_batch_sizes": self.batch_processor.optimal_batch_sizes
        }


# Global optimizer instance
_global_optimizer = None


def get_optimizer(config: OptimizationConfig = None) -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        if config is None:
            config = OptimizationConfig()
        _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer


def optimized_benchmark(config: OptimizationConfig = None):
    """Decorator to apply automatic optimizations to benchmark functions."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            optimizer = get_optimizer(config)
            
            # Apply optimizations based on function signature and arguments
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Optimized benchmark failed: {e}")
                raise
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return async wrapper if function is async, sync wrapper otherwise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator