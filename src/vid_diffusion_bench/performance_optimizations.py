"""Advanced performance optimizations for video diffusion benchmarks."""

import time
import logging
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import functools
import pickle
import hashlib
from pathlib import Path
import json
import gc
import weakref

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
        
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp


class IntelligentCache:
    """High-performance intelligent cache with LRU and adaptive policies."""
    
    def __init__(self, max_size_mb: int = 1024, max_entries: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cache = {}  # key -> CacheEntry
        self.access_times = deque()  # (key, timestamp) for LRU
        self.size_bytes = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Adaptive policies
        self.access_patterns = defaultdict(list)  # key -> [access_times]
        self.popularity_threshold = 3  # Minimum accesses to be "popular"
        
    def get(self, key: str) -> Any:
        """Get value from cache."""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
                
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self.cache[key]
                self.size_bytes -= entry.size_bytes
                self.stats['misses'] += 1
                return None
                
            # Update access patterns
            entry.access_count += 1
            self.access_times.append((key, time.time()))
            self.access_patterns[key].append(time.time())
            
            # Keep only recent access history
            cutoff = time.time() - 3600  # 1 hour
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff
            ]
            
            self.stats['hits'] += 1
            return entry.value
            
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default size if can't serialize
                
            # Check if too large for cache
            if size_bytes > self.max_size_bytes * 0.5:  # Don't cache if >50% of total
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return
                
            # Remove existing entry if present
            if key in self.cache:
                self.size_bytes -= self.cache[key].size_bytes
                
            # Make space if necessary
            self._evict_if_necessary(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self.cache[key] = entry
            self.size_bytes += size_bytes
            self.access_times.append((key, time.time()))
            
    def _evict_if_necessary(self, incoming_size: int):
        """Evict items to make space for new entry."""
        while (self.size_bytes + incoming_size > self.max_size_bytes or
               len(self.cache) >= self.max_entries):
            
            if not self.cache:
                break
                
            # Find victim using intelligent policy
            victim_key = self._select_victim()
            if victim_key:
                victim_entry = self.cache[victim_key]
                self.size_bytes -= victim_entry.size_bytes
                del self.cache[victim_key]
                self.stats['evictions'] += 1
            else:
                break  # No victim found
                
    def _select_victim(self) -> Optional[str]:
        """Select cache victim using intelligent policy."""
        if not self.cache:
            return None
            
        now = time.time()
        candidates = []
        
        for key, entry in self.cache.items():
            # Calculate victim score (higher = more likely to evict)
            age_score = entry.age_seconds / 3600  # Normalize by hour
            size_score = entry.size_bytes / (1024 * 1024)  # Normalize by MB
            access_score = 1.0 / max(entry.access_count, 1)  # Less accessed = higher score
            
            # Recent access patterns
            recent_accesses = len([t for t in self.access_patterns[key] if now - t < 300])  # 5 minutes
            recency_score = 1.0 / max(recent_accesses, 1)
            
            # Combine scores
            victim_score = age_score * 0.3 + size_score * 0.2 + access_score * 0.3 + recency_score * 0.2
            
            candidates.append((victim_score, key))
            
        # Select highest scoring victim
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else None
        
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_patterns.clear()
            self.size_bytes = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.stats['hits'] / max(self.stats['total_requests'], 1)
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'size_mb': self.size_bytes / (1024 * 1024),
                'entries': len(self.cache),
                'popular_keys': self._get_popular_keys()
            }
            
    def _get_popular_keys(self) -> List[str]:
        """Get most popular cache keys."""
        now = time.time()
        popularity = []
        
        for key in self.cache:
            recent_accesses = len([t for t in self.access_patterns[key] if now - t < 3600])
            if recent_accesses >= self.popularity_threshold:
                popularity.append((recent_accesses, key))
                
        popularity.sort(reverse=True)
        return [key for _, key in popularity[:10]]


class BatchProcessor:
    """Intelligent batch processing for model inference."""
    
    def __init__(self, 
                 batch_size: int = 4,
                 max_wait_time: float = 1.0,
                 adaptive: bool = True):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.adaptive = adaptive
        self.pending_requests = deque()
        self.processing_lock = threading.Lock()
        self.condition = threading.Condition(self.processing_lock)
        
        # Adaptive batching state
        self.throughput_history = deque(maxlen=100)  # (timestamp, throughput)
        self.latency_history = deque(maxlen=100)
        self.optimal_batch_size = batch_size
        
        # Processing thread
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()
        
    def submit(self, request_data: Any, callback: Callable[[Any], None]):
        """Submit request for batch processing."""
        with self.condition:
            self.pending_requests.append({
                'data': request_data,
                'callback': callback,
                'timestamp': time.time()
            })
            self.condition.notify()
            
    def _process_batches(self):
        """Background thread for processing batches."""
        while True:
            with self.condition:
                # Wait for requests or timeout
                if not self.pending_requests:
                    self.condition.wait(timeout=0.1)
                    continue
                    
                # Collect batch
                batch = []
                start_time = time.time()
                
                while (len(batch) < self.optimal_batch_size and 
                       self.pending_requests and
                       time.time() - start_time < self.max_wait_time):
                    
                    batch.append(self.pending_requests.popleft())
                    
                    # If we don't have enough for a batch, wait a bit more
                    if len(batch) < self.optimal_batch_size and self.pending_requests:
                        self.condition.wait(timeout=0.01)  # Short wait
                        
            if batch:
                self._process_batch(batch)
                
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests."""
        start_time = time.time()
        
        try:
            # Extract data for processing
            batch_data = [req['data'] for req in batch]
            
            # Simulated batch processing (replace with actual model inference)
            results = self._simulate_batch_inference(batch_data)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            throughput = len(batch) / processing_time
            avg_latency = processing_time / len(batch)
            
            # Update adaptive parameters
            if self.adaptive:
                self._update_adaptive_parameters(throughput, avg_latency, len(batch))
            
            # Send results to callbacks
            for req, result in zip(batch, results):
                try:
                    req['callback'](result)
                except Exception as e:
                    logger.error(f"Callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Send errors to callbacks
            for req in batch:
                try:
                    req['callback'](None)  # Or appropriate error result
                except Exception as callback_error:
                    logger.error(f"Error callback failed: {callback_error}")
                    
    def _simulate_batch_inference(self, batch_data: List[Any]) -> List[Any]:
        """Simulate batch inference (replace with actual model calls)."""
        # This would be replaced with actual model inference
        time.sleep(0.1 * len(batch_data))  # Simulate processing time
        return [f"result_{i}" for i in range(len(batch_data))]
        
    def _update_adaptive_parameters(self, throughput: float, latency: float, batch_size: int):
        """Update adaptive batching parameters based on performance."""
        self.throughput_history.append((time.time(), throughput))
        self.latency_history.append((time.time(), latency))
        
        # Only adapt if we have enough history
        if len(self.throughput_history) < 10:
            return
            
        # Calculate recent average throughput
        recent_throughputs = [tp for _, tp in list(self.throughput_history)[-10:]]
        avg_throughput = sum(recent_throughputs) / len(recent_throughputs)
        
        # Adaptive logic: increase batch size if throughput is good and latency acceptable
        target_latency = 2.0  # 2 seconds max acceptable latency
        
        if latency < target_latency and throughput > avg_throughput * 0.9:
            # Good performance, can try larger batches
            if batch_size == self.optimal_batch_size:
                self.optimal_batch_size = min(self.optimal_batch_size + 1, 16)
        elif latency > target_latency:
            # Too slow, reduce batch size
            self.optimal_batch_size = max(self.optimal_batch_size - 1, 1)
            
        logger.debug(f"Adaptive batching: optimal_size={self.optimal_batch_size}, "
                    f"throughput={throughput:.2f}, latency={latency:.2f}")


class MemoryOptimizer:
    """Memory optimization and management."""
    
    def __init__(self):
        self.memory_pools = {}  # size -> deque of available tensors
        self.pool_lock = threading.Lock()
        self.allocated_tensors = weakref.WeakSet()
        
        # Memory tracking
        self.peak_memory = 0
        self.current_memory = 0
        self.allocation_count = 0
        
    def get_tensor_pool(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        """Get tensor from memory pool or allocate new one."""
        pool_key = (shape, dtype)
        
        with self.pool_lock:
            if pool_key not in self.memory_pools:
                self.memory_pools[pool_key] = deque()
                
            pool = self.memory_pools[pool_key]
            
            if pool:
                tensor = pool.popleft()
                logger.debug(f"Reused tensor from pool: {shape}")
                return tensor
            else:
                # Allocate new tensor (simulate)
                tensor_size = 1
                for dim in shape:
                    tensor_size *= dim
                if dtype == "float32":
                    tensor_size *= 4
                elif dtype == "float16":
                    tensor_size *= 2
                    
                self.current_memory += tensor_size
                self.allocation_count += 1
                self.peak_memory = max(self.peak_memory, self.current_memory)
                
                # Create mock tensor (in real implementation would be actual tensor)
                tensor = {
                    'shape': shape,
                    'dtype': dtype,
                    'size_bytes': tensor_size,
                    'id': self.allocation_count
                }
                
                self.allocated_tensors.add(id(tensor))
                logger.debug(f"Allocated new tensor: {shape}, total_memory={self.current_memory}")
                return tensor
                
    def return_tensor_to_pool(self, tensor: Any):
        """Return tensor to memory pool for reuse."""
        if not isinstance(tensor, dict):
            return
            
        pool_key = (tensor['shape'], tensor['dtype'])
        
        with self.pool_lock:
            if pool_key not in self.memory_pools:
                self.memory_pools[pool_key] = deque()
                
            pool = self.memory_pools[pool_key]
            
            # Limit pool size to prevent memory leaks
            if len(pool) < 10:  # Max 10 tensors per pool
                pool.append(tensor)
                logger.debug(f"Returned tensor to pool: {tensor['shape']}")
            else:
                # Pool full, just deallocate
                self.current_memory -= tensor['size_bytes']
                logger.debug(f"Deallocated tensor: {tensor['shape']}")
                
    def cleanup_pools(self):
        """Cleanup memory pools."""
        with self.pool_lock:
            total_freed = 0
            for pool_key, pool in self.memory_pools.items():
                while pool:
                    tensor = pool.popleft()
                    total_freed += tensor['size_bytes']
                    
            self.current_memory -= total_freed
            logger.info(f"Freed {total_freed} bytes from memory pools")
            
    def force_gc(self):
        """Force garbage collection and cleanup."""
        self.cleanup_pools()
        gc.collect()  # Python garbage collection
        
        # Simulate GPU memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cache cleared")
        except ImportError:
            pass
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'current_memory_mb': self.current_memory / (1024 * 1024),
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'allocation_count': self.allocation_count,
            'active_tensors': len(self.allocated_tensors),
            'pool_counts': {str(k): len(v) for k, v in self.memory_pools.items()}
        }


class ModelCompiler:
    """Model compilation and optimization."""
    
    def __init__(self):
        self.compiled_models = {}
        self.compilation_cache = IntelligentCache(max_size_mb=512)
        
    def compile_model(self, model: Any, optimization_level: str = "default") -> Any:
        """Compile model with optimizations."""
        model_id = self._get_model_id(model)
        cache_key = f"compiled_{model_id}_{optimization_level}"
        
        # Check cache first
        cached_model = self.compilation_cache.get(cache_key)
        if cached_model:
            logger.info(f"Using cached compiled model: {model_id}")
            return cached_model
            
        logger.info(f"Compiling model: {model_id} with {optimization_level} optimization")
        start_time = time.time()
        
        try:
            compiled_model = self._perform_compilation(model, optimization_level)
            compilation_time = time.time() - start_time
            
            # Cache compiled model
            self.compilation_cache.put(cache_key, compiled_model, ttl=3600)  # 1 hour TTL
            
            logger.info(f"Model compilation completed in {compilation_time:.2f}s")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return model  # Return original model on failure
            
    def _perform_compilation(self, model: Any, optimization_level: str) -> Any:
        """Perform actual model compilation."""
        # Simulate compilation process
        optimizations = {
            "basic": ["operator_fusion"],
            "default": ["operator_fusion", "memory_optimization"],
            "aggressive": ["operator_fusion", "memory_optimization", "quantization", "tensorrt"]
        }
        
        selected_opts = optimizations.get(optimization_level, optimizations["default"])
        
        # Apply optimizations
        compiled_model = {"original_model": model, "optimizations": selected_opts}
        
        for opt in selected_opts:
            self._apply_optimization(compiled_model, opt)
            
        return compiled_model
        
    def _apply_optimization(self, model: Dict[str, Any], optimization: str):
        """Apply specific optimization."""
        if optimization == "operator_fusion":
            # Simulate operator fusion
            model["fused_ops"] = True
            time.sleep(0.1)  # Simulate compilation time
            
        elif optimization == "memory_optimization":
            model["memory_optimized"] = True
            time.sleep(0.05)
            
        elif optimization == "quantization":
            model["quantized"] = True
            time.sleep(0.2)
            
        elif optimization == "tensorrt":
            model["tensorrt_optimized"] = True
            time.sleep(0.3)
            
        logger.debug(f"Applied optimization: {optimization}")
        
    def _get_model_id(self, model: Any) -> str:
        """Get unique identifier for model."""
        # In real implementation, would use model parameters hash
        return hashlib.md5(str(model).encode()).hexdigest()[:8]


class ParallelExecutor:
    """Parallel execution manager for benchmarks."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.use_processes = use_processes
        self.executor = None
        
    def __enter__(self):
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
            
    def map_parallel(self, func: Callable, items: List[Any], 
                    callback: Callable[[Any], None] = None) -> List[Any]:
        """Execute function in parallel on list of items."""
        if not self.executor:
            raise RuntimeError("ParallelExecutor not in context manager")
            
        # Submit all tasks
        future_to_item = {
            self.executor.submit(func, item): (i, item) 
            for i, item in enumerate(items)
        }
        
        results = [None] * len(items)
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            index, item = future_to_item[future]
            
            try:
                result = future.result()
                results[index] = result
                
                if callback:
                    callback(result)
                    
            except Exception as e:
                logger.error(f"Parallel execution failed for item {index}: {e}")
                results[index] = None
                
        return results
        
    def submit_async(self, func: Callable, *args, **kwargs):
        """Submit function for async execution."""
        if not self.executor:
            raise RuntimeError("ParallelExecutor not in context manager")
            
        return self.executor.submit(func, *args, **kwargs)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.cache = IntelligentCache(
            max_size_mb=self.config.get('cache_size_mb', 1024),
            max_entries=self.config.get('cache_max_entries', 10000)
        )
        
        self.batch_processor = BatchProcessor(
            batch_size=self.config.get('batch_size', 4),
            max_wait_time=self.config.get('batch_wait_time', 1.0),
            adaptive=self.config.get('adaptive_batching', True)
        )
        
        self.memory_optimizer = MemoryOptimizer()
        self.model_compiler = ModelCompiler()
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.optimization_stats = defaultdict(int)
        
        logger.info("Performance optimizer initialized")
        
    def optimize_model_loading(self, model_name: str, model_loader: Callable) -> Any:
        """Optimize model loading with caching and compilation."""
        cache_key = f"model_{model_name}"
        
        # Check cache first
        cached_model = self.cache.get(cache_key)
        if cached_model:
            logger.info(f"Using cached model: {model_name}")
            self.optimization_stats['cache_hits'] += 1
            return cached_model
            
        # Load and compile model
        logger.info(f"Loading and optimizing model: {model_name}")
        start_time = time.time()
        
        model = model_loader()
        compiled_model = self.model_compiler.compile_model(
            model, self.config.get('compilation_level', 'default')
        )
        
        # Cache compiled model
        self.cache.put(cache_key, compiled_model, ttl=3600)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded and cached in {load_time:.2f}s")
        
        self.optimization_stats['model_loads'] += 1
        return compiled_model
        
    def optimize_inference(self, 
                         model: Any, 
                         inputs: List[Any], 
                         batch_inference_func: Callable) -> List[Any]:
        """Optimize inference with batching and memory management."""
        
        # Use batch processor for optimal throughput
        results = []
        remaining_inputs = list(inputs)
        
        def process_result(result):
            results.append(result)
            
        # Submit all inputs for batch processing
        for input_data in remaining_inputs:
            self.batch_processor.submit(input_data, process_result)
            
        # Wait for all results (in real implementation would be more sophisticated)
        while len(results) < len(inputs):
            time.sleep(0.01)
            
        self.optimization_stats['inferences'] += len(inputs)
        return results
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        cache_stats = self.cache.get_stats()
        memory_stats = self.memory_optimizer.get_memory_stats()
        
        return {
            'optimization_stats': dict(self.optimization_stats),
            'cache_stats': cache_stats,
            'memory_stats': memory_stats,
            'performance_history_size': len(self.performance_history)
        }
        
    def cleanup(self):
        """Cleanup optimization resources."""
        self.cache.clear()
        self.memory_optimizer.cleanup_pools()
        logger.info("Performance optimizer cleanup completed")


# Utility functions
def memoize_with_ttl(ttl: float = 3600):
    """Memoization decorator with TTL."""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl:
                    return result
                    
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
            
        return wrapper
    return decorator


def profile_performance(func):
    """Decorator to profile function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
            
        end_time = time.time()
        end_memory = _get_memory_usage()
        
        performance_data = {
            'function': func.__name__,
            'duration': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'success': success,
            'timestamp': start_time
        }
        
        logger.debug(f"Performance profile: {performance_data}")
        
        if not success:
            raise result
            
        return result
        
    return wrapper


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0