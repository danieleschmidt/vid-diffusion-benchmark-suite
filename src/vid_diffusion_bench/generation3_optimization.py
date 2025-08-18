"""Generation 3: Advanced performance optimization and scaling features."""

import logging
import time
import asyncio
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import json
import pickle
import hashlib
import psutil
import os
import signal
import queue
import heapq
from functools import wraps, lru_cache
from contextlib import contextmanager
import math

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_gpu_optimization: bool = True
    enable_memory_pooling: bool = True
    enable_model_caching: bool = True
    enable_async_processing: bool = True
    enable_batch_optimization: bool = True
    enable_distributed_computing: bool = False
    max_concurrent_models: int = 4
    max_memory_usage_gb: float = 32.0
    cache_size_gb: float = 8.0
    prefetch_queue_size: int = 10
    optimization_level: str = "balanced"  # conservative, balanced, aggressive


@dataclass 
class PerformanceMetrics:
    """Advanced performance metrics tracking."""
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_videos_per_second: float = 0.0
    gpu_utilization_percent: float = 0.0
    memory_efficiency_percent: float = 0.0
    cache_hit_rate_percent: float = 0.0
    parallel_efficiency: float = 0.0
    energy_efficiency_videos_per_watt: float = 0.0
    bottleneck_analysis: Dict[str, float] = field(default_factory=dict)


class IntelligentCaching:
    """Advanced caching system with prediction and optimization."""
    
    def __init__(self, max_size_gb: float = 8.0, prediction_enabled: bool = True):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cache = {}
        self.metadata = {}
        self.access_history = deque(maxlen=10000)
        self.prediction_enabled = prediction_enabled
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
        
        # Prediction model for cache optimization
        self.access_patterns = defaultdict(list)
        self.prediction_model = None
        
    def get(self, key: str, default=None):
        """Get item from cache with prediction."""
        with self._lock:
            if key in self.cache:
                self.hit_count += 1
                self.metadata[key]['last_access'] = time.time()
                self.metadata[key]['access_count'] += 1
                self.access_history.append(('hit', key, time.time()))
                return self.cache[key]
            else:
                self.miss_count += 1
                self.access_history.append(('miss', key, time.time()))
                return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set item in cache with intelligent eviction."""
        with self._lock:
            value_size = self._estimate_size(value)
            
            # Check if we need to evict items
            if self._get_total_size() + value_size > self.max_size_bytes:
                self._intelligent_eviction(value_size)
            
            self.cache[key] = value
            self.metadata[key] = {
                'size': value_size,
                'created': time.time(),
                'last_access': time.time(),
                'access_count': 1,
                'ttl': ttl,
                'priority_score': self._calculate_priority(key)
            }
            
            # Update prediction patterns
            if self.prediction_enabled:
                self.access_patterns[key].append(time.time())
    
    def _intelligent_eviction(self, needed_space: int):
        """Evict items using intelligent algorithm."""
        # Calculate eviction scores for all items
        eviction_candidates = []
        
        for key, metadata in self.metadata.items():
            score = self._calculate_eviction_score(key, metadata)
            heapq.heappush(eviction_candidates, (score, key))
        
        # Evict items until we have enough space
        freed_space = 0
        while eviction_candidates and freed_space < needed_space:
            _, key_to_evict = heapq.heappop(eviction_candidates)
            freed_space += self.metadata[key_to_evict]['size']
            del self.cache[key_to_evict]
            del self.metadata[key_to_evict]
    
    def _calculate_eviction_score(self, key: str, metadata: Dict[str, Any]) -> float:
        """Calculate eviction score (lower = more likely to evict)."""
        age = time.time() - metadata['created']
        recency = time.time() - metadata['last_access']
        frequency = metadata['access_count']
        size = metadata['size']
        
        # Weighted score considering multiple factors
        score = (
            frequency * 0.4 +                    # Higher frequency = keep
            (1.0 / (recency + 1)) * 0.3 +         # More recent = keep
            (1.0 / (age + 1)) * 0.2 +             # Newer = keep
            (1.0 / (size + 1)) * 0.1              # Smaller = slight preference
        )
        
        return score
    
    def _calculate_priority(self, key: str) -> float:
        """Calculate priority score for cache item."""
        # Simple priority based on key patterns
        if 'model' in key:
            return 1.0  # High priority for models
        elif 'metrics' in key:
            return 0.8  # Medium-high priority for metrics
        elif 'temp' in key:
            return 0.2  # Low priority for temporary items
        return 0.5
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            import pickle
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default fallback
    
    def _get_total_size(self) -> int:
        """Get total cache size."""
        return sum(meta['size'] for meta in self.metadata.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            'hit_rate_percent': hit_rate,
            'total_items': len(self.cache),
            'total_size_mb': self._get_total_size() / (1024**2),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count
        }


class AsyncBenchmarkExecutor:
    """Asynchronous benchmark execution with advanced scheduling."""
    
    def __init__(self, max_concurrent: int = 4, optimization_level: str = "balanced"):
        self.max_concurrent = max_concurrent
        self.optimization_level = optimization_level
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.active_tasks = {}
        self.performance_history = deque(maxlen=1000)
        self._scheduler_running = False
        
    async def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit task for async execution."""
        task_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        task_info = {
            'id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'submitted': time.time(),
            'priority': kwargs.pop('priority', 0.5),
            'estimated_duration': kwargs.pop('estimated_duration', 60.0)
        }
        
        await self.task_queue.put(task_info)
        return task_id
    
    async def start_scheduler(self):
        """Start the async task scheduler."""
        if self._scheduler_running:
            return
            
        self._scheduler_running = True
        
        async def scheduler_loop():
            while self._scheduler_running:
                try:
                    # Dynamic concurrency adjustment
                    optimal_concurrency = self._calculate_optimal_concurrency()
                    
                    # Start new tasks if we have capacity
                    if len(self.active_tasks) < optimal_concurrency:
                        try:
                            task_info = await asyncio.wait_for(
                                self.task_queue.get(), timeout=1.0
                            )
                            await self._execute_task(task_info)
                        except asyncio.TimeoutError:
                            continue
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    await asyncio.sleep(1.0)
        
        # Start scheduler in background
        asyncio.create_task(scheduler_loop())
    
    async def _execute_task(self, task_info: Dict[str, Any]):
        """Execute individual task."""
        task_id = task_info['id']
        start_time = time.time()
        
        try:
            self.active_tasks[task_id] = {
                'start_time': start_time,
                'task_info': task_info
            }
            
            # Execute task
            if asyncio.iscoroutinefunction(task_info['func']):
                result = await task_info['func'](*task_info['args'], **task_info['kwargs'])
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, task_info['func'], *task_info['args']
                )
            
            execution_time = time.time() - start_time
            
            # Store performance metrics
            self.performance_history.append({
                'task_id': task_id,
                'execution_time': execution_time,
                'estimated_duration': task_info['estimated_duration'],
                'accuracy': abs(execution_time - task_info['estimated_duration']) / task_info['estimated_duration']
            })
            
            # Put result in result queue
            await self.result_queue.put({
                'task_id': task_id,
                'result': result,
                'execution_time': execution_time,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            await self.result_queue.put({
                'task_id': task_id,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'success': False
            })
            
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    def _calculate_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on system resources."""
        # Get system metrics
        cpu_count = psutil.cpu_count()
        memory_available = psutil.virtual_memory().available / (1024**3)  # GB
        
        # Get current performance
        if len(self.performance_history) > 10:
            recent_perf = list(self.performance_history)[-10:]
            avg_accuracy = sum(p['accuracy'] for p in recent_perf) / len(recent_perf)
            
            # Adjust based on prediction accuracy
            if avg_accuracy < 0.2:  # Good accuracy
                base_concurrency = min(cpu_count, self.max_concurrent)
            elif avg_accuracy < 0.5:  # Medium accuracy  
                base_concurrency = min(cpu_count // 2, self.max_concurrent)
            else:  # Poor accuracy, conservative
                base_concurrency = min(2, self.max_concurrent)
        else:
            base_concurrency = min(cpu_count // 2, self.max_concurrent)
        
        # Adjust based on memory
        if memory_available < 8:  # Low memory
            base_concurrency = min(base_concurrency, 2)
        elif memory_available > 32:  # High memory
            base_concurrency = min(base_concurrency * 2, self.max_concurrent)
        
        return max(1, base_concurrency)
    
    async def get_result(self, task_id: str, timeout: float = None) -> Dict[str, Any]:
        """Get result for specific task."""
        start_time = time.time()
        
        while True:
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                if result['task_id'] == task_id:
                    return result
                else:
                    # Put back result for other consumers
                    await self.result_queue.put(result)
                    
            except asyncio.TimeoutError:
                if timeout and (time.time() - start_time) > timeout:
                    raise asyncio.TimeoutError(f"Task {task_id} timed out")
                continue
    
    def stop_scheduler(self):
        """Stop the async scheduler."""
        self._scheduler_running = False


class ModelMemoryPool:
    """Intelligent model memory pooling and management."""
    
    def __init__(self, max_pool_size_gb: float = 16.0):
        self.max_pool_size_bytes = int(max_pool_size_gb * 1024**3)
        self.pool = {}
        self.model_metadata = {}
        self.access_tracker = defaultdict(list)
        self._lock = threading.RLock()
        
    def get_model(self, model_name: str, loader_func: Callable) -> Any:
        """Get model from pool or load if not cached."""
        with self._lock:
            if model_name in self.pool:
                self.access_tracker[model_name].append(time.time())
                logger.debug(f"Model {model_name} retrieved from pool")
                return self.pool[model_name]
            
            # Load model
            model = loader_func()
            model_size = self._estimate_model_size(model)
            
            # Check if we need to evict models
            if self._get_pool_size() + model_size > self.max_pool_size_bytes:
                self._evict_models(model_size)
            
            # Add to pool
            self.pool[model_name] = model
            self.model_metadata[model_name] = {
                'size': model_size,
                'loaded_at': time.time(),
                'access_count': 1
            }
            self.access_tracker[model_name].append(time.time())
            
            logger.info(f"Model {model_name} loaded and added to pool ({model_size / 1024**2:.1f}MB)")
            return model
    
    def _evict_models(self, needed_space: int):
        """Evict least recently used models."""
        # Calculate LRU scores
        eviction_candidates = []
        current_time = time.time()
        
        for model_name in self.pool.keys():
            last_access = max(self.access_tracker[model_name]) if self.access_tracker[model_name] else 0
            lru_score = current_time - last_access
            model_size = self.model_metadata[model_name]['size']
            
            eviction_candidates.append((lru_score, model_size, model_name))
        
        # Sort by LRU score (highest first = least recently used)
        eviction_candidates.sort(reverse=True)
        
        freed_space = 0
        for lru_score, model_size, model_name in eviction_candidates:
            if freed_space >= needed_space:
                break
                
            logger.info(f"Evicting model {model_name} from pool (freed {model_size / 1024**2:.1f}MB)")
            del self.pool[model_name]
            del self.model_metadata[model_name]
            del self.access_tracker[model_name]
            freed_space += model_size
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate model memory usage."""
        try:
            if hasattr(model, 'parameters'):
                # PyTorch model
                total_params = sum(p.numel() for p in model.parameters())
                return total_params * 4  # Assume float32
            elif hasattr(model, '__sizeof__'):
                return model.__sizeof__()
            else:
                return 1024**3  # 1GB default fallback
        except:
            return 1024**3
    
    def _get_pool_size(self) -> int:
        """Get total pool size in bytes."""
        return sum(meta['size'] for meta in self.model_metadata.values())
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'total_models': len(self.pool),
            'total_size_gb': self._get_pool_size() / (1024**3),
            'utilization_percent': (self._get_pool_size() / self.max_pool_size_bytes) * 100,
            'models': {name: {
                'size_mb': meta['size'] / (1024**2),
                'loaded_at': meta['loaded_at'],
                'access_count': len(self.access_tracker[name])
            } for name, meta in self.model_metadata.items()}
        }


class BatchOptimizer:
    """Intelligent batch processing optimization."""
    
    def __init__(self, target_latency_ms: float = 5000.0):
        self.target_latency_ms = target_latency_ms
        self.batch_history = deque(maxlen=100)
        self.optimal_batch_sizes = {}
        
    def optimize_batch_size(self, model_name: str, data_size: int, 
                          available_memory_gb: float) -> int:
        """Calculate optimal batch size for model and data."""
        # Start with a reasonable base batch size
        base_batch_size = self._calculate_base_batch_size(data_size, available_memory_gb)
        
        # Adjust based on historical performance
        if model_name in self.optimal_batch_sizes:
            historical_optimal = self.optimal_batch_sizes[model_name]
            base_batch_size = int((base_batch_size + historical_optimal) / 2)
        
        # Ensure batch size is reasonable
        batch_size = max(1, min(base_batch_size, data_size))
        
        logger.debug(f"Optimal batch size for {model_name}: {batch_size}")
        return batch_size
    
    def _calculate_base_batch_size(self, data_size: int, available_memory_gb: float) -> int:
        """Calculate base batch size from system resources."""
        # Estimate memory per item (rough approximation)
        memory_per_item_mb = 200  # Conservative estimate for video processing
        
        # Calculate max batch size based on memory
        max_batch_from_memory = int((available_memory_gb * 1024 * 0.7) / memory_per_item_mb)
        
        # Calculate batch size for target latency (very rough heuristic)
        target_batch_from_latency = max(1, int(self.target_latency_ms / 1000))
        
        return min(max_batch_from_memory, target_batch_from_latency, data_size)
    
    def update_performance(self, model_name: str, batch_size: int, 
                         latency_ms: float, success_rate: float):
        """Update performance history for batch optimization."""
        performance_record = {
            'model_name': model_name,
            'batch_size': batch_size,
            'latency_ms': latency_ms,
            'success_rate': success_rate,
            'efficiency': success_rate / (latency_ms / 1000),  # success per second
            'timestamp': time.time()
        }
        
        self.batch_history.append(performance_record)
        
        # Update optimal batch size
        model_records = [r for r in self.batch_history if r['model_name'] == model_name]
        if len(model_records) >= 5:
            # Find batch size with best efficiency
            best_record = max(model_records, key=lambda r: r['efficiency'])
            self.optimal_batch_sizes[model_name] = best_record['batch_size']


class PerformanceProfiler:
    """Advanced performance profiling and bottleneck analysis."""
    
    def __init__(self):
        self.profiles = defaultdict(list)
        self.bottleneck_detector = BottleneckDetector()
        
    @contextmanager
    def profile(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        try:
            import torch
            gpu_start = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        except ImportError:
            gpu_start = 0
            
        start_cpu = psutil.cpu_percent()
        
        yield
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)
        
        try:
            import torch
            gpu_end = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        except ImportError:
            gpu_end = 0
            
        end_cpu = psutil.cpu_percent()
        
        profile_data = {
            'operation': operation_name,
            'duration_ms': (end_time - start_time) * 1000,
            'memory_delta_mb': end_memory - start_memory,
            'gpu_memory_delta_mb': gpu_end - gpu_start,
            'cpu_usage_percent': (start_cpu + end_cpu) / 2,
            'timestamp': start_time,
            'metadata': metadata or {}
        }
        
        self.profiles[operation_name].append(profile_data)
        
        # Analyze bottlenecks
        self.bottleneck_detector.analyze(profile_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        for operation, profiles in self.profiles.items():
            if not profiles:
                continue
                
            durations = [p['duration_ms'] for p in profiles]
            memory_deltas = [p['memory_delta_mb'] for p in profiles]
            
            summary[operation] = {
                'call_count': len(profiles),
                'avg_duration_ms': sum(durations) / len(durations),
                'p95_duration_ms': self._percentile(durations, 95),
                'p99_duration_ms': self._percentile(durations, 99),
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'total_time_s': sum(durations) / 1000
            }
        
        # Add bottleneck analysis
        summary['bottlenecks'] = self.bottleneck_detector.get_bottlenecks()
        
        return summary
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class BottleneckDetector:
    """Automatic bottleneck detection and analysis."""
    
    def __init__(self):
        self.bottlenecks = defaultdict(list)
        self.thresholds = {
            'cpu_high': 80.0,      # CPU usage %
            'memory_high': 1000.0,  # Memory delta MB
            'duration_high': 10000.0,  # Duration ms
            'gpu_memory_high': 2000.0   # GPU memory delta MB
        }
    
    def analyze(self, profile_data: Dict[str, Any]):
        """Analyze profile data for bottlenecks."""
        operation = profile_data['operation']
        
        # CPU bottleneck
        if profile_data['cpu_usage_percent'] > self.thresholds['cpu_high']:
            self.bottlenecks[operation].append({
                'type': 'cpu',
                'severity': profile_data['cpu_usage_percent'],
                'timestamp': profile_data['timestamp']
            })
        
        # Memory bottleneck
        if profile_data['memory_delta_mb'] > self.thresholds['memory_high']:
            self.bottlenecks[operation].append({
                'type': 'memory',
                'severity': profile_data['memory_delta_mb'],
                'timestamp': profile_data['timestamp']
            })
        
        # Duration bottleneck
        if profile_data['duration_ms'] > self.thresholds['duration_high']:
            self.bottlenecks[operation].append({
                'type': 'duration',
                'severity': profile_data['duration_ms'],
                'timestamp': profile_data['timestamp']
            })
        
        # GPU memory bottleneck
        if profile_data['gpu_memory_delta_mb'] > self.thresholds['gpu_memory_high']:
            self.bottlenecks[operation].append({
                'type': 'gpu_memory',
                'severity': profile_data['gpu_memory_delta_mb'],
                'timestamp': profile_data['timestamp']
            })
    
    def get_bottlenecks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get detected bottlenecks."""
        return dict(self.bottlenecks)


# GPU optimization utilities
def optimize_gpu_memory():
    """Optimize GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("GPU memory optimized")
    except ImportError:
        logger.info("PyTorch not available for GPU optimization")


def warm_up_gpu():
    """Warm up GPU for optimal performance."""
    try:
        import torch
        if torch.cuda.is_available():
            # Run a small operation to warm up GPU
            x = torch.randn(100, 100, device='cuda')
            y = torch.mm(x, x.T)
            del x, y
            torch.cuda.synchronize()
            logger.info("GPU warmed up")
    except ImportError:
        pass