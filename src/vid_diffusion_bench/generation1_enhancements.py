"""Generation 1 enhancements: Basic functionality improvements."""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json
import traceback

logger = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    """Progress tracking information."""
    current: int = 0
    total: int = 0
    start_time: float = 0.0
    eta_seconds: Optional[float] = None
    current_task: str = "Initializing"
    
    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        return (self.current / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def update_eta(self):
        """Update estimated time to arrival."""
        if self.current > 0 and self.elapsed_time > 0:
            rate = self.current / self.elapsed_time
            if rate > 0:
                remaining = self.total - self.current
                self.eta_seconds = remaining / rate


class BenchmarkProgressTracker:
    """Progress tracking for benchmark operations."""
    
    def __init__(self, callback: Optional[Callable[[ProgressInfo], None]] = None):
        self.callback = callback
        self.progress = ProgressInfo()
        self._lock = threading.Lock()
    
    def start(self, total: int, task_name: str = "Processing"):
        """Start progress tracking."""
        with self._lock:
            self.progress.current = 0
            self.progress.total = total
            self.progress.start_time = time.time()
            self.progress.current_task = task_name
            self.progress.eta_seconds = None
        self._notify()
    
    def update(self, current: int, task_name: str = None):
        """Update progress."""
        with self._lock:
            self.progress.current = current
            if task_name:
                self.progress.current_task = task_name
            self.progress.update_eta()
        self._notify()
    
    def increment(self, task_name: str = None):
        """Increment progress by 1."""
        with self._lock:
            self.progress.current += 1
            if task_name:
                self.progress.current_task = task_name
            self.progress.update_eta()
        self._notify()
    
    def _notify(self):
        """Notify callback of progress update."""
        if self.callback:
            try:
                self.callback(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")


class RetryHandler:
    """Enhanced retry handler with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def __call__(self, func: Callable):
        """Decorator to add retry logic."""
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < self.max_attempts - 1:
                        delay = min(
                            self.base_delay * (self.exponential_base ** attempt),
                            self.max_delay
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {self.max_attempts} attempts failed")
            
            raise last_exception
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


@contextmanager
def resource_monitor(memory_threshold_gb: float = 30.0, 
                     gpu_utilization_threshold: float = 0.95):
    """Context manager for resource monitoring."""
    import psutil
    
    # Check initial resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < memory_threshold_gb:
        logger.warning(f"Low system memory: {memory_gb:.1f}GB < {memory_threshold_gb}GB")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory available: {gpu_memory:.1f}GB")
    except ImportError:
        logger.info("PyTorch not available for GPU monitoring")
    
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Resource monitor: Operation completed in {elapsed:.2f}s")
        
        # Final memory check
        current_memory = psutil.virtual_memory().percent
        if current_memory > 90:
            logger.warning(f"High memory usage detected: {current_memory:.1f}%")


class SafetyValidator:
    """Input validation and safety checks."""
    
    @staticmethod
    def validate_prompts(prompts: List[str], max_length: int = 1000, 
                        max_count: int = 100) -> List[str]:
        """Validate and sanitize prompts."""
        if len(prompts) > max_count:
            logger.warning(f"Too many prompts ({len(prompts)}), limiting to {max_count}")
            prompts = prompts[:max_count]
        
        validated = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                logger.warning(f"Skipping non-string prompt at index {i}")
                continue
                
            if len(prompt) > max_length:
                logger.warning(f"Truncating long prompt at index {i}")
                prompt = prompt[:max_length] + "..."
            
            # Basic safety filtering
            if any(pattern in prompt.lower() for pattern in ['<script', 'javascript:', 'eval(', 'exec(', '__import__']):
                logger.warning(f"Skipping potentially unsafe prompt at index {i}")
                continue
                
            validated.append(prompt.strip())
        
        return validated
    
    @staticmethod
    def validate_model_params(num_frames: int = 16, fps: int = 8, 
                            resolution: tuple = (512, 512), 
                            batch_size: int = 1) -> dict:
        """Validate model generation parameters."""
        # Reasonable limits for video generation
        num_frames = max(1, min(num_frames, 200))
        fps = max(1, min(fps, 60))
        
        # Resolution validation
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            logger.warning("Invalid resolution format, using default (512, 512)")
            resolution = (512, 512)
        
        width, height = resolution
        width = max(64, min(width, 2048))
        height = max(64, min(height, 2048))
        
        # Batch size limits
        batch_size = max(1, min(batch_size, 16))
        
        return {
            'num_frames': num_frames,
            'fps': fps, 
            'resolution': (width, height),
            'batch_size': batch_size
        }


class BasicCacheManager:
    """Simple in-memory cache for generated content."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get_cache_key(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate cache key for prompt and parameters."""
        import hashlib
        key_data = f"{model_name}:{prompt}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get(self, key: str):
        """Get cached result."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached result."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            self.cache.pop(oldest_key, None)
            self.access_times.pop(oldest_key, None)
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()
    
    @property
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
        
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': 0.0  # Would need hit/miss tracking for real implementation
        }