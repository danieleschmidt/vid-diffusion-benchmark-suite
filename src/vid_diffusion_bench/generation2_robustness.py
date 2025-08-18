"""Generation 2: Comprehensive robustness and reliability enhancements."""

import logging
import time
import threading
import queue
import json
import pickle
import asyncio
import signal
import functools
import traceback
from typing import Dict, List, Any, Optional, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import socket
import psutil
import os
import subprocess
import shutil

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """System health check result."""
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    gpu_available: bool = False
    gpu_memory_total: Optional[float] = None
    gpu_memory_free: Optional[float] = None
    gpu_temperature: Optional[float] = None
    network_latency: Optional[float] = None
    is_healthy: bool = True
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.history = deque(maxlen=1440)  # Keep 24h of data (1 minute intervals)
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'gpu_memory_percent': 90.0,
            'gpu_temperature': 85.0
        }
        self._monitoring = False
        self._monitor_thread = None
        
    def check_health(self) -> HealthCheckResult:
        """Perform comprehensive health check."""
        result = HealthCheckResult(timestamp=time.time())
        
        try:
            # CPU and Memory
            result.cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            result.memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            result.disk_usage_percent = (disk.used / disk.total) * 100
            
            # GPU check
            try:
                import torch
                result.gpu_available = torch.cuda.is_available()
                if result.gpu_available:
                    result.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    result.gpu_memory_free = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    # Try to get GPU temperature (NVIDIA only)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        result.gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        pass  # GPU temperature not available
                        
            except ImportError:
                result.gpu_available = False
                
            # Network latency check
            try:
                start = time.time()
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                result.network_latency = (time.time() - start) * 1000
            except:
                result.network_latency = None
                result.warnings.append("Network connectivity check failed")
                
            # Evaluate thresholds
            if result.cpu_percent > self.thresholds['cpu_percent']:
                result.warnings.append(f"High CPU usage: {result.cpu_percent:.1f}%")
                
            if result.memory_percent > self.thresholds['memory_percent']:
                result.warnings.append(f"High memory usage: {result.memory_percent:.1f}%")
                
            if result.disk_usage_percent > self.thresholds['disk_usage_percent']:
                result.errors.append(f"High disk usage: {result.disk_usage_percent:.1f}%")
                result.is_healthy = False
                
            if result.gpu_temperature and result.gpu_temperature > self.thresholds['gpu_temperature']:
                result.warnings.append(f"High GPU temperature: {result.gpu_temperature:.1f}°C")
                
        except Exception as e:
            result.errors.append(f"Health check failed: {str(e)}")
            result.is_healthy = False
            
        return result
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                health = self.check_health()
                self.history.append(health)
                
                # Log warnings and errors
                for warning in health.warnings:
                    logger.warning(f"Health warning: {warning}")
                for error in health.errors:
                    logger.error(f"Health error: {error}")
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
            time.sleep(self.check_interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        if not self.history:
            return {"status": "no_data"}
            
        recent = list(self.history)[-10:]  # Last 10 checks
        avg_cpu = sum(h.cpu_percent for h in recent) / len(recent)
        avg_memory = sum(h.memory_percent for h in recent) / len(recent)
        
        warnings_count = sum(len(h.warnings) for h in recent)
        errors_count = sum(len(h.errors) for h in recent)
        
        return {
            "status": "healthy" if errors_count == 0 else "unhealthy",
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "warnings_count": warnings_count,
            "errors_count": errors_count,
            "checks_count": len(recent),
            "last_check": recent[-1].timestamp if recent else None
        }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, 
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self._failure_count = 0
        self._last_failure_time = None
        self._state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable):
        """Decorator to add circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self._state == 'OPEN':
                    if self._should_attempt_reset():
                        self._state = 'HALF_OPEN'
                        logger.info(f"Circuit breaker for {func.__name__} moving to HALF_OPEN")
                    else:
                        raise Exception(f"Circuit breaker OPEN for {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                    
                except self.expected_exception as e:
                    self._on_failure()
                    raise e
                    
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (time.time() - self._last_failure_time) >= self.timeout
    
    def _on_success(self):
        """Handle successful call."""
        self._failure_count = 0
        if self._state == 'HALF_OPEN':
            self._state = 'CLOSED'
            logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = 'OPEN'
            logger.warning(f"Circuit breaker OPENED after {self._failure_count} failures")
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state


class BenchmarkRecovery:
    """Benchmark failure recovery and resilience."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.recovery_strategies = {}
        self.failure_history = defaultdict(list)
        
    def register_recovery_strategy(self, exception_type: type, strategy: Callable):
        """Register recovery strategy for specific exception type."""
        self.recovery_strategies[exception_type] = strategy
        
    def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with automatic recovery."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self.failure_history[func.__name__].append({
                    'timestamp': time.time(),
                    'exception': str(e),
                    'attempt': attempt
                })
                
                # Try recovery strategy
                recovery_strategy = self._get_recovery_strategy(type(e))
                if recovery_strategy:
                    try:
                        logger.info(f"Attempting recovery for {func.__name__} after {type(e).__name__}")
                        recovery_strategy(e, attempt)
                    except Exception as recovery_error:
                        logger.error(f"Recovery failed: {recovery_error}")
                
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for {func.__name__}")
        
        raise last_exception
    
    def _get_recovery_strategy(self, exception_type: type) -> Optional[Callable]:
        """Get recovery strategy for exception type."""
        for exc_type, strategy in self.recovery_strategies.items():
            if issubclass(exception_type, exc_type):
                return strategy
        return None


class DataBackupManager:
    """Automatic data backup and recovery."""
    
    def __init__(self, backup_dir: str = "./backups", max_backups: int = 10):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        
    def backup_data(self, data: Any, backup_id: str) -> str:
        """Backup data with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_id}_{timestamp}.json"
        backup_path = self.backup_dir / backup_filename
        
        try:
            with open(backup_path, 'w') as f:
                if hasattr(data, 'to_dict'):
                    json.dump(data.to_dict(), f, indent=2)
                elif isinstance(data, dict):
                    json.dump(data, f, indent=2)
                else:
                    json.dump(str(data), f, indent=2)
            
            logger.info(f"Data backed up to {backup_path}")
            self._cleanup_old_backups(backup_id)
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
    
    def restore_latest(self, backup_id: str) -> Optional[Any]:
        """Restore latest backup for given ID."""
        pattern = f"{backup_id}_*.json"
        backup_files = list(self.backup_dir.glob(pattern))
        
        if not backup_files:
            logger.warning(f"No backups found for {backup_id}")
            return None
            
        # Get latest backup
        latest_backup = max(backup_files, key=os.path.getctime)
        
        try:
            with open(latest_backup, 'r') as f:
                data = json.load(f)
            logger.info(f"Data restored from {latest_backup}")
            return data
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return None
    
    def _cleanup_old_backups(self, backup_id: str):
        """Clean up old backups keeping only max_backups."""
        pattern = f"{backup_id}_*.json"
        backup_files = sorted(self.backup_dir.glob(pattern), key=os.path.getctime)
        
        while len(backup_files) > self.max_backups:
            old_backup = backup_files.pop(0)
            try:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")


class AdvancedLoggingManager:
    """Advanced logging with structured output and monitoring."""
    
    def __init__(self, log_dir: str = "./logs", structured: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.structured = structured
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        # Create formatter
        if self.structured:
            formatter = self._create_structured_formatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # File handler for all logs
        all_handler = logging.FileHandler(self.log_dir / "benchmark.log")
        all_handler.setLevel(logging.DEBUG)
        all_handler.setFormatter(formatter)
        
        # Error handler for errors only
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Performance handler for metrics
        perf_handler = logging.FileHandler(self.log_dir / "performance.log")
        perf_handler.setLevel(logging.INFO)
        perf_handler.addFilter(self._performance_filter)
        perf_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(all_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(perf_handler)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s: %(message)s'
        ))
        root_logger.addHandler(console_handler)
        
    def _create_structured_formatter(self):
        """Create structured JSON formatter."""
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_obj['exception'] = self.formatException(record.exc_info)
                    
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                                  'pathname', 'filename', 'module', 'lineno',
                                  'funcName', 'created', 'msecs', 'relativeCreated',
                                  'thread', 'threadName', 'processName', 'process',
                                  'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                        log_obj[key] = value
                        
                return json.dumps(log_obj)
                
        return StructuredFormatter()
    
    def _performance_filter(self, record):
        """Filter for performance-related logs."""
        performance_keywords = ['latency', 'throughput', 'fps', 'memory', 'gpu', 'benchmark']
        return any(keyword in record.getMessage().lower() for keyword in performance_keywords)


# Recovery strategies for common failures
def gpu_memory_recovery(exception: Exception, attempt: int):
    """Recovery strategy for GPU memory errors."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared for recovery")
        import gc
        gc.collect()
        logger.info("Garbage collection performed for recovery")
    except ImportError:
        logger.warning("PyTorch not available for GPU memory recovery")


def disk_space_recovery(exception: Exception, attempt: int):
    """Recovery strategy for disk space errors."""
    try:
        # Clean up temporary files
        temp_dirs = ['/tmp', './temp', './cache']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    if file.startswith('benchmark_temp_'):
                        try:
                            os.remove(os.path.join(temp_dir, file))
                        except:
                            pass
        logger.info("Temporary files cleaned for recovery")
    except Exception as e:
        logger.warning(f"Disk cleanup failed: {e}")


def network_recovery(exception: Exception, attempt: int):
    """Recovery strategy for network errors."""
    time.sleep(min(2 ** attempt, 30))  # Exponential backoff with cap
    logger.info(f"Network recovery: waited {min(2 ** attempt, 30)}s")