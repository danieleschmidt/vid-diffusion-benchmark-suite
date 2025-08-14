"""Enhanced monitoring, logging and observability for video diffusion benchmarks."""

import logging
import time
import threading
import json
import traceback
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
from contextlib import contextmanager
import sys
import os

# Setup structured logging
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement point."""
    timestamp: float
    name: str
    value: float
    labels: Dict[str, str]
    unit: str = ""


@dataclass 
class PerformanceProfile:
    """Performance profiling data."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    gpu_memory_before: Optional[float] = None
    gpu_memory_after: Optional[float] = None
    gpu_memory_peak: Optional[float] = None
    metadata: Dict[str, Any] = None


class MetricsCollector:
    """Thread-safe metrics collection system."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.Lock()
        self._collectors = []
        self._collection_thread = None
        self._stop_collection = threading.Event()
        
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None, unit: str = ""):
        """Record a single metric point."""
        with self.lock:
            point = MetricPoint(
                timestamp=time.time(),
                name=name,
                value=value,
                labels=labels or {},
                unit=unit
            )
            self.metrics[name].append(point)
            
    def record_counter(self, name: str, labels: Dict[str, str] = None):
        """Record a counter increment."""
        self.record_metric(name, 1.0, labels, "count")
        
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None, unit: str = ""):
        """Record a gauge value."""
        self.record_metric(name, value, labels, unit)
        
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None, unit: str = ""):
        """Record a histogram value (duration, size, etc.)."""
        self.record_metric(name, value, labels, unit)
        
    def get_metrics(self, name: str = None, since: float = None) -> Dict[str, List[MetricPoint]]:
        """Get collected metrics."""
        with self.lock:
            if name:
                metrics = {name: list(self.metrics.get(name, []))}
            else:
                metrics = {k: list(v) for k, v in self.metrics.items()}
                
            if since:
                for metric_name in metrics:
                    metrics[metric_name] = [
                        p for p in metrics[metric_name] 
                        if p.timestamp >= since
                    ]
                    
            return metrics
    
    def start_background_collection(self, interval: float = 30.0):
        """Start background system metrics collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            return
            
        self._stop_collection.clear()
        self._collection_thread = threading.Thread(
            target=self._background_collector,
            args=(interval,),
            daemon=True
        )
        self._collection_thread.start()
        logger.info("Started background metrics collection")
        
    def stop_background_collection(self):
        """Stop background metrics collection."""
        if self._collection_thread:
            self._stop_collection.set()
            self._collection_thread.join(timeout=5.0)
            logger.info("Stopped background metrics collection")
            
    def _background_collector(self, interval: float):
        """Background thread for system metrics collection."""
        while not self._stop_collection.wait(interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.record_gauge("system.cpu.utilization", cpu_percent, unit="%")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system.memory.used", memory.used / (1024**3), unit="GB")
            self.record_gauge("system.memory.available", memory.available / (1024**3), unit="GB")
            self.record_gauge("system.memory.percent", memory.percent, unit="%")
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            self.record_gauge("system.disk.used", disk_usage.used / (1024**3), unit="GB")
            self.record_gauge("system.disk.free", disk_usage.free / (1024**3), unit="GB")
            
        except ImportError:
            pass  # psutil not available
        
        # GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    labels = {"device": str(i)}
                    
                    # Memory
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    self.record_gauge("gpu.memory.allocated", allocated, labels, "GB")
                    self.record_gauge("gpu.memory.reserved", reserved, labels, "GB")
                    
                    # Utilization (simplified)
                    try:
                        utilization = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                        self.record_gauge("gpu.utilization", utilization, labels, "%")
                    except:
                        pass
                        
        except ImportError:
            pass  # torch not available


class PerformanceProfiler:
    """Detailed performance profiling."""
    
    def __init__(self):
        self.profiles = []
        self.current_profiles = {}  # operation_name -> start_data
        self.lock = threading.Lock()
        
    @contextmanager
    def profile(self, operation: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling operations."""
        profile_data = self._start_profile(operation, metadata)
        try:
            yield profile_data
        except Exception as e:
            profile_data.metadata = profile_data.metadata or {}
            profile_data.metadata['exception'] = str(e)
            profile_data.metadata['traceback'] = traceback.format_exc()
            raise
        finally:
            self._end_profile(profile_data)
            
    def _start_profile(self, operation: str, metadata: Dict[str, Any] = None) -> PerformanceProfile:
        """Start profiling an operation."""
        profile = PerformanceProfile(
            operation=operation,
            start_time=time.time(),
            end_time=0,
            duration=0,
            metadata=metadata or {}
        )
        
        # Collect initial metrics
        profile.memory_before = self._get_memory_usage()
        profile.gpu_memory_before = self._get_gpu_memory_usage()
        
        with self.lock:
            self.current_profiles[operation] = profile
            
        return profile
        
    def _end_profile(self, profile: PerformanceProfile):
        """End profiling an operation."""
        profile.end_time = time.time()
        profile.duration = profile.end_time - profile.start_time
        profile.memory_after = self._get_memory_usage()
        profile.gpu_memory_after = self._get_gpu_memory_usage()
        
        with self.lock:
            if profile.operation in self.current_profiles:
                del self.current_profiles[profile.operation]
            self.profiles.append(profile)
            
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in GB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            return None
            
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
        except ImportError:
            pass
        return None
        
    def get_profiles(self, operation: str = None, since: float = None) -> List[PerformanceProfile]:
        """Get performance profiles."""
        with self.lock:
            profiles = self.profiles.copy()
            
        if operation:
            profiles = [p for p in profiles if p.operation == operation]
            
        if since:
            profiles = [p for p in profiles if p.start_time >= since]
            
        return profiles
        
    def get_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        profiles = self.get_profiles(operation)
        
        if not profiles:
            return {}
            
        durations = [p.duration for p in profiles]
        memory_usage = [p.memory_after - p.memory_before for p in profiles 
                       if p.memory_before and p.memory_after]
        
        summary = {
            'count': len(profiles),
            'total_duration': sum(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
        }
        
        if memory_usage:
            summary.update({
                'avg_memory_delta': sum(memory_usage) / len(memory_usage),
                'max_memory_delta': max(memory_usage)
            })
            
        return summary


class StructuredLogger:
    """Structured logging with context and correlation."""
    
    def __init__(self, name: str, output_dir: str = "./logs"):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.correlation_id = None
        self.context = {}
        
        # Setup file handler
        log_file = self.output_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # JSON formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(name)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking."""
        self.correlation_id = correlation_id
        
    def set_context(self, **kwargs):
        """Set logging context."""
        self.context.update(kwargs)
        
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context."""
        log_data = {
            'message': message,
            'correlation_id': self.correlation_id,
            'context': self.context,
            **kwargs
        }
        return json.dumps(log_data)
        
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
        
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
        
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))
        
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message, **kwargs))


class HealthChecker:
    """System health monitoring and alerts."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks = {}
        self.alerts = []
        self.thresholds = {
            'memory_usage_percent': 90.0,
            'disk_usage_percent': 85.0,
            'gpu_memory_usage_percent': 95.0,
            'consecutive_failures': 5
        }
        
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        
    def check_system_health(self) -> Dict[str, Any]:
        """Run all health checks."""
        health_status = {
            'timestamp': time.time(),
            'overall_healthy': True,
            'checks': {},
            'alerts': []
        }
        
        # Run registered health checks
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                health_status['checks'][name] = {
                    'healthy': result,
                    'timestamp': time.time()
                }
                if not result:
                    health_status['overall_healthy'] = False
                    
            except Exception as e:
                health_status['checks'][name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
                health_status['overall_healthy'] = False
                
        # Check system metrics
        recent_metrics = self.metrics.get_metrics(since=time.time() - 300)  # Last 5 minutes
        
        # Memory check
        memory_metrics = recent_metrics.get('system.memory.percent', [])
        if memory_metrics:
            latest_memory = memory_metrics[-1].value
            if latest_memory > self.thresholds['memory_usage_percent']:
                alert = {
                    'type': 'high_memory_usage',
                    'severity': 'warning',
                    'value': latest_memory,
                    'threshold': self.thresholds['memory_usage_percent'],
                    'timestamp': time.time()
                }
                health_status['alerts'].append(alert)
                
        # GPU memory check  
        gpu_metrics = recent_metrics.get('gpu.memory.allocated', [])
        if gpu_metrics:
            for metric in gpu_metrics:
                try:
                    import torch
                    device_props = torch.cuda.get_device_properties(int(metric.labels.get('device', '0')))
                    total_memory = device_props.total_memory / (1024**3)
                    usage_percent = (metric.value / total_memory) * 100
                    
                    if usage_percent > self.thresholds['gpu_memory_usage_percent']:
                        alert = {
                            'type': 'high_gpu_memory_usage',
                            'severity': 'critical',
                            'device': metric.labels.get('device', '0'),
                            'value': usage_percent,
                            'threshold': self.thresholds['gpu_memory_usage_percent'],
                            'timestamp': time.time()
                        }
                        health_status['alerts'].append(alert)
                except:
                    pass
                    
        return health_status
        
    def is_healthy(self) -> bool:
        """Check if system is currently healthy."""
        health = self.check_system_health()
        return health['overall_healthy'] and len(health['alerts']) == 0


class BenchmarkMonitor:
    """Comprehensive monitoring for benchmark operations."""
    
    def __init__(self, output_dir: str = "./monitoring"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = MetricsCollector()
        self.profiler = PerformanceProfiler()
        self.health = HealthChecker(self.metrics)
        self.logger = StructuredLogger("benchmark", str(self.output_dir))
        
        # Start background collection
        self.metrics.start_background_collection()
        
        # Register basic health checks
        self._setup_health_checks()
        
    def _setup_health_checks(self):
        """Setup basic health checks."""
        def check_disk_space():
            try:
                import shutil
                total, used, free = shutil.disk_usage(self.output_dir)
                usage_percent = (used / total) * 100
                return usage_percent < 85.0
            except:
                return True  # Assume healthy if can't check
                
        def check_python_health():
            return sys.version_info >= (3, 8)
            
        self.health.register_health_check('disk_space', check_disk_space)
        self.health.register_health_check('python_version', check_python_health)
        
    def start_benchmark_monitoring(self, benchmark_id: str, **context):
        """Start monitoring a benchmark run."""
        self.logger.set_correlation_id(benchmark_id)
        self.logger.set_context(**context)
        self.logger.info("Starting benchmark monitoring", benchmark_id=benchmark_id)
        
        self.metrics.record_counter("benchmark.started", {"benchmark_id": benchmark_id})
        
    def record_model_generation(self, model_name: str, duration: float, success: bool, **metadata):
        """Record model generation metrics."""
        labels = {"model": model_name, "status": "success" if success else "failure"}
        
        self.metrics.record_histogram("generation.duration", duration, labels, "seconds")
        self.metrics.record_counter("generation.total", labels)
        
        if success:
            self.metrics.record_counter("generation.success", {"model": model_name})
            self.logger.info("Generation successful", 
                           model=model_name, duration=duration, **metadata)
        else:
            self.metrics.record_counter("generation.failure", {"model": model_name})
            self.logger.error("Generation failed", 
                            model=model_name, duration=duration, **metadata)
            
    def record_metric_computation(self, metric_name: str, value: float, duration: float):
        """Record metric computation."""
        self.metrics.record_histogram("metric.computation_time", duration, 
                                    {"metric": metric_name}, "seconds")
        self.metrics.record_gauge("metric.value", value, {"metric": metric_name})
        
        self.logger.debug("Metric computed", 
                         metric=metric_name, value=value, duration=duration)
        
    def get_monitoring_report(self, since: float = None) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        if since is None:
            since = time.time() - 3600  # Last hour
            
        report = {
            'timestamp': time.time(),
            'period_start': since,
            'metrics': self.metrics.get_metrics(since=since),
            'performance': {
                'profiles': [asdict(p) for p in self.profiler.get_profiles(since=since)],
                'summary': self.profiler.get_summary()
            },
            'health': self.health.check_system_health(),
        }
        
        # Add summary statistics
        generation_metrics = report['metrics'].get('generation.duration', [])
        if generation_metrics:
            durations = [m.value for m in generation_metrics]
            report['summary'] = {
                'total_generations': len(durations),
                'avg_generation_time': sum(durations) / len(durations),
                'min_generation_time': min(durations),
                'max_generation_time': max(durations)
            }
            
        return report
        
    def save_monitoring_report(self, report: Dict[str, Any] = None) -> Path:
        """Save monitoring report to file."""
        if report is None:
            report = self.get_monitoring_report()
            
        timestamp = datetime.fromtimestamp(report['timestamp']).strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"monitoring_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info("Monitoring report saved", file=str(report_file))
        return report_file
        
    def cleanup(self):
        """Cleanup monitoring resources."""
        self.metrics.stop_background_collection()
        self.logger.info("Monitoring cleanup completed")


# Global monitoring instance
_global_monitor: Optional[BenchmarkMonitor] = None

def get_global_monitor() -> BenchmarkMonitor:
    """Get global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = BenchmarkMonitor()
    return _global_monitor

def init_monitoring(output_dir: str = "./monitoring") -> BenchmarkMonitor:
    """Initialize global monitoring."""
    global _global_monitor
    _global_monitor = BenchmarkMonitor(output_dir)
    return _global_monitor