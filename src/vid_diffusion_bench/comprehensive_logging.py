"""Comprehensive logging and monitoring system for video diffusion benchmarking.

This module provides advanced logging capabilities including:
- Structured logging with context
- Performance metrics collection
- Security event logging
- Audit trails
- Real-time monitoring integration
"""

import json
import time
import logging
import threading
import functools
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import uuid

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class EventType(Enum):
    """Event types for structured logging."""
    BENCHMARK_START = "benchmark_start"
    BENCHMARK_END = "benchmark_end"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    INFERENCE_START = "inference_start"
    INFERENCE_END = "inference_end"
    METRIC_COMPUTATION = "metric_computation"
    ERROR_OCCURRED = "error_occurred"
    RESOURCE_USAGE = "resource_usage"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_ALERT = "performance_alert"
    SYSTEM_HEALTH = "system_health"


class LogLevel(Enum):
    """Extended log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60


@dataclass
class StructuredLogEntry:
    """Structured log entry with rich metadata."""
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.BENCHMARK_START
    level: LogLevel = LogLevel.INFO
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    model_name: Optional[str] = None
    benchmark_id: Optional[str] = None
    duration_ms: Optional[float] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class PerformanceTimer:
    """High-precision performance timer with context tracking."""
    
    def __init__(self, name: str, logger: 'StructuredLogger' = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.context = {}
    
    def start(self, **context) -> 'PerformanceTimer':
        """Start the timer with optional context."""
        self.start_time = time.perf_counter()
        self.context.update(context)
        
        if self.logger:
            self.logger.log_event(
                event_type=EventType.INFERENCE_START,
                message=f"Started timer: {self.name}",
                context=self.context
            )
        
        return self
    
    def stop(self, **additional_context) -> float:
        """Stop the timer and return duration in milliseconds."""
        if self.start_time is None:
            raise ValueError("Timer was not started")
        
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        self.context.update(additional_context)
        
        if self.logger:
            self.logger.log_event(
                event_type=EventType.INFERENCE_END,
                message=f"Completed timer: {self.name}",
                context=self.context,
                duration_ms=duration_ms
            )
        
        return duration_ms
    
    def __enter__(self) -> 'PerformanceTimer':
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.context['exception'] = str(exc_val)
            self.context['exception_type'] = exc_type.__name__
        
        self.stop()


class ResourceTracker:
    """Track system resource usage over time."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.is_tracking = False
        self.samples = []
        self._tracking_thread = None
    
    def start_tracking(self):
        """Start tracking resources in background thread."""
        if self.is_tracking:
            return
        
        self.is_tracking = True
        self.samples = []
        self._tracking_thread = threading.Thread(target=self._track_resources, daemon=True)
        self._tracking_thread.start()
    
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop tracking and return summary statistics."""
        self.is_tracking = False
        
        if self._tracking_thread:
            self._tracking_thread.join(timeout=2.0)
        
        if not self.samples:
            return {}
        
        # Calculate statistics
        memory_usage = [s['memory_percent'] for s in self.samples]
        cpu_usage = [s['cpu_percent'] for s in self.samples]
        
        stats = {
            "sample_count": len(self.samples),
            "duration_seconds": len(self.samples) * self.sampling_interval,
            "memory_usage": {
                "avg": sum(memory_usage) / len(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            },
            "cpu_usage": {
                "avg": sum(cpu_usage) / len(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            }
        }
        
        # Add GPU stats if available
        gpu_memory = [s.get('gpu_memory_gb', 0) for s in self.samples if 'gpu_memory_gb' in s]
        if gpu_memory:
            stats["gpu_memory_usage"] = {
                "avg": sum(gpu_memory) / len(gpu_memory),
                "max": max(gpu_memory),
                "min": min(gpu_memory)
            }
        
        return stats
    
    def _track_resources(self):
        """Background thread function to track resources."""
        try:
            import psutil
        except ImportError:
            return
        
        while self.is_tracking:
            sample = {
                "timestamp": time.time(),
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
            
            # Add GPU information if available
            try:
                import torch
                if torch.cuda.is_available():
                    sample["gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
                    sample["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            except ImportError:
                pass
            
            self.samples.append(sample)
            time.sleep(self.sampling_interval)


class StructuredLogger:
    """Advanced structured logger with context management."""
    
    def __init__(
        self,
        name: str,
        session_id: str = None,
        output_file: Path = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True
    ):
        self.name = name
        self.session_id = session_id or str(uuid.uuid4())
        self.output_file = output_file
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        
        # Standard Python logger for fallback
        self.logger = logging.getLogger(name)
        
        # Context stack for hierarchical logging
        self._context_stack = []
        self._current_context = {}
        
        # Setup file logging
        if self.enable_file and self.output_file:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file logging handlers."""
        if self.output_file:
            handler = logging.FileHandler(self.output_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @contextmanager
    def context(self, **context_data):
        """Add context for nested log entries."""
        self._context_stack.append(context_data)
        self._current_context.update(context_data)
        
        try:
            yield
        finally:
            # Remove the context
            if self._context_stack:
                removed_context = self._context_stack.pop()
                # Restore previous context
                self._current_context = {}
                for ctx in self._context_stack:
                    self._current_context.update(ctx)
    
    def log_event(
        self,
        event_type: EventType,
        message: str,
        level: LogLevel = LogLevel.INFO,
        context: Dict[str, Any] = None,
        **kwargs
    ):
        """Log a structured event."""
        # Merge contexts
        full_context = {}
        full_context.update(self._current_context)
        if context:
            full_context.update(context)
        
        # Create structured log entry
        entry = StructuredLogEntry(
            event_type=event_type,
            level=level,
            message=message,
            context=full_context,
            session_id=self.session_id,
            **kwargs
        )
        
        # Log to console
        if self.enable_console:
            log_level = getattr(logging, level.name, logging.INFO)
            self.logger.log(log_level, f"[{event_type.value}] {message}")
            
            if full_context:
                self.logger.log(log_level, f"Context: {json.dumps(full_context, default=str)}")
        
        # Log to JSON file
        if self.enable_json and self.enable_file:
            self._write_json_log(entry)
    
    def _write_json_log(self, entry: StructuredLogEntry):
        """Write structured log entry to JSON file."""
        if not self.output_file:
            return
        
        json_file = self.output_file.with_suffix('.json')
        
        try:
            # Append to JSON lines file
            with open(json_file, 'a') as f:
                f.write(entry.to_json() + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write JSON log: {e}")
    
    def create_timer(self, name: str) -> PerformanceTimer:
        """Create a performance timer with this logger."""
        return PerformanceTimer(name, self)
    
    def create_resource_tracker(self, sampling_interval: float = 1.0) -> ResourceTracker:
        """Create a resource tracker."""
        return ResourceTracker(sampling_interval)
    
    def benchmark_start(self, model_name: str, prompts: List[str], **config):
        """Log benchmark start event."""
        self.log_event(
            event_type=EventType.BENCHMARK_START,
            message=f"Starting benchmark for model '{model_name}'",
            context={
                "model_name": model_name,
                "prompt_count": len(prompts),
                "benchmark_config": config
            },
            model_name=model_name
        )
    
    def benchmark_end(self, model_name: str, results: Dict[str, Any], duration_ms: float):
        """Log benchmark completion event."""
        self.log_event(
            event_type=EventType.BENCHMARK_END,
            message=f"Completed benchmark for model '{model_name}'",
            context={
                "model_name": model_name,
                "results": results
            },
            model_name=model_name,
            duration_ms=duration_ms
        )
    
    def model_load(self, model_name: str, load_time_ms: float):
        """Log model loading event."""
        self.log_event(
            event_type=EventType.MODEL_LOAD,
            message=f"Loaded model '{model_name}'",
            context={"model_name": model_name},
            model_name=model_name,
            duration_ms=load_time_ms
        )
    
    def error_occurred(self, error: Exception, context: Dict[str, Any] = None):
        """Log error event with full context."""
        self.log_event(
            event_type=EventType.ERROR_OCCURRED,
            level=LogLevel.ERROR,
            message=f"Error occurred: {error}",
            context={
                "error_type": type(error).__name__,
                "error_message": str(error),
                **(context or {})
            }
        )
    
    def security_event(self, event_description: str, severity: str = "medium", **details):
        """Log security-related event."""
        self.log_event(
            event_type=EventType.SECURITY_EVENT,
            level=LogLevel.SECURITY,
            message=f"Security event: {event_description}",
            context={
                "severity": severity,
                "security_details": details
            }
        )


class BenchmarkAuditor:
    """Audit trail system for benchmark operations."""
    
    def __init__(self, audit_file: Path):
        self.audit_file = audit_file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize audit log
        if not self.audit_file.exists():
            with open(self.audit_file, 'w') as f:
                json.dump({"audit_log_created": time.time()}, f)
                f.write('\n')
    
    def record_action(
        self,
        action: str,
        user_id: str = "system",
        resource: str = None,
        outcome: str = "success",
        **metadata
    ):
        """Record an auditable action."""
        audit_entry = {
            "timestamp": time.time(),
            "audit_id": str(uuid.uuid4()),
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "outcome": outcome,
            "metadata": metadata
        }
        
        with open(self.audit_file, 'a') as f:
            json.dump(audit_entry, f, default=str)
            f.write('\n')
    
    def get_audit_trail(
        self,
        start_time: float = None,
        end_time: float = None,
        action_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail entries matching criteria."""
        entries = []
        
        try:
            with open(self.audit_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        
                        # Apply filters
                        if start_time and entry.get('timestamp', 0) < start_time:
                            continue
                        if end_time and entry.get('timestamp', float('inf')) > end_time:
                            continue
                        if action_filter and action_filter not in entry.get('action', ''):
                            continue
                        
                        entries.append(entry)
        except FileNotFoundError:
            pass
        
        return entries


def performance_logged(logger: StructuredLogger = None):
    """Decorator to automatically log function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_default_logger()
            
            with func_logger.context(function=func.__name__, module=func.__module__):
                timer = func_logger.create_timer(f"{func.__module__}.{func.__name__}")
                
                with timer:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        func_logger.error_occurred(e)
                        raise
        
        return wrapper
    return decorator


def resource_monitored(logger: StructuredLogger = None, sampling_interval: float = 1.0):
    """Decorator to monitor resource usage during function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_default_logger()
            tracker = func_logger.create_resource_tracker(sampling_interval)
            
            tracker.start_tracking()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                resource_stats = tracker.stop_tracking()
                
                func_logger.log_event(
                    event_type=EventType.RESOURCE_USAGE,
                    message=f"Resource usage for {func.__name__}",
                    context={
                        "function": func.__name__,
                        "resource_stats": resource_stats
                    }
                )
        
        return wrapper
    return decorator


# Global logger instances
_default_logger = None
_global_auditor = None


def get_default_logger() -> StructuredLogger:
    """Get the default global logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = StructuredLogger(
            name="vid_diffusion_bench",
            output_file=Path("./logs/benchmark.log")
        )
    return _default_logger


def get_global_auditor() -> BenchmarkAuditor:
    """Get the global audit system."""
    global _global_auditor
    if _global_auditor is None:
        _global_auditor = BenchmarkAuditor(Path("./logs/audit.jsonl"))
    return _global_auditor


def setup_logging(
    log_level: str = "INFO",
    log_file: Path = None,
    enable_json: bool = True,
    enable_audit: bool = True
):
    """Setup comprehensive logging configuration."""
    # Set global log level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Initialize default logger
    global _default_logger
    _default_logger = StructuredLogger(
        name="vid_diffusion_bench",
        output_file=log_file or Path("./logs/benchmark.log"),
        enable_json=enable_json
    )
    
    # Initialize auditor
    if enable_audit:
        global _global_auditor
        _global_auditor = BenchmarkAuditor(Path("./logs/audit.jsonl"))
    
    _default_logger.log_event(
        event_type=EventType.SYSTEM_HEALTH,
        message="Logging system initialized",
        context={"log_level": log_level, "json_enabled": enable_json, "audit_enabled": enable_audit}
    )