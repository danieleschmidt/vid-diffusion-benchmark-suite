"""Health monitoring and checks."""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from ..monitoring.logging import get_structured_logger

logger = get_structured_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


class SystemHealthChecker:
    """System-level health monitoring."""
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._last_results: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
        
        # Register default checks
        self.register_check("cpu", self._check_cpu)
        self.register_check("memory", self._check_memory)
        self.register_check("disk", self._check_disk)
        if TORCH_AVAILABLE:
            self.register_check("gpu", self._check_gpu)
        self.register_check("database", self._check_database)
        
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check function."""
        self._checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
        
    def run_check(self, name: str) -> Optional[HealthCheck]:
        """Run a specific health check."""
        if name not in self._checks:
            return None
            
        start_time = time.time()
        try:
            result = self._checks[name]()
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            
            with self._lock:
                self._last_results[name] = result
                
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms
            )
            
            with self._lock:
                self._last_results[name] = error_result
                
            logger.error(f"Health check '{name}' failed", error=str(e))
            return error_result
            
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        for name in self._checks:
            results[name] = self.run_check(name)
        return results
        
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            if not self._last_results:
                return HealthStatus.UNHEALTHY
                
            statuses = [check.status for check in self._last_results.values()]
            
            if HealthStatus.CRITICAL in statuses:
                return HealthStatus.CRITICAL
            elif HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
                
    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        with self._lock:
            overall_status = self.get_overall_status()
            
            return {
                "overall_status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {
                    name: {
                        "status": check.status.value,
                        "message": check.message,
                        "timestamp": check.timestamp.isoformat(),
                        "duration_ms": check.duration_ms
                    }
                    for name, check in self._last_results.items()
                }
            }
            
    def _check_cpu(self) -> HealthCheck:
        """Check CPU health."""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        
        # Determine status
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
            message = f"CPU usage critically high: {cpu_percent:.1f}%"
        elif cpu_percent > 80:
            status = HealthStatus.DEGRADED
            message = f"CPU usage high: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"
            
        details = {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "load_avg": load_avg
        }
        
        return HealthCheck(
            name="cpu",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.utcnow(),
            duration_ms=0  # Will be set by caller
        )
        
    def _check_memory(self) -> HealthCheck:
        """Check memory health."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_percent = memory.percent
        
        # Determine status
        if memory_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Memory usage critically high: {memory_percent:.1f}%"
        elif memory_percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Memory usage high: {memory_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory_percent:.1f}%"
            
        details = {
            "memory_percent": memory_percent,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "swap_percent": swap.percent,
            "swap_total_gb": swap.total / (1024**3)
        }
        
        return HealthCheck(
            name="memory",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.utcnow(),
            duration_ms=0
        )
        
    def _check_disk(self) -> HealthCheck:
        """Check disk health."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Determine status
        if disk_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Disk usage critically high: {disk_percent:.1f}%"
        elif disk_percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Disk usage high: {disk_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {disk_percent:.1f}%"
            
        details = {
            "disk_percent": disk_percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_used_gb": disk.used / (1024**3)
        }
        
        return HealthCheck(
            name="disk",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.utcnow(),
            duration_ms=0
        )
        
    def _check_gpu(self) -> HealthCheck:
        """Check GPU health."""
        if not TORCH_AVAILABLE:
            return HealthCheck(
                name="gpu",
                status=HealthStatus.DEGRADED,
                message="PyTorch not available",
                details={"torch_available": False},
                timestamp=datetime.utcnow(),
                duration_ms=0
            )
            
        if not torch.cuda.is_available():
            return HealthCheck(
                name="gpu",
                status=HealthStatus.DEGRADED,
                message="CUDA not available",
                details={"cuda_available": False},
                timestamp=datetime.utcnow(),
                duration_ms=0
            )
            
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = props.total_memory / (1024**3)
            
            gpu_info.append({
                "device": i,
                "name": props.name,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_total_gb": memory_total,
                "memory_percent": (memory_reserved / memory_total) * 100
            })
            
        # Check for memory issues
        max_memory_percent = max(gpu["memory_percent"] for gpu in gpu_info) if gpu_info else 0
        
        if max_memory_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"GPU memory critically high: {max_memory_percent:.1f}%"
        elif max_memory_percent > 85:
            status = HealthStatus.DEGRADED
            message = f"GPU memory high: {max_memory_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"GPU healthy, {gpu_count} device(s) available"
            
        details = {
            "gpu_count": gpu_count,
            "cuda_version": torch.version.cuda,
            "gpus": gpu_info
        }
        
        return HealthCheck(
            name="gpu",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.utcnow(),
            duration_ms=0
        )
        
    def _check_database(self) -> HealthCheck:
        """Check database health."""
        try:
            from ..database.connection import db_manager
            
            db_health = db_manager.health_check()
            
            if db_health["status"] == "healthy":
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Database unhealthy: {db_health.get('error', 'Unknown error')}"
                
            return HealthCheck(
                name="database",
                status=status,
                message=message,
                details=db_health,
                timestamp=datetime.utcnow(),
                duration_ms=0
            )
            
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow(),
                duration_ms=0
            )


class BenchmarkHealthChecker:
    """Benchmark-specific health monitoring."""
    
    def __init__(self):
        self._active_benchmarks: Dict[str, Dict[str, Any]] = {}
        self._recent_failures: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def start_monitoring(self, run_id: str, model_name: str, timeout_minutes: int = 30):
        """Start monitoring a benchmark run."""
        with self._lock:
            self._active_benchmarks[run_id] = {
                "model_name": model_name,
                "start_time": datetime.utcnow(),
                "timeout_minutes": timeout_minutes,
                "last_activity": datetime.utcnow(),
                "prompts_completed": 0,
                "prompts_failed": 0
            }
            
    def update_progress(self, run_id: str, prompts_completed: int, prompts_failed: int):
        """Update benchmark progress."""
        with self._lock:
            if run_id in self._active_benchmarks:
                benchmark = self._active_benchmarks[run_id]
                benchmark["last_activity"] = datetime.utcnow()
                benchmark["prompts_completed"] = prompts_completed
                benchmark["prompts_failed"] = prompts_failed
                
    def complete_monitoring(self, run_id: str, success: bool, error: str = None):
        """Complete benchmark monitoring."""
        with self._lock:
            if run_id in self._active_benchmarks:
                benchmark = self._active_benchmarks[run_id]
                
                if not success:
                    self._recent_failures.append({
                        "run_id": run_id,
                        "model_name": benchmark["model_name"],
                        "error": error,
                        "timestamp": datetime.utcnow()
                    })
                    
                    # Keep only last 10 failures
                    self._recent_failures = self._recent_failures[-10:]
                    
                del self._active_benchmarks[run_id]
                
    def check_timeouts(self) -> List[str]:
        """Check for timed out benchmarks."""
        timed_out = []
        current_time = datetime.utcnow()
        
        with self._lock:
            for run_id, benchmark in list(self._active_benchmarks.items()):
                timeout_delta = timedelta(minutes=benchmark["timeout_minutes"])
                if current_time - benchmark["start_time"] > timeout_delta:
                    timed_out.append(run_id)
                    
                    # Record as failure
                    self._recent_failures.append({
                        "run_id": run_id,
                        "model_name": benchmark["model_name"],
                        "error": "Benchmark timeout",
                        "timestamp": current_time
                    })
                    
                    del self._active_benchmarks[run_id]
                    
        return timed_out
        
    def get_status(self) -> Dict[str, Any]:
        """Get benchmark health status."""
        with self._lock:
            current_time = datetime.utcnow()
            
            # Check for stalled benchmarks
            stalled_count = 0
            for benchmark in self._active_benchmarks.values():
                if current_time - benchmark["last_activity"] > timedelta(minutes=5):
                    stalled_count += 1
                    
            recent_failure_count = len([
                f for f in self._recent_failures
                if current_time - f["timestamp"] < timedelta(hours=1)
            ])
            
            return {
                "active_benchmarks": len(self._active_benchmarks),
                "stalled_benchmarks": stalled_count,
                "recent_failures_1h": recent_failure_count,
                "recent_failures": self._recent_failures[-5:],  # Last 5 failures
                "benchmark_details": [
                    {
                        "run_id": run_id,
                        "model_name": bench["model_name"],
                        "duration_minutes": (current_time - bench["start_time"]).total_seconds() / 60,
                        "prompts_completed": bench["prompts_completed"],
                        "prompts_failed": bench["prompts_failed"]
                    }
                    for run_id, bench in self._active_benchmarks.items()
                ]
            }


# Global health checker instances
_system_health = SystemHealthChecker()
_benchmark_health = BenchmarkHealthChecker()


def get_system_health() -> SystemHealthChecker:
    """Get global system health checker."""
    return _system_health


def get_benchmark_health() -> BenchmarkHealthChecker:
    """Get global benchmark health checker."""
    return _benchmark_health


class HealthChecker:
    """Combined health checker."""
    
    def __init__(self):
        self.system = get_system_health()
        self.benchmark = get_benchmark_health()
        
    def get_full_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        system_summary = self.system.get_summary()
        benchmark_status = self.benchmark.get_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": system_summary["overall_status"],
            "system": system_summary,
            "benchmarks": benchmark_status
        }