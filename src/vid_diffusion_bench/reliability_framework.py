"""Reliability framework for robust video diffusion benchmarking.

This module implements comprehensive reliability features including:
- Circuit breakers for external services
- Health checks for system components
- Graceful degradation mechanisms
- Recovery procedures
- Performance monitoring and alerting
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    status: HealthStatus
    response_time_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthChecker(ABC):
    """Abstract base class for health checkers."""
    
    def __init__(self, service_name: str, timeout_seconds: float = 5.0):
        self.service_name = service_name
        self.timeout_seconds = timeout_seconds
    
    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """Perform health check and return result."""
        pass


class GPUHealthChecker(HealthChecker):
    """Health checker for GPU resources."""
    
    def check_health(self) -> HealthCheckResult:
        """Check GPU health and availability."""
        start_time = time.time()
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message="CUDA not available",
                    details={"cuda_available": False}
                )
            
            device_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_total = props.total_memory / (1024**3)
                
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "memory_total_gb": memory_total,
                    "memory_allocated_gb": memory_allocated,
                    "memory_reserved_gb": memory_reserved,
                    "memory_free_gb": memory_total - memory_reserved,
                    "compute_capability": f"{props.major}.{props.minor}"
                })
            
            # Determine status based on available memory
            min_free_memory = min(gpu["memory_free_gb"] for gpu in gpu_info)
            
            if min_free_memory > 2.0:  # At least 2GB free
                status = HealthStatus.HEALTHY
                message = f"{device_count} GPU(s) available with sufficient memory"
            elif min_free_memory > 0.5:  # At least 500MB free
                status = HealthStatus.DEGRADED
                message = f"{device_count} GPU(s) available but memory is low"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"{device_count} GPU(s) available but insufficient memory"
            
            return HealthCheckResult(
                service_name=self.service_name,
                status=status,
                response_time_ms=(time.time() - start_time) * 1000,
                message=message,
                details={
                    "device_count": device_count,
                    "gpus": gpu_info,
                    "min_free_memory_gb": min_free_memory
                }
            )
            
        except ImportError:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message="PyTorch not available",
                details={"torch_available": False}
            )
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"GPU health check failed: {e}",
                details={"error": str(e)}
            )


class ModelHealthChecker(HealthChecker):
    """Health checker for model loading and inference."""
    
    def __init__(self, service_name: str, model_name: str, timeout_seconds: float = 30.0):
        super().__init__(service_name, timeout_seconds)
        self.model_name = model_name
    
    def check_health(self) -> HealthCheckResult:
        """Check if model can be loaded and perform basic inference."""
        start_time = time.time()
        
        try:
            # This would be implemented to actually test model loading
            # For now, we simulate the check
            
            # Simulate model loading time
            time.sleep(0.1)
            
            response_time = (time.time() - start_time) * 1000
            
            if response_time > 5000:  # More than 5 seconds
                status = HealthStatus.DEGRADED
                message = f"Model '{self.model_name}' loading is slow"
            else:
                status = HealthStatus.HEALTHY
                message = f"Model '{self.model_name}' is responsive"
            
            return HealthCheckResult(
                service_name=self.service_name,
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    "model_name": self.model_name,
                    "load_time_ms": response_time
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Model '{self.model_name}' health check failed: {e}",
                details={"model_name": self.model_name, "error": str(e)}
            )


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_duration: float = 60.0,
        call_timeout: float = 30.0
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_duration = timeout_duration
        self.call_timeout = call_timeout
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        
        self._lock = threading.Lock()
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.state != CircuitBreakerState.OPEN:
            return False
        
        if self.next_attempt_time is None:
            return True
        
        return time.time() >= self.next_attempt_time
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")
    
    def _record_failure(self):
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = time.time() + self.timeout_duration
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
                )
    
    @contextmanager
    def call(self):
        """Context manager for making calls through the circuit breaker."""
        # Check if we should block the call
        if self.state == CircuitBreakerState.OPEN:
            if not self._should_attempt_reset():
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
            else:
                with self._lock:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
        
        start_time = time.time()
        try:
            yield
            
            # Record success
            call_duration = time.time() - start_time
            if call_duration > self.call_timeout:
                raise TimeoutError(f"Call exceeded timeout of {self.call_timeout}s")
            
            self._record_success()
            
        except Exception as e:
            self._record_failure()
            raise


class ReliabilityManager:
    """Central manager for reliability features."""
    
    def __init__(self):
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._health_check_results: Dict[str, HealthCheckResult] = {}
        self._health_check_lock = threading.Lock()
    
    def register_health_checker(self, checker: HealthChecker):
        """Register a health checker."""
        self.health_checkers[checker.service_name] = checker
        logger.info(f"Registered health checker for '{checker.service_name}'")
    
    def register_circuit_breaker(self, breaker: CircuitBreaker):
        """Register a circuit breaker."""
        self.circuit_breakers[breaker.name] = breaker
        logger.info(f"Registered circuit breaker for '{breaker.name}'")
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, checker in self.health_checkers.items():
            try:
                result = checker.check_health()
                results[name] = result
                
                with self._health_check_lock:
                    self._health_check_results[name] = result
                
                logger.debug(f"Health check '{name}': {result.status.value}")
                
            except Exception as e:
                error_result = HealthCheckResult(
                    service_name=name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    message=f"Health check failed: {e}",
                    details={"error": str(e)}
                )
                results[name] = error_result
                
                with self._health_check_lock:
                    self._health_check_results[name] = error_result
                
                logger.error(f"Health check '{name}' failed: {e}")
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._health_check_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self._health_check_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        overall_status = self.get_overall_health()
        
        service_statuses = {}
        for name, result in self._health_check_results.items():
            service_statuses[name] = {
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "message": result.message,
                "last_check": result.timestamp
            }
        
        circuit_breaker_statuses = {}
        for name, breaker in self.circuit_breakers.items():
            circuit_breaker_statuses[name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count,
                "last_failure_time": breaker.last_failure_time
            }
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "services": service_statuses,
            "circuit_breakers": circuit_breaker_statuses
        }
    
    def export_health_report(self, output_path: Path):
        """Export detailed health report to file."""
        health_summary = self.get_health_summary()
        
        # Add detailed information
        health_summary["detailed_results"] = {}
        for name, result in self._health_check_results.items():
            health_summary["detailed_results"][name] = {
                "service_name": result.service_name,
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "message": result.message,
                "details": result.details,
                "timestamp": result.timestamp
            }
        
        with open(output_path, 'w') as f:
            json.dump(health_summary, f, indent=2, default=str)
        
        logger.info(f"Health report exported to {output_path}")


class GracefulDegradation:
    """Implement graceful degradation strategies."""
    
    @staticmethod
    def fallback_to_cpu():
        """Fallback strategy when GPU is unavailable."""
        logger.warning("Falling back to CPU computation due to GPU unavailability")
        # Implementation would set device to CPU
        return "cpu"
    
    @staticmethod
    def reduce_batch_size(original_batch_size: int, reduction_factor: float = 0.5) -> int:
        """Reduce batch size when memory is limited."""
        new_batch_size = max(1, int(original_batch_size * reduction_factor))
        logger.warning(f"Reducing batch size from {original_batch_size} to {new_batch_size}")
        return new_batch_size
    
    @staticmethod
    def skip_expensive_metrics(metric_list: List[str]) -> List[str]:
        """Skip computationally expensive metrics when resources are limited."""
        expensive_metrics = ["fvd", "inception_score"]  # These are typically expensive
        filtered_metrics = [m for m in metric_list if m not in expensive_metrics]
        
        if len(filtered_metrics) < len(metric_list):
            skipped = set(metric_list) - set(filtered_metrics)
            logger.warning(f"Skipping expensive metrics: {skipped}")
        
        return filtered_metrics
    
    @staticmethod
    def use_lower_resolution(original_resolution: tuple, scale_factor: float = 0.5) -> tuple:
        """Use lower resolution when resources are limited."""
        width, height = original_resolution
        new_width = max(64, int(width * scale_factor))
        new_height = max(64, int(height * scale_factor))
        
        logger.warning(f"Reducing resolution from {original_resolution} to ({new_width}, {new_height})")
        return (new_width, new_height)


# Global reliability manager instance
_global_reliability_manager = None


def get_reliability_manager() -> ReliabilityManager:
    """Get the global reliability manager instance."""
    global _global_reliability_manager
    if _global_reliability_manager is None:
        _global_reliability_manager = ReliabilityManager()
        
        # Register default health checkers
        _global_reliability_manager.register_health_checker(
            GPUHealthChecker("gpu_resources")
        )
        
        # Register default circuit breakers
        _global_reliability_manager.register_circuit_breaker(
            CircuitBreaker("model_inference", failure_threshold=3, timeout_duration=30.0)
        )
        _global_reliability_manager.register_circuit_breaker(
            CircuitBreaker("metric_computation", failure_threshold=5, timeout_duration=60.0)
        )
    
    return _global_reliability_manager


def setup_reliability_monitoring(check_interval: float = 60.0):
    """Setup automated reliability monitoring."""
    import threading
    
    def monitoring_loop():
        manager = get_reliability_manager()
        
        while True:
            try:
                health_results = manager.run_health_checks()
                overall_health = manager.get_overall_health()
                
                if overall_health == HealthStatus.UNHEALTHY:
                    logger.critical("System health is UNHEALTHY")
                elif overall_health == HealthStatus.DEGRADED:
                    logger.warning("System health is DEGRADED")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitor_thread.start()
    
    logger.info(f"Reliability monitoring started with {check_interval}s interval")