"""Fault tolerance mechanisms for robust benchmarking.

This module provides comprehensive fault tolerance including circuit breakers,
retry mechanisms, fallback strategies, and health checking to ensure robust
operation even in the presence of failures.
"""

import time
import threading
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import functools
import random

from .error_handling import BenchmarkException, RetryableError, ErrorSeverity


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, calls rejected
    HALF_OPEN = "half_open"  # Testing recovery


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 3
    timeout: float = 30.0
    expected_failure_rate: float = 0.1  # 10%
    minimum_requests: int = 10


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[type, ...] = field(
        default_factory=lambda: (RetryableError, ConnectionError, TimeoutError)
    )


@dataclass
class HealthCheckResult:
    """Result of health check."""
    status: HealthStatus
    timestamp: datetime
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Name of the circuit
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.recent_requests = deque(maxlen=100)  # Keep last 100 requests
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"CircuitBreaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            BenchmarkException: When circuit is open
        """
        with self._lock:
            self.total_requests += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self._record_request(success=False, blocked=True)
                    raise BenchmarkException(
                        f"Circuit breaker '{self.name}' is OPEN",
                        severity=ErrorSeverity.HIGH
                    )
            
            # Record attempt time
            start_time = time.time()
        
        try:
            # Execute function with timeout
            result = self._execute_with_timeout(func, args, kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
    
    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with timeout."""
        if self.config.timeout <= 0:
            return func(*args, **kwargs)
        
        # Simple timeout implementation
        start_time = time.time()
        result = func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        if execution_time > self.config.timeout:
            raise TimeoutError(f"Function execution exceeded {self.config.timeout}s")
        
        return result
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.next_attempt_time is None:
            return True
        
        return datetime.now() >= self.next_attempt_time
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    def _record_success(self, execution_time: float):
        """Record successful execution."""
        with self._lock:
            self.total_successes += 1
            self.failure_count = 0
            
            self._record_request(success=True, execution_time=execution_time)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def _record_failure(self, error: Exception, execution_time: float):
        """Record failed execution."""
        with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            self._record_request(
                success=False, 
                execution_time=execution_time,
                error=str(error)
            )
            
            # Check if circuit should open
            if self._should_trip_circuit():
                self._transition_to_open()
    
    def _record_request(
        self,
        success: bool,
        execution_time: float = 0.0,
        error: Optional[str] = None,
        blocked: bool = False
    ):
        """Record request statistics."""
        request_record = {
            'timestamp': datetime.now(),
            'success': success,
            'execution_time': execution_time,
            'error': error,
            'blocked': blocked
        }
        
        self.recent_requests.append(request_record)
    
    def _should_trip_circuit(self) -> bool:
        """Determine if circuit should trip to open state."""
        # Check failure threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate if we have enough requests
        if self.total_requests >= self.config.minimum_requests:
            recent_failure_rate = self._calculate_recent_failure_rate()
            if recent_failure_rate > self.config.expected_failure_rate:
                return True
        
        return False
    
    def _calculate_recent_failure_rate(self) -> float:
        """Calculate recent failure rate."""
        if not self.recent_requests:
            return 0.0
        
        recent_failures = sum(1 for req in self.recent_requests 
                             if not req['success'] and not req['blocked'])
        recent_total = len([req for req in self.recent_requests 
                           if not req['blocked']])
        
        return recent_failures / recent_total if recent_total > 0 else 0.0
    
    def _transition_to_open(self):
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self.next_attempt_time = datetime.now() + timedelta(
            seconds=self.config.recovery_timeout
        )
        logger.warning(f"Circuit breaker '{self.name}' tripped to OPEN state")
    
    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED state")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'total_requests': self.total_requests,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'success_rate': (self.total_successes / self.total_requests 
                               if self.total_requests > 0 else 0.0),
                'recent_failure_rate': self._calculate_recent_failure_rate(),
                'last_failure_time': (self.last_failure_time.isoformat() 
                                     if self.last_failure_time else None),
                'next_attempt_time': (self.next_attempt_time.isoformat() 
                                    if self.next_attempt_time else None)
            }
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.next_attempt_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class RetryManager:
    """Advanced retry mechanism with backoff and jitter."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry manager.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        logger.info("RetryManager initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry(func, *args, **kwargs)
        return wrapper
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.info(f"Exception {type(e).__name__} is not retryable")
                    raise
                
                # If this is the last attempt, don't wait
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.info(
                    f"Attempt {attempt + 1} failed with {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)
        
        # All attempts failed
        logger.error(
            f"All {self.config.max_attempts} attempts failed. "
            f"Last error: {last_exception}"
        )
        raise last_exception
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return isinstance(exception, self.config.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay += jitter
        
        return max(0, delay)


class FallbackHandler:
    """Fallback handler for graceful degradation."""
    
    def __init__(self):
        """Initialize fallback handler."""
        self.fallback_strategies = {}
        self.fallback_stats = {}
        logger.info("FallbackHandler initialized")
    
    def register_fallback(
        self,
        primary_name: str,
        fallback_func: Callable,
        condition: Optional[Callable[[Exception], bool]] = None
    ):
        """Register fallback strategy.
        
        Args:
            primary_name: Name of primary operation
            fallback_func: Fallback function to execute
            condition: Optional condition to determine when to use fallback
        """
        self.fallback_strategies[primary_name] = {
            'func': fallback_func,
            'condition': condition or (lambda e: True)
        }
        
        self.fallback_stats[primary_name] = {
            'fallback_count': 0,
            'success_count': 0,
            'failure_count': 0
        }
        
        logger.info(f"Registered fallback for '{primary_name}'")
    
    def execute_with_fallback(
        self,
        primary_name: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with fallback support.
        
        Args:
            primary_name: Name of primary operation
            primary_func: Primary function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from primary or fallback function
        """
        try:
            # Try primary function
            result = primary_func(*args, **kwargs)
            
            # Update stats
            if primary_name in self.fallback_stats:
                self.fallback_stats[primary_name]['success_count'] += 1
            
            return result
            
        except Exception as e:
            logger.warning(f"Primary function '{primary_name}' failed: {e}")
            
            # Check if fallback is available and condition is met
            if primary_name in self.fallback_strategies:
                strategy = self.fallback_strategies[primary_name]
                
                if strategy['condition'](e):
                    try:
                        logger.info(f"Executing fallback for '{primary_name}'")
                        result = strategy['func'](*args, **kwargs)
                        
                        # Update stats
                        self.fallback_stats[primary_name]['fallback_count'] += 1
                        
                        return result
                        
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback for '{primary_name}' also failed: {fallback_error}"
                        )
                        self.fallback_stats[primary_name]['failure_count'] += 1
                        raise fallback_error
            
            # No fallback available or condition not met
            if primary_name in self.fallback_stats:
                self.fallback_stats[primary_name]['failure_count'] += 1
            
            raise
    
    def get_fallback_stats(self) -> Dict[str, Dict[str, int]]:
        """Get fallback usage statistics."""
        return self.fallback_stats.copy()


class HealthChecker:
    """Health checker for monitoring system components."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health checker.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_history = {}
        self.running = False
        self._check_thread = None
        logger.info("HealthChecker initialized")
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
        critical: bool = False
    ):
        """Register health check.
        
        Args:
            name: Name of health check
            check_func: Function that performs health check
            critical: Whether this check is critical for overall health
        """
        self.health_checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_result': None
        }
        
        self.health_history[name] = deque(maxlen=100)
        logger.info(f"Registered health check '{name}' (critical={critical})")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            return
        
        self.running = True
        self._check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._check_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.running = False
        if self._check_thread:
            self._check_thread.join()
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.run_all_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        
        for name, check_info in self.health_checks.items():
            try:
                result = check_info['func']()
                check_info['last_result'] = result
                self.health_history[name].append(result)
                results[name] = result
                
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                
                # Create error result
                error_result = HealthCheckResult(
                    status=HealthStatus.CRITICAL,
                    timestamp=datetime.now(),
                    response_time=-1,
                    error_message=str(e)
                )
                
                check_info['last_result'] = error_result
                self.health_history[name].append(error_result)
                results[name] = error_result
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status.
        
        Returns:
            Overall health status
        """
        if not self.health_checks:
            return HealthStatus.HEALTHY
        
        statuses = []
        critical_failed = False
        
        for name, check_info in self.health_checks.items():
            last_result = check_info['last_result']
            
            if last_result is None:
                continue
            
            if check_info['critical'] and last_result.status == HealthStatus.CRITICAL:
                critical_failed = True
            
            statuses.append(last_result.status)
        
        if critical_failed:
            return HealthStatus.CRITICAL
        
        if not statuses:
            return HealthStatus.HEALTHY
        
        # Count status occurrences
        status_counts = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.UNHEALTHY, 0) > len(statuses) * 0.5:
            return HealthStatus.UNHEALTHY
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        overall_status = self.get_overall_health()
        
        checks = {}
        for name, check_info in self.health_checks.items():
            last_result = check_info['last_result']
            
            if last_result:
                checks[name] = {
                    'status': last_result.status.value,
                    'timestamp': last_result.timestamp.isoformat(),
                    'response_time': last_result.response_time,
                    'critical': check_info['critical'],
                    'details': last_result.details,
                    'error_message': last_result.error_message
                }
            else:
                checks[name] = {
                    'status': 'unknown',
                    'critical': check_info['critical'],
                    'error_message': 'No check results available'
                }
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': checks,
            'summary': {
                'total_checks': len(self.health_checks),
                'critical_checks': sum(1 for info in self.health_checks.values() 
                                      if info['critical']),
                'failing_checks': sum(1 for info in self.health_checks.values() 
                                     if info['last_result'] and 
                                     info['last_result'].status in [
                                         HealthStatus.UNHEALTHY, HealthStatus.CRITICAL
                                     ])
            }
        }


# Utility functions for creating health checks
def create_gpu_health_check() -> Callable[[], HealthCheckResult]:
    """Create GPU health check function."""
    def gpu_health_check() -> HealthCheckResult:
        start_time = time.time()
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(),
                    response_time=time.time() - start_time,
                    details={'gpu_available': False},
                    error_message='CUDA not available'
                )
            
            # Test GPU memory
            gpu_count = torch.cuda.device_count()
            gpu_info = {}
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_cached = torch.cuda.memory_reserved(i)
                
                gpu_info[f'gpu_{i}'] = {
                    'name': props.name,
                    'memory_total': props.total_memory,
                    'memory_allocated': memory_allocated,
                    'memory_cached': memory_cached,
                    'memory_free': props.total_memory - memory_cached
                }
            
            # Determine status based on memory usage
            max_memory_usage = max(
                info['memory_cached'] / info['memory_total'] 
                for info in gpu_info.values()
            )
            
            if max_memory_usage > 0.95:
                status = HealthStatus.CRITICAL
            elif max_memory_usage > 0.85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                status=status,
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                details={
                    'gpu_count': gpu_count,
                    'max_memory_usage': max_memory_usage,
                    'gpu_info': gpu_info
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    return gpu_health_check


def create_memory_health_check() -> Callable[[], HealthCheckResult]:
    """Create system memory health check function."""
    def memory_health_check() -> HealthCheckResult:
        start_time = time.time()
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            # Determine status based on memory usage
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                status=status,
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                details={
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'memory_total': memory.total,
                    'memory_used': memory.used
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    return memory_health_check


# Composite fault tolerance manager
class FaultToleranceManager:
    """Composite fault tolerance manager."""
    
    def __init__(self):
        """Initialize fault tolerance manager."""
        self.circuit_breakers = {}
        self.retry_manager = RetryManager()
        self.fallback_handler = FallbackHandler()
        self.health_checker = HealthChecker()
        
        # Setup default health checks
        self.health_checker.register_check(
            'gpu_health', 
            create_gpu_health_check(), 
            critical=True
        )
        self.health_checker.register_check(
            'memory_health', 
            create_memory_health_check(), 
            critical=False
        )
        
        logger.info("FaultToleranceManager initialized")
    
    def get_circuit_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Optional configuration
            
        Returns:
            Circuit breaker instance
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        
        return self.circuit_breakers[name]
    
    def start_monitoring(self):
        """Start health monitoring."""
        self.health_checker.start_monitoring()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.health_checker.stop_monitoring()
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status.
        
        Returns:
            Status dictionary
        """
        # Circuit breaker stats
        circuit_stats = {}
        for name, breaker in self.circuit_breakers.items():
            circuit_stats[name] = breaker.get_stats()
        
        # Fallback stats
        fallback_stats = self.fallback_handler.get_fallback_stats()
        
        # Health report
        health_report = self.health_checker.get_health_report()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'circuit_breakers': circuit_stats,
            'fallback_usage': fallback_stats,
            'health_status': health_report,
            'overall_status': health_report['overall_status']
        }


# Global fault tolerance manager instance
_global_fault_tolerance_manager = None


def get_fault_tolerance_manager() -> FaultToleranceManager:
    """Get global fault tolerance manager instance."""
    global _global_fault_tolerance_manager
    if _global_fault_tolerance_manager is None:
        _global_fault_tolerance_manager = FaultToleranceManager()
    return _global_fault_tolerance_manager