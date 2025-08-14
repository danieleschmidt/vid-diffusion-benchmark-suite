"""Adaptive resilience framework for autonomous fault tolerance.

This module implements self-healing mechanisms, circuit breakers, and adaptive
retry strategies to ensure maximum system uptime and reliability.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Callable, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import statistics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """System health metrics container."""
    status: HealthStatus = HealthStatus.HEALTHY
    response_time: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 1.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    total_requests: int = 0


class CircuitBreaker:
    """Intelligent circuit breaker with adaptive thresholds."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        monitoring_window: float = 300.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.monitoring_window = monitoring_window
        
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.request_history = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN for {func.__name__}"
                    )
                else:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            self._record_success(start_time)
            return result
        except Exception as e:
            self._record_failure(start_time, e)
            raise
            
    def _record_success(self, start_time: float):
        """Record successful execution."""
        with self._lock:
            duration = time.time() - start_time
            self.request_history.append({
                'timestamp': start_time,
                'duration': duration,
                'success': True
            })
            
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker recovered to CLOSED state")
            elif self.state == "CLOSED":
                self.failure_count = max(0, self.failure_count - 1)
                
    def _record_failure(self, start_time: float, exception: Exception):
        """Record failed execution."""
        with self._lock:
            duration = time.time() - start_time
            self.request_history.append({
                'timestamp': start_time,
                'duration': duration,
                'success': False,
                'error': str(exception)
            })
            
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    f"Circuit breaker OPENED after {self.failure_count} failures"
                )
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            recent_requests = [
                r for r in self.request_history 
                if time.time() - r['timestamp'] < self.monitoring_window
            ]
            
            if not recent_requests:
                return {
                    'state': self.state,
                    'failure_count': self.failure_count,
                    'success_rate': 1.0,
                    'avg_response_time': 0.0,
                    'total_requests': 0
                }
                
            successes = [r for r in recent_requests if r['success']]
            success_rate = len(successes) / len(recent_requests)
            avg_response_time = statistics.mean([r['duration'] for r in recent_requests])
            
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'total_requests': len(recent_requests),
                'recent_errors': [r['error'] for r in recent_requests if not r['success']][-5:]
            }


class AdaptiveRetryStrategy:
    """Intelligent retry strategy with adaptive backoff."""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_factor: float = 2.0,
        jitter: bool = True,
        max_attempts: int = 5
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_factor = exponential_factor
        self.jitter = jitter
        self.max_attempts = max_attempts
        self.attempt_history = defaultdict(list)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add adaptive retry logic."""
        async def async_wrapper(*args, **kwargs):
            return await self._retry_async(func, *args, **kwargs)
            
        def sync_wrapper(*args, **kwargs):
            return self._retry_sync(func, *args, **kwargs)
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    def _retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Synchronous retry logic."""
        func_name = func.__name__
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                self._record_success(func_name, attempt)
                return result
            except Exception as e:
                last_exception = e
                self._record_failure(func_name, attempt, e)
                
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt, func_name)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_attempts} for {func_name} "
                        f"after {delay:.1f}s delay. Error: {e}"
                    )
                    time.sleep(delay)
                    
        raise last_exception
        
    async def _retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Asynchronous retry logic."""
        func_name = func.__name__
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs)
                self._record_success(func_name, attempt)
                return result
            except Exception as e:
                last_exception = e
                self._record_failure(func_name, attempt, e)
                
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt, func_name)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_attempts} for {func_name} "
                        f"after {delay:.1f}s delay. Error: {e}"
                    )
                    await asyncio.sleep(delay)
                    
        raise last_exception
        
    def _calculate_delay(self, attempt: int, func_name: str) -> float:
        """Calculate adaptive delay based on historical performance."""
        base_delay = self.base_delay * (self.exponential_factor ** attempt)
        
        # Adaptive component based on historical success rates
        history = self.attempt_history[func_name]
        if history:
            recent_failures = [h for h in history[-20:] if not h['success']]
            failure_rate = len(recent_failures) / len(history[-20:])
            
            # Increase delay for functions with high failure rates
            adaptive_multiplier = 1.0 + (failure_rate * 2.0)
            base_delay *= adaptive_multiplier
            
        delay = min(base_delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            import secrets
            delay *= (0.5 + secrets.SystemRandom().random() * 0.5)
            
        return delay
        
    def _record_success(self, func_name: str, attempt: int):
        """Record successful attempt."""
        self.attempt_history[func_name].append({
            'timestamp': time.time(),
            'attempt': attempt,
            'success': True
        })
        
    def _record_failure(self, func_name: str, attempt: int, exception: Exception):
        """Record failed attempt."""
        self.attempt_history[func_name].append({
            'timestamp': time.time(),
            'attempt': attempt,
            'success': False,
            'error': str(exception)
        })


class HealthMonitor:
    """Continuous health monitoring with predictive alerts."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.alerts = []
        
    def register_health_check(self, name: str, check_func: Callable[[], HealthMetrics]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
        
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            return
            
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
        
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
                
    async def _run_health_checks(self):
        """Execute all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                metrics = check_func()
                self.metrics_history[name].append({
                    'timestamp': time.time(),
                    'metrics': metrics
                })
                
                # Check for degradation trends
                await self._analyze_trends(name, metrics)
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                
    async def _analyze_trends(self, service_name: str, current_metrics: HealthMetrics):
        """Analyze health trends and generate predictive alerts."""
        history = self.metrics_history[service_name]
        if len(history) < 5:  # Need minimum history
            return
            
        # Analyze response time trend
        recent_times = [h['metrics'].response_time for h in list(history)[-10:]]
        if len(recent_times) >= 3:
            trend = statistics.linear_regression(range(len(recent_times)), recent_times)
            if trend.slope > 0.1:  # Response time increasing
                self.alerts.append({
                    'timestamp': time.time(),
                    'service': service_name,
                    'type': 'performance_degradation',
                    'message': f'Response time increasing trend detected: {trend.slope:.3f}s/check',
                    'severity': 'warning'
                })
                
        # Analyze error rate trend
        recent_errors = [h['metrics'].error_rate for h in list(history)[-10:]]
        current_error_rate = current_metrics.error_rate
        if current_error_rate > 0.1 and current_error_rate > statistics.mean(recent_errors) * 2:
            self.alerts.append({
                'timestamp': time.time(),
                'service': service_name,
                'type': 'error_spike',
                'message': f'Error rate spike detected: {current_error_rate:.2%}',
                'severity': 'error'
            })
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.metrics_history:
            return {'status': 'unknown', 'services': {}}
            
        service_health = {}
        overall_status = HealthStatus.HEALTHY
        
        for service_name, history in self.metrics_history.items():
            if not history:
                continue
                
            latest = history[-1]['metrics']
            service_health[service_name] = {
                'status': latest.status.value,
                'response_time': latest.response_time,
                'error_rate': latest.error_rate,
                'success_rate': latest.success_rate,
                'last_check': latest.last_check
            }
            
            # Determine worst status for overall health
            if latest.status.value == 'critical':
                overall_status = HealthStatus.CRITICAL
            elif latest.status.value == 'unhealthy' and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif latest.status.value == 'degraded' and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
                
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 300]
        
        return {
            'status': overall_status.value,
            'services': service_health,
            'recent_alerts': recent_alerts,
            'alert_count': len(recent_alerts),
            'timestamp': time.time()
        }


class SelfHealingManager:
    """Autonomous self-healing capabilities."""
    
    def __init__(self):
        self.healing_strategies: Dict[str, Callable] = {}
        self.healing_history: List[Dict] = []
        self.auto_healing_enabled = True
        
    def register_healing_strategy(self, error_pattern: str, strategy: Callable):
        """Register a self-healing strategy for specific error patterns."""
        self.healing_strategies[error_pattern] = strategy
        logger.info(f"Registered healing strategy for: {error_pattern}")
        
    async def attempt_healing(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to heal from an error automatically."""
        if not self.auto_healing_enabled:
            return False
            
        error_type = type(error).__name__
        error_message = str(error)
        
        # Find matching healing strategy
        for pattern, strategy in self.healing_strategies.items():
            if pattern in error_type or pattern in error_message:
                try:
                    logger.info(f"Attempting self-healing for {error_type} using {strategy.__name__}")
                    success = await self._execute_healing_strategy(strategy, error, context)
                    
                    self.healing_history.append({
                        'timestamp': time.time(),
                        'error_type': error_type,
                        'error_message': error_message,
                        'strategy': strategy.__name__,
                        'success': success,
                        'context': context
                    })
                    
                    if success:
                        logger.info(f"Self-healing successful for {error_type}")
                        return True
                    else:
                        logger.warning(f"Self-healing failed for {error_type}")
                        
                except Exception as healing_error:
                    logger.error(f"Error during self-healing: {healing_error}")
                    
        return False
        
    async def _execute_healing_strategy(
        self,
        strategy: Callable,
        error: Exception,
        context: Dict[str, Any]
    ) -> bool:
        """Execute a healing strategy safely."""
        if asyncio.iscoroutinefunction(strategy):
            return await strategy(error, context)
        else:
            return strategy(error, context)
            
    def get_healing_stats(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        if not self.healing_history:
            return {'total_attempts': 0, 'success_rate': 0.0}
            
        total = len(self.healing_history)
        successes = sum(1 for h in self.healing_history if h['success'])
        success_rate = successes / total
        
        recent_history = [
            h for h in self.healing_history 
            if time.time() - h['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'total_attempts': total,
            'success_rate': success_rate,
            'recent_attempts': len(recent_history),
            'recent_success_rate': sum(1 for h in recent_history if h['success']) / max(1, len(recent_history)),
            'common_errors': self._get_common_errors(),
            'most_effective_strategies': self._get_effective_strategies()
        }
        
    def _get_common_errors(self) -> List[Dict[str, Any]]:
        """Get most common error types."""
        error_counts = defaultdict(int)
        for h in self.healing_history:
            error_counts[h['error_type']] += 1
            
        return [
            {'error_type': error_type, 'count': count}
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
    def _get_effective_strategies(self) -> List[Dict[str, Any]]:
        """Get most effective healing strategies."""
        strategy_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        
        for h in self.healing_history:
            strategy = h['strategy']
            strategy_stats[strategy]['attempts'] += 1
            if h['success']:
                strategy_stats[strategy]['successes'] += 1
                
        effectiveness = []
        for strategy, stats in strategy_stats.items():
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts']
                effectiveness.append({
                    'strategy': strategy,
                    'success_rate': success_rate,
                    'attempts': stats['attempts']
                })
                
        return sorted(effectiveness, key=lambda x: x['success_rate'], reverse=True)[:5]


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global resilience manager instance
resilience_manager = SelfHealingManager()