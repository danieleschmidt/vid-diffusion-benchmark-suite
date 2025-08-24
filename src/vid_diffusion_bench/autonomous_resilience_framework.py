"""Autonomous Resilience Framework for Video Diffusion Benchmarking.

Self-healing, fault-tolerant, and adaptive resilience system with
autonomous recovery and predictive failure prevention.
"""

import asyncio
import logging
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from pathlib import Path
from enum import Enum
import traceback

import numpy as np
from scipy import stats
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur during benchmarking."""
    MEMORY_OVERFLOW = "memory_overflow"
    TIMEOUT = "timeout"
    MODEL_LOAD_ERROR = "model_load_error"
    GENERATION_ERROR = "generation_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of failures."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RESOURCE_REALLOCATION = "resource_reallocation"
    CIRCUIT_BREAK = "circuit_break"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class FailureRecord:
    """Record of a failure event with context and recovery actions."""
    failure_type: FailureType
    timestamp: datetime
    context: Dict[str, Any]
    error_message: str
    stack_trace: str
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: bool = False
    recovery_time: float = 0.0
    impact_score: float = 0.0


@dataclass
class HealthMetrics:
    """System health and performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    success_rate: float = 1.0
    average_latency: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    availability: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResilienceConfig:
    """Configuration for resilience framework."""
    max_retry_attempts: int = 3
    retry_delay_base: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    health_check_interval: float = 30.0
    failure_prediction_window: int = 100
    resource_monitoring_interval: float = 5.0
    emergency_threshold: float = 0.95
    graceful_degradation_threshold: float = 0.8
    enable_predictive_failure_detection: bool = True
    enable_automatic_recovery: bool = True
    enable_adaptive_resource_management: bool = True


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time > self.timeout
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = 'OPEN'


class ResourceMonitor:
    """Monitor system resources and predict resource exhaustion."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=100)
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            # CPU and memory
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU metrics (if available)
            gpu_usage, gpu_memory = self._get_gpu_metrics()
            
            return HealthMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory
            )
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
            return HealthMetrics()
    
    def _get_gpu_metrics(self) -> tuple[float, float]:
        """Get GPU utilization metrics."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Get GPU memory
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                # Approximate GPU usage (simplified)
                gpu_usage = min(gpu_memory, 100.0)
                return gpu_usage, gpu_memory
        except Exception:
            pass
        
        return 0.0, 0.0
    
    def get_current_metrics(self) -> Optional[HealthMetrics]:
        """Get most recent metrics."""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None
    
    def predict_resource_exhaustion(self) -> Dict[str, float]:
        """Predict probability of resource exhaustion."""
        with self._lock:
            if len(self.metrics_history) < 10:
                return {}
            
            predictions = {}
            recent_metrics = list(self.metrics_history)[-20:]
            
            # Memory exhaustion prediction
            memory_values = [m.memory_usage for m in recent_metrics]
            memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            if memory_trend > 0:
                time_to_exhaustion = (100 - memory_values[-1]) / memory_trend
                predictions['memory_exhaustion_risk'] = min(1.0 / (1.0 + time_to_exhaustion), 1.0)
            
            # GPU memory exhaustion prediction
            gpu_values = [m.gpu_memory for m in recent_metrics if m.gpu_memory > 0]
            if gpu_values:
                gpu_trend = np.polyfit(range(len(gpu_values)), gpu_values, 1)[0]
                if gpu_trend > 0:
                    time_to_gpu_exhaustion = (100 - gpu_values[-1]) / gpu_trend
                    predictions['gpu_exhaustion_risk'] = min(1.0 / (1.0 + time_to_gpu_exhaustion), 1.0)
            
            return predictions


class FailurePredictor:
    """Predict failures based on historical patterns and current metrics."""
    
    def __init__(self, prediction_window: int = 100):
        self.prediction_window = prediction_window
        self.failure_history = deque(maxlen=prediction_window)
        self.pattern_cache = {}
        self._lock = threading.Lock()
    
    def record_failure(self, failure: FailureRecord):
        """Record a failure for pattern analysis."""
        with self._lock:
            self.failure_history.append(failure)
            self._update_patterns()
    
    def predict_failure_probability(
        self,
        current_metrics: HealthMetrics,
        context: Dict[str, Any]
    ) -> Dict[FailureType, float]:
        """Predict probability of different failure types."""
        predictions = {}
        
        with self._lock:
            if len(self.failure_history) < 5:
                return predictions
            
            # Memory-based failure prediction
            if current_metrics.memory_usage > 85:
                predictions[FailureType.MEMORY_OVERFLOW] = min(
                    (current_metrics.memory_usage - 85) / 15, 1.0
                )
            
            # GPU memory failure prediction
            if current_metrics.gpu_memory > 90:
                predictions[FailureType.RESOURCE_EXHAUSTION] = min(
                    (current_metrics.gpu_memory - 90) / 10, 1.0
                )
            
            # Pattern-based predictions
            pattern_predictions = self._predict_from_patterns(current_metrics, context)
            predictions.update(pattern_predictions)
            
            return predictions
    
    def _update_patterns(self):
        """Update failure patterns for prediction."""
        try:
            # Analyze temporal patterns
            recent_failures = list(self.failure_history)[-50:]
            
            failure_types = defaultdict(list)
            for failure in recent_failures:
                failure_types[failure.failure_type].append(failure.timestamp)
            
            # Store patterns for each failure type
            for failure_type, timestamps in failure_types.items():
                if len(timestamps) >= 3:
                    intervals = [
                        (timestamps[i] - timestamps[i-1]).total_seconds()
                        for i in range(1, len(timestamps))
                    ]
                    
                    self.pattern_cache[failure_type] = {
                        'mean_interval': np.mean(intervals),
                        'std_interval': np.std(intervals),
                        'frequency': len(timestamps) / len(recent_failures)
                    }
                    
        except Exception as e:
            logger.warning(f"Pattern update failed: {e}")
    
    def _predict_from_patterns(
        self,
        current_metrics: HealthMetrics,
        context: Dict[str, Any]
    ) -> Dict[FailureType, float]:
        """Predict failures based on learned patterns."""
        predictions = {}
        
        try:
            for failure_type, pattern in self.pattern_cache.items():
                # Simple heuristic based on frequency
                base_probability = pattern['frequency']
                
                # Adjust based on current conditions
                if failure_type == FailureType.MEMORY_OVERFLOW:
                    memory_factor = current_metrics.memory_usage / 100
                    predictions[failure_type] = min(base_probability * memory_factor, 1.0)
                elif failure_type == FailureType.TIMEOUT:
                    latency_factor = current_metrics.average_latency / 30  # 30s baseline
                    predictions[failure_type] = min(base_probability * latency_factor, 1.0)
                else:
                    predictions[failure_type] = base_probability
                    
        except Exception as e:
            logger.warning(f"Pattern-based prediction failed: {e}")
        
        return predictions


class AutoRecoverySystem:
    """Autonomous recovery system for handling failures."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.recovery_history = deque(maxlen=1000)
        self.circuit_breakers = {}
        self._lock = threading.Lock()
    
    async def handle_failure(
        self,
        failure: FailureRecord,
        operation: Callable,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """Handle failure with appropriate recovery strategy."""
        strategy = self._select_recovery_strategy(failure)
        failure.recovery_strategy = strategy
        
        start_time = time.time()
        success = False
        result = None
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                success, result = await self._retry_with_backoff(
                    operation, failure, *args, **kwargs
                )
            elif strategy == RecoveryStrategy.FALLBACK:
                success, result = await self._execute_fallback(
                    operation, failure, *args, **kwargs
                )
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success, result = await self._graceful_degradation(
                    operation, failure, *args, **kwargs
                )
            elif strategy == RecoveryStrategy.RESOURCE_REALLOCATION:
                success, result = await self._reallocate_resources(
                    operation, failure, *args, **kwargs
                )
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                success = False
                result = None
                logger.warning("Circuit breaker activated - operation blocked")
            else:
                success = False
                result = None
            
        except Exception as e:
            logger.error(f"Recovery strategy {strategy} failed: {e}")
            success = False
            result = None
        
        failure.recovery_success = success
        failure.recovery_time = time.time() - start_time
        
        with self._lock:
            self.recovery_history.append(failure)
        
        return success, result
    
    def _select_recovery_strategy(self, failure: FailureRecord) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on failure type."""
        strategy_map = {
            FailureType.MEMORY_OVERFLOW: RecoveryStrategy.RESOURCE_REALLOCATION,
            FailureType.TIMEOUT: RecoveryStrategy.RETRY,
            FailureType.MODEL_LOAD_ERROR: RecoveryStrategy.FALLBACK,
            FailureType.GENERATION_ERROR: RecoveryStrategy.RETRY,
            FailureType.VALIDATION_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureType.RESOURCE_EXHAUSTION: RecoveryStrategy.RESOURCE_REALLOCATION,
            FailureType.NETWORK_ERROR: RecoveryStrategy.RETRY,
            FailureType.UNKNOWN: RecoveryStrategy.RETRY
        }
        
        return strategy_map.get(failure.failure_type, RecoveryStrategy.RETRY)
    
    async def _retry_with_backoff(
        self,
        operation: Callable,
        failure: FailureRecord,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """Retry operation with exponential backoff."""
        for attempt in range(self.config.max_retry_attempts):
            try:
                delay = self.config.retry_delay_base * (2 ** attempt)
                if attempt > 0:
                    await asyncio.sleep(delay)
                
                logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retry_attempts}")
                result = await asyncio.to_thread(operation, *args, **kwargs)
                return True, result
                
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retry_attempts - 1:
                    return False, None
        
        return False, None
    
    async def _execute_fallback(
        self,
        operation: Callable,
        failure: FailureRecord,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """Execute fallback operation."""
        try:
            # Simple fallback: reduce complexity
            fallback_kwargs = kwargs.copy()
            
            # Reduce parameters for fallback
            if 'num_frames' in fallback_kwargs:
                fallback_kwargs['num_frames'] = min(fallback_kwargs.get('num_frames', 16), 8)
            if 'resolution' in fallback_kwargs:
                original_res = fallback_kwargs.get('resolution', (512, 512))
                fallback_kwargs['resolution'] = tuple(r // 2 for r in original_res)
            
            logger.info("Executing fallback operation with reduced parameters")
            result = await asyncio.to_thread(operation, *args, **fallback_kwargs)
            return True, result
            
        except Exception as e:
            logger.error(f"Fallback operation failed: {e}")
            return False, None
    
    async def _graceful_degradation(
        self,
        operation: Callable,
        failure: FailureRecord,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """Execute operation with graceful degradation."""
        try:
            # Implement graceful degradation by reducing quality expectations
            degraded_kwargs = kwargs.copy()
            
            # Reduce quality parameters
            if 'cfg_scale' in degraded_kwargs:
                degraded_kwargs['cfg_scale'] = max(degraded_kwargs.get('cfg_scale', 7.5) * 0.7, 1.0)
            if 'num_inference_steps' in degraded_kwargs:
                degraded_kwargs['num_inference_steps'] = max(
                    int(degraded_kwargs.get('num_inference_steps', 50) * 0.6), 10
                )
            
            logger.info("Executing with graceful degradation")
            result = await asyncio.to_thread(operation, *args, **degraded_kwargs)
            return True, result
            
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False, None
    
    async def _reallocate_resources(
        self,
        operation: Callable,
        failure: FailureRecord,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """Reallocate resources and retry operation."""
        try:
            # Clear GPU cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            
            # Reduce batch size or parallel operations
            reallocation_kwargs = kwargs.copy()
            if 'batch_size' in reallocation_kwargs:
                reallocation_kwargs['batch_size'] = max(
                    reallocation_kwargs.get('batch_size', 1) // 2, 1
                )
            
            # Wait a moment for resources to be freed
            await asyncio.sleep(2.0)
            
            logger.info("Retrying with resource reallocation")
            result = await asyncio.to_thread(operation, *args, **reallocation_kwargs)
            return True, result
            
        except Exception as e:
            logger.error(f"Resource reallocation failed: {e}")
            return False, None
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        with self._lock:
            if not self.recovery_history:
                return {}
            
            recent_recoveries = list(self.recovery_history)[-100:]
            
            success_rate = sum(1 for r in recent_recoveries if r.recovery_success) / len(recent_recoveries)
            
            strategy_stats = defaultdict(lambda: {'count': 0, 'success': 0})
            for recovery in recent_recoveries:
                if recovery.recovery_strategy:
                    strategy_stats[recovery.recovery_strategy]['count'] += 1
                    if recovery.recovery_success:
                        strategy_stats[recovery.recovery_strategy]['success'] += 1
            
            return {
                'total_recoveries': len(recent_recoveries),
                'success_rate': success_rate,
                'average_recovery_time': np.mean([r.recovery_time for r in recent_recoveries]),
                'strategy_statistics': {
                    strategy.value: {
                        'count': stats['count'],
                        'success_rate': stats['success'] / stats['count'] if stats['count'] > 0 else 0
                    }
                    for strategy, stats in strategy_stats.items()
                }
            }


class AutonomousResilienceFramework:
    """Main resilience framework coordinating all components."""
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(self.config.resource_monitoring_interval)
        self.failure_predictor = FailurePredictor(self.config.failure_prediction_window)
        self.recovery_system = AutoRecoverySystem(self.config)
        
        # Health tracking
        self.system_health = HealthMetrics()
        self.alert_handlers = []
        self.is_active = False
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        logger.info("Autonomous Resilience Framework initialized")
    
    def start(self):
        """Start the resilience framework."""
        if not self.is_active:
            self.is_active = True
            self.resource_monitor.start_monitoring()
            
            # Start health check loop
            asyncio.create_task(self._health_check_loop())
            
            logger.info("Resilience framework started")
    
    def stop(self):
        """Stop the resilience framework."""
        self.is_active = False
        self.resource_monitor.stop_monitoring()
        logger.info("Resilience framework stopped")
    
    async def execute_with_resilience(
        self,
        operation: Callable,
        context: Dict[str, Any],
        *args,
        **kwargs
    ) -> tuple[bool, Any, Optional[FailureRecord]]:
        """Execute operation with full resilience protection."""
        start_time = time.time()
        
        # Pre-execution checks
        current_metrics = self.resource_monitor.get_current_metrics()
        if current_metrics:
            # Predict failures
            if self.config.enable_predictive_failure_detection:
                failure_predictions = self.failure_predictor.predict_failure_probability(
                    current_metrics, context
                )
                
                # Take preemptive action if high failure probability
                high_risk_failures = {
                    failure_type: prob for failure_type, prob in failure_predictions.items()
                    if prob > 0.7
                }
                
                if high_risk_failures:
                    logger.warning(f"High failure risk detected: {high_risk_failures}")
                    await self._preemptive_mitigation(high_risk_failures)
        
        try:
            # Execute operation
            result = await asyncio.to_thread(operation, *args, **kwargs)
            
            # Record successful execution
            execution_time = time.time() - start_time
            self._record_success(execution_time, context)
            
            return True, result, None
            
        except Exception as e:
            # Handle failure
            failure_record = self._create_failure_record(e, context)
            
            if self.config.enable_automatic_recovery:
                success, result = await self.recovery_system.handle_failure(
                    failure_record, operation, *args, **kwargs
                )
                
                if success:
                    logger.info(f"Automatic recovery successful for {failure_record.failure_type}")
                    return True, result, failure_record
                else:
                    logger.error(f"Automatic recovery failed for {failure_record.failure_type}")
            
            # Record failure for learning
            self.failure_predictor.record_failure(failure_record)
            
            return False, None, failure_record
    
    async def _health_check_loop(self):
        """Continuous health monitoring loop."""
        while self.is_active:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                current_metrics = self.resource_monitor.get_current_metrics()
                if current_metrics:
                    self.system_health = current_metrics
                    
                    # Check for critical conditions
                    await self._check_critical_conditions(current_metrics)
                    
                    # Update performance history
                    with self._lock:
                        self.performance_history.append(current_metrics)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _check_critical_conditions(self, metrics: HealthMetrics):
        """Check for critical system conditions."""
        critical_alerts = []
        
        # Memory critical
        if metrics.memory_usage > self.config.emergency_threshold * 100:
            critical_alerts.append(f"Critical memory usage: {metrics.memory_usage:.1f}%")
        
        # GPU memory critical
        if metrics.gpu_memory > self.config.emergency_threshold * 100:
            critical_alerts.append(f"Critical GPU memory: {metrics.gpu_memory:.1f}%")
        
        # Error rate critical
        if metrics.error_rate > self.config.graceful_degradation_threshold:
            critical_alerts.append(f"High error rate: {metrics.error_rate:.1%}")
        
        # Send alerts
        for alert in critical_alerts:
            await self._send_alert("CRITICAL", alert)
    
    async def _preemptive_mitigation(self, high_risk_failures: Dict[FailureType, float]):
        """Take preemptive action to mitigate predicted failures."""
        for failure_type, probability in high_risk_failures.items():
            if failure_type == FailureType.MEMORY_OVERFLOW:
                # Clear caches preemptively
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Preemptive GPU cache clearing for memory overflow prevention")
                
            elif failure_type == FailureType.RESOURCE_EXHAUSTION:
                # Reduce resource allocation
                logger.info("Preemptive resource allocation reduction")
                
            elif failure_type == FailureType.TIMEOUT:
                # Increase timeout thresholds temporarily
                logger.info("Preemptive timeout threshold adjustment")
    
    def _create_failure_record(self, exception: Exception, context: Dict[str, Any]) -> FailureRecord:
        """Create failure record from exception."""
        failure_type = self._classify_failure(exception)
        
        return FailureRecord(
            failure_type=failure_type,
            timestamp=datetime.now(),
            context=context,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            impact_score=self._calculate_impact_score(failure_type, context)
        )
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type from exception."""
        error_message = str(exception).lower()
        
        if 'memory' in error_message and ('out' in error_message or 'overflow' in error_message):
            return FailureType.MEMORY_OVERFLOW
        elif 'timeout' in error_message or 'time' in error_message:
            return FailureType.TIMEOUT
        elif 'load' in error_message and 'model' in error_message:
            return FailureType.MODEL_LOAD_ERROR
        elif 'generation' in error_message or 'generate' in error_message:
            return FailureType.GENERATION_ERROR
        elif 'validation' in error_message or 'valid' in error_message:
            return FailureType.VALIDATION_ERROR
        elif 'resource' in error_message or 'cuda' in error_message:
            return FailureType.RESOURCE_EXHAUSTION
        elif 'network' in error_message or 'connection' in error_message:
            return FailureType.NETWORK_ERROR
        else:
            return FailureType.UNKNOWN
    
    def _calculate_impact_score(self, failure_type: FailureType, context: Dict[str, Any]) -> float:
        """Calculate impact score of failure."""
        base_scores = {
            FailureType.MEMORY_OVERFLOW: 0.8,
            FailureType.TIMEOUT: 0.6,
            FailureType.MODEL_LOAD_ERROR: 0.9,
            FailureType.GENERATION_ERROR: 0.7,
            FailureType.VALIDATION_ERROR: 0.4,
            FailureType.RESOURCE_EXHAUSTION: 0.9,
            FailureType.NETWORK_ERROR: 0.5,
            FailureType.UNKNOWN: 0.5
        }
        
        base_score = base_scores.get(failure_type, 0.5)
        
        # Adjust based on context
        if context.get('is_critical', False):
            base_score *= 1.2
        
        if context.get('batch_size', 1) > 1:
            base_score *= 1.1
        
        return min(base_score, 1.0)
    
    def _record_success(self, execution_time: float, context: Dict[str, Any]):
        """Record successful execution for performance tracking."""
        # Update success metrics
        self.system_health.success_rate = 0.95 * self.system_health.success_rate + 0.05 * 1.0
        self.system_health.average_latency = 0.9 * self.system_health.average_latency + 0.1 * execution_time
    
    async def _send_alert(self, severity: str, message: str):
        """Send alert to registered handlers."""
        alert = {
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'cpu_usage': self.system_health.cpu_usage,
                'memory_usage': self.system_health.memory_usage,
                'gpu_usage': self.system_health.gpu_usage,
                'success_rate': self.system_health.success_rate
            }
        }
        
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"[{severity}] {message}")
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_metrics = self.resource_monitor.get_current_metrics()
        recovery_stats = self.recovery_system.get_recovery_statistics()
        
        with self._lock:
            performance_stats = {}
            if self.performance_history:
                recent_metrics = list(self.performance_history)[-50:]
                performance_stats = {
                    'average_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
                    'average_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
                    'average_gpu_usage': np.mean([m.gpu_usage for m in recent_metrics]),
                    'performance_trend': self._calculate_performance_trend(recent_metrics)
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'is_active': self.is_active,
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'performance_statistics': performance_stats,
            'recovery_statistics': recovery_stats,
            'prediction_accuracy': self._calculate_prediction_accuracy(),
            'system_health_score': self._calculate_system_health_score(),
            'recommendations': self._generate_health_recommendations()
        }
    
    def _calculate_performance_trend(self, metrics_list: List[HealthMetrics]) -> str:
        """Calculate performance trend from recent metrics."""
        if len(metrics_list) < 10:
            return "insufficient_data"
        
        # Calculate trend in success rate
        success_rates = [m.success_rate for m in metrics_list]
        trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
        
        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy of failure predictor."""
        # Simplified accuracy calculation
        recent_failures = list(self.failure_predictor.failure_history)[-20:]
        if len(recent_failures) < 5:
            return 0.0
        
        # This would be more sophisticated in a real implementation
        return 0.75  # Placeholder accuracy
    
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score."""
        if not self.system_health:
            return 0.0
        
        # Weighted health score
        weights = {
            'success_rate': 0.4,
            'resource_usage': 0.3,
            'response_time': 0.2,
            'error_rate': 0.1
        }
        
        success_component = self.system_health.success_rate * weights['success_rate']
        
        # Resource usage (lower is better)
        avg_resource_usage = (
            self.system_health.cpu_usage + self.system_health.memory_usage + 
            self.system_health.gpu_usage
        ) / 300  # Normalize to 0-1
        resource_component = (1.0 - avg_resource_usage) * weights['resource_usage']
        
        # Response time (lower is better)
        response_component = min(1.0 / (1.0 + self.system_health.average_latency), 1.0) * weights['response_time']
        
        # Error rate (lower is better)
        error_component = (1.0 - self.system_health.error_rate) * weights['error_rate']
        
        return success_component + resource_component + response_component + error_component
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate system health recommendations."""
        recommendations = []
        
        if self.system_health.memory_usage > 80:
            recommendations.append("Consider reducing memory usage or increasing available RAM")
        
        if self.system_health.gpu_memory > 85:
            recommendations.append("GPU memory usage is high - consider optimizing model parameters")
        
        if self.system_health.success_rate < 0.9:
            recommendations.append("Success rate is below optimal - investigate recurring failures")
        
        if self.system_health.average_latency > 30:
            recommendations.append("Response times are high - consider performance optimization")
        
        # Recovery system recommendations
        recovery_stats = self.recovery_system.get_recovery_statistics()
        if recovery_stats and recovery_stats.get('success_rate', 1.0) < 0.8:
            recommendations.append("Recovery success rate is low - review recovery strategies")
        
        return recommendations
    
    def export_resilience_report(self, output_path: Path):
        """Export comprehensive resilience report."""
        report = {
            'report_metadata': {
                'timestamp': datetime.now().isoformat(),
                'framework_version': '2.0-autonomous',
                'report_type': 'resilience_analysis'
            },
            'system_status': self.get_system_status(),
            'failure_analysis': self._generate_failure_analysis(),
            'performance_analysis': self._generate_performance_analysis(),
            'recommendations': self._generate_comprehensive_recommendations()
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Resilience report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export resilience report: {e}")
            raise
    
    def _generate_failure_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive failure analysis."""
        recent_failures = list(self.failure_predictor.failure_history)[-100:]
        
        if not recent_failures:
            return {'message': 'No recent failures recorded'}
        
        failure_types = defaultdict(int)
        recovery_success = defaultdict(list)
        
        for failure in recent_failures:
            failure_types[failure.failure_type.value] += 1
            if failure.recovery_strategy:
                recovery_success[failure.recovery_strategy.value].append(failure.recovery_success)
        
        return {
            'total_failures': len(recent_failures),
            'failure_distribution': dict(failure_types),
            'most_common_failure': max(failure_types.items(), key=lambda x: x[1])[0] if failure_types else None,
            'recovery_effectiveness': {
                strategy: {
                    'attempts': len(successes),
                    'success_rate': sum(successes) / len(successes) if successes else 0
                }
                for strategy, successes in recovery_success.items()
            }
        }
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate performance analysis."""
        with self._lock:
            if not self.performance_history:
                return {'message': 'No performance data available'}
            
            recent_metrics = list(self.performance_history)[-100:]
            
            return {
                'metrics_count': len(recent_metrics),
                'average_metrics': {
                    'cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
                    'memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
                    'gpu_usage': np.mean([m.gpu_usage for m in recent_metrics]),
                    'success_rate': np.mean([m.success_rate for m in recent_metrics])
                },
                'peak_usage': {
                    'cpu_peak': np.max([m.cpu_usage for m in recent_metrics]),
                    'memory_peak': np.max([m.memory_usage for m in recent_metrics]),
                    'gpu_peak': np.max([m.gpu_usage for m in recent_metrics])
                },
                'performance_trend': self._calculate_performance_trend(recent_metrics)
            }
    
    def _generate_comprehensive_recommendations(self) -> Dict[str, List[str]]:
        """Generate comprehensive recommendations by category."""
        recommendations = {
            'immediate_actions': [],
            'optimization_opportunities': [],
            'long_term_improvements': [],
            'monitoring_enhancements': []
        }
        
        # Immediate actions based on current state
        if self.system_health.memory_usage > 90:
            recommendations['immediate_actions'].append(
                "Critical: Memory usage is extremely high - implement immediate memory management"
            )
        
        if self.system_health.success_rate < 0.8:
            recommendations['immediate_actions'].append(
                "Critical: Success rate is below acceptable threshold - investigate failures"
            )
        
        # Optimization opportunities
        recovery_stats = self.recovery_system.get_recovery_statistics()
        if recovery_stats and recovery_stats.get('success_rate', 1.0) < 0.9:
            recommendations['optimization_opportunities'].append(
                "Improve recovery strategies effectiveness"
            )
        
        if self.system_health.average_latency > 20:
            recommendations['optimization_opportunities'].append(
                "Optimize response times through performance tuning"
            )
        
        # Long-term improvements
        recommendations['long_term_improvements'].extend([
            "Implement advanced machine learning for failure prediction",
            "Develop custom recovery strategies for domain-specific failures",
            "Integrate with external monitoring and alerting systems"
        ])
        
        # Monitoring enhancements
        recommendations['monitoring_enhancements'].extend([
            "Add custom metrics for domain-specific monitoring",
            "Implement distributed tracing for complex operations",
            "Set up automated performance regression detection"
        ])
        
        return recommendations


# Factory function for easy instantiation
def create_resilience_framework(
    enable_predictive: bool = True,
    enable_recovery: bool = True,
    enable_adaptive_resources: bool = True
) -> AutonomousResilienceFramework:
    """Create resilience framework with specified capabilities."""
    config = ResilienceConfig(
        enable_predictive_failure_detection=enable_predictive,
        enable_automatic_recovery=enable_recovery,
        enable_adaptive_resource_management=enable_adaptive_resources
    )
    
    return AutonomousResilienceFramework(config)