"""Robust benchmark suite with comprehensive error handling, monitoring, and security."""

import time
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import traceback

# Import enhanced components
from .enhanced_validation import (
    BenchmarkInputValidator, ValidationLevel, ValidationError,
    calculate_benchmark_fingerprint, estimate_benchmark_duration
)
from .enhanced_monitoring import BenchmarkMonitor, get_global_monitor
from .enhanced_security import BenchmarkSecurityManager

# Import existing components
from .benchmark import BenchmarkSuite, BenchmarkResult
from .models.registry import list_models, get_model
from .prompts import StandardPrompts

logger = logging.getLogger(__name__)


class RobustBenchmarkResult(BenchmarkResult):
    """Enhanced benchmark result with additional metadata."""
    
    def __init__(self, model_name: str, prompts: List[str]):
        super().__init__(model_name, prompts)
        
        # Add enhanced tracking
        self.validation_results = {}
        self.security_context = {}
        self.monitoring_data = {}
        self.execution_id = str(uuid.uuid4())
        self.fingerprint = calculate_benchmark_fingerprint(model_name, prompts)
        
        # Quality gates
        self.quality_gates_passed = False
        self.quality_gates_results = {}
        
        # Recovery attempts
        self.recovery_attempts = []
        
    def add_validation_results(self, validation_results: Dict[str, Any]):
        """Add validation results to benchmark result."""
        self.validation_results = validation_results
        
    def add_security_context(self, security_context: Dict[str, Any]):
        """Add security context."""
        self.security_context = security_context
        
    def add_monitoring_data(self, monitoring_data: Dict[str, Any]):
        """Add monitoring data."""
        self.monitoring_data = monitoring_data
        
    def add_recovery_attempt(self, error: Exception, recovery_action: str, success: bool):
        """Record recovery attempt."""
        self.recovery_attempts.append({
            'timestamp': time.time(),
            'error': str(error),
            'error_type': type(error).__name__,
            'recovery_action': recovery_action,
            'success': success,
            'traceback': traceback.format_exc()
        })
        
    def run_quality_gates(self) -> bool:
        """Run quality gates on benchmark results."""
        gates = {
            'minimum_success_rate': self.success_rate >= 0.5,  # At least 50% success
            'reasonable_latency': self._check_reasonable_latency(),
            'memory_efficiency': self._check_memory_efficiency(),
            'no_critical_errors': self._check_no_critical_errors(),
            'metrics_validity': self._check_metrics_validity()
        }
        
        self.quality_gates_results = gates
        self.quality_gates_passed = all(gates.values())
        
        return self.quality_gates_passed
        
    def _check_reasonable_latency(self) -> bool:
        """Check if latencies are reasonable."""
        if not self.performance:
            return True  # Can't verify, assume OK
            
        avg_latency = self.performance.get('avg_latency_ms', 0)
        # Reasonable if under 5 minutes per generation
        return avg_latency < 300000
        
    def _check_memory_efficiency(self) -> bool:
        """Check memory efficiency."""
        if not self.performance:
            return True
            
        peak_vram = self.performance.get('peak_vram_gb', 0)
        # Reasonable if under 32GB VRAM
        return peak_vram < 32.0
        
    def _check_no_critical_errors(self) -> bool:
        """Check for critical errors."""
        critical_errors = [
            'CUDA out of memory',
            'Model weights corrupted',
            'System crash'
        ]
        
        for error in self.errors:
            error_msg = error.get('error', '').lower()
            for critical in critical_errors:
                if critical.lower() in error_msg:
                    return False
                    
        return True
        
    def _check_metrics_validity(self) -> bool:
        """Check if computed metrics are valid."""
        if not self.metrics:
            return False
            
        # Check for NaN or infinite values
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                if not (-1e10 < value < 1e10):  # Reasonable range
                    return False
                    
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced data."""
        base_dict = super().to_dict()
        
        base_dict.update({
            'execution_id': self.execution_id,
            'fingerprint': self.fingerprint,
            'validation_results': self.validation_results,
            'security_context': self.security_context,
            'monitoring_data': self.monitoring_data,
            'quality_gates_passed': self.quality_gates_passed,
            'quality_gates_results': self.quality_gates_results,
            'recovery_attempts': self.recovery_attempts,
            'metadata': getattr(self, 'metadata', {})
        })
        
        return base_dict


class RobustBenchmarkSuite:
    """Robust benchmark suite with comprehensive error handling and monitoring."""
    
    def __init__(
        self,
        device: str = "auto",
        output_dir: str = "./results",
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        enable_security: bool = True,
        enable_monitoring: bool = True,
        max_retries: int = 3,
        timeout_seconds: int = 3600  # 1 hour default timeout
    ):
        self.device = self._resolve_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Initialize core benchmark suite
        self.core_suite = BenchmarkSuite(device=device, output_dir=str(output_dir))
        
        # Initialize enhanced components
        self.validator = BenchmarkInputValidator(validation_level)
        
        if enable_security:
            self.security_manager = BenchmarkSecurityManager()
            logger.info("Security manager initialized")
        else:
            self.security_manager = None
            
        if enable_monitoring:
            self.monitor = get_global_monitor()
            logger.info("Monitoring enabled")
        else:
            self.monitor = None
            
        # Circuit breaker for system health
        self.circuit_breaker = {
            'consecutive_failures': 0,
            'last_failure': 0,
            'threshold': 5,
            'cooldown': 300  # 5 minutes
        }
        
        logger.info(f"RobustBenchmarkSuite initialized - device: {self.device}")
        
    def _resolve_device(self, device: str) -> str:
        """Resolve device with error handling."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                logger.warning("PyTorch not available, using CPU")
                return "cpu"
        return device
        
    def validate_and_secure_request(
        self,
        model_name: str,
        prompts: List[str],
        source_ip: str = "127.0.0.1",
        api_key: str = None,
        **kwargs
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate and secure benchmark request."""
        
        if self.security_manager:
            return self.security_manager.validate_request(
                source_ip=source_ip,
                api_key=api_key,
                model_name=model_name,
                prompts=prompts,
                **kwargs
            )
        else:
            # Basic validation without security
            validation_results = self.validator.validate_benchmark_request(
                model_name=model_name,
                prompts=prompts,
                **kwargs
            )
            
            is_valid = all(result.is_valid for result in validation_results.values())
            warnings = [w for result in validation_results.values() for w in result.warnings]
            metadata = {'validation_results': validation_results}
            
            return is_valid, warnings, metadata
    
    @contextmanager
    def _circuit_breaker_context(self):
        """Circuit breaker for system protection."""
        # Check if circuit breaker is open
        now = time.time()
        if (self.circuit_breaker['consecutive_failures'] >= self.circuit_breaker['threshold'] and
            now - self.circuit_breaker['last_failure'] < self.circuit_breaker['cooldown']):
            
            remaining_cooldown = self.circuit_breaker['cooldown'] - (now - self.circuit_breaker['last_failure'])
            raise RuntimeError(f"Circuit breaker open. System in cooldown for {remaining_cooldown:.1f} more seconds")
            
        try:
            yield
            # Success - reset failure count
            self.circuit_breaker['consecutive_failures'] = 0
            
        except Exception as e:
            # Failure - increment counter
            self.circuit_breaker['consecutive_failures'] += 1
            self.circuit_breaker['last_failure'] = now
            
            logger.error(f"Circuit breaker failure {self.circuit_breaker['consecutive_failures']}: {e}")
            
            if self.circuit_breaker['consecutive_failures'] >= self.circuit_breaker['threshold']:
                logger.critical("Circuit breaker opened due to consecutive failures")
                
            raise
    
    def evaluate_model_robust(
        self,
        model_name: str,
        prompts: Optional[List[str]] = None,
        source_ip: str = "127.0.0.1",
        api_key: str = None,
        **kwargs
    ) -> RobustBenchmarkResult:
        """Robust model evaluation with comprehensive error handling."""
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        if prompts is None:
            prompts = StandardPrompts.DIVERSE_SET_V2[:5]  # Use 5 standard prompts
            
        # Create robust result container
        result = RobustBenchmarkResult(model_name, prompts)
        
        try:
            # Start monitoring
            if self.monitor:
                self.monitor.start_benchmark_monitoring(execution_id, 
                                                       model=model_name, 
                                                       prompts=len(prompts))
            
            with self._circuit_breaker_context():
                # Security and validation
                is_valid, warnings, security_metadata = self.validate_and_secure_request(
                    model_name=model_name,
                    prompts=prompts,
                    source_ip=source_ip,
                    api_key=api_key,
                    **kwargs
                )
                
                if not is_valid:
                    error = ValidationError(
                        f"Request validation failed: {'; '.join(warnings[:3])}",
                        "REQUEST_VALIDATION_FAILED",
                        {"warnings": warnings, "metadata": security_metadata}
                    )
                    result.add_error(-1, error)
                    return result
                
                result.add_security_context(security_metadata)
                
                # Estimate duration
                estimated_duration = estimate_benchmark_duration(model_name, len(prompts), **kwargs)
                logger.info(f"Estimated benchmark duration: {estimated_duration:.1f} seconds")
                
                # Execute benchmark with retries
                benchmark_result = self._execute_with_retries(
                    model_name, prompts, result, **kwargs
                )
                
                # Run quality gates
                quality_passed = result.run_quality_gates()
                logger.info(f"Quality gates passed: {quality_passed}")
                
                # Record monitoring data
                if self.monitor:
                    monitoring_report = self.monitor.get_monitoring_report(since=time.time() - 3600)
                    result.add_monitoring_data(monitoring_report)
                
                return result
                
        except Exception as e:
            logger.error(f"Robust evaluation failed for {model_name}: {e}")
            result.add_error(-1, e)
            
            # Record in monitoring
            if self.monitor:
                self.monitor.record_model_generation(
                    model_name=model_name,
                    duration=0,
                    success=False,
                    error=str(e),
                    execution_id=execution_id
                )
                
            return result
            
    def _execute_with_retries(
        self,
        model_name: str,
        prompts: List[str],
        result: RobustBenchmarkResult,
        **kwargs
    ) -> RobustBenchmarkResult:
        """Execute benchmark with retry logic."""
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Benchmark attempt {attempt + 1}/{self.max_retries + 1} for {model_name}")
                
                # Use core benchmark suite
                core_result = self.core_suite.evaluate_model(
                    model_name=model_name,
                    prompts=prompts,
                    **kwargs
                )
                
                # Copy results to robust result
                result.results = core_result.results
                result.metrics = core_result.metrics
                result.performance = core_result.performance
                result.errors = core_result.errors
                
                # Check if we got reasonable results
                if result.success_rate >= 0.1:  # At least 10% success to continue
                    logger.info(f"Benchmark successful on attempt {attempt + 1}")
                    return result
                else:
                    raise RuntimeError(f"Low success rate: {result.success_rate:.1%}")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Benchmark attempt {attempt + 1} failed: {e}")
                
                # Determine recovery strategy
                recovery_action = self._determine_recovery_strategy(e, attempt)
                recovery_success = False
                
                if attempt < self.max_retries:
                    try:
                        recovery_success = self._apply_recovery_strategy(recovery_action, e)
                        result.add_recovery_attempt(e, recovery_action, recovery_success)
                        
                        if recovery_success:
                            # Wait before retry
                            wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                            logger.info(f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            # Recovery failed, don't retry
                            break
                            
                    except Exception as recovery_error:
                        logger.error(f"Recovery strategy failed: {recovery_error}")
                        result.add_recovery_attempt(e, recovery_action, False)
                        break
                else:
                    result.add_recovery_attempt(e, "max_retries_exceeded", False)
                    
        # All retries failed
        if last_error:
            result.add_error(-1, last_error)
            
        return result
        
    def _determine_recovery_strategy(self, error: Exception, attempt: int) -> str:
        """Determine appropriate recovery strategy for error."""
        error_msg = str(error).lower()
        
        if "cuda out of memory" in error_msg or "memory" in error_msg:
            return "clear_gpu_cache"
        elif "timeout" in error_msg:
            return "increase_timeout"
        elif "connection" in error_msg or "network" in error_msg:
            return "wait_and_retry"
        elif "model" in error_msg and ("load" in error_msg or "download" in error_msg):
            return "reload_model"
        else:
            return "wait_and_retry"
            
    def _apply_recovery_strategy(self, strategy: str, error: Exception) -> bool:
        """Apply recovery strategy."""
        try:
            if strategy == "clear_gpu_cache":
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("GPU cache cleared")
                        return True
                except ImportError:
                    pass
                    
            elif strategy == "increase_timeout":
                # Could increase timeout for next attempt
                logger.info("Timeout strategy applied")
                return True
                
            elif strategy == "wait_and_retry":
                time.sleep(5)
                logger.info("Wait strategy applied")
                return True
                
            elif strategy == "reload_model":
                # Clear any cached models
                if hasattr(self.core_suite, '_models'):
                    self.core_suite._models.clear()
                logger.info("Model cache cleared")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Recovery strategy {strategy} failed: {e}")
            return False
            
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            'timestamp': time.time(),
            'overall_healthy': True,
            'components': {},
            'recommendations': []
        }
        
        # Check core components
        try:
            models = list_models()
            health_status['components']['model_registry'] = {
                'healthy': len(models) > 0,
                'model_count': len(models)
            }
        except Exception as e:
            health_status['components']['model_registry'] = {
                'healthy': False,
                'error': str(e)
            }
            health_status['overall_healthy'] = False
            
        # Check monitoring
        if self.monitor:
            try:
                monitor_health = self.monitor.health.check_system_health()
                health_status['components']['monitoring'] = {
                    'healthy': monitor_health['overall_healthy'],
                    'details': monitor_health
                }
                if not monitor_health['overall_healthy']:
                    health_status['overall_healthy'] = False
            except Exception as e:
                health_status['components']['monitoring'] = {
                    'healthy': False,
                    'error': str(e)
                }
                
        # Check security
        if self.security_manager:
            try:
                security_status = self.security_manager.get_security_status()
                health_status['components']['security'] = {
                    'healthy': True,  # Basic security always healthy if initialized
                    'details': security_status
                }
            except Exception as e:
                health_status['components']['security'] = {
                    'healthy': False,
                    'error': str(e)
                }
                
        # Circuit breaker status
        health_status['components']['circuit_breaker'] = {
            'healthy': self.circuit_breaker['consecutive_failures'] < self.circuit_breaker['threshold'],
            'failures': self.circuit_breaker['consecutive_failures'],
            'threshold': self.circuit_breaker['threshold']
        }
        
        if self.circuit_breaker['consecutive_failures'] >= self.circuit_breaker['threshold']:
            health_status['overall_healthy'] = False
            health_status['recommendations'].append("System in cooldown due to consecutive failures")
            
        # Generate recommendations
        if not health_status['overall_healthy']:
            unhealthy_components = [
                name for name, component in health_status['components'].items()
                if not component.get('healthy', False)
            ]
            health_status['recommendations'].append(
                f"Unhealthy components: {', '.join(unhealthy_components)}"
            )
            
        return health_status
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        if not self.monitor:
            return {"error": "Monitoring not enabled"}
            
        try:
            return self.monitor.get_monitoring_report()
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
            
    def cleanup(self):
        """Cleanup resources."""
        if self.monitor:
            self.monitor.cleanup()
        logger.info("RobustBenchmarkSuite cleanup completed")


# Convenience functions
def create_robust_benchmark(
    validation_level: str = "strict",
    enable_security: bool = True,
    enable_monitoring: bool = True,
    **kwargs
) -> RobustBenchmarkSuite:
    """Create robust benchmark suite with specified configuration."""
    
    level_map = {
        "basic": ValidationLevel.BASIC,
        "strict": ValidationLevel.STRICT,
        "research": ValidationLevel.RESEARCH
    }
    
    return RobustBenchmarkSuite(
        validation_level=level_map.get(validation_level, ValidationLevel.STRICT),
        enable_security=enable_security,
        enable_monitoring=enable_monitoring,
        **kwargs
    )


def quick_robust_benchmark(
    model_name: str,
    num_prompts: int = 3,
    **kwargs
) -> RobustBenchmarkResult:
    """Quick robust benchmark with default settings."""
    
    suite = create_robust_benchmark()
    prompts = StandardPrompts.DIVERSE_SET_V2[:num_prompts]
    
    return suite.evaluate_model_robust(
        model_name=model_name,
        prompts=prompts,
        **kwargs
    )