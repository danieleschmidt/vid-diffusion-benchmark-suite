"""Advanced error handling and recovery mechanisms.

This module provides comprehensive error handling for video diffusion benchmarking,
including structured exception hierarchy, recovery strategies, and error analytics.
"""

import logging
import traceback
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import functools


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    NETWORK = "network"
    MEMORY = "memory"
    GPU = "gpu"
    MODEL = "model"
    DATA = "data"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    traceback_info: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    impact_assessment: str = ""
    recommendations: List[str] = field(default_factory=list)


class BenchmarkException(Exception):
    """Base exception for benchmark-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now()


class ModelLoadError(BenchmarkException):
    """Error during model loading."""
    
    def __init__(self, model_name: str, message: str, **kwargs):
        super().__init__(
            f"Failed to load model '{model_name}': {message}",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL,
            **kwargs
        )
        self.model_name = model_name


class EvaluationError(BenchmarkException):
    """Error during model evaluation."""
    
    def __init__(self, model_name: str, prompt: str, message: str, **kwargs):
        super().__init__(
            f"Evaluation failed for model '{model_name}' with prompt '{prompt}': {message}",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.MODEL,
            **kwargs
        )
        self.model_name = model_name
        self.prompt = prompt


class ValidationError(BenchmarkException):
    """Error during input validation."""
    
    def __init__(self, field: str, message: str, **kwargs):
        super().__init__(
            f"Validation failed for field '{field}': {message}",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        self.field = field


class RetryableError(BenchmarkException):
    """Error that can be retried."""
    
    def __init__(self, message: str, max_retries: int = 3, **kwargs):
        super().__init__(message, **kwargs)
        self.max_retries = max_retries
        self.retry_count = 0


class MemoryError(BenchmarkException):
    """Memory-related error."""
    
    def __init__(self, message: str, memory_usage: Optional[float] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            **kwargs
        )
        self.memory_usage = memory_usage


class GPUError(BenchmarkException):
    """GPU-related error."""
    
    def __init__(self, message: str, gpu_id: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.GPU,
            **kwargs
        )
        self.gpu_id = gpu_id


class TimeoutError(BenchmarkException):
    """Timeout error."""
    
    def __init__(self, operation: str, timeout: float, **kwargs):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout} seconds",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            **kwargs
        )
        self.operation = operation
        self.timeout = timeout


class ErrorHandler:
    """Advanced error handler with recovery strategies and analytics."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        enable_analytics: bool = True,
        max_error_history: int = 1000
    ):
        """Initialize error handler.
        
        Args:
            log_file: Optional file path for error logging
            enable_analytics: Whether to enable error analytics
            max_error_history: Maximum number of errors to keep in history
        """
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.enable_analytics = enable_analytics
        self.max_error_history = max_error_history
        
        # Error tracking
        self.error_history: List[ErrorReport] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        
        # Setup file logging if specified
        if log_file:
            self._setup_file_logging(log_file)
        
        # Register default recovery strategies
        self._register_default_strategies()
        
        self.logger.info("ErrorHandler initialized")
    
    def _setup_file_logging(self, log_file: str):
        """Setup file logging for errors."""
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        # Memory error strategy
        def memory_recovery_strategy(error: MemoryError, context: Dict[str, Any]) -> bool:
            """Attempt to recover from memory errors."""
            try:
                import gc
                import torch
                
                # Clear Python garbage
                gc.collect()
                
                # Clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                self.logger.info("Memory recovery strategy executed")
                return True
                
            except Exception as e:
                self.logger.error(f"Memory recovery failed: {e}")
                return False
        
        # GPU error strategy
        def gpu_recovery_strategy(error: GPUError, context: Dict[str, Any]) -> bool:
            """Attempt to recover from GPU errors."""
            try:
                import torch
                
                if torch.cuda.is_available():
                    # Reset GPU state
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Try to reinitialize CUDA context
                    if error.gpu_id is not None:
                        with torch.cuda.device(error.gpu_id):
                            torch.cuda.init()
                
                self.logger.info("GPU recovery strategy executed")
                return True
                
            except Exception as e:
                self.logger.error(f"GPU recovery failed: {e}")
                return False
        
        # Model loading error strategy
        def model_recovery_strategy(error: ModelLoadError, context: Dict[str, Any]) -> bool:
            """Attempt to recover from model loading errors."""
            try:
                # Clear memory before retry
                import gc
                import torch
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Add delay before retry
                time.sleep(2)
                
                self.logger.info(f"Model loading recovery for {error.model_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Model recovery failed: {e}")
                return False
        
        # Register strategies
        self.register_recovery_strategy(MemoryError, memory_recovery_strategy)
        self.register_recovery_strategy(GPUError, gpu_recovery_strategy)
        self.register_recovery_strategy(ModelLoadError, model_recovery_strategy)
    
    def register_recovery_strategy(
        self,
        exception_type: Type[Exception],
        strategy: Callable[[Exception, Dict[str, Any]], bool]
    ):
        """Register recovery strategy for specific exception type.
        
        Args:
            exception_type: Exception type to handle
            strategy: Recovery function returning success boolean
        """
        self.recovery_strategies[exception_type] = strategy
        self.logger.info(f"Registered recovery strategy for {exception_type.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> ErrorReport:
        """Handle error with comprehensive reporting and recovery.
        
        Args:
            error: Exception to handle
            context: Additional context information
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            Error report with recovery information
        """
        context = context or {}
        
        # Generate error report
        error_report = self._create_error_report(error, context)
        
        # Attempt recovery if enabled
        if attempt_recovery:
            error_report.recovery_attempted = True
            error_report.recovery_successful = self._attempt_recovery(error, context)
        
        # Log error
        self._log_error(error_report)
        
        # Update analytics
        if self.enable_analytics:
            self._update_analytics(error_report)
        
        # Store in history
        self._store_error_report(error_report)
        
        return error_report
    
    def _create_error_report(self, error: Exception, context: Dict[str, Any]) -> ErrorReport:
        """Create comprehensive error report."""
        # Extract error information
        error_id = f"{int(time.time() * 1000)}_{hash(str(error)) % 10000}"
        
        if isinstance(error, BenchmarkException):
            severity = error.severity
            category = error.category
            message = error.message
            context.update(error.context)
        else:
            severity = self._classify_severity(error)
            category = self._classify_category(error)
            message = str(error)
        
        # Generate traceback
        traceback_info = traceback.format_exc()
        
        # Impact assessment
        impact_assessment = self._assess_impact(error, category, severity)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(error, category, severity)
        
        return ErrorReport(
            error_id=error_id,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=message,
            severity=severity,
            category=category,
            traceback_info=traceback_info,
            context=context,
            impact_assessment=impact_assessment,
            recommendations=recommendations
        )
    
    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ['out of memory', 'cuda error', 'system failure']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['RuntimeError', 'OSError', 'ImportError']:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _classify_category(self, error: Exception) -> ErrorCategory:
        """Classify error category based on type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Memory errors
        if 'memory' in error_message or 'out of memory' in error_message:
            return ErrorCategory.MEMORY
        
        # GPU errors
        if any(keyword in error_message for keyword in ['cuda', 'gpu', 'device']):
            return ErrorCategory.GPU
        
        # Network errors
        if any(keyword in error_message for keyword in ['network', 'connection', 'timeout']):
            return ErrorCategory.NETWORK
        
        # Model errors
        if any(keyword in error_message for keyword in ['model', 'checkpoint', 'weights']):
            return ErrorCategory.MODEL
        
        # Data errors
        if any(keyword in error_message for keyword in ['data', 'tensor', 'shape']):
            return ErrorCategory.DATA
        
        # Validation errors
        if error_type in ['ValueError', 'TypeError']:
            return ErrorCategory.VALIDATION
        
        return ErrorCategory.UNKNOWN
    
    def _assess_impact(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> str:
        """Assess error impact on benchmarking process."""
        if severity == ErrorSeverity.CRITICAL:
            return "Critical impact - benchmark execution halted"
        elif severity == ErrorSeverity.HIGH:
            if category in [ErrorCategory.MODEL, ErrorCategory.GPU]:
                return "High impact - model evaluation affected"
            else:
                return "High impact - significant functionality impaired"
        elif severity == ErrorSeverity.MEDIUM:
            return "Medium impact - partial functionality affected"
        else:
            return "Low impact - minor functionality affected"
    
    def _generate_recommendations(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> List[str]:
        """Generate recommendations based on error characteristics."""
        recommendations = []
        
        if category == ErrorCategory.MEMORY:
            recommendations.extend([
                "Reduce batch size or input resolution",
                "Enable gradient checkpointing if available",
                "Close unused applications to free memory",
                "Consider using CPU offloading for large models"
            ])
        
        elif category == ErrorCategory.GPU:
            recommendations.extend([
                "Check GPU driver compatibility",
                "Verify CUDA installation",
                "Monitor GPU temperature and power",
                "Try restarting GPU processes"
            ])
        
        elif category == ErrorCategory.MODEL:
            recommendations.extend([
                "Verify model checkpoint integrity", 
                "Check model compatibility with current framework version",
                "Ensure sufficient disk space for model files",
                "Try redownloading model weights"
            ])
        
        elif category == ErrorCategory.VALIDATION:
            recommendations.extend([
                "Check input data format and types",
                "Verify parameter ranges and constraints",
                "Review configuration file syntax",
                "Ensure all required fields are provided"
            ])
        
        elif category == ErrorCategory.NETWORK:
            recommendations.extend([
                "Check internet connection",
                "Verify firewall and proxy settings",
                "Try different download mirrors",
                "Use cached models if available"
            ])
        
        # General recommendations based on severity
        if severity == ErrorSeverity.CRITICAL:
            recommendations.append("Consider using fallback models or configurations")
        
        return recommendations
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from error using registered strategies."""
        error_type = type(error)
        
        # Check for specific recovery strategy
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    success = strategy(error, context)
                    if success:
                        self.logger.info(f"Recovery successful for {error_type.__name__}")
                        return True
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # Generic recovery attempts
        return self._generic_recovery(error, context)
    
    def _generic_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Generic recovery attempts."""
        try:
            # Clear caches and garbage collect
            import gc
            gc.collect()
            
            # Small delay to allow system recovery
            time.sleep(1)
            
            self.logger.info("Generic recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Generic recovery failed: {e}")
            return False
    
    def _log_error(self, error_report: ErrorReport):
        """Log error with appropriate level."""
        log_message = (
            f"[{error_report.error_id}] {error_report.error_type}: "
            f"{error_report.error_message}"
        )
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_report.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _update_analytics(self, error_report: ErrorReport):
        """Update error analytics."""
        # Count by type
        error_type = error_report.error_type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Count by category
        category_key = f"category_{error_report.category.value}"
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1
        
        # Count by severity
        severity_key = f"severity_{error_report.severity.value}"
        self.error_counts[severity_key] = self.error_counts.get(severity_key, 0) + 1
    
    def _store_error_report(self, error_report: ErrorReport):
        """Store error report in history."""
        self.error_history.append(error_report)
        
        # Maintain maximum history size
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {'total_errors': 0}
        
        total_errors = len(self.error_history)
        
        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            count = sum(1 for report in self.error_history 
                       if report.severity == severity)
            severity_counts[severity.value] = count
        
        # Count by category
        category_counts = {}
        for category in ErrorCategory:
            count = sum(1 for report in self.error_history 
                       if report.category == category)
            category_counts[category.value] = count
        
        # Recovery success rate
        recovery_attempts = sum(1 for report in self.error_history 
                               if report.recovery_attempted)
        recovery_successes = sum(1 for report in self.error_history 
                                if report.recovery_successful)
        
        recovery_rate = (recovery_successes / recovery_attempts * 100 
                        if recovery_attempts > 0 else 0)
        
        # Most common errors
        error_type_counts = {}
        for report in self.error_history:
            error_type = report.error_type
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        most_common_errors = sorted(error_type_counts.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_errors': total_errors,
            'severity_distribution': severity_counts,
            'category_distribution': category_counts,
            'recovery_rate': recovery_rate,
            'recovery_attempts': recovery_attempts,
            'recovery_successes': recovery_successes,
            'most_common_errors': most_common_errors,
            'error_rate_per_hour': self._calculate_error_rate()
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate per hour."""
        if not self.error_history:
            return 0.0
        
        # Get time span of error history
        earliest = min(report.timestamp for report in self.error_history)
        latest = max(report.timestamp for report in self.error_history)
        
        time_span_hours = (latest - earliest).total_seconds() / 3600
        
        if time_span_hours == 0:
            return len(self.error_history)  # All errors in same hour
        
        return len(self.error_history) / time_span_hours
    
    def export_error_reports(self, output_path: str, format: str = "json"):
        """Export error reports to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Convert reports to dictionaries
            reports_data = []
            for report in self.error_history:
                report_dict = {
                    'error_id': report.error_id,
                    'timestamp': report.timestamp.isoformat(),
                    'error_type': report.error_type,
                    'error_message': report.error_message,
                    'severity': report.severity.value,
                    'category': report.category.value,
                    'recovery_attempted': report.recovery_attempted,
                    'recovery_successful': report.recovery_successful,
                    'impact_assessment': report.impact_assessment,
                    'recommendations': report.recommendations,
                    'context': report.context
                }
                reports_data.append(report_dict)
            
            with open(output_path, 'w') as f:
                json.dump(reports_data, f, indent=2)
        
        elif format == "csv":
            import pandas as pd
            
            # Create DataFrame
            data = []
            for report in self.error_history:
                data.append({
                    'error_id': report.error_id,
                    'timestamp': report.timestamp.isoformat(),
                    'error_type': report.error_type,
                    'error_message': report.error_message,
                    'severity': report.severity.value,
                    'category': report.category.value,
                    'recovery_attempted': report.recovery_attempted,
                    'recovery_successful': report.recovery_successful,
                    'impact_assessment': report.impact_assessment
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Error reports exported to {output_path}")


def error_handler_decorator(
    error_handler: ErrorHandler,
    retries: int = 0,
    backoff_factor: float = 1.0
):
    """Decorator for automatic error handling with retries.
    
    Args:
        error_handler: Error handler instance
        retries: Number of retry attempts
        backoff_factor: Exponential backoff factor
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # Handle error
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_retries': retries,
                        'args': str(args)[:100],  # Truncate for logging
                        'kwargs': str(kwargs)[:100]
                    }
                    
                    error_report = error_handler.handle_error(e, context)
                    
                    # If this is the last attempt or error is not retryable, re-raise
                    if attempt == retries or not isinstance(e, RetryableError):
                        raise
                    
                    # Wait before retry with exponential backoff
                    if backoff_factor > 0:
                        wait_time = backoff_factor * (2 ** attempt)
                        time.sleep(wait_time)
            
            # This should not be reached, but just in case
            raise last_error
        
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorReport:
    """Convenience function to handle errors with global handler."""
    return get_error_handler().handle_error(error, context)