"""Advanced error handling and recovery mechanisms.

This module provides sophisticated error handling, recovery strategies,
and fault tolerance for video diffusion benchmarking operations.

Key features:
1. Hierarchical error classification and handling
2. Automatic error recovery with backoff strategies
3. Resource cleanup and leak prevention
4. Error pattern analysis and prediction
5. Graceful degradation mechanisms
6. Comprehensive error reporting and analysis
"""

import torch
import numpy as np
import logging
import traceback
import functools
import time
import threading
import gc
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import weakref
from enum import Enum
import json
import pickle

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    WARNING = "warning"


class ErrorCategory(Enum):
    """Error categories for classification."""
    MEMORY = "memory"
    COMPUTATION = "computation"
    IO = "io"
    NETWORK = "network"
    DATA = "data"
    MODEL = "model"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Comprehensive error information container."""
    error_id: str
    timestamp: float
    error_type: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    traceback_str: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: str = ""
    affected_resources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'traceback': self.traceback_str,
            'context': self.context,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_strategy': self.recovery_strategy,
            'affected_resources': self.affected_resources
        }


class ErrorClassifier:
    """Classifies errors by type and severity."""
    
    def __init__(self):
        self.error_patterns = self._init_error_patterns()
        
    def _init_error_patterns(self) -> Dict[str, Tuple[ErrorCategory, ErrorSeverity]]:
        """Initialize error classification patterns."""
        return {
            # Memory errors
            'OutOfMemoryError': (ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            'CUDA out of memory': (ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            'RuntimeError.*memory': (ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            'MemoryError': (ErrorCategory.MEMORY, ErrorSeverity.CRITICAL),
            
            # Computation errors
            'RuntimeError.*CUDA': (ErrorCategory.COMPUTATION, ErrorSeverity.HIGH),
            'RuntimeError.*tensor': (ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM),
            'ValueError.*shape': (ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM),
            'TypeError.*tensor': (ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM),
            
            # IO errors
            'FileNotFoundError': (ErrorCategory.IO, ErrorSeverity.MEDIUM),
            'PermissionError': (ErrorCategory.IO, ErrorSeverity.MEDIUM),
            'OSError.*disk': (ErrorCategory.IO, ErrorSeverity.HIGH),
            'IOError': (ErrorCategory.IO, ErrorSeverity.MEDIUM),
            
            # Network errors
            'ConnectionError': (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            'TimeoutError': (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            'URLError': (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            
            # Data errors
            'ValueError.*data': (ErrorCategory.DATA, ErrorSeverity.MEDIUM),
            'KeyError': (ErrorCategory.DATA, ErrorSeverity.LOW),
            'IndexError': (ErrorCategory.DATA, ErrorSeverity.LOW),
            
            # Model errors
            'ModuleNotFoundError.*model': (ErrorCategory.MODEL, ErrorSeverity.HIGH),
            'AttributeError.*model': (ErrorCategory.MODEL, ErrorSeverity.MEDIUM),
            'RuntimeError.*model': (ErrorCategory.MODEL, ErrorSeverity.MEDIUM),
            
            # Configuration errors
            'KeyError.*config': (ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
            'ValueError.*config': (ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
            
            # System errors
            'SystemError': (ErrorCategory.SYSTEM, ErrorSeverity.HIGH),
            'OSError': (ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM),
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity."""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Check specific patterns
        for pattern, (category, severity) in self.error_patterns.items():
            if pattern in error_str or pattern in error_type:
                return category, severity
        
        # Context-based classification
        if context:
            if 'memory_usage' in context and context['memory_usage'] > 0.9:
                return ErrorCategory.MEMORY, ErrorSeverity.HIGH
            
            if 'gpu_memory' in context and context['gpu_memory'] > 0.95:
                return ErrorCategory.MEMORY, ErrorSeverity.CRITICAL
        
        # Default classification
        if isinstance(error, (RuntimeError, SystemError)):
            return ErrorCategory.SYSTEM, ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.DATA, ErrorSeverity.MEDIUM
        else:
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3):
        self.name = name
        self.max_attempts = max_attempts
        
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can recover from the error."""
        raise NotImplementedError
        
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt to recover from the error."""
        raise NotImplementedError


class MemoryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for memory-related errors."""
    
    def __init__(self):
        super().__init__("memory_recovery", max_attempts=3)
        
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.MEMORY
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt memory recovery."""
        try:
            logger.info(f"Attempting memory recovery for error: {error_info.error_id}")
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Clear any cached tensors in context
            if 'cached_tensors' in context:
                del context['cached_tensors']
                context['cached_tensors'] = {}
            
            # Reduce batch size if specified
            if 'batch_size' in context and context['batch_size'] > 1:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                logger.info(f"Reduced batch size to {context['batch_size']}")
            
            # Check available memory
            if torch.cuda.is_available():
                memory_stats = torch.cuda.memory_stats()
                allocated = memory_stats.get('allocated_bytes.all.current', 0)
                reserved = memory_stats.get('reserved_bytes.all.current', 0)
                
                if allocated > 0 or reserved > 0:
                    logger.info(f"Memory after cleanup - Allocated: {allocated / 1024**3:.2f}GB, "
                              f"Reserved: {reserved / 1024**3:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory recovery failed: {str(e)}")
            return False


class ComputationRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for computation errors."""
    
    def __init__(self):
        super().__init__("computation_recovery", max_attempts=3)
        
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.COMPUTATION
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt computation recovery."""
        try:
            logger.info(f"Attempting computation recovery for error: {error_info.error_id}")
            
            # Reset CUDA context if CUDA error
            if 'cuda' in error_info.message.lower() and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Reset current device
                    current_device = torch.cuda.current_device()
                    torch.cuda.set_device(current_device)
                except Exception as e:
                    logger.warning(f"CUDA context reset failed: {e}")
            
            # Adjust precision if specified
            if 'use_fp16' in context and context['use_fp16']:
                context['use_fp16'] = False
                context['use_fp32'] = True
                logger.info("Switched from FP16 to FP32 precision")
            
            # Reduce model complexity if possible
            if 'num_inference_steps' in context and context['num_inference_steps'] > 20:
                context['num_inference_steps'] = max(20, context['num_inference_steps'] // 2)
                logger.info(f"Reduced inference steps to {context['num_inference_steps']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Computation recovery failed: {str(e)}")
            return False


class IORecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for IO errors."""
    
    def __init__(self):
        super().__init__("io_recovery", max_attempts=5)
        
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.IO
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt IO recovery."""
        try:
            logger.info(f"Attempting IO recovery for error: {error_info.error_id}")
            
            # Wait and retry for temporary issues
            time.sleep(1.0)
            
            # Check disk space
            if 'output_path' in context:
                output_path = Path(context['output_path'])
                if output_path.exists():
                    free_space = psutil.disk_usage(output_path.parent).free
                    if free_space < 1024**3:  # Less than 1GB
                        logger.error("Insufficient disk space for recovery")
                        return False
            
            # Create directory if it doesn't exist
            if 'output_path' in context:
                output_path = Path(context['output_path'])
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"IO recovery failed: {str(e)}")
            return False


class NetworkRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for network errors."""
    
    def __init__(self):
        super().__init__("network_recovery", max_attempts=5)
        
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.NETWORK
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt network recovery."""
        try:
            logger.info(f"Attempting network recovery for error: {error_info.error_id}")
            
            # Exponential backoff
            attempt = context.get('recovery_attempt', 1)
            wait_time = min(2 ** attempt, 30)  # Max 30 seconds
            
            logger.info(f"Waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
            
            # Switch to offline mode if available
            if 'allow_offline' in context:
                context['offline_mode'] = True
                logger.info("Switched to offline mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Network recovery failed: {str(e)}")
            return False


class ResourceTracker:
    """Tracks and manages system resources."""
    
    def __init__(self):
        self.tracked_resources = {}
        self.resource_lock = threading.Lock()
        
    def register_resource(self, resource_id: str, resource: Any, cleanup_func: Callable = None):
        """Register a resource for tracking."""
        with self.resource_lock:
            self.tracked_resources[resource_id] = {
                'resource': weakref.ref(resource) if hasattr(resource, '__weakref__') else resource,
                'cleanup_func': cleanup_func,
                'created_at': time.time()
            }
    
    def cleanup_resource(self, resource_id: str) -> bool:
        """Clean up a specific resource."""
        with self.resource_lock:
            if resource_id in self.tracked_resources:
                resource_info = self.tracked_resources[resource_id]
                
                try:
                    # Call custom cleanup function if provided
                    if resource_info['cleanup_func']:
                        resource_info['cleanup_func']()
                    
                    # Generic cleanup for common types
                    resource = resource_info['resource']
                    if isinstance(resource, weakref.ref):
                        resource = resource()
                    
                    if resource is not None:
                        if hasattr(resource, 'close'):
                            resource.close()
                        elif hasattr(resource, '__del__'):
                            del resource
                    
                    del self.tracked_resources[resource_id]
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to cleanup resource {resource_id}: {e}")
                    return False
                    
        return False
    
    def cleanup_all_resources(self):
        """Clean up all tracked resources."""
        with self.resource_lock:
            resource_ids = list(self.tracked_resources.keys())
            
        for resource_id in resource_ids:
            self.cleanup_resource(resource_id)
            
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked resources."""
        with self.resource_lock:
            stats = {
                'total_resources': len(self.tracked_resources),
                'resource_types': {},
                'oldest_resource_age': 0.0
            }
            
            current_time = time.time()
            oldest_age = 0.0
            
            for resource_info in self.tracked_resources.values():
                # Count resource types
                resource = resource_info['resource']
                if isinstance(resource, weakref.ref):
                    resource = resource()
                    
                if resource is not None:
                    resource_type = type(resource).__name__
                    stats['resource_types'][resource_type] = stats['resource_types'].get(resource_type, 0) + 1
                
                # Track oldest resource
                age = current_time - resource_info['created_at']
                oldest_age = max(oldest_age, age)
            
            stats['oldest_resource_age'] = oldest_age
            
        return stats


class AdvancedErrorHandler:
    """Advanced error handler with recovery mechanisms."""
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.recovery_strategies = [
            MemoryRecoveryStrategy(),
            ComputationRecoveryStrategy(),
            IORecoveryStrategy(),
            NetworkRecoveryStrategy()
        ]
        self.resource_tracker = ResourceTracker()
        self.error_history = []
        self.error_patterns = {}
        self.recovery_stats = {}
        
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any] = None,
                    operation_name: str = "unknown") -> ErrorInfo:
        """Handle an error with recovery attempts."""
        if context is None:
            context = {}
            
        # Generate unique error ID
        error_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        # Classify error
        category, severity = self.error_classifier.classify_error(error, context)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=time.time(),
            error_type=type(error).__name__,
            category=category,
            severity=severity,
            message=str(error),
            traceback_str=traceback.format_exc(),
            context=context.copy()
        )
        
        # Add to error history
        self.error_history.append(error_info)
        
        # Attempt recovery if not critical
        if severity != ErrorSeverity.CRITICAL:
            self._attempt_recovery(error_info, context)
        
        # Update error patterns
        self._update_error_patterns(error_info)
        
        # Log error
        self._log_error(error_info)
        
        return error_info
    
    def _attempt_recovery(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt to recover from the error."""
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_info):
                try:
                    error_info.recovery_attempted = True
                    error_info.recovery_strategy = strategy.name
                    
                    # Set recovery attempt number
                    context['recovery_attempt'] = context.get('recovery_attempt', 0) + 1
                    
                    if context['recovery_attempt'] > strategy.max_attempts:
                        logger.warning(f"Max recovery attempts ({strategy.max_attempts}) "
                                     f"exceeded for strategy {strategy.name}")
                        continue
                    
                    success = strategy.recover(error_info, context)
                    error_info.recovery_successful = success
                    
                    # Update recovery stats
                    strategy_stats = self.recovery_stats.setdefault(strategy.name, {
                        'attempts': 0, 'successes': 0, 'failures': 0
                    })
                    strategy_stats['attempts'] += 1
                    
                    if success:
                        strategy_stats['successes'] += 1
                        logger.info(f"Recovery successful using strategy: {strategy.name}")
                        return True
                    else:
                        strategy_stats['failures'] += 1
                        
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                    
        return False
    
    def _update_error_patterns(self, error_info: ErrorInfo):
        """Update error pattern analysis."""
        pattern_key = f"{error_info.category.value}_{error_info.error_type}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                'count': 0,
                'first_seen': error_info.timestamp,
                'last_seen': error_info.timestamp,
                'recovery_success_rate': 0.0
            }
        
        pattern = self.error_patterns[pattern_key]
        pattern['count'] += 1
        pattern['last_seen'] = error_info.timestamp
        
        # Update success rate
        if error_info.recovery_attempted:
            # Calculate new success rate
            total_attempts = sum(1 for e in self.error_history 
                               if f"{e.category.value}_{e.error_type}" == pattern_key 
                               and e.recovery_attempted)
            
            successful_attempts = sum(1 for e in self.error_history 
                                    if f"{e.category.value}_{e.error_type}" == pattern_key 
                                    and e.recovery_successful)
            
            if total_attempts > 0:
                pattern['recovery_success_rate'] = successful_attempts / total_attempts
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = (f"Error {error_info.error_id}: {error_info.message} "
                      f"[{error_info.category.value}/{error_info.severity.value}]")
        
        if error_info.recovery_attempted:
            recovery_status = "successful" if error_info.recovery_successful else "failed"
            log_message += f" (Recovery {recovery_status})"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    @contextmanager
    def error_context(self, operation_name: str, context: Dict[str, Any] = None):
        """Context manager for error handling."""
        if context is None:
            context = {}
            
        # Add system context
        context.update({
            'memory_usage': psutil.virtual_memory().percent / 100.0,
            'cpu_usage': psutil.cpu_percent(),
            'operation_name': operation_name
        })
        
        if torch.cuda.is_available():
            context['gpu_memory'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        try:
            yield context
        except Exception as e:
            error_info = self.handle_error(e, context, operation_name)
            
            # If recovery was successful, continue; otherwise re-raise
            if not error_info.recovery_successful:
                raise
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {'total_errors': 0}
        
        total_errors = len(self.error_history)
        
        # Category breakdown
        category_counts = {}
        severity_counts = {}
        
        for error_info in self.error_history:
            category_counts[error_info.category.value] = category_counts.get(
                error_info.category.value, 0) + 1
            severity_counts[error_info.severity.value] = severity_counts.get(
                error_info.severity.value, 0) + 1
        
        # Recovery statistics
        recovery_attempted = sum(1 for e in self.error_history if e.recovery_attempted)
        recovery_successful = sum(1 for e in self.error_history if e.recovery_successful)
        
        recovery_rate = recovery_successful / recovery_attempted if recovery_attempted > 0 else 0.0
        
        # Recent error trends
        recent_time = time.time() - 3600  # Last hour
        recent_errors = [e for e in self.error_history if e.timestamp > recent_time]
        
        return {
            'total_errors': total_errors,
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'recovery_attempted': recovery_attempted,
            'recovery_successful': recovery_successful,
            'recovery_rate': recovery_rate,
            'recent_errors': len(recent_errors),
            'error_patterns': self.error_patterns,
            'recovery_strategy_stats': self.recovery_stats,
            'resource_stats': self.resource_tracker.get_resource_stats()
        }
    
    def generate_error_report(self) -> str:
        """Generate human-readable error report."""
        stats = self.get_error_statistics()
        
        if stats['total_errors'] == 0:
            return "No errors recorded."
        
        report = ["=== ADVANCED ERROR HANDLING REPORT ===\n"]
        
        # Summary
        report.append(f"Total Errors: {stats['total_errors']}")
        report.append(f"Recovery Rate: {stats['recovery_rate']:.1%}")
        report.append(f"Recent Errors (1h): {stats['recent_errors']}")
        report.append("")
        
        # Category breakdown
        report.append("Error Categories:")
        for category, count in stats['category_breakdown'].items():
            percentage = count / stats['total_errors'] * 100
            report.append(f"  {category.title()}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Severity breakdown
        report.append("Error Severity:")
        for severity, count in stats['severity_breakdown'].items():
            percentage = count / stats['total_errors'] * 100
            report.append(f"  {severity.title()}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Recovery strategies
        if stats['recovery_strategy_stats']:
            report.append("Recovery Strategy Performance:")
            for strategy, strategy_stats in stats['recovery_strategy_stats'].items():
                success_rate = (strategy_stats['successes'] / 
                               strategy_stats['attempts'] * 100 
                               if strategy_stats['attempts'] > 0 else 0)
                report.append(f"  {strategy}: {success_rate:.1f}% "
                            f"({strategy_stats['successes']}/{strategy_stats['attempts']})")
        
        return "\n".join(report)
    
    def cleanup_all_resources(self):
        """Clean up all tracked resources."""
        self.resource_tracker.cleanup_all_resources()


# Decorator for automatic error handling
def with_error_handling(operation_name: str = None, 
                       context: Dict[str, Any] = None,
                       handler: AdvancedErrorHandler = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal handler, operation_name, context
            
            if handler is None:
                handler = AdvancedErrorHandler()
            
            if operation_name is None:
                operation_name = func.__name__
                
            if context is None:
                context = {}
            
            with handler.error_context(operation_name, context):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Example usage of advanced error handling
    handler = AdvancedErrorHandler()
    
    # Test memory error handling
    @with_error_handling("test_operation", handler=handler)
    def test_memory_operation():
        # Simulate memory error
        raise RuntimeError("CUDA out of memory")
    
    try:
        test_memory_operation()
    except Exception as e:
        print(f"Operation failed after recovery attempts: {e}")
    
    # Print error statistics
    print(handler.generate_error_report())