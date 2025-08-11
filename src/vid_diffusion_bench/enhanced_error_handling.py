"""Enhanced error handling and validation for video diffusion benchmarking.

This module provides comprehensive error handling, recovery mechanisms,
and validation to make the benchmarking suite robust and reliable.
"""

import logging
import functools
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Base exception for benchmark-related errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.error_code = error_code or "BENCHMARK_ERROR"
        self.details = details or {}
        self.timestamp = time.time()


class ModelLoadError(BenchmarkError):
    """Error loading or initializing a model."""
    
    def __init__(self, model_name: str, message: str):
        super().__init__(
            f"Failed to load model '{model_name}': {message}",
            "MODEL_LOAD_ERROR",
            {"model_name": model_name}
        )


class ValidationError(BenchmarkError):
    """Data validation error."""
    
    def __init__(self, field: str, value: Any, expected: str):
        super().__init__(
            f"Validation failed for '{field}': got {value}, expected {expected}",
            "VALIDATION_ERROR",
            {"field": field, "value": str(value), "expected": expected}
        )


class ResourceError(BenchmarkError):
    """Resource availability error (GPU, memory, etc.)."""
    
    def __init__(self, resource_type: str, required: str, available: str):
        super().__init__(
            f"Insufficient {resource_type}: required {required}, available {available}",
            "RESOURCE_ERROR",
            {"resource_type": resource_type, "required": required, "available": available}
        )


class MetricComputationError(BenchmarkError):
    """Error computing evaluation metrics."""
    
    def __init__(self, metric_name: str, message: str):
        super().__init__(
            f"Failed to compute metric '{metric_name}': {message}",
            "METRIC_ERROR",
            {"metric_name": metric_name}
        )


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to retry function calls on failure."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def handle_benchmark_errors(func: Callable) -> Callable:
    """Decorator for comprehensive error handling in benchmark functions."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BenchmarkError:
            # Re-raise our custom errors
            raise
        except ImportError as e:
            raise BenchmarkError(
                f"Missing dependency in {func.__name__}: {e}",
                "DEPENDENCY_ERROR",
                {"function": func.__name__, "missing_dependency": str(e)}
            )
        except FileNotFoundError as e:
            raise BenchmarkError(
                f"Required file not found in {func.__name__}: {e}",
                "FILE_NOT_FOUND",
                {"function": func.__name__, "file_path": str(e)}
            )
        except MemoryError as e:
            raise ResourceError("memory", "more", "insufficient")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise BenchmarkError(
                f"Unexpected error in {func.__name__}: {e}",
                "UNEXPECTED_ERROR",
                {"function": func.__name__, "original_error": str(e)}
            )
    
    return wrapper


class InputValidator:
    """Comprehensive input validation for benchmark parameters."""
    
    @staticmethod
    def validate_model_name(model_name: str) -> str:
        """Validate model name format and security."""
        if not isinstance(model_name, str):
            raise ValidationError("model_name", type(model_name), "str")
        
        if not model_name.strip():
            raise ValidationError("model_name", model_name, "non-empty string")
        
        # Security: prevent path traversal and injection
        if any(char in model_name for char in ['/', '\\', '..', ';', '|', '&']):
            raise ValidationError("model_name", model_name, "safe model name without special characters")
        
        return model_name.strip()
    
    @staticmethod
    def validate_prompts(prompts: Union[str, List[str]]) -> List[str]:
        """Validate and normalize prompt inputs."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if not isinstance(prompts, list):
            raise ValidationError("prompts", type(prompts), "str or List[str]")
        
        if not prompts:
            raise ValidationError("prompts", "empty list", "non-empty list")
        
        validated_prompts = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise ValidationError(f"prompts[{i}]", type(prompt), "str")
            
            prompt = prompt.strip()
            if not prompt:
                raise ValidationError(f"prompts[{i}]", "empty string", "non-empty string")
            
            # Security: limit prompt length to prevent abuse
            if len(prompt) > 1000:
                raise ValidationError(f"prompts[{i}]", f"{len(prompt)} characters", "<=1000 characters")
            
            validated_prompts.append(prompt)
        
        return validated_prompts
    
    @staticmethod
    def validate_resolution(resolution: tuple) -> tuple:
        """Validate video resolution parameters."""
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValidationError("resolution", resolution, "tuple of (width, height)")
        
        width, height = resolution
        if not all(isinstance(x, int) and x > 0 for x in [width, height]):
            raise ValidationError("resolution", resolution, "positive integers")
        
        # Reasonable limits for video generation
        if width > 4096 or height > 4096:
            raise ValidationError("resolution", resolution, "width and height <= 4096")
        
        if width < 64 or height < 64:
            raise ValidationError("resolution", resolution, "width and height >= 64")
        
        return (width, height)
    
    @staticmethod
    def validate_num_frames(num_frames: int) -> int:
        """Validate number of frames parameter."""
        if not isinstance(num_frames, int):
            raise ValidationError("num_frames", type(num_frames), "int")
        
        if num_frames <= 0:
            raise ValidationError("num_frames", num_frames, "positive integer")
        
        if num_frames > 512:  # Reasonable upper limit
            raise ValidationError("num_frames", num_frames, "<=512 frames")
        
        return num_frames
    
    @staticmethod
    def validate_fps(fps: Union[int, float]) -> float:
        """Validate frames per second parameter."""
        if not isinstance(fps, (int, float)):
            raise ValidationError("fps", type(fps), "int or float")
        
        if fps <= 0:
            raise ValidationError("fps", fps, "positive number")
        
        if fps > 120:  # Reasonable upper limit
            raise ValidationError("fps", fps, "<=120 FPS")
        
        return float(fps)


class ResourceMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.initial_state = self._get_resource_state()
    
    def _get_resource_state(self) -> Dict[str, Any]:
        """Get current system resource state."""
        import psutil
        
        state = {
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "disk_usage_percent": psutil.disk_usage('/').percent,
        }
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                state.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "gpu_count": torch.cuda.device_count(),
                })
        except ImportError:
            pass
        
        return state
    
    def check_resources(self, requirements: Dict[str, Any]) -> None:
        """Check if system has required resources."""
        current_state = self._get_resource_state()
        
        # Memory check
        if "memory_gb" in requirements:
            required_gb = requirements["memory_gb"]
            available_gb = current_state["memory_available_gb"]
            if available_gb < required_gb:
                raise ResourceError("memory", f"{required_gb}GB", f"{available_gb:.1f}GB")
        
        # GPU memory check
        if "gpu_memory_gb" in requirements and "gpu_memory_reserved_gb" in current_state:
            required_gpu_gb = requirements["gpu_memory_gb"]
            # Estimate available GPU memory (total - reserved)
            try:
                import torch
                total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                reserved_gb = current_state["gpu_memory_reserved_gb"]
                available_gpu_gb = total_gpu_gb - reserved_gb
                
                if available_gpu_gb < required_gpu_gb:
                    raise ResourceError(
                        "GPU memory", 
                        f"{required_gpu_gb}GB", 
                        f"{available_gpu_gb:.1f}GB"
                    )
            except ImportError:
                pass
    
    def get_resource_usage_delta(self) -> Dict[str, float]:
        """Get change in resource usage since initialization."""
        current_state = self._get_resource_state()
        delta = {}
        
        for key, current_value in current_state.items():
            if key in self.initial_state:
                initial_value = self.initial_state[key]
                delta[f"delta_{key}"] = current_value - initial_value
        
        return delta


@contextmanager
def error_context(operation: str, **context_data):
    """Context manager for enriched error reporting."""
    try:
        logger.debug(f"Starting operation: {operation}")
        yield
        logger.debug(f"Completed operation: {operation}")
    except Exception as e:
        # Enrich the error with context
        if isinstance(e, BenchmarkError):
            e.details.update(context_data)
            e.details["operation"] = operation
        
        logger.error(f"Error in operation '{operation}': {e}")
        if context_data:
            logger.error(f"Context: {context_data}")
        
        raise


class ErrorReportGenerator:
    """Generate comprehensive error reports for debugging."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./error_reports")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self, error: BenchmarkError, **additional_context) -> Path:
        """Generate a detailed error report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"error_report_{timestamp}.json"
        
        report_data = {
            "timestamp": timestamp,
            "error_type": type(error).__name__,
            "error_code": getattr(error, 'error_code', 'UNKNOWN'),
            "message": str(error),
            "details": getattr(error, 'details', {}),
            "additional_context": additional_context,
            "traceback": traceback.format_exc(),
        }
        
        # Add system information
        try:
            import psutil
            report_data["system_info"] = {
                "python_version": __import__('sys').version,
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "platform": __import__('platform').platform(),
            }
            
            # Add GPU information if available
            try:
                import torch
                if torch.cuda.is_available():
                    report_data["gpu_info"] = {
                        "gpu_count": torch.cuda.device_count(),
                        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        "cuda_version": torch.version.cuda,
                    }
            except ImportError:
                pass
            
        except ImportError:
            pass
        
        # Write report to file
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Error report generated: {report_file}")
        return report_file


# Global error handler configuration
def setup_global_error_handling(log_level: str = "INFO", report_dir: Path = None):
    """Setup global error handling configuration."""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create global error reporter
    global _global_error_reporter
    _global_error_reporter = ErrorReportGenerator(report_dir)
    
    # Install global exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't catch keyboard interrupt
            __import__('sys').__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        if isinstance(exc_value, BenchmarkError):
            _global_error_reporter.generate_report(exc_value)
    
    __import__('sys').excepthook = handle_exception


# Global error reporter instance
_global_error_reporter = None


def get_error_reporter() -> ErrorReportGenerator:
    """Get the global error reporter instance."""
    global _global_error_reporter
    if _global_error_reporter is None:
        _global_error_reporter = ErrorReportGenerator()
    return _global_error_reporter


# Validation decorators for common use cases
def validate_inputs(validator_func: Callable):
    """Decorator to validate function inputs using a validator function."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run validation
            validator_func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_gpu(func: Callable) -> Callable:
    """Decorator to ensure GPU is available for the function."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import torch
            if not torch.cuda.is_available():
                raise ResourceError("GPU", "CUDA-capable GPU", "none available")
        except ImportError:
            raise BenchmarkError(
                "PyTorch not available, cannot check GPU",
                "DEPENDENCY_ERROR"
            )
        
        return func(*args, **kwargs)
    
    return wrapper