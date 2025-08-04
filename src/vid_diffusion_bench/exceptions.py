"""Custom exceptions for the Video Diffusion Benchmark Suite."""

from typing import Optional, Dict, Any, List


class VidBenchError(Exception):
    """Base exception for all vid-bench errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ModelError(VidBenchError):
    """Errors related to model operations."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    
    def __init__(self, model_name: str, available_models: Optional[List[str]] = None):
        self.model_name = model_name
        self.available_models = available_models or []
        
        message = f"Model '{model_name}' not found"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
            
        details = {
            "model_name": model_name,
            "available_models": self.available_models
        }
        
        super().__init__(message, details)


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    
    def __init__(self, model_name: str, cause: Exception):
        self.model_name = model_name
        self.cause = cause
        
        message = f"Failed to load model '{model_name}': {str(cause)}"
        details = {
            "model_name": model_name,
            "cause_type": type(cause).__name__,
            "cause_message": str(cause)
        }
        
        super().__init__(message, details)


class InferenceError(ModelError):
    """Raised when model inference fails."""
    
    def __init__(self, model_name: str, prompt: str, cause: Exception):
        self.model_name = model_name
        self.prompt = prompt
        self.cause = cause
        
        message = f"Inference failed for model '{model_name}': {str(cause)}"
        details = {
            "model_name": model_name,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "cause_type": type(cause).__name__,
            "cause_message": str(cause)
        }
        
        super().__init__(message, details)


class ResourceError(VidBenchError):
    """Errors related to system resources."""
    pass


class InsufficientMemoryError(ResourceError):
    """Raised when there's insufficient memory for operation."""
    
    def __init__(self, required_gb: float, available_gb: float, resource_type: str = "VRAM"):
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.resource_type = resource_type
        
        message = f"Insufficient {resource_type}: required {required_gb:.1f}GB, available {available_gb:.1f}GB"
        details = {
            "required_gb": required_gb,
            "available_gb": available_gb,
            "resource_type": resource_type
        }
        
        super().__init__(message, details)


class GPUNotAvailableError(ResourceError):
    """Raised when GPU is required but not available."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        
        if model_name:
            message = f"GPU required for model '{model_name}' but not available"
            details = {"model_name": model_name}
        else:
            message = "GPU required but not available"
            details = {}
            
        super().__init__(message, details)


class ValidationError(VidBenchError):
    """Errors related to input validation."""
    pass


class InvalidPromptError(ValidationError):
    """Raised when a prompt is invalid."""
    
    def __init__(self, prompt: str, reason: str):
        self.prompt = prompt
        self.reason = reason
        
        message = f"Invalid prompt: {reason}"
        details = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "reason": reason
        }
        
        super().__init__(message, details)


class InvalidConfigError(ValidationError):
    """Raised when benchmark configuration is invalid."""
    
    def __init__(self, config_key: str, config_value: Any, reason: str):
        self.config_key = config_key
        self.config_value = config_value
        self.reason = reason
        
        message = f"Invalid configuration '{config_key}': {reason}"
        details = {
            "config_key": config_key,
            "config_value": config_value,
            "reason": reason
        }
        
        super().__init__(message, details)


class MetricsError(VidBenchError):
    """Errors related to metrics computation."""
    pass


class MetricsComputationError(MetricsError):
    """Raised when metrics computation fails."""
    
    def __init__(self, metric_name: str, cause: Exception):
        self.metric_name = metric_name
        self.cause = cause
        
        message = f"Failed to compute {metric_name}: {str(cause)}"
        details = {
            "metric_name": metric_name,
            "cause_type": type(cause).__name__,
            "cause_message": str(cause)
        }
        
        super().__init__(message, details)


class ReferenceDataError(MetricsError):
    """Raised when reference data cannot be loaded."""
    
    def __init__(self, dataset_name: str, cause: Exception):
        self.dataset_name = dataset_name
        self.cause = cause
        
        message = f"Failed to load reference data '{dataset_name}': {str(cause)}"
        details = {
            "dataset_name": dataset_name,
            "cause_type": type(cause).__name__,
            "cause_message": str(cause)
        }
        
        super().__init__(message, details)


class DatabaseError(VidBenchError):
    """Errors related to database operations."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, database_url: str, cause: Exception):
        self.database_url = database_url
        self.cause = cause
        
        # Hide sensitive connection details
        safe_url = database_url.split('@')[-1] if '@' in database_url else database_url
        
        message = f"Database connection failed: {str(cause)}"
        details = {
            "database_url": safe_url,
            "cause_type": type(cause).__name__,
            "cause_message": str(cause)
        }
        
        super().__init__(message, details)


class BenchmarkError(VidBenchError):
    """Errors related to benchmark execution."""
    pass


class BenchmarkTimeoutError(BenchmarkError):
    """Raised when benchmark times out."""
    
    def __init__(self, model_name: str, timeout_minutes: int):
        self.model_name = model_name
        self.timeout_minutes = timeout_minutes
        
        message = f"Benchmark timed out after {timeout_minutes} minutes for model '{model_name}'"
        details = {
            "model_name": model_name,
            "timeout_minutes": timeout_minutes
        }
        
        super().__init__(message, details)


class BenchmarkAbortedError(BenchmarkError):
    """Raised when benchmark is aborted."""
    
    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        
        message = f"Benchmark aborted for model '{model_name}': {reason}"
        details = {
            "model_name": model_name,
            "reason": reason
        }
        
        super().__init__(message, details)


class APIError(VidBenchError):
    """Errors related to API operations."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, reason: str = "Invalid credentials"):
        message = f"Authentication failed: {reason}"
        details = {"reason": reason}
        super().__init__(message, details)


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, action: str, resource: Optional[str] = None):
        self.action = action
        self.resource = resource
        
        if resource:
            message = f"Not authorized to {action} {resource}"
            details = {"action": action, "resource": resource}
        else:
            message = f"Not authorized to {action}"
            details = {"action": action}
            
        super().__init__(message, details)


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window_seconds: int, retry_after: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        
        message = f"Rate limit exceeded: {limit} requests per {window_seconds}s. Retry after {retry_after}s"
        details = {
            "limit": limit,
            "window_seconds": window_seconds,
            "retry_after": retry_after
        }
        
        super().__init__(message, details)


# Exception handling utilities

def handle_model_errors(func):
    """Decorator to handle common model errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            if "diffusers" in str(e) or "transformers" in str(e):
                raise ModelLoadError("unknown", e) from e
            raise
        except torch.cuda.OutOfMemoryError as e:
            # Extract memory info if possible
            raise InsufficientMemoryError(
                required_gb=0,  # Unknown
                available_gb=0,  # Unknown
                resource_type="VRAM"
            ) from e
        except Exception as e:
            # Wrap unexpected errors
            if hasattr(args[0], 'name'):
                model_name = args[0].name
            else:
                model_name = "unknown"
            raise ModelError(f"Unexpected error in {func.__name__}: {str(e)}", {
                "function": func.__name__,
                "model_name": model_name,
                "original_error": str(e)
            }) from e
    return wrapper


def handle_inference_errors(model_name: str, prompt: str):
    """Decorator to handle inference errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                raise InsufficientMemoryError(
                    required_gb=0,  # Unknown
                    available_gb=0,  # Unknown
                    resource_type="VRAM"
                ) from e
            except Exception as e:
                raise InferenceError(model_name, prompt, e) from e
        return wrapper
    return decorator


def handle_metrics_errors(metric_name: str):
    """Decorator to handle metrics computation errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise MetricsComputationError(metric_name, e) from e
        return wrapper
    return decorator