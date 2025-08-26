"""Circuit breaker pattern implementation for robust error handling."""

import time
import threading
import logging
from enum import Enum
from typing import Callable, Any, Dict, Optional
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before trying recovery
    success_threshold: int = 2          # Successes before closing
    timeout: float = 10.0               # Call timeout in seconds


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, last_failure: str = ""):
        self.service_name = service_name
        self.last_failure = last_failure
        super().__init__(f"Circuit breaker OPEN for {service_name}. Last failure: {last_failure}")


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_failure_reason = ""
        
        # Threading protection
        self.lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(self.name, self.last_failure_reason)
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure(str(e))
                raise
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _transition_to_half_open(self):
        """Transition to half-open state for testing recovery."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
    
    def _on_success(self):
        """Handle successful function execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            logger.debug(f"Circuit breaker '{self.name}' success {self.success_count}/{self.config.success_threshold}")
            
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self, error: str):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.last_failure_reason = error
        
        logger.warning(f"Circuit breaker '{self.name}' failure {self.failure_count}/{self.config.failure_threshold}: {error}")
        
        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_closed(self):
        """Transition to closed state (normal operation)."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def _transition_to_open(self):
        """Transition to open state (rejecting calls)."""
        logger.error(f"Circuit breaker '{self.name}' transitioning to OPEN")
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_failure_reason": self.last_failure_reason,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self.lock:
            logger.info(f"Circuit breaker '{self.name}' manually reset")
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0
            self.last_failure_reason = ""


class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()
    
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self.lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config)
            return self.breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        with self.lock:
            return {name: breaker.status for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self.lock:
            for breaker in self.breakers.values():
                breaker.reset()


# Global registry instance
_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker to a function."""
    def decorator(func: Callable) -> Callable:
        breaker = _registry.get_or_create(name, config)
        return breaker(func)
    return decorator


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get circuit breaker instance."""
    return _registry.get_or_create(name, config)


def get_all_circuit_breakers_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    return _registry.get_all_status()


@contextmanager
def circuit_breaker_context(name: str, config: CircuitBreakerConfig = None):
    """Context manager for circuit breaker protection."""
    breaker = _registry.get_or_create(name, config)
    
    with breaker.lock:
        if breaker.state == CircuitBreakerState.OPEN:
            if breaker._should_attempt_recovery():
                breaker._transition_to_half_open()
            else:
                raise CircuitBreakerError(name, breaker.last_failure_reason)
    
    try:
        yield
        breaker._on_success()
    except Exception as e:
        breaker._on_failure(str(e))
        raise


# Example usage and testing
if __name__ == "__main__":
    import secrets
    
    # Example: Model loading with circuit breaker
    @circuit_breaker("model_loader", CircuitBreakerConfig(failure_threshold=3))
    def load_model(model_name: str):
        """Example model loading function."""
        if secrets.SystemRandom().random() < 0.7:  # 70% failure rate for testing
            raise Exception(f"Failed to load {model_name}")
        return f"Model {model_name} loaded successfully"
    
    # Test circuit breaker behavior
    for i in range(10):
        try:
            result = load_model("test-model")
            print(f"Attempt {i+1}: {result}")
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
        
        # Print circuit breaker status
        status = get_all_circuit_breakers_status()
        print(f"Circuit breaker state: {status['model_loader']['state']}")
        print("-" * 50)
        
        time.sleep(1)