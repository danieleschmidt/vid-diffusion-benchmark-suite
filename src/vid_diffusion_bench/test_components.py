"""Test components without heavy dependencies."""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class MockCircuitBreakerConfig:
    """Mock circuit breaker configuration for testing."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2
    timeout: float = 10.0


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""
    
    def __init__(self, name: str, config: MockCircuitBreakerConfig = None):
        self.name = name
        self.config = config or MockCircuitBreakerConfig()
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "closed"
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get status for testing."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout
            }
        }


@dataclass
class MockHealthData:
    """Mock health data for testing."""
    cpu_percent: float = 50.0
    memory_percent: float = 60.0
    disk_usage_percent: float = 40.0
    alerts: List[str] = None
    
    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class MockHealthMonitor:
    """Mock health monitor for testing."""
    
    def __init__(self):
        self.monitoring = False
    
    def check_health(self) -> MockHealthData:
        """Mock health check."""
        return MockHealthData()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Mock health summary."""
        return {
            "cpu": {"current": 50.0, "average": 45.0},
            "memory": {"current": 60.0, "available_gb": 8.0},
            "disk": {"usage_percent": 40.0, "free_gb": 100.0}
        }


@dataclass
class MockResilientConfig:
    """Mock resilient configuration."""
    max_retries: int = 3
    auto_recovery: bool = True
    health_check_enabled: bool = True


class MockResilientBenchmarkSuite:
    """Mock resilient benchmark suite for testing."""
    
    def __init__(self, config: MockResilientConfig = None):
        self.config = config or MockResilientConfig()
    
    def evaluate_model_resilient(self, model_name: str, prompts: List[str]) -> Dict[str, Any]:
        """Mock resilient evaluation."""
        return {
            "model_name": model_name,
            "total_prompts": len(prompts),
            "successful_prompts": len(prompts),
            "failed_prompts": 0
        }


@dataclass 
class MockScalingConfig:
    """Mock scaling configuration."""
    min_workers: int = 1
    max_workers: int = 8
    target_cpu_utilization: float = 70.0


class MockAdaptiveScaler:
    """Mock adaptive scaler for testing."""
    
    def __init__(self, config: MockScalingConfig = None):
        self.config = config or MockScalingConfig()
        self.current_workers = self.config.min_workers
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Mock scaling status."""
        return {
            "current_workers": self.current_workers,
            "target_workers": self.current_workers,
            "total_tasks": 0,
            "success_rate": 1.0
        }


class MockPerformanceAccelerator:
    """Mock performance accelerator for testing."""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
    
    def accelerate_function(self, enable_cache: bool = True):
        """Mock function acceleration decorator."""
        def decorator(func):
            return func  # Just return function unchanged for testing
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Mock performance summary."""
        return {
            "total_operations": 0,
            "averages": {
                "duration": 1.0,
                "throughput": 1.0,
                "cache_hit_rate": 0.8
            }
        }


class MockIntelligentCache:
    """Mock intelligent cache for testing."""
    
    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def put(self, key: str, value: Any):
        """Put value in mock cache."""
        self.cache[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from mock cache."""
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache stats."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total)
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "memory_entries": len(self.cache)
        }