"""Resilient benchmark execution with fault tolerance and recovery."""

import time
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
import json
import traceback
from datetime import datetime, timedelta

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, circuit_breaker
from .health_monitoring import HealthMonitor, get_health_monitor
from .enhanced_error_handling import retry_on_failure, BenchmarkError
from .benchmark import BenchmarkSuite, BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkAttempt:
    """Record of a benchmark attempt."""
    timestamp: float
    model_name: str
    prompt: str
    attempt_number: int
    success: bool
    duration: Optional[float] = None
    error: Optional[str] = None
    health_before: Optional[Dict] = None
    health_after: Optional[Dict] = None
    recovery_action: Optional[str] = None


@dataclass
class ResilientConfig:
    """Configuration for resilient benchmarking."""
    max_retries: int = 3
    retry_delay: float = 5.0
    backoff_multiplier: float = 2.0
    timeout_seconds: float = 300.0
    health_check_enabled: bool = True
    circuit_breaker_enabled: bool = True
    auto_recovery: bool = True
    save_failed_attempts: bool = True
    parallel_models: int = 1
    memory_cleanup_threshold: float = 0.9  # GPU memory usage threshold for cleanup


class ResilientBenchmarkSuite:
    """Fault-tolerant benchmark suite with recovery mechanisms."""
    
    def __init__(self, config: ResilientConfig = None):
        self.config = config or ResilientConfig()
        self.base_suite = BenchmarkSuite()
        self.health_monitor = get_health_monitor()
        self.attempt_history: List[BenchmarkAttempt] = []
        self.recovery_strategies = {}
        self.lock = threading.Lock()
        
        # Setup circuit breakers for each operation
        self._setup_circuit_breakers()
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
        logger.info("Resilient benchmark suite initialized")
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for different operations."""
        self.circuit_breakers = {
            "model_load": CircuitBreaker(
                "model_load",
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0)
            ),
            "video_generation": CircuitBreaker(
                "video_generation", 
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
            ),
            "metric_computation": CircuitBreaker(
                "metric_computation",
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout=20.0)
            )
        }
    
    def _register_recovery_strategies(self):
        """Register recovery strategies for different error types."""
        self.recovery_strategies = {
            "cuda_out_of_memory": self._recover_from_oom,
            "model_load_failure": self._recover_from_model_failure,
            "timeout": self._recover_from_timeout,
            "generic": self._generic_recovery
        }
    
    @retry_on_failure(max_attempts=3, delay=2.0, backoff_factor=2.0)
    def evaluate_model_resilient(
        self, 
        model_name: str, 
        prompts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate model with full resilience and recovery."""
        
        # Pre-execution health check
        if self.config.health_check_enabled:
            health_before = self.health_monitor.check_health()
            if health_before.alerts:
                logger.warning(f"Health alerts before evaluation: {health_before.alerts}")
        else:
            health_before = None
        
        results = {
            "model_name": model_name,
            "total_prompts": len(prompts),
            "successful_prompts": 0,
            "failed_prompts": 0,
            "attempts": [],
            "overall_metrics": {},
            "health_summary": {},
            "recovery_actions": []
        }
        
        successful_results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Evaluating prompt {i+1}/{len(prompts)} for {model_name}")
            
            attempt_result = self._evaluate_single_prompt_resilient(
                model_name, prompt, i+1, health_before
            )
            
            results["attempts"].append(asdict(attempt_result))
            
            if attempt_result.success:
                results["successful_prompts"] += 1
                # In a real implementation, you'd collect the actual results here
                successful_results.append(attempt_result)
            else:
                results["failed_prompts"] += 1
                
                # Apply recovery if needed
                if self.config.auto_recovery and attempt_result.error:
                    recovery_action = self._apply_recovery(attempt_result.error)
                    if recovery_action:
                        results["recovery_actions"].append(recovery_action)
        
        # Compute overall metrics from successful results
        if successful_results:
            results["overall_metrics"] = self._compute_aggregate_metrics(successful_results)
        
        # Post-execution health check
        if self.config.health_check_enabled:
            health_after = self.health_monitor.check_health()
            results["health_summary"] = {
                "before": asdict(health_before) if health_before else None,
                "after": asdict(health_after),
                "degradation": self._compute_health_degradation(health_before, health_after)
            }
        
        return results
    
    def _evaluate_single_prompt_resilient(
        self, 
        model_name: str, 
        prompt: str, 
        prompt_num: int,
        health_before: Any
    ) -> BenchmarkAttempt:
        """Evaluate single prompt with resilience."""
        
        for attempt in range(1, self.config.max_retries + 1):
            start_time = time.time()
            
            attempt_record = BenchmarkAttempt(
                timestamp=start_time,
                model_name=model_name,
                prompt=prompt,
                attempt_number=attempt,
                success=False,
                health_before=asdict(health_before) if health_before else None
            )
            
            try:
                # Use circuit breaker for model operations
                with self.circuit_breakers["video_generation"]:
                    # In real implementation, this would call the actual benchmark
                    result = self._simulate_benchmark_execution(model_name, prompt)
                    
                duration = time.time() - start_time
                attempt_record.duration = duration
                attempt_record.success = True
                
                logger.info(f"✅ {model_name} prompt {prompt_num} succeeded on attempt {attempt}")
                
                with self.lock:
                    self.attempt_history.append(attempt_record)
                
                return attempt_record
                
            except Exception as e:
                duration = time.time() - start_time
                error_str = str(e)
                
                attempt_record.duration = duration
                attempt_record.error = error_str
                
                logger.warning(f"❌ {model_name} prompt {prompt_num} failed on attempt {attempt}: {error_str}")
                
                # Determine recovery action
                recovery_action = self._determine_recovery_action(error_str)
                attempt_record.recovery_action = recovery_action
                
                with self.lock:
                    self.attempt_history.append(attempt_record)
                
                # Apply recovery if not the last attempt
                if attempt < self.config.max_retries:
                    if recovery_action and self.config.auto_recovery:
                        self._apply_recovery_action(recovery_action)
                    
                    # Wait before retry with exponential backoff
                    delay = self.config.retry_delay * (self.config.backoff_multiplier ** (attempt - 1))
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(f"❌ {model_name} prompt {prompt_num} failed after {attempt} attempts")
        
        return attempt_record
    
    def _simulate_benchmark_execution(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Simulate benchmark execution (replace with actual implementation)."""
        # This is a placeholder - in real implementation, call self.base_suite.evaluate_model
        import random
        
        # Simulate various failure scenarios
        rand = random.random()
        if rand < 0.1:  # 10% CUDA OOM
            raise RuntimeError("CUDA out of memory")
        elif rand < 0.15:  # 5% model load failure
            raise RuntimeError("Failed to load model")
        elif rand < 0.2:  # 5% timeout
            raise TimeoutError("Model generation timed out")
        
        # Simulate processing time
        time.sleep(0.5)
        
        return {
            "fvd": random.uniform(80, 120),
            "latency": random.uniform(2, 10),
            "vram_gb": random.uniform(8, 32)
        }
    
    def _determine_recovery_action(self, error: str) -> Optional[str]:
        """Determine appropriate recovery action for error."""
        error_lower = error.lower()
        
        if "cuda out of memory" in error_lower or "out of memory" in error_lower:
            return "cuda_out_of_memory"
        elif "failed to load" in error_lower or "model" in error_lower:
            return "model_load_failure"
        elif "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        else:
            return "generic"
    
    def _apply_recovery_action(self, recovery_action: str):
        """Apply specific recovery action."""
        if recovery_action in self.recovery_strategies:
            try:
                self.recovery_strategies[recovery_action]()
                logger.info(f"Applied recovery action: {recovery_action}")
            except Exception as e:
                logger.error(f"Recovery action {recovery_action} failed: {e}")
    
    def _recover_from_oom(self):
        """Recover from CUDA out of memory error."""
        logger.info("Recovering from CUDA OOM...")
        
        try:
            import torch
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info("CUDA cache cleared")
        except ImportError:
            logger.warning("PyTorch not available for CUDA cleanup")
    
    def _recover_from_model_failure(self):
        """Recover from model loading failure."""
        logger.info("Recovering from model failure...")
        
        # Clear any cached models
        import gc
        gc.collect()
        
        # Wait a bit for system to stabilize
        time.sleep(2.0)
    
    def _recover_from_timeout(self):
        """Recover from timeout error."""
        logger.info("Recovering from timeout...")
        
        # Just wait for system to stabilize
        time.sleep(5.0)
    
    def _generic_recovery(self):
        """Generic recovery actions."""
        logger.info("Applying generic recovery...")
        
        import gc
        gc.collect()
        time.sleep(1.0)
    
    def _apply_recovery(self, error: str) -> Optional[str]:
        """Apply recovery strategy for error."""
        recovery_action = self._determine_recovery_action(error)
        if recovery_action:
            self._apply_recovery_action(recovery_action)
        return recovery_action
    
    def _compute_aggregate_metrics(self, attempts: List[BenchmarkAttempt]) -> Dict[str, float]:
        """Compute aggregate metrics from successful attempts."""
        # Placeholder implementation
        return {
            "success_rate": len(attempts) / len(self.attempt_history) if self.attempt_history else 0.0,
            "average_duration": sum(a.duration for a in attempts if a.duration) / len(attempts),
            "total_attempts": len(self.attempt_history)
        }
    
    def _compute_health_degradation(self, before: Any, after: Any) -> Dict[str, Any]:
        """Compute health degradation metrics."""
        if not before or not after:
            return {}
        
        return {
            "cpu_increase": after.cpu_percent - before.cpu_percent,
            "memory_increase": after.memory_percent - before.memory_percent,
            "new_alerts": len(after.alerts or []) - len(before.alerts or [])
        }
    
    def get_benchmark_statistics(self) -> Dict[str, Any]:
        """Get comprehensive benchmark statistics."""
        with self.lock:
            total_attempts = len(self.attempt_history)
            successful_attempts = sum(1 for a in self.attempt_history if a.success)
            
            if total_attempts == 0:
                return {"error": "No benchmark attempts recorded"}
            
            success_rate = successful_attempts / total_attempts
            
            # Group by model
            model_stats = {}
            for attempt in self.attempt_history:
                model = attempt.model_name
                if model not in model_stats:
                    model_stats[model] = {"total": 0, "successful": 0, "failures": []}
                
                model_stats[model]["total"] += 1
                if attempt.success:
                    model_stats[model]["successful"] += 1
                else:
                    model_stats[model]["failures"].append(attempt.error)
            
            # Circuit breaker status
            cb_status = {name: cb.status for name, cb in self.circuit_breakers.items()}
            
            return {
                "overall": {
                    "total_attempts": total_attempts,
                    "successful_attempts": successful_attempts,
                    "success_rate": success_rate,
                    "average_duration": sum(a.duration for a in self.attempt_history if a.duration and a.success) / successful_attempts if successful_attempts > 0 else 0
                },
                "by_model": model_stats,
                "circuit_breakers": cb_status,
                "recent_attempts": [asdict(a) for a in self.attempt_history[-10:]]  # Last 10 attempts
            }
    
    def export_results(self, file_path: str):
        """Export comprehensive results to file."""
        stats = self.get_benchmark_statistics()
        health_summary = self.health_monitor.get_health_summary()
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "statistics": stats,
            "health_summary": health_summary,
            "full_attempt_history": [asdict(a) for a in self.attempt_history]
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Resilient benchmark results exported to {file_path}")


# Factory function
def create_resilient_suite(
    max_retries: int = 3,
    auto_recovery: bool = True,
    health_monitoring: bool = True
) -> ResilientBenchmarkSuite:
    """Create a resilient benchmark suite with specified configuration."""
    config = ResilientConfig(
        max_retries=max_retries,
        auto_recovery=auto_recovery,
        health_check_enabled=health_monitoring
    )
    return ResilientBenchmarkSuite(config)


# Example usage
if __name__ == "__main__":
    # Create resilient benchmark suite
    suite = create_resilient_suite(max_retries=3, auto_recovery=True)
    
    # Run resilient benchmark
    results = suite.evaluate_model_resilient(
        "mock-fast",
        ["A cat playing piano", "A dog dancing", "A bird flying"]
    )
    
    print("Resilient Benchmark Results:")
    print(json.dumps(results, indent=2))
    
    # Export results
    suite.export_results("resilient_benchmark_results.json")