"""Adaptive auto-scaling system for video diffusion benchmarks."""

import time
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import multiprocessing as mp
from pathlib import Path
import json

from .health_monitoring import get_health_monitor, HealthMonitor
from .circuit_breaker import get_circuit_breaker

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for adaptive scaling."""
    min_workers: int = 1
    max_workers: int = 8
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 50.0
    scale_up_cooldown: float = 60.0  # seconds
    scale_down_cooldown: float = 120.0  # seconds
    queue_length_threshold: int = 5
    worker_timeout: float = 300.0


@dataclass
class WorkerMetrics:
    """Metrics for a single worker."""
    worker_id: str
    start_time: float
    end_time: Optional[float] = None
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: float = 0.0


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    event_type: str  # scale_up, scale_down, worker_added, worker_removed
    worker_count_before: int
    worker_count_after: int
    reason: str
    metrics: Dict[str, Any]


class AdaptiveScaler:
    """Adaptive auto-scaling system for distributed benchmarking."""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.health_monitor = get_health_monitor()
        
        # Worker management
        self.workers: Dict[str, WorkerMetrics] = {}
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Scaling state
        self.current_workers = 0
        self.target_workers = self.config.min_workers
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.scaling_events: List[ScalingEvent] = []
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        logger.info("Adaptive scaler initialized")
    
    def start(self):
        """Start the adaptive scaling system."""
        with self.lock:
            if self.monitoring:
                logger.warning("Adaptive scaler already running")
                return
            
            self.monitoring = True
            self.current_workers = self.config.min_workers
            
            # Initialize worker pool
            self._initialize_worker_pool()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info(f"Adaptive scaler started with {self.current_workers} workers")
    
    def stop(self):
        """Stop the adaptive scaling system."""
        with self.lock:
            self.monitoring = False
            
            if self.worker_pool:
                self.worker_pool.shutdown(wait=True)
                self.worker_pool = None
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            
            logger.info("Adaptive scaler stopped")
    
    def _initialize_worker_pool(self):
        """Initialize the worker thread pool."""
        if self.worker_pool:
            self.worker_pool.shutdown(wait=False)
        
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="benchmark_worker"
        )
        
        # Initialize worker metrics
        for i in range(self.current_workers):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                start_time=time.time(),
                last_heartbeat=time.time()
            )
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while self.monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_scaling_metrics()
                
                # Determine if scaling is needed
                scaling_decision = self._make_scaling_decision(metrics)
                
                if scaling_decision:
                    self._execute_scaling_decision(scaling_decision, metrics)
                
                # Update worker heartbeats
                self._update_worker_heartbeats()
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Adaptive scaler monitoring error: {e}")
                time.sleep(30.0)
    
    def _collect_scaling_metrics(self) -> Dict[str, Any]:
        """Collect metrics for scaling decisions."""
        health = self.health_monitor.check_health()
        
        # Queue metrics
        queue_size = self.task_queue.qsize()
        
        # Worker metrics
        active_workers = sum(1 for w in self.workers.values() if w.end_time is None)
        avg_cpu = sum(w.cpu_usage for w in self.workers.values()) / max(1, active_workers)
        avg_memory = sum(w.memory_usage for w in self.workers.values()) / max(1, active_workers)
        
        # Task throughput
        total_tasks = sum(w.task_count for w in self.workers.values())
        total_successes = sum(w.success_count for w in self.workers.values())
        success_rate = total_successes / max(1, total_tasks)
        
        return {
            "timestamp": time.time(),
            "system_cpu": health.cpu_percent,
            "system_memory": health.memory_percent,
            "queue_size": queue_size,
            "active_workers": active_workers,
            "target_workers": self.target_workers,
            "worker_cpu": avg_cpu,
            "worker_memory": avg_memory,
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "health_alerts": len(health.alerts or [])
        }
    
    def _make_scaling_decision(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Make scaling decision based on current metrics."""
        current_time = time.time()
        
        # Check cooldown periods
        if (current_time - self.last_scale_up) < self.config.scale_up_cooldown:
            scale_up_allowed = False
        else:
            scale_up_allowed = True
        
        if (current_time - self.last_scale_down) < self.config.scale_down_cooldown:
            scale_down_allowed = False
        else:
            scale_down_allowed = True
        
        # Scale up conditions
        should_scale_up = (
            scale_up_allowed and
            self.current_workers < self.config.max_workers and
            (
                metrics["system_cpu"] > self.config.scale_up_threshold or
                metrics["queue_size"] > self.config.queue_length_threshold or
                metrics["worker_cpu"] > self.config.scale_up_threshold
            )
        )
        
        # Scale down conditions
        should_scale_down = (
            scale_down_allowed and
            self.current_workers > self.config.min_workers and
            metrics["system_cpu"] < self.config.scale_down_threshold and
            metrics["queue_size"] == 0 and
            metrics["worker_cpu"] < self.config.scale_down_threshold
        )
        
        if should_scale_up:
            return "scale_up"
        elif should_scale_down:
            return "scale_down"
        else:
            return None
    
    def _execute_scaling_decision(self, decision: str, metrics: Dict[str, Any]):
        """Execute the scaling decision."""
        with self.lock:
            old_worker_count = self.current_workers
            
            if decision == "scale_up":
                new_worker_count = min(self.current_workers + 1, self.config.max_workers)
                self._scale_up(new_worker_count)
                self.last_scale_up = time.time()
                reason = f"High utilization: CPU={metrics['system_cpu']:.1f}%, Queue={metrics['queue_size']}"
                
            elif decision == "scale_down":
                new_worker_count = max(self.current_workers - 1, self.config.min_workers)
                self._scale_down(new_worker_count)
                self.last_scale_down = time.time()
                reason = f"Low utilization: CPU={metrics['system_cpu']:.1f}%, Queue={metrics['queue_size']}"
            
            else:
                return
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                event_type=decision,
                worker_count_before=old_worker_count,
                worker_count_after=new_worker_count,
                reason=reason,
                metrics=metrics.copy()
            )
            self.scaling_events.append(event)
            
            logger.info(f"Scaled {decision}: {old_worker_count} -> {new_worker_count} workers. Reason: {reason}")
    
    def _scale_up(self, new_worker_count: int):
        """Scale up to new worker count."""
        if new_worker_count <= self.current_workers:
            return
        
        # Add new workers
        for i in range(self.current_workers, new_worker_count):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                start_time=time.time(),
                last_heartbeat=time.time()
            )
        
        self.current_workers = new_worker_count
        self.target_workers = new_worker_count
        
        # Recreate worker pool with new size
        self._initialize_worker_pool()
    
    def _scale_down(self, new_worker_count: int):
        """Scale down to new worker count."""
        if new_worker_count >= self.current_workers:
            return
        
        # Mark excess workers for removal
        workers_to_remove = self.current_workers - new_worker_count
        removed_count = 0
        
        for worker_id, worker in list(self.workers.items()):
            if removed_count >= workers_to_remove:
                break
            
            if worker.end_time is None:  # Active worker
                worker.end_time = time.time()
                removed_count += 1
        
        self.current_workers = new_worker_count
        self.target_workers = new_worker_count
        
        # Recreate worker pool with new size
        self._initialize_worker_pool()
    
    def _update_worker_heartbeats(self):
        """Update worker heartbeat timestamps."""
        current_time = time.time()
        
        for worker in self.workers.values():
            if worker.end_time is None:  # Active worker
                worker.last_heartbeat = current_time
                
                # Simulate CPU/memory usage (in real implementation, collect from actual workers)
                import secrets
                worker.cpu_usage = secrets.SystemRandom().uniform(30, 90)
                worker.memory_usage = secrets.SystemRandom().uniform(40, 85)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit a task for execution with adaptive scaling."""
        if not self.worker_pool:
            raise RuntimeError("Adaptive scaler not started")
        
        # Submit to thread pool
        future = self.worker_pool.submit(self._execute_task, func, *args, **kwargs)
        
        # Update task count for a random worker (in real implementation, track by worker)
        import secrets
        active_workers = [w for w in self.workers.values() if w.end_time is None]
        if active_workers:
            worker = secrets.SystemRandom().choice(active_workers)
            worker.task_count += 1
        
        return future
    
    def _execute_task(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a single task with metrics collection."""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Update success metrics
            import secrets
            active_workers = [w for w in self.workers.values() if w.end_time is None]
            if active_workers:
                worker = secrets.SystemRandom().choice(active_workers)
                worker.success_count += 1
                worker.total_duration += time.time() - start_time
            
            return result
            
        except Exception as e:
            # Update failure metrics
            import secrets
            active_workers = [w for w in self.workers.values() if w.end_time is None]
            if active_workers:
                worker = secrets.SystemRandom().choice(active_workers)
                worker.failure_count += 1
            
            raise
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        with self.lock:
            active_workers = [w for w in self.workers.values() if w.end_time is None]
            
            total_tasks = sum(w.task_count for w in self.workers.values())
            total_successes = sum(w.success_count for w in self.workers.values())
            total_failures = sum(w.failure_count for w in self.workers.values())
            
            return {
                "timestamp": time.time(),
                "current_workers": self.current_workers,
                "target_workers": self.target_workers,
                "active_workers": len(active_workers),
                "queue_size": self.task_queue.qsize(),
                "total_tasks": total_tasks,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "success_rate": total_successes / max(1, total_tasks),
                "recent_scaling_events": [asdict(e) for e in self.scaling_events[-5:]],
                "worker_metrics": [asdict(w) for w in self.workers.values()],
                "config": asdict(self.config)
            }
    
    def get_scaling_history(self) -> List[ScalingEvent]:
        """Get complete scaling event history."""
        return self.scaling_events.copy()
    
    def export_scaling_metrics(self, file_path: str):
        """Export scaling metrics to file."""
        data = {
            "export_time": datetime.now().isoformat(),
            "status": self.get_scaling_status(),
            "scaling_history": [asdict(e) for e in self.scaling_events],
            "worker_details": [asdict(w) for w in self.workers.values()]
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Scaling metrics exported to {file_path}")


class DistributedBenchmarkRunner:
    """Distributed benchmark runner with adaptive scaling."""
    
    def __init__(self, scaling_config: ScalingConfig = None):
        self.scaler = AdaptiveScaler(scaling_config)
        self.results = []
        self.lock = threading.Lock()
    
    def start(self):
        """Start the distributed runner."""
        self.scaler.start()
        logger.info("Distributed benchmark runner started")
    
    def stop(self):
        """Stop the distributed runner."""
        self.scaler.stop()
        logger.info("Distributed benchmark runner stopped")
    
    def run_distributed_benchmark(
        self, 
        models: List[str], 
        prompts: List[str],
        benchmark_func: Callable
    ) -> Dict[str, Any]:
        """Run distributed benchmark across multiple models and prompts."""
        
        start_time = time.time()
        futures = []
        
        # Submit all model/prompt combinations as tasks
        for model in models:
            for prompt in prompts:
                future = self.scaler.submit_task(benchmark_func, model, prompt)
                futures.append((model, prompt, future))
        
        # Collect results
        results = {}
        completed = 0
        total_tasks = len(futures)
        
        for model, prompt, future in futures:
            try:
                result = future.result(timeout=300.0)  # 5 minute timeout
                
                if model not in results:
                    results[model] = {"prompts": {}, "summary": {}}
                
                results[model]["prompts"][prompt] = result
                completed += 1
                
                logger.info(f"Completed {completed}/{total_tasks}: {model} - {prompt[:50]}...")
                
            except Exception as e:
                logger.error(f"Task failed for {model} - {prompt}: {e}")
                
                if model not in results:
                    results[model] = {"prompts": {}, "summary": {}}
                
                results[model]["prompts"][prompt] = {"error": str(e)}
        
        # Compute summary statistics
        for model in results:
            prompts_data = results[model]["prompts"]
            successful = [p for p in prompts_data.values() if "error" not in p]
            
            if successful:
                results[model]["summary"] = {
                    "total_prompts": len(prompts_data),
                    "successful_prompts": len(successful),
                    "success_rate": len(successful) / len(prompts_data),
                    "average_duration": sum(p.get("duration", 0) for p in successful) / len(successful)
                }
            else:
                results[model]["summary"] = {
                    "total_prompts": len(prompts_data),
                    "successful_prompts": 0,
                    "success_rate": 0.0,
                    "average_duration": 0.0
                }
        
        total_duration = time.time() - start_time
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "models": list(models),
            "prompt_count": len(prompts),
            "results": results,
            "scaling_status": self.scaler.get_scaling_status()
        }


# Factory functions
def create_adaptive_scaler(
    min_workers: int = 1,
    max_workers: int = 8,
    target_cpu: float = 70.0
) -> AdaptiveScaler:
    """Create adaptive scaler with specified configuration."""
    config = ScalingConfig(
        min_workers=min_workers,
        max_workers=max_workers,
        target_cpu_utilization=target_cpu
    )
    return AdaptiveScaler(config)


def create_distributed_runner(
    min_workers: int = 2,
    max_workers: int = 16
) -> DistributedBenchmarkRunner:
    """Create distributed benchmark runner."""
    config = ScalingConfig(min_workers=min_workers, max_workers=max_workers)
    return DistributedBenchmarkRunner(config)


# Example usage
if __name__ == "__main__":
    # Example benchmark function
    def mock_benchmark(model: str, prompt: str) -> Dict[str, Any]:
        import secrets
        time.sleep(secrets.SystemRandom().uniform(1, 5))  # Simulate work
        
        return {
            "model": model,
            "prompt": prompt,
            "fvd": secrets.SystemRandom().uniform(80, 120),
            "duration": secrets.SystemRandom().uniform(1, 5),
            "timestamp": time.time()
        }
    
    # Create and start distributed runner
    runner = create_distributed_runner(min_workers=2, max_workers=6)
    runner.start()
    
    try:
        # Run distributed benchmark
        results = runner.run_distributed_benchmark(
            models=["mock-fast", "mock-high-quality"],
            prompts=["A cat playing piano", "A dog dancing", "A bird flying"],
            benchmark_func=mock_benchmark
        )
        
        print("Distributed Benchmark Results:")
        print(json.dumps(results, indent=2))
        
        # Export scaling metrics
        runner.scaler.export_scaling_metrics("scaling_metrics.json")
        
    finally:
        runner.stop()