"""Auto-scaling and load balancing system for video diffusion benchmarking.

This module implements intelligent auto-scaling features including:
- Dynamic resource allocation based on workload
- Load balancing across multiple workers/processes
- Cloud integration for elastic scaling
- Queue management and priority scheduling
- Resource usage prediction and preemptive scaling
"""

import asyncio
import time
import logging
import threading
import psutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from pathlib import Path
import subprocess
import shutil

logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Scaling policy options."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    PREDICTIVE = "predictive"
    COST_OPTIMIZED = "cost_optimized"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU_WORKERS = "cpu_workers"
    GPU_WORKERS = "gpu_workers"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class ResourceMetrics:
    """Resource usage metrics for scaling decisions."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    active_workers: int = 0
    queue_length: int = 0
    average_processing_time: float = 0.0
    throughput_per_second: float = 0.0


@dataclass
class ScalingDecision:
    """Represents a scaling decision."""
    resource_type: ResourceType
    action: str  # "scale_up", "scale_down", "no_action"
    target_count: int
    confidence: float
    reasoning: str
    estimated_cost_impact: float = 0.0


class ResourceMonitor:
    """Monitor system resources for scaling decisions."""
    
    def __init__(self, sampling_interval: float = 5.0, history_size: int = 100):
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.is_monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        
        gpu_memory_percent = 0.0
        gpu_utilization_percent = 0.0
        
        # Try to get GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_percent = (reserved / total) * 100
                    
                    # GPU utilization would need nvidia-ml-py or similar
                    # For now, estimate based on memory usage
                    gpu_utilization_percent = min(100, gpu_memory_percent * 1.2)
        except ImportError:
            pass
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization_percent=gpu_utilization_percent,
            active_workers=0,  # Would be set by worker pool
            queue_length=0,    # Would be set by queue manager
            average_processing_time=0.0,  # Would be calculated from job history
            throughput_per_second=0.0      # Would be calculated from job history
        )
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_average_metrics(self, window_size: int = 10) -> Optional[ResourceMetrics]:
        """Get average metrics over a time window."""
        if len(self.metrics_history) < window_size:
            recent_metrics = list(self.metrics_history)
        else:
            recent_metrics = list(self.metrics_history)[-window_size:]
        
        if not recent_metrics:
            return None
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu_memory = sum(m.gpu_memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu_util = sum(m.gpu_utilization_percent for m in recent_metrics) / len(recent_metrics)
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            gpu_memory_percent=avg_gpu_memory,
            gpu_utilization_percent=avg_gpu_util,
            active_workers=recent_metrics[-1].active_workers,
            queue_length=recent_metrics[-1].queue_length,
            average_processing_time=sum(m.average_processing_time for m in recent_metrics) / len(recent_metrics),
            throughput_per_second=sum(m.throughput_per_second for m in recent_metrics) / len(recent_metrics)
        )


class ScalingDecisionEngine:
    """Make intelligent scaling decisions based on metrics and policies."""
    
    def __init__(self, policy: ScalingPolicy = ScalingPolicy.CONSERVATIVE):
        self.policy = policy
        self.decision_history = []
        self.thresholds = self._get_policy_thresholds(policy)
    
    def _get_policy_thresholds(self, policy: ScalingPolicy) -> Dict[str, Any]:
        """Get thresholds based on scaling policy."""
        base_thresholds = {
            "cpu_scale_up": 75.0,
            "cpu_scale_down": 30.0,
            "memory_scale_up": 80.0,
            "memory_scale_down": 40.0,
            "gpu_scale_up": 70.0,
            "gpu_scale_down": 25.0,
            "queue_length_scale_up": 10,
            "queue_length_scale_down": 2,
            "min_workers": 1,
            "max_workers": 10,
            "scale_up_cooldown": 300,  # 5 minutes
            "scale_down_cooldown": 600  # 10 minutes
        }
        
        policy_adjustments = {
            ScalingPolicy.CONSERVATIVE: {
                "cpu_scale_up": 85.0,
                "memory_scale_up": 85.0,
                "gpu_scale_up": 80.0,
                "scale_up_cooldown": 600,
                "scale_down_cooldown": 900
            },
            ScalingPolicy.AGGRESSIVE: {
                "cpu_scale_up": 60.0,
                "memory_scale_up": 65.0,
                "gpu_scale_up": 55.0,
                "queue_length_scale_up": 5,
                "scale_up_cooldown": 120,
                "scale_down_cooldown": 300
            },
            ScalingPolicy.PREDICTIVE: {
                # Would use ML models for prediction
                "prediction_window": 300,
                "prediction_confidence_threshold": 0.8
            },
            ScalingPolicy.COST_OPTIMIZED: {
                "cost_per_worker_hour": 1.0,
                "max_cost_per_hour": 10.0,
                "prefer_scale_down": True
            }
        }
        
        if policy in policy_adjustments:
            base_thresholds.update(policy_adjustments[policy])
        
        return base_thresholds
    
    def make_scaling_decision(
        self,
        current_metrics: ResourceMetrics,
        current_worker_count: int,
        resource_type: ResourceType = ResourceType.CPU_WORKERS
    ) -> ScalingDecision:
        """Make a scaling decision based on current metrics."""
        
        # Check cooldown periods
        if self._is_in_cooldown(resource_type):
            return ScalingDecision(
                resource_type=resource_type,
                action="no_action",
                target_count=current_worker_count,
                confidence=1.0,
                reasoning="In cooldown period"
            )
        
        # Determine if scaling is needed
        if resource_type == ResourceType.CPU_WORKERS:
            return self._decide_cpu_worker_scaling(current_metrics, current_worker_count)
        elif resource_type == ResourceType.GPU_WORKERS:
            return self._decide_gpu_worker_scaling(current_metrics, current_worker_count)
        else:
            return ScalingDecision(
                resource_type=resource_type,
                action="no_action",
                target_count=current_worker_count,
                confidence=0.0,
                reasoning="Unsupported resource type"
            )
    
    def _decide_cpu_worker_scaling(
        self,
        metrics: ResourceMetrics,
        current_workers: int
    ) -> ScalingDecision:
        """Decide CPU worker scaling."""
        
        # Scale up conditions
        scale_up_reasons = []
        if metrics.cpu_percent > self.thresholds["cpu_scale_up"]:
            scale_up_reasons.append(f"CPU usage {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_scale_up"]:
            scale_up_reasons.append(f"Memory usage {metrics.memory_percent:.1f}%")
        
        if metrics.queue_length > self.thresholds["queue_length_scale_up"]:
            scale_up_reasons.append(f"Queue length {metrics.queue_length}")
        
        # Scale down conditions
        scale_down_reasons = []
        if (metrics.cpu_percent < self.thresholds["cpu_scale_down"] and 
            metrics.memory_percent < self.thresholds["memory_scale_down"] and
            metrics.queue_length <= self.thresholds["queue_length_scale_down"]):
            scale_down_reasons.append("Low resource utilization")
        
        # Make decision
        if scale_up_reasons and current_workers < self.thresholds["max_workers"]:
            target_workers = min(
                current_workers + self._calculate_scale_amount(metrics, "up"),
                self.thresholds["max_workers"]
            )
            return ScalingDecision(
                resource_type=ResourceType.CPU_WORKERS,
                action="scale_up",
                target_count=target_workers,
                confidence=0.8,
                reasoning=f"Scale up due to: {', '.join(scale_up_reasons)}"
            )
        
        elif scale_down_reasons and current_workers > self.thresholds["min_workers"]:
            target_workers = max(
                current_workers - self._calculate_scale_amount(metrics, "down"),
                self.thresholds["min_workers"]
            )
            return ScalingDecision(
                resource_type=ResourceType.CPU_WORKERS,
                action="scale_down",
                target_count=target_workers,
                confidence=0.7,
                reasoning=f"Scale down due to: {', '.join(scale_down_reasons)}"
            )
        
        else:
            return ScalingDecision(
                resource_type=ResourceType.CPU_WORKERS,
                action="no_action",
                target_count=current_workers,
                confidence=0.9,
                reasoning="Resource utilization within acceptable range"
            )
    
    def _decide_gpu_worker_scaling(
        self,
        metrics: ResourceMetrics,
        current_workers: int
    ) -> ScalingDecision:
        """Decide GPU worker scaling."""
        
        # Similar to CPU scaling but focused on GPU metrics
        scale_up_reasons = []
        if metrics.gpu_utilization_percent > self.thresholds["gpu_scale_up"]:
            scale_up_reasons.append(f"GPU utilization {metrics.gpu_utilization_percent:.1f}%")
        
        if metrics.gpu_memory_percent > self.thresholds["gpu_scale_up"]:
            scale_up_reasons.append(f"GPU memory {metrics.gpu_memory_percent:.1f}%")
        
        scale_down_reasons = []
        if (metrics.gpu_utilization_percent < self.thresholds["gpu_scale_down"] and 
            metrics.gpu_memory_percent < self.thresholds["gpu_scale_down"]):
            scale_down_reasons.append("Low GPU utilization")
        
        # Make decision (similar logic to CPU scaling)
        if scale_up_reasons and current_workers < self.thresholds["max_workers"]:
            target_workers = min(current_workers + 1, self.thresholds["max_workers"])
            return ScalingDecision(
                resource_type=ResourceType.GPU_WORKERS,
                action="scale_up",
                target_count=target_workers,
                confidence=0.8,
                reasoning=f"Scale up due to: {', '.join(scale_up_reasons)}"
            )
        
        elif scale_down_reasons and current_workers > self.thresholds["min_workers"]:
            target_workers = max(current_workers - 1, self.thresholds["min_workers"])
            return ScalingDecision(
                resource_type=ResourceType.GPU_WORKERS,
                action="scale_down",
                target_count=target_workers,
                confidence=0.7,
                reasoning=f"Scale down due to: {', '.join(scale_down_reasons)}"
            )
        
        else:
            return ScalingDecision(
                resource_type=ResourceType.GPU_WORKERS,
                action="no_action",
                target_count=current_workers,
                confidence=0.9,
                reasoning="GPU utilization within acceptable range"
            )
    
    def _calculate_scale_amount(self, metrics: ResourceMetrics, direction: str) -> int:
        """Calculate how many workers to add/remove."""
        if self.policy == ScalingPolicy.AGGRESSIVE:
            return 2 if direction == "up" else 1
        elif self.policy == ScalingPolicy.CONSERVATIVE:
            return 1
        else:
            # Default scaling amount
            return 1
    
    def _is_in_cooldown(self, resource_type: ResourceType) -> bool:
        """Check if resource is in cooldown period."""
        if not self.decision_history:
            return False
        
        # Find last decision for this resource type
        for decision in reversed(self.decision_history):
            if decision.resource_type == resource_type and decision.action != "no_action":
                time_since_decision = time.time() - getattr(decision, 'timestamp', 0)
                
                if decision.action == "scale_up":
                    return time_since_decision < self.thresholds["scale_up_cooldown"]
                elif decision.action == "scale_down":
                    return time_since_decision < self.thresholds["scale_down_cooldown"]
        
        return False
    
    def record_decision(self, decision: ScalingDecision):
        """Record a scaling decision for history tracking."""
        decision.timestamp = time.time()
        self.decision_history.append(decision)
        
        # Keep only recent history
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-50:]


class WorkerPool:
    """Dynamic worker pool that can scale based on demand."""
    
    def __init__(self, initial_workers: int = 2, max_workers: int = 10):
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        self.current_workers = 0
        self.worker_processes = []
        self.work_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.is_running = False
    
    async def start(self):
        """Start the worker pool."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start initial workers
        for _ in range(self.initial_workers):
            await self._add_worker()
        
        logger.info(f"Worker pool started with {self.current_workers} workers")
    
    async def stop(self):
        """Stop all workers and cleanup."""
        self.is_running = False
        
        # Stop all workers
        while self.worker_processes:
            await self._remove_worker()
        
        logger.info("Worker pool stopped")
    
    async def _add_worker(self):
        """Add a new worker to the pool."""
        if self.current_workers >= self.max_workers:
            logger.warning("Cannot add worker: max workers reached")
            return
        
        worker_id = f"worker_{self.current_workers}"
        
        # Start worker process (simplified implementation)
        worker = WorkerProcess(worker_id, self.work_queue, self.result_queue)
        await worker.start()
        
        self.worker_processes.append(worker)
        self.current_workers += 1
        
        logger.info(f"Added worker {worker_id}, total workers: {self.current_workers}")
    
    async def _remove_worker(self):
        """Remove a worker from the pool."""
        if self.current_workers <= 1:
            logger.warning("Cannot remove worker: minimum workers required")
            return
        
        if not self.worker_processes:
            return
        
        # Remove least recently used worker
        worker = self.worker_processes.pop()
        await worker.stop()
        
        self.current_workers -= 1
        
        logger.info(f"Removed worker {worker.worker_id}, total workers: {self.current_workers}")
    
    async def scale_to(self, target_workers: int):
        """Scale worker pool to target size."""
        target_workers = max(1, min(target_workers, self.max_workers))
        
        while self.current_workers < target_workers:
            await self._add_worker()
        
        while self.current_workers > target_workers:
            await self._remove_worker()
        
        logger.info(f"Scaled worker pool to {self.current_workers} workers")
    
    async def submit_work(self, work_item: Any) -> Any:
        """Submit work to the pool."""
        await self.work_queue.put(work_item)
        
        # Get result
        result = await self.result_queue.get()
        return result
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.work_queue.qsize()
    
    def get_worker_count(self) -> int:
        """Get current worker count."""
        return self.current_workers


class WorkerProcess:
    """Individual worker process."""
    
    def __init__(self, worker_id: str, work_queue: asyncio.Queue, result_queue: asyncio.Queue):
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.is_running = False
        self.worker_task = None
    
    async def start(self):
        """Start the worker."""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._work_loop())
        logger.debug(f"Started worker {self.worker_id}")
    
    async def stop(self):
        """Stop the worker."""
        self.is_running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        logger.debug(f"Stopped worker {self.worker_id}")
    
    async def _work_loop(self):
        """Main work loop."""
        while self.is_running:
            try:
                # Wait for work with timeout
                work_item = await asyncio.wait_for(
                    self.work_queue.get(),
                    timeout=1.0
                )
                
                # Process work item
                result = await self._process_work(work_item)
                
                # Return result
                await self.result_queue.put(result)
                
            except asyncio.TimeoutError:
                # No work available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                await self.result_queue.put(None)
    
    async def _process_work(self, work_item: Any) -> Any:
        """Process a work item (mock implementation)."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Return processed result
        return {
            "worker_id": self.worker_id,
            "input": work_item,
            "result": f"processed_{work_item}",
            "timestamp": time.time()
        }


class AutoScaler:
    """Main auto-scaling coordinator."""
    
    def __init__(
        self,
        policy: ScalingPolicy = ScalingPolicy.CONSERVATIVE,
        monitoring_interval: float = 30.0
    ):
        self.policy = policy
        self.monitoring_interval = monitoring_interval
        
        self.resource_monitor = ResourceMonitor()
        self.decision_engine = ScalingDecisionEngine(policy)
        self.worker_pool = WorkerPool()
        
        self.is_auto_scaling = False
        self._scaling_task = None
    
    async def start_auto_scaling(self):
        """Start automatic scaling."""
        if self.is_auto_scaling:
            return
        
        self.is_auto_scaling = True
        
        # Start components
        self.resource_monitor.start_monitoring()
        await self.worker_pool.start()
        
        # Start scaling loop
        self._scaling_task = asyncio.create_task(self._auto_scaling_loop())
        
        logger.info(f"Auto-scaling started with {self.policy.value} policy")
    
    async def stop_auto_scaling(self):
        """Stop automatic scaling."""
        self.is_auto_scaling = False
        
        # Stop scaling loop
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        self.resource_monitor.stop_monitoring()
        await self.worker_pool.stop()
        
        logger.info("Auto-scaling stopped")
    
    async def _auto_scaling_loop(self):
        """Main auto-scaling loop."""
        while self.is_auto_scaling:
            try:
                # Get current metrics
                metrics = self.resource_monitor.get_average_metrics(window_size=5)
                if metrics is None:
                    await asyncio.sleep(self.monitoring_interval)
                    continue
                
                # Update metrics with current worker pool state
                metrics.active_workers = self.worker_pool.get_worker_count()
                metrics.queue_length = self.worker_pool.get_queue_size()
                
                # Make scaling decision
                decision = self.decision_engine.make_scaling_decision(
                    metrics, 
                    metrics.active_workers,
                    ResourceType.CPU_WORKERS
                )
                
                # Execute scaling decision
                if decision.action == "scale_up":
                    await self.worker_pool.scale_to(decision.target_count)
                    logger.info(f"Scaled up to {decision.target_count} workers: {decision.reasoning}")
                
                elif decision.action == "scale_down":
                    await self.worker_pool.scale_to(decision.target_count)
                    logger.info(f"Scaled down to {decision.target_count} workers: {decision.reasoning}")
                
                else:
                    logger.debug(f"No scaling action: {decision.reasoning}")
                
                # Record decision
                self.decision_engine.record_decision(decision)
                
                # Wait before next iteration
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def submit_work(self, work_item: Any) -> Any:
        """Submit work to the auto-scaling worker pool."""
        return await self.worker_pool.submit_work(work_item)
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        current_metrics = self.resource_monitor.get_current_metrics()
        
        return {
            "is_auto_scaling": self.is_auto_scaling,
            "policy": self.policy.value,
            "current_workers": self.worker_pool.get_worker_count(),
            "queue_size": self.worker_pool.get_queue_size(),
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "recent_decisions": [d.__dict__ for d in self.decision_engine.decision_history[-10:]],
            "metrics_history_size": len(self.resource_monitor.metrics_history)
        }