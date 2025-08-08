"""Auto-scaling and adaptive load management for benchmark suite.

This module provides intelligent auto-scaling capabilities including:
- Dynamic resource allocation based on workload
- Adaptive batch sizing for optimal throughput
- Load balancing across multiple GPUs/nodes
- Performance-based scaling decisions
- Cost-aware scaling strategies
"""

import torch
import torch.distributed as dist
import numpy as np
import logging
import time
import psutil
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from datetime import datetime, timedelta
import json
from pathlib import Path
import math

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Container for system resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_count: int
    gpu_utilization: List[float]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    network_io: Dict[str, float]
    disk_io: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_gpu_utilization(self) -> float:
        """Average GPU utilization across all devices."""
        return np.mean(self.gpu_utilization) if self.gpu_utilization else 0.0
    
    @property
    def avg_gpu_memory_percent(self) -> float:
        """Average GPU memory usage percentage."""
        if not self.gpu_memory_used or not self.gpu_memory_total:
            return 0.0
        percentages = [used/total*100 for used, total in zip(self.gpu_memory_used, self.gpu_memory_total)]
        return np.mean(percentages)
    
    @property
    def bottleneck_resource(self) -> str:
        """Identify the primary bottleneck resource."""
        bottlenecks = []
        
        if self.cpu_percent > 90:
            bottlenecks.append('cpu')
        if self.memory_percent > 90:
            bottlenecks.append('memory')
        if self.avg_gpu_memory_percent > 90:
            bottlenecks.append('gpu_memory')
        if self.avg_gpu_utilization > 95:
            bottlenecks.append('gpu_compute')
            
        if bottlenecks:
            return ','.join(bottlenecks)
        elif self.cpu_percent > 70 or self.memory_percent > 70 or self.avg_gpu_memory_percent > 70:
            return 'approaching_limits'
        else:
            return 'healthy'


@dataclass
class ScalingDecision:
    """Container for auto-scaling decisions."""
    action: str  # 'scale_up', 'scale_down', 'maintain', 'optimize'
    target_batch_size: int
    target_workers: int
    reasoning: str
    confidence: float
    expected_improvement: float
    resource_requirements: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkloadProfile:
    """Container for workload characteristics."""
    avg_prompt_length: float
    complexity_score: float
    estimated_memory_per_sample: float
    estimated_compute_per_sample: float
    parallelizable: bool
    io_bound: bool
    compute_bound: bool
    
    @classmethod
    def from_prompts(cls, prompts: List[str]) -> 'WorkloadProfile':
        """Create workload profile from prompts."""
        if not prompts:
            return cls(0, 0, 0, 0, True, False, True)
            
        avg_length = np.mean([len(p) for p in prompts])
        
        # Estimate complexity based on prompt characteristics
        complexity_factors = []
        for prompt in prompts:
            # Length factor
            length_factor = min(1.0, len(prompt) / 100)
            
            # Complexity keywords
            complex_words = ['detailed', 'intricate', 'complex', 'elaborate', 'sophisticated']
            complexity_words = sum(1 for word in complex_words if word in prompt.lower())
            complexity_factor = complexity_words / 5.0
            
            # Scene complexity
            scene_words = ['scene', 'environment', 'landscape', 'cityscape', 'crowd']
            scene_complexity = sum(1 for word in scene_words if word in prompt.lower())
            scene_factor = scene_complexity / 3.0
            
            overall_complexity = (length_factor + complexity_factor + scene_factor) / 3.0
            complexity_factors.append(overall_complexity)
        
        complexity_score = np.mean(complexity_factors)
        
        # Estimate resource requirements
        base_memory = 2.0  # GB base memory per sample
        memory_per_sample = base_memory * (1 + complexity_score)
        
        base_compute = 1.0  # Relative compute units
        compute_per_sample = base_compute * (1 + complexity_score * 2)
        
        return cls(
            avg_prompt_length=avg_length,
            complexity_score=complexity_score,
            estimated_memory_per_sample=memory_per_sample,
            estimated_compute_per_sample=compute_per_sample,
            parallelizable=True,  # Video generation is typically parallelizable
            io_bound=False,
            compute_bound=True
        )


class ResourceMonitor:
    """Monitors system resources in real-time."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        """Initialize resource monitor.
        
        Args:
            monitoring_interval: Interval between resource checks in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history = 100  # Keep last 100 measurements
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics_queue = queue.Queue()
        
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self._metrics_queue.put(metrics)
                
                # Add to history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                    
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent if net_io else 0,
            'bytes_recv': net_io.bytes_recv if net_io else 0
        }
        
        # Disk I/O
        disk_io_counters = psutil.disk_io_counters()
        disk_io = {
            'read_bytes': disk_io_counters.read_bytes if disk_io_counters else 0,
            'write_bytes': disk_io_counters.write_bytes if disk_io_counters else 0
        }
        
        # GPU metrics
        gpu_count = 0
        gpu_utilization = []
        gpu_memory_used = []
        gpu_memory_total = []
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            for i in range(gpu_count):
                # Memory info
                memory_info = torch.cuda.memory_stats(i)
                allocated = memory_info.get('allocated_bytes.all.current', 0) / (1024**3)
                
                # Get total memory
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)
                
                gpu_memory_used.append(allocated)
                gpu_memory_total.append(total_memory)
                
                # For utilization, we use allocated memory as proxy
                # (actual GPU utilization requires nvidia-ml-py)
                utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0
                gpu_utilization.append(utilization)
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            gpu_count=gpu_count,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            network_io=network_io,
            disk_io=disk_io
        )
    
    def get_latest_metrics(self) -> Optional[ResourceMetrics]:
        """Get the latest resource metrics."""
        try:
            return self._metrics_queue.get_nowait()
        except queue.Empty:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, window_minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over a time window."""
        if not self.metrics_history:
            return None
            
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_memory_available = np.mean([m.memory_available_gb for m in recent_metrics])
        
        # GPU averages
        if recent_metrics[0].gpu_count > 0:
            all_gpu_util = [util for m in recent_metrics for util in m.gpu_utilization]
            all_gpu_mem_used = [mem for m in recent_metrics for mem in m.gpu_memory_used]
            all_gpu_mem_total = [mem for m in recent_metrics for mem in m.gpu_memory_total]
            
            avg_gpu_util = [np.mean(all_gpu_util)] if all_gpu_util else []
            avg_gpu_mem_used = [np.mean(all_gpu_mem_used)] if all_gpu_mem_used else []
            avg_gpu_mem_total = [np.mean(all_gpu_mem_total)] if all_gpu_mem_total else []
        else:
            avg_gpu_util = []
            avg_gpu_mem_used = []
            avg_gpu_mem_total = []
        
        return ResourceMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            memory_available_gb=avg_memory_available,
            gpu_count=recent_metrics[0].gpu_count,
            gpu_utilization=avg_gpu_util,
            gpu_memory_used=avg_gpu_mem_used,
            gpu_memory_total=avg_gpu_mem_total,
            network_io=recent_metrics[-1].network_io,  # Latest values
            disk_io=recent_metrics[-1].disk_io
        )


class AdaptiveBatchSizer:
    """Dynamically adjusts batch sizes based on available resources."""
    
    def __init__(self, 
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 target_memory_utilization: float = 0.8):
        """Initialize adaptive batch sizer.
        
        Args:
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size  
            target_memory_utilization: Target GPU memory utilization (0-1)
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_utilization = target_memory_utilization
        
        self.performance_history: Dict[int, List[float]] = {}  # batch_size -> throughput history
        self.memory_usage_history: Dict[int, List[float]] = {}  # batch_size -> memory usage history
        
    def suggest_batch_size(self, 
                          workload: WorkloadProfile,
                          current_metrics: ResourceMetrics,
                          current_batch_size: int) -> Tuple[int, str]:
        """Suggest optimal batch size based on workload and resources.
        
        Returns:
            Tuple of (suggested_batch_size, reasoning)
        """
        reasoning_parts = []
        
        # Calculate memory-constrained batch size
        if current_metrics.gpu_count > 0 and current_metrics.gpu_memory_total:
            available_memory = min(current_metrics.gpu_memory_total) * self.target_memory_utilization
            memory_per_sample = workload.estimated_memory_per_sample
            
            memory_constrained_batch = int(available_memory / memory_per_sample)
            memory_constrained_batch = max(self.min_batch_size, min(self.max_batch_size, memory_constrained_batch))
            
            reasoning_parts.append(f"Memory constraint allows batch size {memory_constrained_batch}")
        else:
            memory_constrained_batch = self.max_batch_size
            reasoning_parts.append("No GPU memory constraint detected")
        
        # Consider current resource utilization
        suggested_batch = memory_constrained_batch
        
        if current_metrics.avg_gpu_memory_percent > 90:
            # Memory pressure - reduce batch size
            suggested_batch = max(self.min_batch_size, current_batch_size - 1)
            reasoning_parts.append("High GPU memory usage, reducing batch size")
            
        elif current_metrics.avg_gpu_memory_percent < 60 and current_metrics.cpu_percent < 80:
            # Underutilization - potentially increase batch size
            suggested_batch = min(memory_constrained_batch, current_batch_size + 1)
            reasoning_parts.append("Low resource utilization, increasing batch size")
        
        # Use performance history for fine-tuning
        if self.performance_history:
            best_batch_size = self._get_best_performing_batch_size()
            if best_batch_size and abs(suggested_batch - best_batch_size) > 2:
                # Bias towards historically best batch size
                suggested_batch = int((suggested_batch + best_batch_size) / 2)
                reasoning_parts.append(f"Adjusted based on historical performance (best: {best_batch_size})")
        
        # Final bounds check
        suggested_batch = max(self.min_batch_size, min(self.max_batch_size, suggested_batch))
        
        reasoning = "; ".join(reasoning_parts)
        return suggested_batch, reasoning
    
    def record_performance(self, batch_size: int, throughput: float, memory_usage: float):
        """Record performance metrics for a batch size."""
        if batch_size not in self.performance_history:
            self.performance_history[batch_size] = []
            self.memory_usage_history[batch_size] = []
        
        self.performance_history[batch_size].append(throughput)
        self.memory_usage_history[batch_size].append(memory_usage)
        
        # Keep only recent history
        max_history = 10
        if len(self.performance_history[batch_size]) > max_history:
            self.performance_history[batch_size] = self.performance_history[batch_size][-max_history:]
            self.memory_usage_history[batch_size] = self.memory_usage_history[batch_size][-max_history:]
    
    def _get_best_performing_batch_size(self) -> Optional[int]:
        """Get batch size with best average throughput."""
        if not self.performance_history:
            return None
        
        batch_throughputs = {}
        for batch_size, throughputs in self.performance_history.items():
            if throughputs:
                batch_throughputs[batch_size] = np.mean(throughputs)
        
        if not batch_throughputs:
            return None
        
        return max(batch_throughputs.items(), key=lambda x: x[1])[0]


class AutoScaler:
    """Main auto-scaling engine that coordinates all components."""
    
    def __init__(self,
                 min_workers: int = 1,
                 max_workers: int = None,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 monitoring_interval: float = 10.0):
        """Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: Resource utilization threshold for scaling up
            scale_down_threshold: Resource utilization threshold for scaling down
            monitoring_interval: Interval between scaling decisions
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or (torch.cuda.device_count() if torch.cuda.is_available() else 4)
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.monitoring_interval = monitoring_interval
        
        # Components
        self.resource_monitor = ResourceMonitor(monitoring_interval / 2)
        self.batch_sizer = AdaptiveBatchSizer()
        
        # State
        self.current_workers = min_workers
        self.current_batch_size = 1
        self.scaling_history: List[ScalingDecision] = []
        self.performance_metrics: List[Dict[str, float]] = []
        
        # Control
        self._auto_scaling_enabled = False
        self._scaling_thread: Optional[threading.Thread] = None
        
        logger.info(f"AutoScaler initialized: {min_workers}-{self.max_workers} workers")
    
    def start_auto_scaling(self):
        """Start automatic scaling."""
        if self._auto_scaling_enabled:
            return
        
        self._auto_scaling_enabled = True
        self.resource_monitor.start_monitoring()
        
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
        
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling."""
        self._auto_scaling_enabled = False
        self.resource_monitor.stop_monitoring()
        
        if self._scaling_thread:
            self._scaling_thread.join(timeout=10.0)
        
        logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self._auto_scaling_enabled:
            try:
                metrics = self.resource_monitor.get_average_metrics(window_minutes=2)
                if metrics:
                    decision = self._make_scaling_decision(metrics)
                    if decision.action != 'maintain':
                        self._execute_scaling_decision(decision)
                        self.scaling_history.append(decision)
                        
                        # Keep only recent history
                        if len(self.scaling_history) > 50:
                            self.scaling_history = self.scaling_history[-50:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _make_scaling_decision(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Make scaling decision based on current metrics."""
        bottleneck = metrics.bottleneck_resource
        
        # Default decision
        decision = ScalingDecision(
            action='maintain',
            target_batch_size=self.current_batch_size,
            target_workers=self.current_workers,
            reasoning="No scaling needed",
            confidence=0.5,
            expected_improvement=0.0,
            resource_requirements={}
        )
        
        # Resource-based scaling decisions
        if 'gpu_memory' in bottleneck or 'memory' in bottleneck:
            # Memory bottleneck - reduce batch size
            if self.current_batch_size > self.batch_sizer.min_batch_size:
                decision.action = 'optimize'
                decision.target_batch_size = max(
                    self.batch_sizer.min_batch_size,
                    self.current_batch_size - 1
                )
                decision.reasoning = "Reducing batch size due to memory pressure"
                decision.confidence = 0.8
                decision.expected_improvement = 0.2
            
        elif 'gpu_compute' in bottleneck:
            # GPU compute bottleneck - check if we can scale up workers
            if self.current_workers < self.max_workers:
                decision.action = 'scale_up'
                decision.target_workers = min(self.max_workers, self.current_workers + 1)
                decision.reasoning = "Adding worker due to GPU compute bottleneck"
                decision.confidence = 0.7
                decision.expected_improvement = 0.3
        
        elif bottleneck == 'healthy':
            # System healthy - potentially optimize for better utilization
            if (metrics.avg_gpu_utilization < 60 and 
                metrics.cpu_percent < 60 and
                self.current_batch_size < self.batch_sizer.max_batch_size):
                
                decision.action = 'optimize'
                decision.target_batch_size = min(
                    self.batch_sizer.max_batch_size,
                    self.current_batch_size + 1
                )
                decision.reasoning = "Increasing batch size due to low resource utilization"
                decision.confidence = 0.6
                decision.expected_improvement = 0.15
        
        return decision
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        if decision.action == 'scale_up':
            self.current_workers = decision.target_workers
            logger.info(f"Scaled up to {self.current_workers} workers")
            
        elif decision.action == 'scale_down':
            self.current_workers = decision.target_workers
            logger.info(f"Scaled down to {self.current_workers} workers")
            
        elif decision.action == 'optimize':
            self.current_batch_size = decision.target_batch_size
            logger.info(f"Adjusted batch size to {self.current_batch_size}")
        
        # Log the decision
        logger.info(f"Scaling decision: {decision.action} - {decision.reasoning}")
    
    def get_optimal_configuration(self, 
                                workload: WorkloadProfile,
                                current_metrics: Optional[ResourceMetrics] = None) -> Dict[str, Any]:
        """Get optimal configuration for current workload and resources."""
        if current_metrics is None:
            current_metrics = self.resource_monitor.get_latest_metrics()
            if current_metrics is None:
                # Fallback metrics
                current_metrics = ResourceMetrics(
                    cpu_percent=50.0, memory_percent=50.0, memory_available_gb=8.0,
                    gpu_count=1, gpu_utilization=[50.0], gpu_memory_used=[4.0], 
                    gpu_memory_total=[8.0], network_io={}, disk_io={}
                )
        
        # Get batch size suggestion
        suggested_batch_size, batch_reasoning = self.batch_sizer.suggest_batch_size(
            workload, current_metrics, self.current_batch_size
        )
        
        effective_workers = min(self.max_workers, max(1, self.current_workers))
        
        return {
            'batch_size': suggested_batch_size,
            'workers': effective_workers,
            'batch_reasoning': batch_reasoning,
            'bottleneck_resource': current_metrics.bottleneck_resource,
            'resource_utilization': {
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'gpu_utilization': current_metrics.avg_gpu_utilization,
                'gpu_memory_percent': current_metrics.avg_gpu_memory_percent
            }
        }