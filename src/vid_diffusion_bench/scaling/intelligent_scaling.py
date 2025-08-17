"""Intelligent auto-scaling system for video diffusion benchmarking.

This module implements advanced auto-scaling mechanisms that dynamically
adjust computational resources based on workload characteristics, performance
metrics, and system constraints.

Key features:
1. AI-driven workload prediction and resource allocation
2. Multi-dimensional scaling (CPU, GPU, memory, storage)
3. Cost-aware scaling with budget optimization
4. Predictive scaling based on usage patterns
5. Cross-platform compatibility (local, cloud, hybrid)
6. Real-time performance monitoring and adjustment
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import threading
import queue
import psutil
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, defaultdict
import subprocess
import os
import pickle
import yaml

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Scaling operation modes."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced" 
    AGGRESSIVE = "aggressive"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


class ResourceType(Enum):
    """Types of scalable resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    WORKERS = "workers"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory_usage: float
    disk_io: float
    network_io: float
    active_workers: int
    pending_tasks: int
    timestamp: float = field(default_factory=time.time)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for ML models."""
        return torch.tensor([
            self.cpu_usage,
            self.memory_usage,
            self.gpu_usage,
            self.gpu_memory_usage,
            self.disk_io,
            self.network_io,
            self.active_workers / 100.0,  # Normalize
            self.pending_tasks / 100.0    # Normalize
        ], dtype=torch.float32)


@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    resource_type: ResourceType
    action: str  # "scale_up", "scale_down", "maintain"
    target_value: Union[int, float]
    current_value: Union[int, float]
    confidence: float
    reasoning: str
    cost_impact: float = 0.0
    performance_impact: float = 0.0
    urgency: float = 0.0


@dataclass
class WorkloadCharacteristics:
    """Characteristics of current workload."""
    model_count: int
    avg_model_size: float  # GB
    avg_inference_time: float  # seconds
    batch_size: int
    video_resolution: Tuple[int, int]
    video_length: int  # frames
    complexity_score: float
    memory_intensity: float
    compute_intensity: float


class ResourceMonitor:
    """Monitors system resource utilization."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = []
        
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback: Callable[[ResourceMetrics], None]):
        """Add callback for metric updates."""
        self.callbacks.append(callback)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.warning(f"Monitoring callback failed: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and Memory
        cpu_usage = psutil.cpu_percent(interval=1.0) / 100.0
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0
        
        # GPU metrics
        gpu_usage = 0.0
        gpu_memory_usage = 0.0
        
        if torch.cuda.is_available():
            try:
                # GPU utilization (simplified)
                gpu_usage = torch.cuda.utilization() / 100.0 if hasattr(torch.cuda, 'utilization') else 0.0
                
                # GPU memory
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_usage = allocated / max_memory
                
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        # Disk I/O
        disk_io = 0.0
        try:
            disk_stats = psutil.disk_io_counters()
            if disk_stats:
                # Simplified disk usage metric
                disk_io = min((disk_stats.read_bytes + disk_stats.write_bytes) / 1024**3, 1.0)
        except Exception:
            pass
        
        # Network I/O
        network_io = 0.0
        try:
            network_stats = psutil.net_io_counters()
            if network_stats:
                # Simplified network usage metric
                network_io = min((network_stats.bytes_sent + network_stats.bytes_recv) / 1024**3, 1.0)
        except Exception:
            pass
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            disk_io=disk_io,
            network_io=network_io,
            active_workers=1,  # Simplified for single-node
            pending_tasks=0    # Would be provided by task queue
        )
    
    def get_recent_metrics(self, window_minutes: int = 10) -> List[ResourceMetrics]:
        """Get recent metrics within time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, window_minutes: int = 10) -> Optional[ResourceMetrics]:
        """Get average metrics over time window."""
        recent_metrics = self.get_recent_metrics(window_minutes)
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = ResourceMetrics(
            cpu_usage=np.mean([m.cpu_usage for m in recent_metrics]),
            memory_usage=np.mean([m.memory_usage for m in recent_metrics]),
            gpu_usage=np.mean([m.gpu_usage for m in recent_metrics]),
            gpu_memory_usage=np.mean([m.gpu_memory_usage for m in recent_metrics]),
            disk_io=np.mean([m.disk_io for m in recent_metrics]),
            network_io=np.mean([m.network_io for m in recent_metrics]),
            active_workers=int(np.mean([m.active_workers for m in recent_metrics])),
            pending_tasks=int(np.mean([m.pending_tasks for m in recent_metrics]))
        )
        
        return avg_metrics


class WorkloadPredictor(nn.Module):
    """Neural network for predicting resource requirements."""
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # Predict resource requirements
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, workload_features: torch.Tensor, 
                current_metrics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict resource requirements.
        
        Args:
            workload_features: Workload characteristics [B, workload_dim]
            current_metrics: Current resource metrics [B, metrics_dim]
            
        Returns:
            predictions: Predicted resource requirements [B, 8]
            confidence: Prediction confidence [B, 1]
        """
        combined = torch.cat([workload_features, current_metrics], dim=1)
        
        # Forward through network
        x = combined
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i == len(self.network) - 3:  # Capture intermediate for confidence
                intermediate = x
        
        predictions = x
        confidence = self.confidence_head(intermediate)
        
        return predictions, confidence


class CostOptimizer:
    """Optimizes resource allocation for cost efficiency."""
    
    def __init__(self, cost_config: Dict[str, float] = None):
        """Initialize with cost configuration.
        
        Args:
            cost_config: Cost per unit for each resource type
        """
        self.cost_config = cost_config or {
            'cpu_core_hour': 0.05,
            'gb_memory_hour': 0.01,
            'gpu_hour': 2.50,
            'gb_storage_month': 0.10,
            'gb_network': 0.09
        }
        
    def calculate_cost(self, resource_allocation: Dict[str, float], 
                      duration_hours: float = 1.0) -> float:
        """Calculate cost for resource allocation."""
        total_cost = 0.0
        
        # CPU cost
        if 'cpu_cores' in resource_allocation:
            total_cost += (resource_allocation['cpu_cores'] * 
                          self.cost_config['cpu_core_hour'] * duration_hours)
        
        # Memory cost
        if 'memory_gb' in resource_allocation:
            total_cost += (resource_allocation['memory_gb'] * 
                          self.cost_config['gb_memory_hour'] * duration_hours)
        
        # GPU cost
        if 'gpu_count' in resource_allocation:
            total_cost += (resource_allocation['gpu_count'] * 
                          self.cost_config['gpu_hour'] * duration_hours)
        
        return total_cost
    
    def optimize_allocation(self, 
                          performance_requirements: Dict[str, float],
                          budget_constraint: float,
                          duration_hours: float = 1.0) -> Dict[str, float]:
        """Optimize resource allocation within budget."""
        # Simplified optimization - in practice would use more sophisticated algorithms
        
        optimized_allocation = {}
        remaining_budget = budget_constraint
        
        # Prioritize based on performance requirements
        resource_priorities = [
            ('gpu_count', 'gpu_hour'),
            ('memory_gb', 'gb_memory_hour'),
            ('cpu_cores', 'cpu_core_hour')
        ]
        
        for resource, cost_key in resource_priorities:
            if resource in performance_requirements:
                required = performance_requirements[resource]
                unit_cost = self.cost_config[cost_key] * duration_hours
                
                # Allocate as much as budget allows, up to requirement
                affordable_units = remaining_budget / unit_cost
                allocated_units = min(required, affordable_units)
                
                if allocated_units > 0:
                    optimized_allocation[resource] = allocated_units
                    remaining_budget -= allocated_units * unit_cost
        
        return optimized_allocation


class IntelligentScaler:
    """Main intelligent scaling controller."""
    
    def __init__(self, 
                 scaling_mode: ScalingMode = ScalingMode.BALANCED,
                 cost_optimizer: CostOptimizer = None):
        self.scaling_mode = scaling_mode
        self.cost_optimizer = cost_optimizer or CostOptimizer()
        
        # Components
        self.resource_monitor = ResourceMonitor()
        self.workload_predictor = WorkloadPredictor()
        
        # State
        self.scaling_history = deque(maxlen=1000)
        self.current_allocation = {}
        self.performance_targets = {}
        self.scaling_lock = threading.Lock()
        
        # Configuration
        self.scaling_thresholds = {
            ScalingMode.CONSERVATIVE: {
                'cpu_upper': 0.8,
                'cpu_lower': 0.3,
                'memory_upper': 0.85,
                'memory_lower': 0.4,
                'gpu_upper': 0.9,
                'gpu_lower': 0.3
            },
            ScalingMode.BALANCED: {
                'cpu_upper': 0.7,
                'cpu_lower': 0.2,
                'memory_upper': 0.8,
                'memory_lower': 0.3,
                'gpu_upper': 0.85,
                'gpu_lower': 0.2
            },
            ScalingMode.AGGRESSIVE: {
                'cpu_upper': 0.6,
                'cpu_lower': 0.15,
                'memory_upper': 0.75,
                'memory_lower': 0.25,
                'gpu_upper': 0.8,
                'gpu_lower': 0.15
            }
        }
        
        # Start monitoring
        self.resource_monitor.add_callback(self._on_metrics_update)
        
    def start(self):
        """Start the intelligent scaling system."""
        self.resource_monitor.start_monitoring()
        logger.info(f"Intelligent scaler started in {self.scaling_mode.value} mode")
    
    def stop(self):
        """Stop the scaling system."""
        self.resource_monitor.stop_monitoring()
        logger.info("Intelligent scaler stopped")
    
    def set_performance_targets(self, targets: Dict[str, float]):
        """Set performance targets for scaling decisions."""
        self.performance_targets = targets.copy()
        logger.info(f"Performance targets updated: {targets}")
    
    def predict_resource_requirements(self, 
                                    workload: WorkloadCharacteristics) -> Dict[str, float]:
        """Predict resource requirements for given workload."""
        # Convert workload to tensor
        workload_tensor = torch.tensor([
            workload.model_count,
            workload.avg_model_size,
            workload.avg_inference_time,
            workload.batch_size,
            workload.video_resolution[0] / 1024.0,  # Normalize
            workload.video_resolution[1] / 1024.0,  # Normalize
            workload.video_length / 100.0,  # Normalize
            workload.complexity_score,
            workload.memory_intensity,
            workload.compute_intensity
        ], dtype=torch.float32).unsqueeze(0)
        
        # Get current metrics
        current_metrics = self.resource_monitor._collect_metrics()
        metrics_tensor = current_metrics.to_tensor().unsqueeze(0)
        
        # Predict requirements
        with torch.no_grad():
            predictions, confidence = self.workload_predictor(workload_tensor, metrics_tensor)
            
        # Convert predictions to resource requirements
        pred_values = predictions[0].tolist()
        conf_value = confidence[0].item()
        
        requirements = {
            'cpu_cores': max(1, int(pred_values[0] * 16)),  # Max 16 cores
            'memory_gb': max(4, pred_values[1] * 64),       # Max 64GB
            'gpu_count': max(1, int(pred_values[2] * 4)),   # Max 4 GPUs
            'gpu_memory_gb': max(8, pred_values[3] * 32),   # Max 32GB per GPU
            'storage_gb': max(10, pred_values[4] * 1000),   # Max 1TB
            'network_mbps': max(100, pred_values[5] * 1000), # Max 1Gbps
            'workers': max(1, int(pred_values[6] * 10)),    # Max 10 workers
            'confidence': conf_value
        }
        
        return requirements
    
    def make_scaling_decision(self, 
                            current_metrics: ResourceMetrics,
                            workload: WorkloadCharacteristics = None) -> List[ScalingDecision]:
        """Make intelligent scaling decisions."""
        decisions = []
        thresholds = self.scaling_thresholds[self.scaling_mode]
        
        with self.scaling_lock:
            # CPU scaling decision
            cpu_decision = self._decide_cpu_scaling(current_metrics, thresholds)
            if cpu_decision:
                decisions.append(cpu_decision)
            
            # Memory scaling decision
            memory_decision = self._decide_memory_scaling(current_metrics, thresholds)
            if memory_decision:
                decisions.append(memory_decision)
            
            # GPU scaling decision
            gpu_decision = self._decide_gpu_scaling(current_metrics, thresholds)
            if gpu_decision:
                decisions.append(gpu_decision)
            
            # Workload-based scaling if workload provided
            if workload:
                workload_decisions = self._decide_workload_scaling(workload, current_metrics)
                decisions.extend(workload_decisions)
        
        return decisions
    
    def _decide_cpu_scaling(self, metrics: ResourceMetrics, 
                          thresholds: Dict[str, float]) -> Optional[ScalingDecision]:
        """Decide CPU scaling action."""
        current_cpu = metrics.cpu_usage
        
        if current_cpu > thresholds['cpu_upper']:
            # Scale up
            target_cores = self.current_allocation.get('cpu_cores', 1) * 2
            return ScalingDecision(
                resource_type=ResourceType.CPU,
                action="scale_up",
                target_value=target_cores,
                current_value=self.current_allocation.get('cpu_cores', 1),
                confidence=min(1.0, (current_cpu - thresholds['cpu_upper']) / 0.2),
                reasoning=f"CPU usage {current_cpu:.1%} exceeds threshold {thresholds['cpu_upper']:.1%}",
                urgency=current_cpu
            )
        elif current_cpu < thresholds['cpu_lower']:
            # Scale down
            target_cores = max(1, self.current_allocation.get('cpu_cores', 1) // 2)
            return ScalingDecision(
                resource_type=ResourceType.CPU,
                action="scale_down",
                target_value=target_cores,
                current_value=self.current_allocation.get('cpu_cores', 1),
                confidence=min(1.0, (thresholds['cpu_lower'] - current_cpu) / 0.2),
                reasoning=f"CPU usage {current_cpu:.1%} below threshold {thresholds['cpu_lower']:.1%}",
                urgency=1.0 - current_cpu
            )
        
        return None
    
    def _decide_memory_scaling(self, metrics: ResourceMetrics, 
                             thresholds: Dict[str, float]) -> Optional[ScalingDecision]:
        """Decide memory scaling action."""
        current_memory = metrics.memory_usage
        
        if current_memory > thresholds['memory_upper']:
            # Scale up
            target_memory = self.current_allocation.get('memory_gb', 8) * 1.5
            return ScalingDecision(
                resource_type=ResourceType.MEMORY,
                action="scale_up",
                target_value=target_memory,
                current_value=self.current_allocation.get('memory_gb', 8),
                confidence=min(1.0, (current_memory - thresholds['memory_upper']) / 0.15),
                reasoning=f"Memory usage {current_memory:.1%} exceeds threshold {thresholds['memory_upper']:.1%}",
                urgency=current_memory
            )
        elif current_memory < thresholds['memory_lower']:
            # Scale down
            target_memory = max(4, self.current_allocation.get('memory_gb', 8) * 0.75)
            return ScalingDecision(
                resource_type=ResourceType.MEMORY,
                action="scale_down",
                target_value=target_memory,
                current_value=self.current_allocation.get('memory_gb', 8),
                confidence=min(1.0, (thresholds['memory_lower'] - current_memory) / 0.2),
                reasoning=f"Memory usage {current_memory:.1%} below threshold {thresholds['memory_lower']:.1%}",
                urgency=1.0 - current_memory
            )
        
        return None
    
    def _decide_gpu_scaling(self, metrics: ResourceMetrics, 
                          thresholds: Dict[str, float]) -> Optional[ScalingDecision]:
        """Decide GPU scaling action."""
        current_gpu = max(metrics.gpu_usage, metrics.gpu_memory_usage)
        
        if current_gpu > thresholds['gpu_upper']:
            # Scale up
            target_gpus = self.current_allocation.get('gpu_count', 1) + 1
            return ScalingDecision(
                resource_type=ResourceType.GPU,
                action="scale_up",
                target_value=target_gpus,
                current_value=self.current_allocation.get('gpu_count', 1),
                confidence=min(1.0, (current_gpu - thresholds['gpu_upper']) / 0.15),
                reasoning=f"GPU usage {current_gpu:.1%} exceeds threshold {thresholds['gpu_upper']:.1%}",
                urgency=current_gpu
            )
        elif current_gpu < thresholds['gpu_lower']:
            # Scale down
            target_gpus = max(1, self.current_allocation.get('gpu_count', 1) - 1)
            return ScalingDecision(
                resource_type=ResourceType.GPU,
                action="scale_down",
                target_value=target_gpus,
                current_value=self.current_allocation.get('gpu_count', 1),
                confidence=min(1.0, (thresholds['gpu_lower'] - current_gpu) / 0.2),
                reasoning=f"GPU usage {current_gpu:.1%} below threshold {thresholds['gpu_lower']:.1%}",
                urgency=1.0 - current_gpu
            )
        
        return None
    
    def _decide_workload_scaling(self, workload: WorkloadCharacteristics,
                               metrics: ResourceMetrics) -> List[ScalingDecision]:
        """Make workload-based scaling decisions."""
        decisions = []
        
        # Predict requirements
        requirements = self.predict_resource_requirements(workload)
        
        # Compare with current allocation and metrics
        if workload.memory_intensity > 0.8 and metrics.memory_usage > 0.7:
            decisions.append(ScalingDecision(
                resource_type=ResourceType.MEMORY,
                action="scale_up",
                target_value=requirements['memory_gb'],
                current_value=self.current_allocation.get('memory_gb', 8),
                confidence=requirements['confidence'],
                reasoning="High memory intensity workload detected",
                urgency=workload.memory_intensity
            ))
        
        if workload.compute_intensity > 0.8 and metrics.gpu_usage > 0.7:
            decisions.append(ScalingDecision(
                resource_type=ResourceType.GPU,
                action="scale_up",
                target_value=requirements['gpu_count'],
                current_value=self.current_allocation.get('gpu_count', 1),
                confidence=requirements['confidence'],
                reasoning="High compute intensity workload detected",
                urgency=workload.compute_intensity
            ))
        
        return decisions
    
    def execute_scaling_decisions(self, decisions: List[ScalingDecision]) -> Dict[str, bool]:
        """Execute scaling decisions."""
        results = {}
        
        # Sort by urgency
        decisions.sort(key=lambda d: d.urgency, reverse=True)
        
        for decision in decisions:
            try:
                success = self._execute_single_decision(decision)
                results[f"{decision.resource_type.value}_{decision.action}"] = success
                
                # Update allocation if successful
                if success:
                    resource_key = f"{decision.resource_type.value}_count" if decision.resource_type == ResourceType.GPU else f"{decision.resource_type.value}_gb"
                    if decision.resource_type == ResourceType.CPU:
                        resource_key = "cpu_cores"
                    elif decision.resource_type == ResourceType.WORKERS:
                        resource_key = "workers"
                        
                    self.current_allocation[resource_key] = decision.target_value
                    
                    # Record scaling history
                    self.scaling_history.append({
                        'timestamp': time.time(),
                        'decision': decision,
                        'success': success
                    })
                    
            except Exception as e:
                logger.error(f"Failed to execute scaling decision: {e}")
                results[f"{decision.resource_type.value}_{decision.action}"] = False
        
        return results
    
    def _execute_single_decision(self, decision: ScalingDecision) -> bool:
        """Execute a single scaling decision."""
        logger.info(f"Executing scaling decision: {decision.action} {decision.resource_type.value} "
                   f"from {decision.current_value} to {decision.target_value}")
        
        # For demo purposes, simulate scaling actions
        # In production, this would interface with actual scaling systems
        
        if decision.resource_type == ResourceType.CPU:
            return self._scale_cpu(decision.target_value)
        elif decision.resource_type == ResourceType.MEMORY:
            return self._scale_memory(decision.target_value)
        elif decision.resource_type == ResourceType.GPU:
            return self._scale_gpu(decision.target_value)
        elif decision.resource_type == ResourceType.WORKERS:
            return self._scale_workers(decision.target_value)
        
        return False
    
    def _scale_cpu(self, target_cores: int) -> bool:
        """Scale CPU resources."""
        # Simulated CPU scaling
        logger.info(f"Scaling CPU to {target_cores} cores")
        return True
    
    def _scale_memory(self, target_memory: float) -> bool:
        """Scale memory resources."""
        # Simulated memory scaling
        logger.info(f"Scaling memory to {target_memory:.1f}GB")
        return True
    
    def _scale_gpu(self, target_gpus: int) -> bool:
        """Scale GPU resources."""
        # Simulated GPU scaling
        logger.info(f"Scaling GPU to {target_gpus} units")
        return True
    
    def _scale_workers(self, target_workers: int) -> bool:
        """Scale worker processes."""
        # Simulated worker scaling
        logger.info(f"Scaling workers to {target_workers} units")
        return True
    
    def _on_metrics_update(self, metrics: ResourceMetrics):
        """Handle metrics update from monitor."""
        # Make scaling decisions based on current metrics
        decisions = self.make_scaling_decision(metrics)
        
        if decisions:
            # Execute decisions
            results = self.execute_scaling_decisions(decisions)
            logger.debug(f"Scaling decisions executed: {results}")
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling system statistics."""
        if not self.scaling_history:
            return {'total_scaling_actions': 0}
        
        total_actions = len(self.scaling_history)
        successful_actions = sum(1 for action in self.scaling_history if action['success'])
        
        # Recent activity
        recent_time = time.time() - 3600  # Last hour
        recent_actions = [a for a in self.scaling_history if a['timestamp'] > recent_time]
        
        # Resource type breakdown
        resource_actions = defaultdict(int)
        for action in self.scaling_history:
            resource_actions[action['decision'].resource_type.value] += 1
        
        return {
            'total_scaling_actions': total_actions,
            'successful_actions': successful_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
            'recent_actions': len(recent_actions),
            'current_allocation': self.current_allocation.copy(),
            'resource_action_breakdown': dict(resource_actions),
            'scaling_mode': self.scaling_mode.value
        }


# Example usage and testing
if __name__ == "__main__":
    # Example usage of intelligent scaling
    scaler = IntelligentScaler(scaling_mode=ScalingMode.BALANCED)
    
    # Set performance targets
    scaler.set_performance_targets({
        'max_latency_ms': 5000,
        'min_throughput_fps': 1.0,
        'max_memory_usage': 0.8
    })
    
    # Create example workload
    workload = WorkloadCharacteristics(
        model_count=3,
        avg_model_size=8.5,
        avg_inference_time=4.2,
        batch_size=2,
        video_resolution=(512, 512),
        video_length=24,
        complexity_score=0.7,
        memory_intensity=0.8,
        compute_intensity=0.9
    )
    
    # Predict requirements
    requirements = scaler.predict_resource_requirements(workload)
    print(f"Predicted requirements: {requirements}")
    
    # Start scaling system
    scaler.start()
    
    try:
        # Let it run for a short time
        time.sleep(10)
        
        # Get statistics
        stats = scaler.get_scaling_statistics()
        print(f"Scaling statistics: {stats}")
        
    finally:
        scaler.stop()