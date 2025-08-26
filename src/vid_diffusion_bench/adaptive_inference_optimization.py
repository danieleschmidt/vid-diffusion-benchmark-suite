"""Real-time Adaptive Inference Optimization for Video Diffusion Models.

This module implements cutting-edge adaptive optimization techniques that dynamically
adjust inference parameters, resource allocation, and model behavior based on
real-time performance metrics and system conditions.
"""

import time
import logging
import asyncio
import threading
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from collections import deque, defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn, F
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Types of adaptive optimization strategies."""
    PERFORMANCE_FIRST = "performance_first"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    ENERGY_EFFICIENT = "energy_efficient"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"


class SystemResource(Enum):
    """System resource types for monitoring."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_MEMORY = "gpu_memory"
    GPU_UTILIZATION = "gpu_utilization"
    TEMPERATURE = "temperature"
    POWER_DRAW = "power_draw"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DISK_IO = "disk_io"


@dataclass
class SystemMetrics:
    """Real-time system metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    temperature: float
    power_draw: float
    network_bandwidth: float
    disk_io_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def get_resource_pressure(self) -> float:
        """Calculate overall resource pressure score (0-1)."""
        pressures = [
            self.cpu_usage / 100.0,
            self.memory_usage / 100.0,
            self.gpu_memory_used / max(self.gpu_memory_total, 1.0),
            self.gpu_utilization / 100.0,
            min(1.0, self.temperature / 85.0),  # Assume 85°C as high temperature
            min(1.0, self.power_draw / 300.0)   # Assume 300W as high power draw
        ]
        return np.mean(pressures)


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run."""
    inference_id: str
    timestamp: float
    latency_ms: float
    memory_peak_mb: float
    energy_consumed_j: float
    quality_score: float
    throughput_fps: float
    model_parameters: Dict[str, Any]
    system_state: SystemMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationAction:
    """An optimization action to be applied."""
    action_id: str
    action_type: str  # "parameter_adjustment", "resource_allocation", "model_switch"
    parameters: Dict[str, Any]
    expected_benefit: float
    confidence: float
    priority: int
    
    def apply_action(self, target_object: Any) -> bool:
        """Apply this optimization action to target object."""
        try:
            if self.action_type == "parameter_adjustment":
                for param_name, param_value in self.parameters.items():
                    if hasattr(target_object, param_name):
                        setattr(target_object, param_name, param_value)
                return True
            elif self.action_type == "resource_allocation":
                # Handle resource allocation changes
                return self._apply_resource_allocation(target_object)
            elif self.action_type == "model_switch":
                # Handle model switching
                return self._apply_model_switch(target_object)
            else:
                logger.warning(f"Unknown action type: {self.action_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to apply optimization action {self.action_id}: {e}")
            return False
    
    def _apply_resource_allocation(self, target_object: Any) -> bool:
        """Apply resource allocation changes."""
        # Simplified implementation
        return True
    
    def _apply_model_switch(self, target_object: Any) -> bool:
        """Apply model switching."""
        # Simplified implementation
        return True


class SystemMonitor:
    """Real-time system monitoring for adaptive optimization."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.is_monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # GPU metrics (simplified - would use nvidia-ml-py in practice)
            gpu_memory_used = 0.0
            gpu_memory_total = 1.0
            gpu_utilization = 0.0
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                gpu_utilization = min(100.0, gpu_memory_used / gpu_memory_total * 100)
            
            # Temperature (simplified)
            temperature = 45.0 + np.random.normal(0, 5)  # Mock temperature
            
            # Power draw (simplified)
            power_draw = 150.0 + np.random.normal(0, 20)  # Mock power draw
            
            # Network and disk I/O
            network_bandwidth = 100.0  # Mock bandwidth in Mbps
            disk_io_rate = 50.0  # Mock disk I/O in MB/s
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_utilization=gpu_utilization,
                temperature=temperature,
                power_draw=power_draw,
                network_bandwidth=network_bandwidth,
                disk_io_rate=disk_io_rate
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=50.0, memory_usage=60.0,
                gpu_memory_used=4.0, gpu_memory_total=8.0,
                gpu_utilization=70.0, temperature=55.0,
                power_draw=180.0, network_bandwidth=100.0,
                disk_io_rate=50.0
            )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, duration_seconds: float = 60.0) -> List[SystemMetrics]:
        """Get metrics history for the specified duration."""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [metrics for metrics in self.metrics_history 
                if metrics.timestamp >= cutoff_time]
    
    def detect_resource_pressure(self) -> Dict[str, float]:
        """Detect current resource pressure levels."""
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return {}
        
        return {
            "cpu_pressure": current_metrics.cpu_usage / 100.0,
            "memory_pressure": current_metrics.memory_usage / 100.0,
            "gpu_memory_pressure": current_metrics.gpu_memory_used / max(current_metrics.gpu_memory_total, 1.0),
            "gpu_utilization_pressure": current_metrics.gpu_utilization / 100.0,
            "thermal_pressure": min(1.0, current_metrics.temperature / 85.0),
            "power_pressure": min(1.0, current_metrics.power_draw / 300.0),
            "overall_pressure": current_metrics.get_resource_pressure()
        }


class PerformancePredictor:
    """ML-based performance predictor for optimization decisions."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.inference_history = deque(maxlen=history_size)
        self.performance_models = {}
        self.feature_extractors = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def add_inference_result(self, metrics: InferenceMetrics):
        """Add inference result to training data."""
        self.inference_history.append(metrics)
        
        # Retrain models periodically
        if len(self.inference_history) % 50 == 0 and len(self.inference_history) >= 100:
            self._update_performance_models()
    
    def predict_performance(self, 
                          model_parameters: Dict[str, Any],
                          system_state: SystemMetrics) -> Dict[str, float]:
        """
        Predict performance metrics for given parameters and system state.
        
        Returns:
            Dictionary with predicted latency, quality, energy consumption, etc.
        """
        try:
            # Extract features
            features = self._extract_features(model_parameters, system_state)
            
            # Predict various metrics
            predictions = {}
            
            # Latency prediction
            predictions["latency_ms"] = self._predict_latency(features)
            
            # Quality prediction
            predictions["quality_score"] = self._predict_quality(features)
            
            # Energy prediction
            predictions["energy_j"] = self._predict_energy(features)
            
            # Memory prediction
            predictions["memory_mb"] = self._predict_memory(features)
            
            # Throughput prediction
            predictions["throughput_fps"] = 1000.0 / max(predictions["latency_ms"], 1.0)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return self._get_default_predictions()
    
    def _extract_features(self, 
                         model_parameters: Dict[str, Any],
                         system_state: SystemMetrics) -> np.ndarray:
        """Extract features for performance prediction."""
        features = []
        
        # Model parameter features
        features.append(model_parameters.get("num_inference_steps", 50))
        features.append(model_parameters.get("guidance_scale", 7.5))
        features.append(model_parameters.get("strength", 0.8))
        features.append(model_parameters.get("eta", 0.0))
        features.append(model_parameters.get("num_frames", 16))
        features.append(model_parameters.get("fps", 8))
        
        # Resolution features
        resolution = model_parameters.get("resolution", (512, 512))
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            features.extend([resolution[0], resolution[1]])
        else:
            features.extend([512, 512])
        
        # System state features
        features.extend([
            system_state.cpu_usage,
            system_state.memory_usage,
            system_state.gpu_memory_used,
            system_state.gpu_utilization,
            system_state.temperature,
            system_state.power_draw
        ])
        
        # Add interaction terms
        features.append(features[0] * features[6] * features[7])  # steps * resolution
        features.append(features[10] * features[11])  # cpu * memory usage
        features.append(features[12] * features[13])  # gpu memory * utilization
        
        return np.array(features, dtype=np.float32)
    
    def _predict_latency(self, features: np.ndarray) -> float:
        """Predict inference latency."""
        # Simplified latency model based on key factors
        
        # Base latency from inference steps and resolution
        base_latency = features[0] * np.prod(features[6:8]) / 50000.0  # Normalize
        
        # System load adjustment
        system_load = (features[10] + features[11] + features[13]) / 300.0
        load_multiplier = 1.0 + system_load * 0.5
        
        # Temperature throttling effect
        temp_effect = 1.0 + max(0, (features[14] - 75) / 75) * 0.3
        
        predicted_latency = base_latency * load_multiplier * temp_effect * 1000  # Convert to ms
        
        # Add some realistic noise and bounds
        predicted_latency *= (1.0 + np.random.normal(0, 0.1))
        return max(100.0, min(30000.0, predicted_latency))
    
    def _predict_quality(self, features: np.ndarray) -> float:
        """Predict output quality score."""
        # Quality generally improves with more inference steps and higher guidance
        base_quality = min(1.0, (features[0] - 10) / 40.0)  # Normalize steps
        guidance_effect = min(1.0, features[1] / 15.0)  # Normalize guidance
        resolution_effect = min(1.0, np.prod(features[6:8]) / (1024 * 1024))
        
        # System stress can degrade quality
        system_stress = (features[10] + features[11] + features[13]) / 300.0
        stress_penalty = system_stress * 0.2
        
        quality_score = 0.4 * base_quality + 0.3 * guidance_effect + 0.2 * resolution_effect - stress_penalty
        
        # Add noise and bounds
        quality_score += np.random.normal(0, 0.05)
        return max(0.0, min(1.0, quality_score))
    
    def _predict_energy(self, features: np.ndarray) -> float:
        """Predict energy consumption."""
        # Energy scales with computational load
        compute_load = features[0] * np.prod(features[6:8]) / 1000.0
        
        # GPU utilization affects power draw
        gpu_power = features[13] / 100.0 * 200.0  # Assume 200W max GPU power
        
        # Base system power
        base_power = 50.0 + features[10] / 100.0 * 100.0  # CPU power component
        
        total_power = gpu_power + base_power
        
        # Energy = Power * Time (from latency prediction)
        latency_seconds = self._predict_latency(features) / 1000.0
        energy_joules = total_power * latency_seconds
        
        return max(1.0, energy_joules)
    
    def _predict_memory(self, features: np.ndarray) -> float:
        """Predict peak memory usage."""
        # Memory scales with resolution and number of frames
        base_memory = np.prod(features[6:8]) * features[4] * 4 / (1024 * 1024)  # MB
        
        # Inference steps affect intermediate storage
        steps_memory = features[0] * 10.0  # MB per step
        
        # System memory pressure affects allocation efficiency
        memory_pressure = features[11] / 100.0
        efficiency_loss = memory_pressure * 0.3
        
        total_memory = (base_memory + steps_memory) * (1.0 + efficiency_loss)
        
        return max(100.0, total_memory)
    
    def _update_performance_models(self):
        """Update ML models based on recent inference history."""
        if len(self.inference_history) < 50:
            return
        
        try:
            # Prepare training data
            X = []
            y_latency = []
            y_quality = []
            y_energy = []
            y_memory = []
            
            for metrics in list(self.inference_history)[-200:]:  # Use recent 200 samples
                features = self._extract_features(metrics.model_parameters, metrics.system_state)
                X.append(features)
                y_latency.append(metrics.latency_ms)
                y_quality.append(metrics.quality_score)
                y_energy.append(metrics.energy_consumed_j)
                y_memory.append(metrics.memory_peak_mb)
            
            X = np.array(X)
            
            # Simple linear regression for each target
            # In practice, would use more sophisticated ML models
            self._train_simple_model("latency", X, np.array(y_latency))
            self._train_simple_model("quality", X, np.array(y_quality))
            self._train_simple_model("energy", X, np.array(y_energy))
            self._train_simple_model("memory", X, np.array(y_memory))
            
            self.logger.info("Performance models updated with latest data")
            
        except Exception as e:
            self.logger.error(f"Failed to update performance models: {e}")
    
    def _train_simple_model(self, target_name: str, X: np.ndarray, y: np.ndarray):
        """Train a simple linear regression model."""
        try:
            # Simple least squares solution: w = (X^T X)^-1 X^T y
            if X.shape[0] < X.shape[1]:  # More features than samples
                return
            
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            
            self.performance_models[target_name] = {
                "weights": weights,
                "mean": np.mean(y),
                "std": np.std(y)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train {target_name} model: {e}")
    
    def _get_default_predictions(self) -> Dict[str, float]:
        """Get default predictions when models fail."""
        return {
            "latency_ms": 2000.0,
            "quality_score": 0.7,
            "energy_j": 300.0,
            "memory_mb": 4000.0,
            "throughput_fps": 0.5
        }


class AdaptiveOptimizer:
    """Main adaptive optimization engine."""
    
    def __init__(self, 
                 strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                 adaptation_rate: float = 0.1):
        self.strategy = strategy
        self.adaptation_rate = adaptation_rate
        self.system_monitor = SystemMonitor()
        self.performance_predictor = PerformancePredictor()
        self.optimization_history = deque(maxlen=500)
        self.current_parameters = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Strategy-specific weights for multi-objective optimization
        self.strategy_weights = self._get_strategy_weights()
        
    def _get_strategy_weights(self) -> Dict[str, float]:
        """Get optimization weights based on strategy."""
        weights = {
            OptimizationStrategy.PERFORMANCE_FIRST: {
                "latency": 0.6, "quality": 0.2, "energy": 0.1, "memory": 0.1
            },
            OptimizationStrategy.QUALITY_FIRST: {
                "latency": 0.2, "quality": 0.6, "energy": 0.1, "memory": 0.1
            },
            OptimizationStrategy.BALANCED: {
                "latency": 0.3, "quality": 0.3, "energy": 0.2, "memory": 0.2
            },
            OptimizationStrategy.ENERGY_EFFICIENT: {
                "latency": 0.2, "quality": 0.2, "energy": 0.5, "memory": 0.1
            },
            OptimizationStrategy.LATENCY_OPTIMIZED: {
                "latency": 0.7, "quality": 0.1, "energy": 0.1, "memory": 0.1
            },
            OptimizationStrategy.THROUGHPUT_OPTIMIZED: {
                "latency": 0.5, "quality": 0.2, "energy": 0.2, "memory": 0.1
            }
        }
        
        return weights.get(self.strategy, weights[OptimizationStrategy.BALANCED])
    
    def start_optimization(self):
        """Start the adaptive optimization engine."""
        self.system_monitor.start_monitoring()
        self.logger.info(f"Adaptive optimization started with strategy: {self.strategy}")
    
    def stop_optimization(self):
        """Stop the adaptive optimization engine."""
        self.system_monitor.stop_monitoring()
        self.logger.info("Adaptive optimization stopped")
    
    async def optimize_inference_parameters(self, 
                                          current_params: Dict[str, Any],
                                          target_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize inference parameters for current system conditions.
        
        Args:
            current_params: Current model parameters
            target_metrics: Target performance metrics (optional)
        
        Returns:
            Optimized parameters
        """
        self.logger.info("Starting parameter optimization")
        
        # Get current system state
        system_state = self.system_monitor.get_current_metrics()
        if not system_state:
            self.logger.warning("No system metrics available, using defaults")
            return current_params
        
        # Detect resource pressure
        resource_pressure = self.system_monitor.detect_resource_pressure()
        
        # Generate optimization candidates
        candidates = await self._generate_parameter_candidates(current_params, resource_pressure)
        
        # Evaluate candidates
        best_params = await self._evaluate_candidates(candidates, system_state, target_metrics)
        
        # Apply gradual adaptation
        optimized_params = self._apply_gradual_adaptation(current_params, best_params)
        
        self.logger.info(f"Parameter optimization complete. Applied {len(optimized_params)} parameter changes")
        
        return optimized_params
    
    async def _generate_parameter_candidates(self, 
                                           current_params: Dict[str, Any],
                                           resource_pressure: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate candidate parameter sets for optimization."""
        candidates = [current_params.copy()]  # Include current as baseline
        
        # Get overall pressure level
        overall_pressure = resource_pressure.get("overall_pressure", 0.5)
        
        # Generate candidates based on pressure levels
        if overall_pressure > 0.8:  # High pressure - aggressive optimization
            candidates.extend(self._generate_aggressive_candidates(current_params))
        elif overall_pressure > 0.5:  # Medium pressure - moderate optimization
            candidates.extend(self._generate_moderate_candidates(current_params))
        else:  # Low pressure - quality optimization
            candidates.extend(self._generate_quality_candidates(current_params))
        
        # Add strategy-specific candidates
        candidates.extend(self._generate_strategy_candidates(current_params))
        
        # Add random exploration candidates
        candidates.extend(self._generate_exploration_candidates(current_params))
        
        return candidates
    
    def _generate_aggressive_candidates(self, current_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidates for high resource pressure."""
        candidates = []
        
        # Reduce inference steps
        for steps in [10, 15, 20]:
            candidate = current_params.copy()
            candidate["num_inference_steps"] = steps
            candidates.append(candidate)
        
        # Reduce resolution
        current_res = current_params.get("resolution", (512, 512))
        if isinstance(current_res, (list, tuple)) and len(current_res) >= 2:
            for scale in [0.5, 0.7]:
                candidate = current_params.copy()
                candidate["resolution"] = (int(current_res[0] * scale), int(current_res[1] * scale))
                candidates.append(candidate)
        
        # Reduce number of frames
        current_frames = current_params.get("num_frames", 16)
        for frames in [8, 12]:
            if frames < current_frames:
                candidate = current_params.copy()
                candidate["num_frames"] = frames
                candidates.append(candidate)
        
        return candidates
    
    def _generate_moderate_candidates(self, current_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidates for medium resource pressure."""
        candidates = []
        
        # Moderate adjustments
        current_steps = current_params.get("num_inference_steps", 50)
        for steps in [max(20, current_steps - 10), max(15, current_steps - 20)]:
            candidate = current_params.copy()
            candidate["num_inference_steps"] = steps
            candidates.append(candidate)
        
        # Adjust guidance scale
        current_guidance = current_params.get("guidance_scale", 7.5)
        for guidance in [current_guidance * 0.8, current_guidance * 1.2]:
            candidate = current_params.copy()
            candidate["guidance_scale"] = max(1.0, min(20.0, guidance))
            candidates.append(candidate)
        
        return candidates
    
    def _generate_quality_candidates(self, current_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidates for low resource pressure (quality focus)."""
        candidates = []
        
        # Increase quality settings
        current_steps = current_params.get("num_inference_steps", 50)
        for steps in [current_steps + 10, current_steps + 20]:
            if steps <= 100:  # Reasonable upper bound
                candidate = current_params.copy()
                candidate["num_inference_steps"] = steps
                candidates.append(candidate)
        
        # Increase resolution
        current_res = current_params.get("resolution", (512, 512))
        if isinstance(current_res, (list, tuple)) and len(current_res) >= 2:
            for scale in [1.2, 1.5]:
                if current_res[0] * scale <= 1024:  # Reasonable upper bound
                    candidate = current_params.copy()
                    candidate["resolution"] = (int(current_res[0] * scale), int(current_res[1] * scale))
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_strategy_candidates(self, current_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategy-specific candidates."""
        candidates = []
        
        if self.strategy == OptimizationStrategy.LATENCY_OPTIMIZED:
            # Focus on minimal latency
            candidate = current_params.copy()
            candidate.update({
                "num_inference_steps": 15,
                "guidance_scale": 5.0,
                "eta": 1.0  # Maximum eta for faster sampling
            })
            candidates.append(candidate)
            
        elif self.strategy == OptimizationStrategy.QUALITY_FIRST:
            # Focus on maximum quality
            candidate = current_params.copy()
            candidate.update({
                "num_inference_steps": 80,
                "guidance_scale": 12.0,
                "eta": 0.0  # Deterministic sampling
            })
            candidates.append(candidate)
            
        elif self.strategy == OptimizationStrategy.ENERGY_EFFICIENT:
            # Balance between performance and quality
            candidate = current_params.copy()
            candidate.update({
                "num_inference_steps": 25,
                "guidance_scale": 6.0,
                "num_frames": min(12, current_params.get("num_frames", 16))
            })
            candidates.append(candidate)
        
        return candidates
    
    def _generate_exploration_candidates(self, current_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate random exploration candidates."""
        candidates = []
        
        # Add some random variations for exploration
        for _ in range(3):
            candidate = current_params.copy()
            
            # Random step count variation
            current_steps = current_params.get("num_inference_steps", 50)
            candidate["num_inference_steps"] = max(10, min(100, 
                current_steps + np.secrets.SystemRandom().randint(-15, 16)))
            
            # Random guidance variation
            current_guidance = current_params.get("guidance_scale", 7.5)
            candidate["guidance_scale"] = max(1.0, min(20.0,
                current_guidance + np.random.normal(0, 2)))
            
            # Random eta variation
            candidate["eta"] = max(0.0, min(1.0, np.secrets.SystemRandom().uniform(0, 1)))
            
            candidates.append(candidate)
        
        return candidates
    
    async def _evaluate_candidates(self, 
                                 candidates: List[Dict[str, Any]],
                                 system_state: SystemMetrics,
                                 target_metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Evaluate candidate parameter sets and select the best."""
        
        candidate_scores = []
        
        for i, candidate in enumerate(candidates):
            # Predict performance for this candidate
            predicted_metrics = self.performance_predictor.predict_performance(
                candidate, system_state
            )
            
            # Calculate composite score based on strategy
            score = self._calculate_composite_score(predicted_metrics, target_metrics)
            
            candidate_scores.append((score, i, candidate, predicted_metrics))
        
        # Sort by score (higher is better)
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Log top candidates
        self.logger.info(f"Evaluated {len(candidates)} candidates:")
        for i, (score, idx, candidate, metrics) in enumerate(candidate_scores[:3]):
            self.logger.info(f"  #{i+1}: Score={score:.3f}, Latency={metrics['latency_ms']:.1f}ms, "
                           f"Quality={metrics['quality_score']:.3f}")
        
        # Return best candidate
        return candidate_scores[0][2]
    
    def _calculate_composite_score(self, 
                                 predicted_metrics: Dict[str, float],
                                 target_metrics: Optional[Dict[str, float]]) -> float:
        """Calculate composite score for a candidate."""
        
        # Normalize metrics to [0, 1] scale
        normalized_metrics = {}
        
        # Latency (lower is better) - normalize to [0, 1] where 0 is best
        normalized_metrics["latency"] = 1.0 - min(1.0, predicted_metrics["latency_ms"] / 10000.0)
        
        # Quality (higher is better) - already in [0, 1]
        normalized_metrics["quality"] = predicted_metrics["quality_score"]
        
        # Energy (lower is better) - normalize to [0, 1] where 0 is best
        normalized_metrics["energy"] = 1.0 - min(1.0, predicted_metrics["energy_j"] / 1000.0)
        
        # Memory (lower is better) - normalize to [0, 1] where 0 is best
        normalized_metrics["memory"] = 1.0 - min(1.0, predicted_metrics["memory_mb"] / 10000.0)
        
        # Calculate weighted score
        score = 0.0
        for metric, weight in self.strategy_weights.items():
            score += weight * normalized_metrics.get(metric, 0.5)
        
        # Apply target metric penalties if specified
        if target_metrics:
            for metric, target_value in target_metrics.items():
                if metric in predicted_metrics:
                    predicted_value = predicted_metrics[metric]
                    
                    if metric == "latency_ms":
                        # Penalty for exceeding target latency
                        if predicted_value > target_value:
                            penalty = (predicted_value - target_value) / target_value * 0.3
                            score -= penalty
                    elif metric == "quality_score":
                        # Penalty for not meeting target quality
                        if predicted_value < target_value:
                            penalty = (target_value - predicted_value) * 0.3
                            score -= penalty
        
        return max(0.0, min(1.0, score))
    
    def _apply_gradual_adaptation(self, 
                                current_params: Dict[str, Any],
                                target_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply gradual adaptation to avoid sudden changes."""
        
        adapted_params = current_params.copy()
        
        for param_name, target_value in target_params.items():
            if param_name in current_params:
                current_value = current_params[param_name]
                
                if isinstance(current_value, (int, float)) and isinstance(target_value, (int, float)):
                    # Gradual numeric adaptation
                    diff = target_value - current_value
                    adapted_value = current_value + diff * self.adaptation_rate
                    
                    # Apply reasonable bounds
                    if param_name == "num_inference_steps":
                        adapted_value = max(5, min(150, int(adapted_value)))
                    elif param_name == "guidance_scale":
                        adapted_value = max(1.0, min(25.0, adapted_value))
                    elif param_name == "eta":
                        adapted_value = max(0.0, min(1.0, adapted_value))
                    elif param_name == "strength":
                        adapted_value = max(0.1, min(1.0, adapted_value))
                    
                    adapted_params[param_name] = adapted_value
                
                elif isinstance(current_value, (list, tuple)) and isinstance(target_value, (list, tuple)):
                    # Gradual adaptation for tuples/lists (e.g., resolution)
                    adapted_list = []
                    for i in range(min(len(current_value), len(target_value))):
                        current_val = current_value[i]
                        target_val = target_value[i]
                        
                        if isinstance(current_val, (int, float)) and isinstance(target_val, (int, float)):
                            diff = target_val - current_val
                            adapted_val = current_val + diff * self.adaptation_rate
                            adapted_list.append(int(adapted_val) if isinstance(current_val, int) else adapted_val)
                        else:
                            adapted_list.append(target_val)
                    
                    adapted_params[param_name] = type(current_value)(adapted_list)
                
                else:
                    # Direct assignment for non-numeric values
                    adapted_params[param_name] = target_value
            else:
                # New parameter
                adapted_params[param_name] = target_value
        
        return adapted_params
    
    def record_inference_result(self, 
                              params: Dict[str, Any],
                              latency_ms: float,
                              quality_score: float,
                              memory_mb: float,
                              energy_j: float):
        """Record inference result for learning."""
        
        system_state = self.system_monitor.get_current_metrics()
        if not system_state:
            return
        
        # Create inference metrics
        inference_metrics = InferenceMetrics(
            inference_id=f"inf_{int(time.time() * 1000)}_{np.secrets.SystemRandom().randint(1000, 9999)}",
            timestamp=time.time(),
            latency_ms=latency_ms,
            memory_peak_mb=memory_mb,
            energy_consumed_j=energy_j,
            quality_score=quality_score,
            throughput_fps=1000.0 / max(latency_ms, 1.0),
            model_parameters=params.copy(),
            system_state=system_state
        )
        
        # Add to performance predictor
        self.performance_predictor.add_inference_result(inference_metrics)
        
        # Store in optimization history
        self.optimization_history.append(inference_metrics)
        
        self.logger.debug(f"Recorded inference result: {latency_ms:.1f}ms, "
                         f"quality={quality_score:.3f}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        if not self.optimization_history:
            return {"error": "No optimization history available"}
        
        recent_results = list(self.optimization_history)[-100:]  # Last 100 results
        
        latencies = [r.latency_ms for r in recent_results]
        qualities = [r.quality_score for r in recent_results]
        energies = [r.energy_consumed_j for r in recent_results]
        throughputs = [r.throughput_fps for r in recent_results]
        
        stats = {
            "total_inferences": len(self.optimization_history),
            "recent_inferences": len(recent_results),
            "strategy": self.strategy.value,
            "adaptation_rate": self.adaptation_rate,
            "performance_trends": {
                "latency": {
                    "mean": np.mean(latencies),
                    "std": np.std(latencies),
                    "min": np.min(latencies),
                    "max": np.max(latencies)
                },
                "quality": {
                    "mean": np.mean(qualities),
                    "std": np.std(qualities),
                    "min": np.min(qualities),
                    "max": np.max(qualities)
                },
                "energy": {
                    "mean": np.mean(energies),
                    "std": np.std(energies),
                    "min": np.min(energies),
                    "max": np.max(energies)
                },
                "throughput": {
                    "mean": np.mean(throughputs),
                    "std": np.std(throughputs),
                    "min": np.min(throughputs),
                    "max": np.max(throughputs)
                }
            }
        }
        
        # Calculate improvement trends
        if len(recent_results) > 20:
            first_half = recent_results[:len(recent_results)//2]
            second_half = recent_results[len(recent_results)//2:]
            
            first_half_latency = np.mean([r.latency_ms for r in first_half])
            second_half_latency = np.mean([r.latency_ms for r in second_half])
            
            first_half_quality = np.mean([r.quality_score for r in first_half])
            second_half_quality = np.mean([r.quality_score for r in second_half])
            
            stats["improvement_trends"] = {
                "latency_improvement": (first_half_latency - second_half_latency) / first_half_latency,
                "quality_improvement": (second_half_quality - first_half_quality) / max(first_half_quality, 0.01)
            }
        
        return stats
    
    def export_optimization_data(self, filepath: str):
        """Export optimization data for analysis."""
        export_data = {
            "strategy": self.strategy.value,
            "adaptation_rate": self.adaptation_rate,
            "strategy_weights": self.strategy_weights,
            "optimization_history": [metrics.to_dict() for metrics in self.optimization_history],
            "statistics": self.get_optimization_statistics(),
            "export_timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported optimization data to {filepath}")


class AdaptiveInferenceEngine:
    """Complete adaptive inference engine that integrates optimization with model execution."""
    
    def __init__(self, 
                 model_adapter,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.model_adapter = model_adapter
        self.optimizer = AdaptiveOptimizer(optimization_strategy)
        self.inference_count = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def start_adaptive_inference(self):
        """Start adaptive inference engine."""
        self.optimizer.start_optimization()
        self.logger.info("Adaptive inference engine started")
    
    def stop_adaptive_inference(self):
        """Stop adaptive inference engine."""
        self.optimizer.stop_optimization()
        self.logger.info("Adaptive inference engine stopped")
    
    async def generate_video_adaptive(self, 
                                    prompt: str,
                                    initial_params: Optional[Dict[str, Any]] = None,
                                    target_metrics: Optional[Dict[str, float]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate video with adaptive optimization.
        
        Args:
            prompt: Text prompt for generation
            initial_params: Initial parameters (will be optimized)
            target_metrics: Target performance metrics
        
        Returns:
            Tuple of (generated_video, final_metrics)
        """
        self.inference_count += 1
        start_time = time.time()
        
        # Use default parameters if none provided
        if initial_params is None:
            initial_params = {
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "strength": 0.8,
                "eta": 0.0,
                "num_frames": 16,
                "fps": 8,
                "resolution": (512, 512)
            }
        
        self.logger.info(f"Starting adaptive video generation #{self.inference_count}")
        
        # Optimize parameters for current system conditions
        optimized_params = await self.optimizer.optimize_inference_parameters(
            initial_params, target_metrics
        )
        
        # Record parameter differences
        param_changes = {}
        for key in optimized_params:
            if key in initial_params and optimized_params[key] != initial_params[key]:
                param_changes[key] = {
                    "initial": initial_params[key],
                    "optimized": optimized_params[key]
                }
        
        if param_changes:
            self.logger.info(f"Applied optimizations: {param_changes}")
        
        # Generate video with optimized parameters
        generation_start = time.time()
        
        try:
            # Mock video generation (replace with actual model call)
            generated_video = await self._generate_with_model(prompt, optimized_params)
            
            generation_time = time.time() - generation_start
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            generation_time = time.time() - generation_start
            
            # Return mock result on failure
            generated_video = self._create_mock_video(optimized_params)
        
        # Calculate metrics
        metrics = await self._calculate_inference_metrics(
            generated_video, optimized_params, generation_time
        )
        
        # Record result for learning
        self.optimizer.record_inference_result(
            optimized_params,
            metrics["latency_ms"],
            metrics["quality_score"],
            metrics["memory_mb"],
            metrics["energy_j"]
        )
        
        total_time = time.time() - start_time
        
        self.logger.info(f"Adaptive generation complete in {total_time:.2f}s "
                        f"(generation: {generation_time:.2f}s)")
        
        # Prepare final metrics
        final_metrics = {
            "generation_id": f"gen_{self.inference_count}_{int(time.time())}",
            "total_time_s": total_time,
            "generation_time_s": generation_time,
            "optimization_time_s": total_time - generation_time,
            "initial_params": initial_params,
            "optimized_params": optimized_params,
            "parameter_changes": param_changes,
            "performance_metrics": metrics,
            "inference_count": self.inference_count
        }
        
        return generated_video, final_metrics
    
    async def _generate_with_model(self, prompt: str, params: Dict[str, Any]) -> Any:
        """Generate video using the model adapter."""
        # Mock implementation - replace with actual model call
        
        # Simulate generation time based on parameters
        base_time = params.get("num_inference_steps", 50) * 0.1
        resolution_factor = np.prod(params.get("resolution", (512, 512))) / (512 * 512)
        frame_factor = params.get("num_frames", 16) / 16.0
        
        generation_time = base_time * resolution_factor * frame_factor
        
        # Add some realistic variation
        generation_time *= (1.0 + np.random.normal(0, 0.2))
        generation_time = max(0.5, generation_time)
        
        # Simulate actual generation delay
        await asyncio.sleep(min(generation_time, 5.0))  # Cap simulation time
        
        # Return mock video tensor
        return self._create_mock_video(params)
    
    def _create_mock_video(self, params: Dict[str, Any]) -> torch.Tensor:
        """Create mock video tensor for testing."""
        num_frames = params.get("num_frames", 16)
        resolution = params.get("resolution", (512, 512))
        
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            height, width = resolution[0], resolution[1]
        else:
            height, width = 512, 512
        
        # Create mock video tensor
        if TORCH_AVAILABLE:
            video = torch.randn(num_frames, 3, height, width)
        else:
            video = type('MockTensor', (), {
                'shape': (num_frames, 3, height, width),
                'dtype': 'float32'
            })()
        
        return video
    
    async def _calculate_inference_metrics(self, 
                                         video: Any,
                                         params: Dict[str, Any],
                                         generation_time: float) -> Dict[str, float]:
        """Calculate comprehensive inference metrics."""
        
        # Latency metrics
        latency_ms = generation_time * 1000
        
        # Memory metrics (simplified)
        if hasattr(video, 'shape'):
            video_size_mb = np.prod(video.shape) * 4 / (1024 * 1024)  # Assume float32
        else:
            video_size_mb = params.get("num_frames", 16) * np.prod(params.get("resolution", (512, 512))) * 3 * 4 / (1024 * 1024)
        
        memory_mb = video_size_mb * 3  # Estimate peak memory usage
        
        # Energy metrics (simplified)
        system_metrics = self.optimizer.system_monitor.get_current_metrics()
        if system_metrics:
            power_draw = system_metrics.power_draw
        else:
            power_draw = 200.0  # Default power draw in watts
        
        energy_j = power_draw * generation_time
        
        # Quality metrics (mock)
        # In practice, would use actual quality evaluation
        base_quality = 0.7
        
        # Quality improves with more steps and higher guidance
        steps_factor = min(1.0, params.get("num_inference_steps", 50) / 80.0)
        guidance_factor = min(1.0, params.get("guidance_scale", 7.5) / 12.0)
        
        quality_score = base_quality + 0.2 * steps_factor + 0.1 * guidance_factor
        quality_score = min(1.0, max(0.0, quality_score + np.random.normal(0, 0.05)))
        
        # Throughput metrics
        throughput_fps = 1000.0 / max(latency_ms, 1.0)
        
        return {
            "latency_ms": latency_ms,
            "memory_mb": memory_mb,
            "energy_j": energy_j,
            "quality_score": quality_score,
            "throughput_fps": throughput_fps,
            "video_size_mb": video_size_mb
        }
    
    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptive inference statistics."""
        optimizer_stats = self.optimizer.get_optimization_statistics()
        
        # Add engine-specific statistics
        engine_stats = {
            "total_generations": self.inference_count,
            "optimization_strategy": self.optimizer.strategy.value,
            "adaptation_rate": self.optimizer.adaptation_rate,
            "system_monitoring_active": self.optimizer.system_monitor.is_monitoring
        }
        
        # Combine statistics
        combined_stats = {
            "engine_stats": engine_stats,
            "optimization_stats": optimizer_stats
        }
        
        return combined_stats


# Example usage and testing
async def run_adaptive_inference_example():
    """Example of running adaptive inference optimization."""
    
    print("=== Adaptive Inference Optimization Example ===")
    
    # Create mock model adapter
    class MockModelAdapter:
        def __init__(self):
            self.name = "mock_video_diffusion_model"
    
    model_adapter = MockModelAdapter()
    
    # Create adaptive inference engine
    engine = AdaptiveInferenceEngine(
        model_adapter=model_adapter,
        optimization_strategy=OptimizationStrategy.BALANCED
    )
    
    # Start adaptive optimization
    engine.start_adaptive_inference()
    
    try:
        # Test multiple generations with different conditions
        test_prompts = [
            "A cat playing in a garden with flowers",
            "A futuristic city with flying cars",
            "Ocean waves crashing on a beach at sunset",
            "A person walking through a forest in autumn"
        ]
        
        # Test different target metrics
        test_targets = [
            None,  # No specific targets
            {"latency_ms": 3000, "quality_score": 0.8},  # Balanced
            {"latency_ms": 1500},  # Speed focused
            {"quality_score": 0.9}  # Quality focused
        ]
        
        results = []
        
        for i, (prompt, targets) in enumerate(zip(test_prompts, test_targets)):
            print(f"\n--- Generation {i+1}/4 ---")
            print(f"Prompt: {prompt}")
            if targets:
                print(f"Targets: {targets}")
            
            # Generate with adaptive optimization
            video, metrics = await engine.generate_video_adaptive(
                prompt=prompt,
                target_metrics=targets
            )
            
            results.append((prompt, targets, metrics))
            
            # Display results
            print(f"Generation time: {metrics['generation_time_s']:.2f}s")
            print(f"Quality score: {metrics['performance_metrics']['quality_score']:.3f}")
            print(f"Memory usage: {metrics['performance_metrics']['memory_mb']:.1f}MB")
            
            if metrics['parameter_changes']:
                print(f"Parameter changes: {len(metrics['parameter_changes'])}")
                for param, change in metrics['parameter_changes'].items():
                    print(f"  {param}: {change['initial']} → {change['optimized']}")
            
            # Short delay to simulate realistic usage
            await asyncio.sleep(1.0)
        
        # Display optimization statistics
        print("\n=== Optimization Statistics ===")
        stats = engine.get_adaptive_statistics()
        
        opt_stats = stats["optimization_stats"]
        if "performance_trends" in opt_stats:
            trends = opt_stats["performance_trends"]
            print(f"Average latency: {trends['latency']['mean']:.1f}ms")
            print(f"Average quality: {trends['quality']['mean']:.3f}")
            print(f"Average throughput: {trends['throughput']['mean']:.2f} FPS")
            
            if "improvement_trends" in opt_stats:
                improvements = opt_stats["improvement_trends"]
                print(f"Latency improvement: {improvements['latency_improvement']*100:.1f}%")
                print(f"Quality improvement: {improvements['quality_improvement']*100:.1f}%")
        
        # Export results
        export_path = "adaptive_inference_results.json"
        engine.optimizer.export_optimization_data(export_path)
        print(f"\nResults exported to {export_path}")
        
        # Also export generation results
        generation_results = {
            "test_prompts": test_prompts,
            "generation_results": [metrics for _, _, metrics in results],
            "optimization_statistics": stats
        }
        
        with open("generation_results.json", "w") as f:
            json.dump(generation_results, f, indent=2)
        
        print("Generation results exported to generation_results.json")
        
        return results
        
    finally:
        # Stop adaptive optimization
        engine.stop_adaptive_inference()


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(run_adaptive_inference_example())