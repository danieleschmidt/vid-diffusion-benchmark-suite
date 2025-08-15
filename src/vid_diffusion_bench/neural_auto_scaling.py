"""Neural network-driven auto-scaling for video diffusion workloads.

Advanced auto-scaling system that uses deep learning to predict resource
needs and automatically scale infrastructure based on workload patterns.
"""

import asyncio
import numpy as np
import time
import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading
import statistics

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    PREEMPTIVE_SCALE = "preemptive_scale"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    disk_io: float
    queue_length: int
    active_requests: int
    response_time: float
    error_rate: float


@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning."""
    action: ScalingAction
    confidence: float
    reasoning: str
    target_instances: int
    predicted_load: float
    expected_improvement: float
    risk_assessment: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkloadPredictor:
    """Neural network-based workload prediction."""
    
    def __init__(self, history_window: int = 100, prediction_horizon: int = 10):
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        
        # Simple neural network weights (would use actual ML library in production)
        self.input_size = 10  # Number of features
        self.hidden_size = 20
        self.output_size = prediction_horizon
        
        # Initialize weights randomly
        np.random.seed(42)
        self.W1 = np.random.normal(0, 0.1, (self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.normal(0, 0.1, (self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))
        
        # Training state
        self.learning_rate = 0.001
        self.metrics_history = deque(maxlen=history_window * 2)
        self.training_data = []
        self.is_trained = False
        
    def add_metrics(self, metrics: ResourceMetrics):
        """Add new metrics for training and prediction."""
        self.metrics_history.append(metrics)
        
        # Update training data
        if len(self.metrics_history) >= self.history_window + self.prediction_horizon:
            self._update_training_data()
            
    def _update_training_data(self):
        """Update training dataset with new metrics."""
        if len(self.metrics_history) < self.history_window + self.prediction_horizon:
            return
            
        # Create features from historical data
        features = self._extract_features(list(self.metrics_history)[-self.history_window-self.prediction_horizon:-self.prediction_horizon])
        
        # Create targets from future data
        targets = [
            self._extract_load_metric(self.metrics_history[-self.prediction_horizon + i])
            for i in range(self.prediction_horizon)
        ]
        
        self.training_data.append((features, targets))
        
        # Keep only recent training data
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-1000:]
            
    def _extract_features(self, metrics_list: List[ResourceMetrics]) -> np.ndarray:
        """Extract features from metrics history."""
        if not metrics_list:
            return np.zeros(self.input_size)
            
        # Extract various statistical features
        cpu_values = [m.cpu_usage for m in metrics_list]
        memory_values = [m.memory_usage for m in metrics_list]
        gpu_values = [m.gpu_usage for m in metrics_list]
        queue_values = [m.queue_length for m in metrics_list]
        response_values = [m.response_time for m in metrics_list]
        
        features = [
            np.mean(cpu_values),
            np.std(cpu_values),
            np.mean(memory_values),
            np.std(memory_values),
            np.mean(gpu_values),
            np.std(gpu_values),
            np.mean(queue_values),
            np.max(queue_values),
            np.mean(response_values),
            self._detect_trend(cpu_values)
        ]
        
        return np.array(features)
        
    def _extract_load_metric(self, metrics: ResourceMetrics) -> float:
        """Extract overall load metric from metrics."""
        # Composite load score
        load = (
            metrics.cpu_usage * 0.3 +
            metrics.memory_usage * 0.2 +
            metrics.gpu_usage * 0.3 +
            min(metrics.queue_length / 10.0, 1.0) * 0.2
        )
        return min(load, 1.0)
        
    def _detect_trend(self, values: List[float]) -> float:
        """Detect trend in time series data."""
        if len(values) < 3:
            return 0.0
            
        # Simple linear trend detection
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope
        
    def train(self, epochs: int = 100):
        """Train the neural network predictor."""
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for workload predictor")
            return
            
        logger.info(f"Training workload predictor with {len(self.training_data)} samples")
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for features, targets in self.training_data:
                # Forward pass
                predictions = self._forward(features)
                
                # Calculate loss
                loss = np.mean((predictions - np.array(targets)) ** 2)
                total_loss += loss
                
                # Backward pass
                self._backward(features, targets, predictions)
                
            if epoch % 20 == 0:
                avg_loss = total_loss / len(self.training_data)
                logger.debug(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
                
        self.is_trained = True
        logger.info("Workload predictor training completed")
        
    def _forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through neural network."""
        features = features.reshape(1, -1)
        
        # Hidden layer
        z1 = np.dot(features, self.W1) + self.b1
        a1 = np.tanh(z1)  # Activation function
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        predictions = np.maximum(0, z2)  # ReLU for positive outputs
        
        return predictions.flatten()
        
    def _backward(self, features: np.ndarray, targets: List[float], predictions: np.ndarray):
        """Backward pass for training."""
        features = features.reshape(1, -1)
        targets = np.array(targets).reshape(1, -1)
        predictions = predictions.reshape(1, -1)
        
        # Calculate gradients
        m = 1  # Batch size
        
        # Output layer gradients
        dz2 = predictions - targets
        dW2 = np.dot(self._get_hidden_activations(features).T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.tanh(np.dot(features, self.W1) + self.b1) ** 2)
        dW1 = np.dot(features.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
    def _get_hidden_activations(self, features: np.ndarray) -> np.ndarray:
        """Get hidden layer activations."""
        features = features.reshape(1, -1)
        z1 = np.dot(features, self.W1) + self.b1
        return np.tanh(z1)
        
    def predict_workload(self, steps_ahead: int = None) -> List[float]:
        """Predict future workload."""
        if not self.is_trained or len(self.metrics_history) < self.history_window:
            # Return simple trend-based prediction
            return self._simple_prediction(steps_ahead or self.prediction_horizon)
            
        # Use neural network prediction
        recent_metrics = list(self.metrics_history)[-self.history_window:]
        features = self._extract_features(recent_metrics)
        predictions = self._forward(features)
        
        steps = steps_ahead or self.prediction_horizon
        return predictions[:steps].tolist()
        
    def _simple_prediction(self, steps: int) -> List[float]:
        """Simple trend-based prediction as fallback."""
        if len(self.metrics_history) < 3:
            return [0.5] * steps  # Default moderate load
            
        recent_loads = [self._extract_load_metric(m) for m in list(self.metrics_history)[-10:]]
        trend = self._detect_trend(recent_loads)
        current_load = recent_loads[-1]
        
        predictions = []
        for i in range(steps):
            predicted_load = current_load + trend * (i + 1)
            predicted_load = max(0.0, min(1.0, predicted_load))
            predictions.append(predicted_load)
            
        return predictions


class IntelligentAutoScaler:
    """AI-driven auto-scaling with predictive capabilities."""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 100,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.5
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_instances = min_instances
        self.predictor = WorkloadPredictor()
        
        # Decision history for learning
        self.decision_history = deque(maxlen=1000)
        self.scaling_events = []
        
        # Performance tracking
        self.performance_metrics = {
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_decisions': 0
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = True
        self.threshold_adaptation_rate = 0.01
        
    async def analyze_and_scale(self, current_metrics: ResourceMetrics) -> ScalingDecision:
        """Analyze current state and make scaling decision."""
        self.predictor.add_metrics(current_metrics)
        
        # Get current utilization
        current_utilization = self._calculate_utilization(current_metrics)
        
        # Predict future workload
        predicted_loads = self.predictor.predict_workload()
        
        # Make scaling decision
        decision = self._make_scaling_decision(
            current_utilization,
            predicted_loads,
            current_metrics
        )
        
        # Execute scaling if needed
        if decision.action != ScalingAction.NO_ACTION:
            await self._execute_scaling(decision)
            
        # Record decision for learning
        self._record_decision(decision, current_metrics)
        
        return decision
        
    def _calculate_utilization(self, metrics: ResourceMetrics) -> float:
        """Calculate overall system utilization."""
        # Weighted average of different resource types
        utilization = (
            metrics.cpu_usage * 0.3 +
            metrics.memory_usage * 0.2 +
            metrics.gpu_usage * 0.4 +
            min(metrics.queue_length / 10.0, 1.0) * 0.1
        )
        return min(utilization, 1.0)
        
    def _make_scaling_decision(
        self,
        current_utilization: float,
        predicted_loads: List[float],
        metrics: ResourceMetrics
    ) -> ScalingDecision:
        """Make intelligent scaling decision using ML insights."""
        
        # Analyze current state
        if current_utilization > self.scale_up_threshold:
            immediate_action = ScalingAction.SCALE_UP
            urgency = "high"
        elif current_utilization < self.scale_down_threshold:
            immediate_action = ScalingAction.SCALE_DOWN
            urgency = "low"
        else:
            immediate_action = ScalingAction.NO_ACTION
            urgency = "none"
            
        # Analyze predictions
        max_predicted = max(predicted_loads) if predicted_loads else current_utilization
        avg_predicted = np.mean(predicted_loads) if predicted_loads else current_utilization
        
        # Predictive scaling logic
        if max_predicted > self.scale_up_threshold and immediate_action == ScalingAction.NO_ACTION:
            predictive_action = ScalingAction.PREEMPTIVE_SCALE
            confidence = min(0.9, max_predicted - self.scale_up_threshold + 0.5)
        else:
            predictive_action = immediate_action
            confidence = 0.8
            
        # Determine final action
        if immediate_action == ScalingAction.SCALE_UP:
            final_action = ScalingAction.SCALE_UP
            reasoning = f"Current utilization ({current_utilization:.2%}) exceeds threshold"
        elif predictive_action == ScalingAction.PREEMPTIVE_SCALE:
            final_action = ScalingAction.SCALE_UP
            reasoning = f"Predicted load spike ({max_predicted:.2%}) in next 10 minutes"
        elif immediate_action == ScalingAction.SCALE_DOWN and avg_predicted < self.scale_down_threshold:
            final_action = ScalingAction.SCALE_DOWN
            reasoning = f"Current and predicted utilization below threshold"
        else:
            final_action = ScalingAction.NO_ACTION
            reasoning = "System operating within normal parameters"
            
        # Calculate target instances
        if final_action == ScalingAction.SCALE_UP:
            # Scale based on predicted load
            target_instances = min(
                self.max_instances,
                max(
                    self.current_instances + 1,
                    int(np.ceil(self.current_instances * max_predicted / self.target_utilization))
                )
            )
        elif final_action == ScalingAction.SCALE_DOWN:
            target_instances = max(
                self.min_instances,
                int(np.floor(self.current_instances * avg_predicted / self.target_utilization))
            )
        else:
            target_instances = self.current_instances
            
        # Risk assessment
        risk_assessment = self._assess_scaling_risk(
            final_action,
            current_utilization,
            predicted_loads,
            metrics
        )
        
        return ScalingDecision(
            action=final_action,
            confidence=confidence,
            reasoning=reasoning,
            target_instances=target_instances,
            predicted_load=max_predicted,
            expected_improvement=self._calculate_expected_improvement(final_action, target_instances),
            risk_assessment=risk_assessment,
            metadata={
                'current_instances': self.current_instances,
                'current_utilization': current_utilization,
                'predicted_loads': predicted_loads,
                'urgency': urgency
            }
        )
        
    def _assess_scaling_risk(
        self,
        action: ScalingAction,
        current_utilization: float,
        predicted_loads: List[float],
        metrics: ResourceMetrics
    ) -> float:
        """Assess risk of scaling decision."""
        risk = 0.0
        
        # High utilization risk
        if current_utilization > 0.9:
            risk += 0.3
            
        # Prediction uncertainty risk
        if predicted_loads:
            load_variance = np.var(predicted_loads)
            risk += min(0.2, load_variance * 5)
            
        # Error rate risk
        if metrics.error_rate > 0.05:
            risk += 0.2
            
        # Queue length risk
        if metrics.queue_length > 20:
            risk += 0.1
            
        # Scale-down risk during high load
        if action == ScalingAction.SCALE_DOWN and current_utilization > 0.6:
            risk += 0.3
            
        return min(1.0, risk)
        
    def _calculate_expected_improvement(self, action: ScalingAction, target_instances: int) -> float:
        """Calculate expected performance improvement."""
        if action == ScalingAction.NO_ACTION:
            return 0.0
            
        instance_ratio = target_instances / max(1, self.current_instances)
        
        if action in [ScalingAction.SCALE_UP, ScalingAction.PREEMPTIVE_SCALE]:
            # Expected improvement from scaling up
            return min(0.5, (instance_ratio - 1) * 0.3)
        else:
            # Cost savings from scaling down
            return min(0.3, (1 - instance_ratio) * 0.2)
            
    async def _execute_scaling(self, decision: ScalingDecision):
        """Execute scaling decision."""
        previous_instances = self.current_instances
        self.current_instances = decision.target_instances
        
        # Record scaling event
        self.scaling_events.append({
            'timestamp': time.time(),
            'action': decision.action.value,
            'from_instances': previous_instances,
            'to_instances': self.current_instances,
            'reasoning': decision.reasoning,
            'confidence': decision.confidence
        })
        
        logger.info(
            f"Scaling {decision.action.value}: {previous_instances} -> {self.current_instances} instances. "
            f"Confidence: {decision.confidence:.2%}. Reason: {decision.reasoning}"
        )
        
        # Simulate scaling delay
        await asyncio.sleep(0.1)
        
    def _record_decision(self, decision: ScalingDecision, metrics: ResourceMetrics):
        """Record decision for performance tracking."""
        self.decision_history.append({
            'timestamp': time.time(),
            'decision': decision,
            'metrics': metrics,
            'instances_before': self.current_instances
        })
        
        self.performance_metrics['total_decisions'] += 1
        
    def train_predictor(self):
        """Train the workload predictor with accumulated data."""
        if self.predictor.training_data:
            self.predictor.train()
            
    def adapt_thresholds(self):
        """Adapt scaling thresholds based on performance."""
        if not self.adaptive_thresholds or len(self.decision_history) < 50:
            return
            
        # Analyze recent performance
        recent_decisions = list(self.decision_history)[-50:]
        
        # Count false positives and negatives
        false_positives = sum(
            1 for d in recent_decisions
            if d['decision'].action == ScalingAction.SCALE_UP and
            self._was_unnecessary_scale_up(d)
        )
        
        false_negatives = sum(
            1 for d in recent_decisions
            if d['decision'].action == ScalingAction.NO_ACTION and
            self._should_have_scaled_up(d)
        )
        
        # Adjust thresholds
        if false_positives > 10:  # Too many unnecessary scale-ups
            self.scale_up_threshold += self.threshold_adaptation_rate
            logger.info(f"Adapted scale-up threshold to {self.scale_up_threshold:.3f}")
            
        if false_negatives > 10:  # Too many missed scale-ups
            self.scale_up_threshold -= self.threshold_adaptation_rate
            logger.info(f"Adapted scale-up threshold to {self.scale_up_threshold:.3f}")
            
        # Keep thresholds in reasonable range
        self.scale_up_threshold = max(0.6, min(0.95, self.scale_up_threshold))
        self.scale_down_threshold = max(0.3, min(0.7, self.scale_down_threshold))
        
    def _was_unnecessary_scale_up(self, decision_record: Dict) -> bool:
        """Check if a scale-up was unnecessary in retrospect."""
        # Would implement logic to check if utilization stayed low after scaling
        return False  # Placeholder
        
    def _should_have_scaled_up(self, decision_record: Dict) -> bool:
        """Check if we should have scaled up when we didn't."""
        # Would implement logic to check if utilization spiked after no-action
        return False  # Placeholder
        
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        total_events = len(self.scaling_events)
        if total_events == 0:
            return {'message': 'No scaling events recorded'}
            
        # Analyze scaling events
        scale_ups = sum(1 for e in self.scaling_events if 'up' in e['action'])
        scale_downs = sum(1 for e in self.scaling_events if 'down' in e['action'])
        
        # Calculate average confidence
        avg_confidence = np.mean([e['confidence'] for e in self.scaling_events])
        
        # Recent events
        recent_events = [e for e in self.scaling_events if time.time() - e['timestamp'] < 3600]
        
        return {
            'total_scaling_events': total_events,
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'current_instances': self.current_instances,
            'average_confidence': avg_confidence,
            'recent_events_1h': len(recent_events),
            'predictor_trained': self.predictor.is_trained,
            'adaptive_thresholds': {
                'scale_up': self.scale_up_threshold,
                'scale_down': self.scale_down_threshold
            },
            'performance_metrics': self.performance_metrics.copy()
        }


# Global auto-scaler instance
auto_scaler = IntelligentAutoScaler()