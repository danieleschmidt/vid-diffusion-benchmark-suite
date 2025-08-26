"""Emergent Capabilities Detection Framework for Video Diffusion Models.

This module implements advanced detection and analysis of emergent capabilities
in video diffusion models, including novel behaviors, unexpected competencies,
and breakthrough performance characteristics that emerge from model scaling
and architectural innovations.
"""

import time
import logging
import asyncio
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
import math
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn, F
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmergentCapabilityType(Enum):
    """Types of emergent capabilities that can be detected."""
    COMPOSITIONAL_REASONING = "compositional_reasoning"
    TEMPORAL_UNDERSTANDING = "temporal_understanding"
    PHYSICS_CONSISTENCY = "physics_consistency"
    ARTISTIC_STYLE_TRANSFER = "artistic_style_transfer"
    NARRATIVE_COHERENCE = "narrative_coherence"
    CROSS_MODAL_ALIGNMENT = "cross_modal_alignment"
    ABSTRACT_CONCEPT_VISUALIZATION = "abstract_concept_visualization"
    NOVEL_OBJECT_GENERATION = "novel_object_generation"
    SCENE_COMPOSITION = "scene_composition"
    MOTION_DYNAMICS = "motion_dynamics"
    CONTEXT_AWARENESS = "context_awareness"
    CREATIVE_EXTRAPOLATION = "creative_extrapolation"


class DetectionMethod(Enum):
    """Methods for detecting emergent capabilities."""
    STATISTICAL_ANALYSIS = "statistical_analysis"
    BEHAVIORAL_PROBING = "behavioral_probing"
    SCALING_LAW_ANALYSIS = "scaling_law_analysis"
    CAPABILITY_CLUSTERING = "capability_clustering"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    NOVELTY_DETECTION = "novelty_detection"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    EMERGENT_METRIC_TRACKING = "emergent_metric_tracking"


@dataclass
class EmergentCapability:
    """Represents a detected emergent capability."""
    capability_id: str
    capability_type: EmergentCapabilityType
    detection_method: DetectionMethod
    confidence_score: float
    evidence_strength: float
    detection_timestamp: float
    model_scale_threshold: Optional[float]
    performance_metrics: Dict[str, float]
    behavioral_evidence: List[Dict[str, Any]]
    statistical_significance: float
    novelty_score: float
    replication_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityEvidence:
    """Evidence supporting an emergent capability detection."""
    evidence_id: str
    capability_id: str
    evidence_type: str  # "performance", "behavioral", "qualitative", "statistical"
    data: Dict[str, Any]
    confidence: float
    timestamp: float
    source_model: str
    validation_status: str  # "pending", "validated", "rejected"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalingLawObservation:
    """Observation point for scaling law analysis."""
    model_size: float  # Model parameters
    compute_flops: float  # Training compute
    data_size: float  # Training data size
    performance_metrics: Dict[str, float]
    emergent_metrics: Dict[str, float]
    timestamp: float
    model_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmergentBehaviorProbe:
    """Probes for detecting specific emergent behaviors."""
    
    def __init__(self, probe_type: EmergentCapabilityType):
        self.probe_type = probe_type
        self.test_cases = self._generate_test_cases()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for this probe type."""
        
        if self.probe_type == EmergentCapabilityType.COMPOSITIONAL_REASONING:
            return [
                {
                    "prompt": "A red cube on top of a blue sphere next to a green cylinder",
                    "expected_behaviors": ["object_placement", "spatial_relationships", "color_accuracy"],
                    "complexity_level": "basic"
                },
                {
                    "prompt": "Three cats sitting in a row, each wearing a different colored hat, with a dog behind the middle cat",
                    "expected_behaviors": ["multiple_objects", "complex_spatial_arrangement", "attribute_assignment"],
                    "complexity_level": "intermediate"
                },
                {
                    "prompt": "A clockwork mechanism where gears turn to power a small bird that emerges from a wooden house every hour",
                    "expected_behaviors": ["mechanical_understanding", "temporal_sequences", "causal_relationships"],
                    "complexity_level": "advanced"
                }
            ]
        
        elif self.probe_type == EmergentCapabilityType.TEMPORAL_UNDERSTANDING:
            return [
                {
                    "prompt": "A flower blooming from bud to full bloom over the course of a day",
                    "expected_behaviors": ["natural_progression", "temporal_consistency", "biological_accuracy"],
                    "complexity_level": "basic"
                },
                {
                    "prompt": "A person aging from child to adult over several decades, showing life milestones",
                    "expected_behaviors": ["long_term_progression", "human_development", "contextual_changes"],
                    "complexity_level": "intermediate"
                },
                {
                    "prompt": "The evolution of a city skyline over 100 years, from rural to modern metropolis",
                    "expected_behaviors": ["historical_progression", "architectural_evolution", "societal_changes"],
                    "complexity_level": "advanced"
                }
            ]
        
        elif self.probe_type == EmergentCapabilityType.PHYSICS_CONSISTENCY:
            return [
                {
                    "prompt": "A ball rolling down a hill and coming to rest",
                    "expected_behaviors": ["gravity_effects", "momentum_conservation", "friction_modeling"],
                    "complexity_level": "basic"
                },
                {
                    "prompt": "Water flowing around obstacles in a stream, creating eddies and currents",
                    "expected_behaviors": ["fluid_dynamics", "conservation_laws", "turbulence_patterns"],
                    "complexity_level": "intermediate"
                },
                {
                    "prompt": "A complex Rube Goldberg machine with multiple interacting components",
                    "expected_behaviors": ["chain_reactions", "energy_transfer", "mechanical_principles"],
                    "complexity_level": "advanced"
                }
            ]
        
        elif self.probe_type == EmergentCapabilityType.CREATIVE_EXTRAPOLATION:
            return [
                {
                    "prompt": "A musical instrument that doesn't exist, being played by an alien musician",
                    "expected_behaviors": ["novel_design", "plausible_functionality", "creative_interpretation"],
                    "complexity_level": "basic"
                },
                {
                    "prompt": "A sport played in zero gravity with rules that make sense for that environment",
                    "expected_behaviors": ["environmental_adaptation", "logical_rule_creation", "physical_plausibility"],
                    "complexity_level": "intermediate"
                },
                {
                    "prompt": "A society where time flows backwards, showing daily life and interactions",
                    "expected_behaviors": ["conceptual_reversal", "logical_consistency", "imaginative_scenarios"],
                    "complexity_level": "advanced"
                }
            ]
        
        # Add more probe types as needed
        return [
            {
                "prompt": f"Test case for {self.probe_type.value}",
                "expected_behaviors": ["general_capability"],
                "complexity_level": "basic"
            }
        ]
    
    async def run_probe(self, model_adapter, evaluation_function: Callable) -> Dict[str, Any]:
        """Run the behavioral probe on a model."""
        results = {
            "probe_type": self.probe_type.value,
            "test_results": [],
            "overall_score": 0.0,
            "capability_evidence": [],
            "timestamp": time.time()
        }
        
        total_score = 0.0
        
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"Running probe test {i+1}/{len(self.test_cases)}: {test_case['complexity_level']}")
            
            try:
                # Generate video for test case
                generated_video = await self._generate_test_video(model_adapter, test_case)
                
                # Evaluate the generated video
                evaluation_results = await self._evaluate_test_result(
                    generated_video, test_case, evaluation_function
                )
                
                test_result = {
                    "test_case_id": i,
                    "prompt": test_case["prompt"],
                    "complexity_level": test_case["complexity_level"],
                    "expected_behaviors": test_case["expected_behaviors"],
                    "evaluation_results": evaluation_results,
                    "capability_score": evaluation_results.get("capability_score", 0.0),
                    "evidence_strength": evaluation_results.get("evidence_strength", 0.0)
                }
                
                results["test_results"].append(test_result)
                total_score += test_result["capability_score"]
                
                # Collect evidence if capability is detected
                if test_result["capability_score"] > 0.7:  # Threshold for capability detection
                    evidence = CapabilityEvidence(
                        evidence_id=f"evidence_{self.probe_type.value}_{i}_{int(time.time())}",
                        capability_id=f"cap_{self.probe_type.value}",
                        evidence_type="behavioral",
                        data={
                            "test_case": test_case,
                            "evaluation_results": evaluation_results,
                            "generated_video_metadata": self._extract_video_metadata(generated_video)
                        },
                        confidence=test_result["capability_score"],
                        timestamp=time.time(),
                        source_model=getattr(model_adapter, 'model_name', 'unknown'),
                        validation_status="pending"
                    )
                    results["capability_evidence"].append(evidence.to_dict())
                
            except Exception as e:
                self.logger.error(f"Probe test {i} failed: {e}")
                test_result = {
                    "test_case_id": i,
                    "error": str(e),
                    "capability_score": 0.0
                }
                results["test_results"].append(test_result)
        
        # Calculate overall score
        if len(self.test_cases) > 0:
            results["overall_score"] = total_score / len(self.test_cases)
        
        return results
    
    async def _generate_test_video(self, model_adapter, test_case: Dict[str, Any]) -> Any:
        """Generate video for a test case."""
        # Mock video generation - replace with actual model call
        prompt = test_case["prompt"]
        
        # Simulate generation time based on complexity
        complexity_delays = {"basic": 1.0, "intermediate": 2.0, "advanced": 3.0}
        delay = complexity_delays.get(test_case["complexity_level"], 1.0)
        await asyncio.sleep(delay)
        
        # Create mock video tensor
        if TORCH_AVAILABLE:
            video = torch.randn(16, 3, 64, 64)  # 16 frames, 3 channels, 64x64
        else:
            video = type('MockTensor', (), {'shape': (16, 3, 64, 64)})()
        
        return video
    
    async def _evaluate_test_result(self, video: Any, test_case: Dict[str, Any], 
                                  evaluation_function: Callable) -> Dict[str, Any]:
        """Evaluate generated video for emergent capabilities."""
        
        # Use evaluation function if provided
        if evaluation_function:
            try:
                eval_results = await asyncio.get_event_loop().run_in_executor(
                    None, evaluation_function, video, test_case
                )
                return eval_results
            except Exception as e:
                self.logger.error(f"Evaluation function failed: {e}")
        
        # Fallback to mock evaluation
        return self._mock_evaluation(video, test_case)
    
    def _mock_evaluation(self, video: Any, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Mock evaluation for testing purposes."""
        
        # Simulate capability assessment based on complexity and probe type
        complexity_multipliers = {"basic": 0.8, "intermediate": 0.6, "advanced": 0.4}
        base_score = complexity_multipliers.get(test_case["complexity_level"], 0.5)
        
        # Add probe-specific scoring
        probe_bonuses = {
            EmergentCapabilityType.COMPOSITIONAL_REASONING: 0.2,
            EmergentCapabilityType.TEMPORAL_UNDERSTANDING: 0.15,
            EmergentCapabilityType.PHYSICS_CONSISTENCY: 0.1,
            EmergentCapabilityType.CREATIVE_EXTRAPOLATION: 0.25
        }
        
        bonus = probe_bonuses.get(self.probe_type, 0.1)
        
        # Calculate scores with some randomness
        capability_score = min(1.0, base_score + bonus + np.random.normal(0, 0.1))
        capability_score = max(0.0, capability_score)
        
        evidence_strength = capability_score * 0.8 + np.secrets.SystemRandom().uniform(0, 0.2)
        evidence_strength = max(0.0, min(1.0, evidence_strength))
        
        # Generate behavioral analysis
        expected_behaviors = test_case.get("expected_behaviors", [])
        behavior_scores = {}
        
        for behavior in expected_behaviors:
            behavior_scores[behavior] = min(1.0, max(0.0, capability_score + np.random.normal(0, 0.15)))
        
        return {
            "capability_score": capability_score,
            "evidence_strength": evidence_strength,
            "behavior_scores": behavior_scores,
            "overall_quality": capability_score * 0.9,
            "novelty_score": np.secrets.SystemRandom().uniform(0.3, 0.8),
            "complexity_handling": base_score + 0.2,
            "detailed_analysis": {
                "prompt_adherence": capability_score * 0.9,
                "technical_quality": capability_score * 0.8,
                "emergent_properties": evidence_strength
            }
        }
    
    def _extract_video_metadata(self, video: Any) -> Dict[str, Any]:
        """Extract metadata from generated video."""
        if hasattr(video, 'shape'):
            return {
                "shape": video.shape,
                "dtype": str(getattr(video, 'dtype', 'unknown')),
                "size_mb": np.prod(video.shape) * 4 / (1024 * 1024) if hasattr(video, 'shape') else 0
            }
        else:
            return {"type": "mock_video", "size_mb": 10.0}


class ScalingLawAnalyzer:
    """Analyzes scaling laws to detect emergent capabilities."""
    
    def __init__(self):
        self.observations: List[ScalingLawObservation] = []
        self.scaling_curves: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.emergence_thresholds: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_observation(self, observation: ScalingLawObservation):
        """Add a scaling law observation."""
        self.observations.append(observation)
        
        # Update scaling curves for each metric
        for metric_name, metric_value in observation.performance_metrics.items():
            self.scaling_curves[metric_name].append((observation.model_size, metric_value))
        
        for metric_name, metric_value in observation.emergent_metrics.items():
            emergent_key = f"emergent_{metric_name}"
            self.scaling_curves[emergent_key].append((observation.model_size, metric_value))
        
        self.logger.debug(f"Added scaling observation for model {observation.model_id} "
                         f"(size: {observation.model_size:.2e} params)")
    
    def detect_emergence_points(self, metric_name: str, 
                               smoothing_window: int = 5) -> List[Dict[str, Any]]:
        """Detect points where emergent capabilities appear in scaling curves."""
        
        if metric_name not in self.scaling_curves:
            return []
        
        curve_data = sorted(self.scaling_curves[metric_name])
        if len(curve_data) < smoothing_window * 2:
            return []
        
        # Extract scales and values
        scales = np.array([point[0] for point in curve_data])
        values = np.array([point[1] for point in curve_data])
        
        # Apply smoothing
        smoothed_values = self._smooth_curve(values, smoothing_window)
        
        # Calculate derivatives to find sharp transitions
        derivatives = np.gradient(smoothed_values, scales)
        second_derivatives = np.gradient(derivatives, scales)
        
        # Find emergence points (sharp increases in capability)
        emergence_points = []
        
        for i in range(smoothing_window, len(derivatives) - smoothing_window):
            # Look for significant positive acceleration
            if (second_derivatives[i] > np.percentile(second_derivatives, 80) and
                derivatives[i] > 0 and
                values[i] > np.percentile(values, 60)):
                
                # Verify this is a genuine emergence (not noise)
                if self._validate_emergence_point(scales, values, i, smoothing_window):
                    emergence_point = {
                        "metric_name": metric_name,
                        "emergence_scale": scales[i],
                        "emergence_value": values[i],
                        "derivative": derivatives[i],
                        "second_derivative": second_derivatives[i],
                        "confidence": self._calculate_emergence_confidence(
                            scales, values, i, smoothing_window
                        ),
                        "data_index": i
                    }
                    emergence_points.append(emergence_point)
        
        return emergence_points
    
    def _smooth_curve(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Apply smoothing to curve data."""
        if len(values) < window_size:
            return values
        
        # Simple moving average smoothing
        smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='same')
        
        # Handle edges
        for i in range(window_size // 2):
            smoothed[i] = np.mean(values[:i + window_size // 2 + 1])
            smoothed[-(i + 1)] = np.mean(values[-(i + window_size // 2 + 1):])
        
        return smoothed
    
    def _validate_emergence_point(self, scales: np.ndarray, values: np.ndarray, 
                                index: int, window_size: int) -> bool:
        """Validate that a detected point represents genuine emergence."""
        
        # Check for sufficient data before and after
        if index < window_size or index >= len(values) - window_size:
            return False
        
        # Compare before and after regions
        before_region = values[index - window_size:index]
        after_region = values[index:index + window_size]
        
        # Emergence should show significant improvement
        before_mean = np.mean(before_region)
        after_mean = np.mean(after_region)
        
        improvement_ratio = after_mean / max(before_mean, 1e-10)
        
        # Require at least 20% improvement
        if improvement_ratio < 1.2:
            return False
        
        # Check for statistical significance
        try:
            t_stat, p_value = stats.ttest_ind(before_region, after_region)
            if p_value > 0.05:  # Not statistically significant
                return False
        except:
            return False
        
        return True
    
    def _calculate_emergence_confidence(self, scales: np.ndarray, values: np.ndarray,
                                      index: int, window_size: int) -> float:
        """Calculate confidence score for an emergence point."""
        
        # Factors contributing to confidence:
        # 1. Magnitude of improvement
        # 2. Statistical significance
        # 3. Consistency of trend
        # 4. Data quality
        
        before_region = values[max(0, index - window_size):index]
        after_region = values[index:min(len(values), index + window_size)]
        
        if len(before_region) == 0 or len(after_region) == 0:
            return 0.0
        
        # Improvement magnitude
        improvement = np.mean(after_region) / max(np.mean(before_region), 1e-10)
        improvement_score = min(1.0, (improvement - 1.0) / 2.0)  # Normalize
        
        # Trend consistency
        before_trend = np.polyfit(range(len(before_region)), before_region, 1)[0]
        after_trend = np.polyfit(range(len(after_region)), after_region, 1)[0]
        
        trend_change = after_trend - before_trend
        trend_score = min(1.0, max(0.0, trend_change / max(np.std(values), 1e-10)))
        
        # Statistical significance
        try:
            _, p_value = stats.ttest_ind(before_region, after_region)
            sig_score = 1.0 - p_value
        except:
            sig_score = 0.0
        
        # Data quality (based on amount of data)
        data_quality = min(1.0, len(before_region) / window_size)
        
        # Combined confidence
        confidence = 0.4 * improvement_score + 0.3 * sig_score + 0.2 * trend_score + 0.1 * data_quality
        
        return max(0.0, min(1.0, confidence))
    
    def analyze_scaling_laws(self) -> Dict[str, Any]:
        """Perform comprehensive scaling law analysis."""
        
        if len(self.observations) < 5:
            return {"error": "Insufficient observations for scaling analysis"}
        
        analysis_results = {
            "total_observations": len(self.observations),
            "scaling_curves": {},
            "emergence_points": {},
            "scaling_law_fits": {},
            "capability_thresholds": {},
            "predictions": {}
        }
        
        # Analyze each metric
        for metric_name in self.scaling_curves:
            if len(self.scaling_curves[metric_name]) < 5:
                continue
            
            # Detect emergence points
            emergence_points = self.detect_emergence_points(metric_name)
            if emergence_points:
                analysis_results["emergence_points"][metric_name] = emergence_points
            
            # Fit scaling law
            scaling_fit = self._fit_scaling_law(metric_name)
            if scaling_fit:
                analysis_results["scaling_law_fits"][metric_name] = scaling_fit
            
            # Predict capability thresholds
            threshold = self._predict_capability_threshold(metric_name)
            if threshold:
                analysis_results["capability_thresholds"][metric_name] = threshold
        
        # Generate predictions for future scaling
        analysis_results["predictions"] = self._generate_scaling_predictions()
        
        return analysis_results
    
    def _fit_scaling_law(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Fit a scaling law (power law) to metric data."""
        
        curve_data = sorted(self.scaling_curves[metric_name])
        if len(curve_data) < 5:
            return None
        
        scales = np.array([point[0] for point in curve_data])
        values = np.array([point[1] for point in curve_data])
        
        # Remove zero or negative values for log fitting
        valid_indices = (scales > 0) & (values > 0)
        if np.sum(valid_indices) < 5:
            return None
        
        log_scales = np.log10(scales[valid_indices])
        log_values = np.log10(values[valid_indices])
        
        try:
            # Fit power law: value = a * scale^b
            coeffs = np.polyfit(log_scales, log_values, 1)
            b, log_a = coeffs[0], coeffs[1]
            a = 10 ** log_a
            
            # Calculate R-squared
            predicted_log_values = np.polyval(coeffs, log_scales)
            r_squared = 1 - np.sum((log_values - predicted_log_values) ** 2) / \
                       np.sum((log_values - np.mean(log_values)) ** 2)
            
            return {
                "coefficient_a": a,
                "exponent_b": b,
                "r_squared": r_squared,
                "equation": f"{metric_name} = {a:.3e} * scale^{b:.3f}",
                "fit_quality": "good" if r_squared > 0.8 else "moderate" if r_squared > 0.6 else "poor"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fit scaling law for {metric_name}: {e}")
            return None
    
    def _predict_capability_threshold(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Predict the scale at which a capability emerges."""
        
        emergence_points = self.detect_emergence_points(metric_name)
        if not emergence_points:
            return None
        
        # Use the first strong emergence point
        strong_emergence = None
        for point in emergence_points:
            if point["confidence"] > 0.7:
                strong_emergence = point
                break
        
        if not strong_emergence:
            strong_emergence = max(emergence_points, key=lambda x: x["confidence"])
        
        # Predict threshold with confidence interval
        threshold_scale = strong_emergence["emergence_scale"]
        confidence = strong_emergence["confidence"]
        
        # Estimate uncertainty based on data density around emergence point
        curve_data = sorted(self.scaling_curves[metric_name])
        scales = np.array([point[0] for point in curve_data])
        
        nearby_scales = scales[np.abs(scales - threshold_scale) / threshold_scale < 0.5]
        data_density = len(nearby_scales) / len(scales)
        
        uncertainty = 0.5 - 0.4 * data_density  # Less uncertainty with more data
        
        return {
            "threshold_scale": threshold_scale,
            "threshold_value": strong_emergence["emergence_value"],
            "confidence": confidence,
            "uncertainty": uncertainty,
            "lower_bound": threshold_scale * (1 - uncertainty),
            "upper_bound": threshold_scale * (1 + uncertainty)
        }
    
    def _generate_scaling_predictions(self) -> Dict[str, Any]:
        """Generate predictions for future scaling behavior."""
        
        predictions = {}
        
        # Predict emergence at larger scales
        max_observed_scale = max(obs.model_size for obs in self.observations)
        future_scales = [max_observed_scale * 2, max_observed_scale * 5, max_observed_scale * 10]
        
        for metric_name, curve_data in self.scaling_curves.items():
            if len(curve_data) < 5:
                continue
            
            scaling_fit = self._fit_scaling_law(metric_name)
            if not scaling_fit or scaling_fit["r_squared"] < 0.6:
                continue
            
            a, b = scaling_fit["coefficient_a"], scaling_fit["exponent_b"]
            
            metric_predictions = []
            for future_scale in future_scales:
                predicted_value = a * (future_scale ** b)
                
                # Estimate prediction confidence based on extrapolation distance
                extrapolation_factor = future_scale / max_observed_scale
                confidence = max(0.1, 1.0 - 0.2 * np.log10(extrapolation_factor))
                
                metric_predictions.append({
                    "scale": future_scale,
                    "predicted_value": predicted_value,
                    "confidence": confidence
                })
            
            predictions[metric_name] = metric_predictions
        
        return predictions


class CapabilityClusterAnalyzer:
    """Analyzes capability emergence through clustering and pattern detection."""
    
    def __init__(self):
        self.capability_vectors: List[Tuple[str, np.ndarray]] = []
        self.cluster_models = {}
        self.capability_patterns = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_capability_vector(self, model_id: str, capabilities: Dict[str, float]):
        """Add a capability vector for a model."""
        
        # Convert capabilities to vector
        capability_names = sorted(capabilities.keys())
        vector = np.array([capabilities[name] for name in capability_names])
        
        self.capability_vectors.append((model_id, vector))
        
        self.logger.debug(f"Added capability vector for {model_id} "
                         f"({len(capabilities)} capabilities)")
    
    def detect_capability_clusters(self, min_samples: int = 3, eps: float = 0.3) -> Dict[str, Any]:
        """Detect clusters of similar capability profiles."""
        
        if len(self.capability_vectors) < min_samples:
            return {"error": "Insufficient data for clustering"}
        
        # Prepare data
        model_ids = [item[0] for item in self.capability_vectors]
        vectors = np.array([item[1] for item in self.capability_vectors])
        
        # Normalize vectors
        scaler = StandardScaler()
        normalized_vectors = scaler.fit_transform(vectors)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(normalized_vectors)
        
        # Analyze clusters
        clusters = defaultdict(list)
        noise_points = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise point
                noise_points.append(model_ids[i])
            else:
                clusters[label].append((model_ids[i], vectors[i]))
        
        # Characterize each cluster
        cluster_analysis = {}
        
        for cluster_id, cluster_members in clusters.items():
            member_ids = [member[0] for member in cluster_members]
            member_vectors = np.array([member[1] for member in cluster_members])
            
            # Calculate cluster statistics
            cluster_center = np.mean(member_vectors, axis=0)
            cluster_std = np.std(member_vectors, axis=0)
            
            # Identify distinctive capabilities
            overall_mean = np.mean(vectors, axis=0)
            distinctive_capabilities = []
            
            for i, (cap_score, overall_score) in enumerate(zip(cluster_center, overall_mean)):
                if cap_score > overall_score + np.std(vectors[:, i]):
                    distinctive_capabilities.append({
                        "capability_index": i,
                        "cluster_score": cap_score,
                        "overall_score": overall_score,
                        "distinctiveness": cap_score - overall_score
                    })
            
            cluster_analysis[cluster_id] = {
                "members": member_ids,
                "size": len(member_ids),
                "center": cluster_center.tolist(),
                "std_dev": cluster_std.tolist(),
                "distinctive_capabilities": distinctive_capabilities,
                "cohesion": self._calculate_cluster_cohesion(member_vectors),
                "separation": self._calculate_cluster_separation(cluster_center, 
                                                               [np.mean(np.array([m[1] for m in other_cluster]))
                                                                for other_cluster in clusters.values()
                                                                if other_cluster != cluster_members])
            }
        
        return {
            "num_clusters": len(clusters),
            "cluster_analysis": cluster_analysis,
            "noise_points": noise_points,
            "silhouette_score": self._calculate_silhouette_score(normalized_vectors, cluster_labels),
            "clustering_params": {"eps": eps, "min_samples": min_samples}
        }
    
    def _calculate_cluster_cohesion(self, cluster_vectors: np.ndarray) -> float:
        """Calculate how cohesive a cluster is (lower is better)."""
        if len(cluster_vectors) < 2:
            return 0.0
        
        cluster_center = np.mean(cluster_vectors, axis=0)
        distances = [np.linalg.norm(vector - cluster_center) for vector in cluster_vectors]
        return np.mean(distances)
    
    def _calculate_cluster_separation(self, cluster_center: np.ndarray, 
                                    other_centers: List[np.ndarray]) -> float:
        """Calculate how separated a cluster is from others (higher is better)."""
        if not other_centers:
            return float('inf')
        
        distances = [np.linalg.norm(cluster_center - other_center) 
                    for other_center in other_centers]
        return np.min(distances)
    
    def _calculate_silhouette_score(self, vectors: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(vectors, labels)
        except ImportError:
            # Simplified silhouette calculation
            return 0.5  # Placeholder
    
    def detect_emergent_patterns(self) -> Dict[str, Any]:
        """Detect patterns in capability emergence across models."""
        
        if len(self.capability_vectors) < 5:
            return {"error": "Insufficient data for pattern detection"}
        
        patterns = {}
        
        # Pattern 1: Capability correlation analysis
        vectors = np.array([item[1] for item in self.capability_vectors])
        correlation_matrix = np.corrcoef(vectors.T)
        
        # Find highly correlated capabilities
        high_correlations = []
        n_capabilities = correlation_matrix.shape[0]
        
        for i in range(n_capabilities):
            for j in range(i + 1, n_capabilities):
                if abs(correlation_matrix[i, j]) > 0.7:
                    high_correlations.append({
                        "capability_1": i,
                        "capability_2": j,
                        "correlation": correlation_matrix[i, j],
                        "relationship": "positive" if correlation_matrix[i, j] > 0 else "negative"
                    })
        
        patterns["capability_correlations"] = high_correlations
        
        # Pattern 2: Capability progression analysis
        # Look for capabilities that tend to emerge together or in sequence
        progression_patterns = self._analyze_capability_progression(vectors)
        patterns["progression_patterns"] = progression_patterns
        
        # Pattern 3: Outlier detection
        outliers = self._detect_capability_outliers(vectors)
        patterns["outlier_models"] = outliers
        
        return patterns
    
    def _analyze_capability_progression(self, vectors: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze how capabilities emerge in progression."""
        
        # Sort models by overall capability level
        overall_scores = np.mean(vectors, axis=1)
        sorted_indices = np.argsort(overall_scores)
        sorted_vectors = vectors[sorted_indices]
        
        progressions = []
        
        # Look for capabilities that emerge at similar points
        for i in range(vectors.shape[1]):  # For each capability
            capability_scores = sorted_vectors[:, i]
            
            # Find the emergence point (where capability significantly increases)
            emergence_point = self._find_emergence_point(capability_scores)
            
            if emergence_point is not None:
                progressions.append({
                    "capability_index": i,
                    "emergence_point": emergence_point,
                    "emergence_model_rank": emergence_point,
                    "pre_emergence_mean": np.mean(capability_scores[:emergence_point]),
                    "post_emergence_mean": np.mean(capability_scores[emergence_point:])
                })
        
        # Group capabilities by emergence point
        emergence_groups = defaultdict(list)
        for prog in progressions:
            emergence_groups[prog["emergence_point"]].append(prog["capability_index"])
        
        return [
            {
                "emergence_point": point,
                "capabilities": caps,
                "group_size": len(caps)
            }
            for point, caps in emergence_groups.items()
            if len(caps) > 1  # Only groups with multiple capabilities
        ]
    
    def _find_emergence_point(self, scores: np.ndarray) -> Optional[int]:
        """Find the point where a capability emerges."""
        
        if len(scores) < 5:
            return None
        
        # Use sliding window to detect significant increases
        window_size = max(2, len(scores) // 5)
        
        for i in range(window_size, len(scores) - window_size):
            before_mean = np.mean(scores[:i])
            after_mean = np.mean(scores[i:])
            
            # Check for significant improvement
            if after_mean > before_mean * 1.5 and after_mean > 0.5:
                return i
        
        return None
    
    def _detect_capability_outliers(self, vectors: np.ndarray) -> List[Dict[str, Any]]:
        """Detect models with unusual capability profiles."""
        
        outliers = []
        
        # Use isolation forest or simple statistical outlier detection
        means = np.mean(vectors, axis=0)
        stds = np.std(vectors, axis=0)
        
        for i, vector in enumerate(vectors):
            # Calculate z-scores for each capability
            z_scores = np.abs((vector - means) / (stds + 1e-10))
            
            # Model is outlier if it has extreme values in multiple capabilities
            extreme_capabilities = np.sum(z_scores > 2.5)
            
            if extreme_capabilities > len(vector) * 0.2:  # 20% of capabilities are extreme
                model_id = self.capability_vectors[i][0]
                outliers.append({
                    "model_id": model_id,
                    "extreme_capabilities": extreme_capabilities,
                    "max_z_score": np.max(z_scores),
                    "outlier_type": "high_performer" if np.mean(vector) > np.mean(means) else "unusual_profile"
                })
        
        return outliers


class EmergentCapabilitiesDetector:
    """Main detector for emergent capabilities in video diffusion models."""
    
    def __init__(self, detection_methods: List[DetectionMethod] = None):
        if detection_methods is None:
            detection_methods = [
                DetectionMethod.BEHAVIORAL_PROBING,
                DetectionMethod.SCALING_LAW_ANALYSIS,
                DetectionMethod.CAPABILITY_CLUSTERING
            ]
        
        self.detection_methods = detection_methods
        self.behavioral_probes = self._initialize_probes()
        self.scaling_analyzer = ScalingLawAnalyzer()
        self.cluster_analyzer = CapabilityClusterAnalyzer()
        self.detected_capabilities: List[EmergentCapability] = []
        self.evidence_database: List[CapabilityEvidence] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _initialize_probes(self) -> Dict[EmergentCapabilityType, EmergentBehaviorProbe]:
        """Initialize behavioral probes for different capability types."""
        probes = {}
        
        capability_types = [
            EmergentCapabilityType.COMPOSITIONAL_REASONING,
            EmergentCapabilityType.TEMPORAL_UNDERSTANDING,
            EmergentCapabilityType.PHYSICS_CONSISTENCY,
            EmergentCapabilityType.CREATIVE_EXTRAPOLATION
        ]
        
        for cap_type in capability_types:
            probes[cap_type] = EmergentBehaviorProbe(cap_type)
        
        return probes
    
    async def comprehensive_capability_detection(self, 
                                               model_adapter,
                                               model_metadata: Dict[str, Any],
                                               evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Perform comprehensive emergent capability detection.
        
        Args:
            model_adapter: Model to analyze
            model_metadata: Model size, training data, etc.
            evaluation_function: Custom evaluation function
        
        Returns:
            Comprehensive detection results
        """
        
        self.logger.info("Starting comprehensive emergent capability detection")
        
        detection_results = {
            "model_id": model_metadata.get("model_id", "unknown"),
            "detection_timestamp": time.time(),
            "detection_methods": [method.value for method in self.detection_methods],
            "behavioral_probe_results": {},
            "scaling_analysis": {},
            "cluster_analysis": {},
            "detected_capabilities": [],
            "evidence_summary": {},
            "overall_assessment": {}
        }
        
        # 1. Behavioral Probing
        if DetectionMethod.BEHAVIORAL_PROBING in self.detection_methods:
            detection_results["behavioral_probe_results"] = await self._run_behavioral_probes(
                model_adapter, evaluation_function
            )
        
        # 2. Scaling Law Analysis
        if DetectionMethod.SCALING_LAW_ANALYSIS in self.detection_methods:
            # Add current model to scaling observations
            if "model_size" in model_metadata:
                observation = ScalingLawObservation(
                    model_size=model_metadata["model_size"],
                    compute_flops=model_metadata.get("compute_flops", 0),
                    data_size=model_metadata.get("data_size", 0),
                    performance_metrics=model_metadata.get("performance_metrics", {}),
                    emergent_metrics=model_metadata.get("emergent_metrics", {}),
                    timestamp=time.time(),
                    model_id=model_metadata.get("model_id", "unknown")
                )
                
                self.scaling_analyzer.add_observation(observation)
                detection_results["scaling_analysis"] = self.scaling_analyzer.analyze_scaling_laws()
        
        # 3. Capability Clustering
        if DetectionMethod.CAPABILITY_CLUSTERING in self.detection_methods:
            # Extract capability scores from behavioral probes
            capability_scores = {}
            for probe_type, probe_results in detection_results["behavioral_probe_results"].items():
                capability_scores[probe_type] = probe_results.get("overall_score", 0.0)
            
            if capability_scores:
                self.cluster_analyzer.add_capability_vector(
                    model_metadata.get("model_id", "unknown"), capability_scores
                )
                
                cluster_results = self.cluster_analyzer.detect_capability_clusters()
                pattern_results = self.cluster_analyzer.detect_emergent_patterns()
                
                detection_results["cluster_analysis"] = {
                    "clustering": cluster_results,
                    "patterns": pattern_results
                }
        
        # 4. Synthesize detected capabilities
        detected_capabilities = await self._synthesize_capabilities(detection_results)
        detection_results["detected_capabilities"] = [cap.to_dict() for cap in detected_capabilities]
        
        # 5. Generate overall assessment
        detection_results["overall_assessment"] = self._generate_overall_assessment(
            detection_results, detected_capabilities
        )
        
        # Store results
        self.detected_capabilities.extend(detected_capabilities)
        
        self.logger.info(f"Detection complete. Found {len(detected_capabilities)} emergent capabilities")
        
        return detection_results
    
    async def _run_behavioral_probes(self, model_adapter, evaluation_function: Optional[Callable]) -> Dict[str, Any]:
        """Run all behavioral probes."""
        
        probe_results = {}
        
        for cap_type, probe in self.behavioral_probes.items():
            self.logger.info(f"Running behavioral probe: {cap_type.value}")
            
            try:
                result = await probe.run_probe(model_adapter, evaluation_function)
                probe_results[cap_type.value] = result
                
                # Collect evidence
                for evidence_dict in result.get("capability_evidence", []):
                    evidence = CapabilityEvidence(**evidence_dict)
                    self.evidence_database.append(evidence)
                
            except Exception as e:
                self.logger.error(f"Behavioral probe {cap_type.value} failed: {e}")
                probe_results[cap_type.value] = {"error": str(e), "overall_score": 0.0}
        
        return probe_results
    
    async def _synthesize_capabilities(self, detection_results: Dict[str, Any]) -> List[EmergentCapability]:
        """Synthesize detected capabilities from all detection methods."""
        
        capabilities = []
        
        # Process behavioral probe results
        for probe_type, probe_results in detection_results.get("behavioral_probe_results", {}).items():
            if probe_results.get("overall_score", 0) > 0.7:  # Threshold for capability detection
                
                # Calculate statistical significance
                test_scores = [test["capability_score"] for test in probe_results.get("test_results", [])
                              if "capability_score" in test]
                
                if test_scores:
                    # Simple statistical test
                    mean_score = np.mean(test_scores)
                    std_score = np.std(test_scores)
                    t_stat = mean_score / (std_score / np.sqrt(len(test_scores)) + 1e-10)
                    
                    # Convert to p-value approximation
                    statistical_significance = 1.0 - 2 * stats.norm.cdf(-abs(t_stat))
                else:
                    statistical_significance = 0.5
                
                capability = EmergentCapability(
                    capability_id=f"cap_{probe_type}_{int(time.time())}",
                    capability_type=EmergentCapabilityType(probe_type),
                    detection_method=DetectionMethod.BEHAVIORAL_PROBING,
                    confidence_score=probe_results["overall_score"],
                    evidence_strength=np.mean([test.get("evidence_strength", 0) 
                                             for test in probe_results.get("test_results", [])]),
                    detection_timestamp=time.time(),
                    model_scale_threshold=None,
                    performance_metrics={"overall_score": probe_results["overall_score"]},
                    behavioral_evidence=probe_results.get("test_results", []),
                    statistical_significance=statistical_significance,
                    novelty_score=np.mean([test.get("novelty_score", 0.5) 
                                         for test in probe_results.get("test_results", [])]),
                    replication_count=len(probe_results.get("test_results", []))
                )
                
                capabilities.append(capability)
        
        # Process scaling law results
        scaling_analysis = detection_results.get("scaling_analysis", {})
        emergence_points = scaling_analysis.get("emergence_points", {})
        
        for metric_name, points in emergence_points.items():
            for point in points:
                if point["confidence"] > 0.7:
                    capability = EmergentCapability(
                        capability_id=f"cap_scaling_{metric_name}_{int(time.time())}",
                        capability_type=self._infer_capability_type_from_metric(metric_name),
                        detection_method=DetectionMethod.SCALING_LAW_ANALYSIS,
                        confidence_score=point["confidence"],
                        evidence_strength=point["confidence"],
                        detection_timestamp=time.time(),
                        model_scale_threshold=point["emergence_scale"],
                        performance_metrics={metric_name: point["emergence_value"]},
                        behavioral_evidence=[],
                        statistical_significance=0.8,  # Scaling laws are generally significant
                        novelty_score=0.6,  # Moderate novelty for scaling emergence
                        replication_count=1
                    )
                    
                    capabilities.append(capability)
        
        return capabilities
    
    def _infer_capability_type_from_metric(self, metric_name: str) -> EmergentCapabilityType:
        """Infer capability type from metric name."""
        
        metric_mappings = {
            "temporal_consistency": EmergentCapabilityType.TEMPORAL_UNDERSTANDING,
            "physics_consistency": EmergentCapabilityType.PHYSICS_CONSISTENCY,
            "compositional_score": EmergentCapabilityType.COMPOSITIONAL_REASONING,
            "creativity_score": EmergentCapabilityType.CREATIVE_EXTRAPOLATION,
            "narrative_coherence": EmergentCapabilityType.NARRATIVE_COHERENCE
        }
        
        for key, cap_type in metric_mappings.items():
            if key in metric_name.lower():
                return cap_type
        
        # Default fallback
        return EmergentCapabilityType.NOVEL_OBJECT_GENERATION
    
    def _generate_overall_assessment(self, detection_results: Dict[str, Any], 
                                   capabilities: List[EmergentCapability]) -> Dict[str, Any]:
        """Generate overall assessment of emergent capabilities."""
        
        assessment = {
            "capability_count": len(capabilities),
            "high_confidence_capabilities": len([c for c in capabilities if c.confidence_score > 0.8]),
            "average_confidence": np.mean([c.confidence_score for c in capabilities]) if capabilities else 0.0,
            "capability_types_detected": list(set(c.capability_type.value for c in capabilities)),
            "detection_strength": "strong" if len(capabilities) > 3 else "moderate" if len(capabilities) > 1 else "weak",
            "statistical_reliability": np.mean([c.statistical_significance for c in capabilities]) if capabilities else 0.0,
            "novelty_assessment": np.mean([c.novelty_score for c in capabilities]) if capabilities else 0.0
        }
        
        # Generate recommendations
        recommendations = []
        
        if assessment["capability_count"] == 0:
            recommendations.append("No significant emergent capabilities detected. Consider larger scale or architectural changes.")
        elif assessment["average_confidence"] < 0.6:
            recommendations.append("Detected capabilities have low confidence. Additional validation recommended.")
        elif assessment["high_confidence_capabilities"] > 2:
            recommendations.append("Multiple strong emergent capabilities detected. Model shows significant advancement.")
        
        if assessment["statistical_reliability"] < 0.7:
            recommendations.append("Statistical reliability is low. Increase sample size or replication count.")
        
        if assessment["novelty_assessment"] > 0.8:
            recommendations.append("High novelty detected. Consider detailed analysis of novel behaviors.")
        
        assessment["recommendations"] = recommendations
        
        return assessment
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of all detected capabilities."""
        
        if not self.detected_capabilities:
            return {"message": "No capabilities detected yet"}
        
        # Group by capability type
        by_type = defaultdict(list)
        for cap in self.detected_capabilities:
            by_type[cap.capability_type.value].append(cap)
        
        summary = {
            "total_capabilities": len(self.detected_capabilities),
            "capability_types": {
                cap_type: {
                    "count": len(caps),
                    "average_confidence": np.mean([c.confidence_score for c in caps]),
                    "high_confidence_count": len([c for c in caps if c.confidence_score > 0.8])
                }
                for cap_type, caps in by_type.items()
            },
            "detection_methods_used": list(set(c.detection_method.value for c in self.detected_capabilities)),
            "evidence_count": len(self.evidence_database),
            "timeline": [
                {
                    "capability_id": cap.capability_id,
                    "type": cap.capability_type.value,
                    "confidence": cap.confidence_score,
                    "timestamp": cap.detection_timestamp
                }
                for cap in sorted(self.detected_capabilities, key=lambda x: x.detection_timestamp)
            ]
        }
        
        return summary
    
    def export_detection_results(self, filepath: str):
        """Export all detection results."""
        
        export_data = {
            "detector_info": {
                "detection_methods": [method.value for method in self.detection_methods],
                "probe_types": list(self.behavioral_probes.keys()),
                "export_timestamp": time.time()
            },
            "detected_capabilities": [cap.to_dict() for cap in self.detected_capabilities],
            "evidence_database": [ev.to_dict() for ev in self.evidence_database],
            "capability_summary": self.get_capability_summary(),
            "scaling_observations": [obs.to_dict() for obs in self.scaling_analyzer.observations],
            "capability_vectors": [(model_id, vector.tolist()) for model_id, vector in self.cluster_analyzer.capability_vectors]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported detection results to {filepath}")


# Example usage and testing
async def run_emergent_capabilities_example():
    """Example of emergent capabilities detection."""
    
    print("=== Emergent Capabilities Detection Example ===")
    
    # Create detector
    detector = EmergentCapabilitiesDetector()
    
    # Mock model adapter
    class MockModelAdapter:
        def __init__(self, model_name: str):
            self.model_name = model_name
    
    # Test multiple models with different scales
    test_models = [
        {
            "adapter": MockModelAdapter("small_model"),
            "metadata": {
                "model_id": "small_model",
                "model_size": 1e8,  # 100M parameters
                "compute_flops": 1e18,
                "data_size": 1e6,
                "performance_metrics": {
                    "fvd_score": 120.0,
                    "inception_score": 25.0,
                    "temporal_consistency": 0.6
                },
                "emergent_metrics": {
                    "compositional_reasoning": 0.3,
                    "physics_consistency": 0.4
                }
            }
        },
        {
            "adapter": MockModelAdapter("medium_model"),
            "metadata": {
                "model_id": "medium_model",
                "model_size": 1e9,  # 1B parameters
                "compute_flops": 1e19,
                "data_size": 1e7,
                "performance_metrics": {
                    "fvd_score": 100.0,
                    "inception_score": 35.0,
                    "temporal_consistency": 0.7
                },
                "emergent_metrics": {
                    "compositional_reasoning": 0.6,
                    "physics_consistency": 0.65
                }
            }
        },
        {
            "adapter": MockModelAdapter("large_model"),
            "metadata": {
                "model_id": "large_model",
                "model_size": 1e10,  # 10B parameters
                "compute_flops": 1e20,
                "data_size": 1e8,
                "performance_metrics": {
                    "fvd_score": 85.0,
                    "inception_score": 45.0,
                    "temporal_consistency": 0.85
                },
                "emergent_metrics": {
                    "compositional_reasoning": 0.8,
                    "physics_consistency": 0.82
                }
            }
        }
    ]
    
    # Custom evaluation function
    def custom_evaluator(video, test_case):
        # Mock evaluation that varies by model and test complexity
        complexity_scores = {"basic": 0.8, "intermediate": 0.6, "advanced": 0.4}
        base_score = complexity_scores.get(test_case.get("complexity_level", "basic"), 0.5)
        
        # Add some randomness
        return {
            "capability_score": min(1.0, base_score + np.random.normal(0, 0.1)),
            "evidence_strength": min(1.0, base_score * 0.9 + np.secrets.SystemRandom().uniform(0, 0.2)),
            "novelty_score": np.secrets.SystemRandom().uniform(0.4, 0.9)
        }
    
    # Run detection on all models
    all_results = []
    
    for i, model_info in enumerate(test_models):
        print(f"\n--- Analyzing Model {i+1}/3: {model_info['metadata']['model_id']} ---")
        
        results = await detector.comprehensive_capability_detection(
            model_adapter=model_info["adapter"],
            model_metadata=model_info["metadata"],
            evaluation_function=custom_evaluator
        )
        
        all_results.append(results)
        
        # Display results
        print(f"Detected capabilities: {len(results['detected_capabilities'])}")
        for cap in results['detected_capabilities']:
            print(f"  - {cap['capability_type']}: confidence={cap['confidence_score']:.3f}")
        
        overall = results['overall_assessment']
        print(f"Overall assessment: {overall['detection_strength']} "
              f"(avg confidence: {overall['average_confidence']:.3f})")
    
    # Display scaling analysis
    print("\n--- Scaling Law Analysis ---")
    scaling_results = detector.scaling_analyzer.analyze_scaling_laws()
    
    if "emergence_points" in scaling_results:
        for metric, points in scaling_results["emergence_points"].items():
            print(f"{metric}: {len(points)} emergence points detected")
            for point in points:
                print(f"  Scale: {point['emergence_scale']:.2e}, "
                      f"Confidence: {point['confidence']:.3f}")
    
    # Display clustering analysis
    print("\n--- Capability Clustering ---")
    cluster_results = detector.cluster_analyzer.detect_capability_clusters()
    
    if "num_clusters" in cluster_results:
        print(f"Found {cluster_results['num_clusters']} capability clusters")
        for cluster_id, analysis in cluster_results.get("cluster_analysis", {}).items():
            print(f"  Cluster {cluster_id}: {analysis['size']} models, "
                  f"cohesion: {analysis['cohesion']:.3f}")
    
    # Pattern analysis
    pattern_results = detector.cluster_analyzer.detect_emergent_patterns()
    if "progression_patterns" in pattern_results:
        print(f"Detected {len(pattern_results['progression_patterns'])} progression patterns")
    
    # Overall summary
    print("\n--- Overall Summary ---")
    summary = detector.get_capability_summary()
    print(f"Total capabilities detected: {summary['total_capabilities']}")
    print(f"Capability types: {len(summary['capability_types'])}")
    print(f"Evidence pieces: {summary['evidence_count']}")
    
    # Export results
    export_path = "emergent_capabilities_results.json"
    detector.export_detection_results(export_path)
    print(f"\nResults exported to {export_path}")
    
    # Generate insights
    print("\n=== Key Insights ===")
    
    # Scaling insights
    if scaling_results and "scaling_law_fits" in scaling_results:
        print("Scaling law insights:")
        for metric, fit in scaling_results["scaling_law_fits"].items():
            if fit["r_squared"] > 0.8:
                print(f"  - {metric}: Strong scaling law (R²={fit['r_squared']:.3f})")
                print(f"    {fit['equation']}")
    
    # Capability progression
    high_conf_caps = [cap for model_results in all_results 
                     for cap in model_results['detected_capabilities']
                     if cap['confidence_score'] > 0.8]
    
    if high_conf_caps:
        print(f"\nHigh-confidence emergent capabilities ({len(high_conf_caps)}):")
        for cap in high_conf_caps:
            print(f"  - {cap['capability_type']}: {cap['confidence_score']:.3f}")
    
    return {
        "detector": detector,
        "model_results": all_results,
        "scaling_analysis": scaling_results,
        "cluster_analysis": cluster_results,
        "pattern_analysis": pattern_results,
        "summary": summary
    }


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(run_emergent_capabilities_example())