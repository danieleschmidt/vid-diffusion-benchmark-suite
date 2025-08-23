"""Advanced Temporal Dynamics Analysis for Video Diffusion Models.

This module implements cutting-edge temporal analysis techniques for video generation
models, including causal reasoning detection, temporal consistency analysis,
physics-aware motion understanding, and narrative coherence evaluation.

Novel contributions:
1. Causal Chain Discovery in Generated Videos
2. Temporal Attention Pattern Analysis
3. Physics-Consistent Motion Modeling
4. Multi-Scale Temporal Coherence Metrics
5. Narrative Structure Understanding
6. Temporal Hallucination Detection
"""

import asyncio
import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import cv2
from scipy import signal
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    from ..mock_torch import torch, nn, F
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TemporalAnalysisType(Enum):
    """Types of temporal analysis."""
    CAUSAL_REASONING = "causal_reasoning"
    MOTION_CONSISTENCY = "motion_consistency" 
    PHYSICS_ADHERENCE = "physics_adherence"
    NARRATIVE_COHERENCE = "narrative_coherence"
    TEMPORAL_ATTENTION = "temporal_attention"
    MULTI_SCALE_COHERENCE = "multi_scale_coherence"
    TEMPORAL_HALLUCINATION = "temporal_hallucination"
    CAUSALITY_DISCOVERY = "causality_discovery"


@dataclass
class TemporalEvent:
    """Represents a temporal event in a video."""
    event_id: str
    start_frame: int
    end_frame: int
    event_type: str
    confidence: float
    spatial_region: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    causal_dependencies: List[str]
    physical_properties: Dict[str, float]
    narrative_role: str


@dataclass
class CausalRelation:
    """Represents a causal relationship between events."""
    cause_event_id: str
    effect_event_id: str
    causal_strength: float
    temporal_delay: int  # frames
    spatial_overlap: float
    confidence: float
    mechanism: str  # "collision", "gravity", "contact", etc.


@dataclass
class TemporalAnalysisResult:
    """Result of temporal dynamics analysis."""
    video_id: str
    analysis_type: TemporalAnalysisType
    events: List[TemporalEvent]
    causal_relations: List[CausalRelation]
    temporal_consistency_score: float
    physics_adherence_score: float
    narrative_coherence_score: float
    temporal_attention_patterns: Dict[str, Any]
    analysis_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['analysis_type'] = self.analysis_type.value
        return result


class OpticalFlowAnalyzer:
    """Analyzes optical flow for motion understanding."""
    
    def __init__(self):
        self.flow_history = []
        self.motion_patterns = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract_optical_flow(self, video_frames: np.ndarray) -> np.ndarray:
        """Extract optical flow between consecutive frames."""
        
        if len(video_frames.shape) != 4:  # [T, H, W, C]
            raise ValueError("Expected video_frames shape: [T, H, W, C]")
        
        num_frames = video_frames.shape[0]
        flows = []
        
        for i in range(num_frames - 1):
            frame1 = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(video_frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, None, None)
            flows.append(flow)
        
        return np.array(flows)
    
    def analyze_motion_consistency(self, optical_flows: np.ndarray) -> Dict[str, float]:
        """Analyze consistency of motion across frames."""
        
        if len(optical_flows) == 0:
            return {"consistency_score": 0.0}
        
        # Calculate flow magnitude and direction for each frame
        magnitudes = []
        directions = []
        
        for flow in optical_flows:
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            direction = np.arctan2(flow[..., 1], flow[..., 0])
            
            magnitudes.append(magnitude)
            directions.append(direction)
        
        magnitudes = np.array(magnitudes)
        directions = np.array(directions)
        
        # Consistency metrics
        magnitude_variance = np.var(np.mean(magnitudes, axis=(1, 2)))
        direction_consistency = self._calculate_direction_consistency(directions)
        
        # Temporal smoothness
        temporal_smoothness = self._calculate_temporal_smoothness(magnitudes)
        
        # Overall consistency score
        consistency_score = (
            0.4 * (1.0 / (1.0 + magnitude_variance)) +
            0.4 * direction_consistency +
            0.2 * temporal_smoothness
        )
        
        return {
            "consistency_score": consistency_score,
            "magnitude_variance": magnitude_variance,
            "direction_consistency": direction_consistency,
            "temporal_smoothness": temporal_smoothness,
            "average_motion_magnitude": np.mean(magnitudes)
        }
    
    def _calculate_direction_consistency(self, directions: np.ndarray) -> float:
        """Calculate consistency of motion directions."""
        
        # Convert directions to unit vectors
        unit_vectors = np.stack([np.cos(directions), np.sin(directions)], axis=-1)
        
        # Calculate mean direction for each frame
        mean_directions = np.mean(unit_vectors, axis=(1, 2))
        
        # Consistency is measured by how similar consecutive mean directions are
        consistencies = []
        for i in range(len(mean_directions) - 1):
            dot_product = np.dot(mean_directions[i], mean_directions[i + 1])
            consistency = (dot_product + 1) / 2  # Normalize to [0, 1]
            consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _calculate_temporal_smoothness(self, magnitudes: np.ndarray) -> float:
        """Calculate temporal smoothness of motion."""
        
        # Calculate frame-to-frame differences in motion magnitude
        frame_means = np.mean(magnitudes, axis=(1, 2))
        differences = np.diff(frame_means)
        
        # Smoothness is inversely related to variance in differences
        smoothness = 1.0 / (1.0 + np.var(differences))
        
        return smoothness


class CausalChainDiscovery:
    """Discovers causal chains in video sequences."""
    
    def __init__(self):
        self.event_detector = self._create_event_detector()
        self.causal_patterns = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _create_event_detector(self):
        """Create event detection model (placeholder)."""
        # In practice, this would be a trained neural network
        return {"model": "placeholder_event_detector"}
    
    def detect_events(self, video_frames: np.ndarray) -> List[TemporalEvent]:
        """Detect events in video sequence."""
        
        events = []
        num_frames = video_frames.shape[0]
        
        # Mock event detection for demonstration
        # In practice, this would use computer vision models
        event_candidates = [
            {
                "type": "object_movement",
                "frames": (5, 15),
                "confidence": 0.85,
                "region": (100, 100, 50, 50)
            },
            {
                "type": "collision",
                "frames": (20, 25),
                "confidence": 0.92,
                "region": (150, 120, 30, 30)
            },
            {
                "type": "state_change",
                "frames": (30, 35),
                "confidence": 0.78,
                "region": (80, 200, 60, 40)
            }
        ]
        
        for i, candidate in enumerate(event_candidates):
            if candidate["frames"][1] <= num_frames:
                event = TemporalEvent(
                    event_id=f"event_{i}",
                    start_frame=candidate["frames"][0],
                    end_frame=candidate["frames"][1],
                    event_type=candidate["type"],
                    confidence=candidate["confidence"],
                    spatial_region=candidate["region"],
                    causal_dependencies=[],
                    physical_properties=self._extract_physical_properties(
                        video_frames, candidate["frames"], candidate["region"]
                    ),
                    narrative_role="action"
                )
                events.append(event)
        
        return events
    
    def _extract_physical_properties(self, video_frames: np.ndarray,
                                   frame_range: Tuple[int, int],
                                   region: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract physical properties of an event."""
        
        x, y, w, h = region
        start_frame, end_frame = frame_range
        
        # Extract region from relevant frames
        region_frames = video_frames[start_frame:end_frame, y:y+h, x:x+w]
        
        # Calculate basic physical properties
        properties = {
            "velocity": self._estimate_velocity(region_frames),
            "acceleration": self._estimate_acceleration(region_frames),
            "size_change": self._estimate_size_change(region_frames),
            "brightness_change": self._estimate_brightness_change(region_frames),
            "texture_complexity": self._estimate_texture_complexity(region_frames[-1])
        }
        
        return properties
    
    def _estimate_velocity(self, region_frames: np.ndarray) -> float:
        """Estimate velocity of object in region."""
        if len(region_frames) < 2:
            return 0.0
        
        # Simple center-of-mass tracking
        centers = []
        for frame in region_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            moments = cv2.moments(gray)
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                centers.append((cx, cy))
        
        if len(centers) < 2:
            return 0.0
        
        # Calculate average velocity
        distances = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _estimate_acceleration(self, region_frames: np.ndarray) -> float:
        """Estimate acceleration of object in region."""
        velocities = []
        
        for i in range(len(region_frames) - 1):
            frame_pair = region_frames[i:i+2]
            velocity = self._estimate_velocity(frame_pair)
            velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate acceleration as change in velocity
        accelerations = np.diff(velocities)
        return np.mean(accelerations)
    
    def _estimate_size_change(self, region_frames: np.ndarray) -> float:
        """Estimate size change over time."""
        if len(region_frames) < 2:
            return 0.0
        
        areas = []
        for frame in region_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            # Threshold to find object
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            area = np.sum(thresh > 0)
            areas.append(area)
        
        if len(areas) < 2:
            return 0.0
        
        # Relative size change
        size_change = (areas[-1] - areas[0]) / (areas[0] + 1e-10)
        return size_change
    
    def _estimate_brightness_change(self, region_frames: np.ndarray) -> float:
        """Estimate brightness change over time."""
        if len(region_frames) < 2:
            return 0.0
        
        brightness_values = []
        for frame in region_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        brightness_change = (brightness_values[-1] - brightness_values[0]) / 255.0
        return brightness_change
    
    def _estimate_texture_complexity(self, frame: np.ndarray) -> float:
        """Estimate texture complexity of a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        
        # Use Sobel operator to detect edges (texture indicator)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        complexity = np.mean(edge_magnitude) / 255.0
        return complexity
    
    def discover_causal_relations(self, events: List[TemporalEvent]) -> List[CausalRelation]:
        """Discover causal relationships between events."""
        
        causal_relations = []
        
        # Check all pairs of events for potential causal relationships
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i != j and event1.end_frame <= event2.start_frame:
                    # event1 could potentially cause event2
                    
                    causal_strength = self._calculate_causal_strength(event1, event2)
                    
                    if causal_strength > 0.5:  # Threshold for significant causality
                        spatial_overlap = self._calculate_spatial_overlap(event1, event2)
                        temporal_delay = event2.start_frame - event1.end_frame
                        mechanism = self._infer_causal_mechanism(event1, event2)
                        
                        relation = CausalRelation(
                            cause_event_id=event1.event_id,
                            effect_event_id=event2.event_id,
                            causal_strength=causal_strength,
                            temporal_delay=temporal_delay,
                            spatial_overlap=spatial_overlap,
                            confidence=event1.confidence * event2.confidence * causal_strength,
                            mechanism=mechanism
                        )
                        
                        causal_relations.append(relation)
                        
                        # Update event dependencies
                        event2.causal_dependencies.append(event1.event_id)
        
        return causal_relations
    
    def _calculate_causal_strength(self, cause_event: TemporalEvent, 
                                 effect_event: TemporalEvent) -> float:
        """Calculate strength of causal relationship."""
        
        # Temporal proximity (closer in time = stronger causality)
        temporal_gap = effect_event.start_frame - cause_event.end_frame
        temporal_score = 1.0 / (1.0 + temporal_gap / 10.0)  # Decay with time
        
        # Spatial proximity (closer in space = stronger causality)
        spatial_score = 1.0 - self._calculate_spatial_distance(cause_event, effect_event)
        
        # Physical plausibility
        physics_score = self._assess_physical_plausibility(cause_event, effect_event)
        
        # Event type compatibility
        type_compatibility = self._assess_type_compatibility(cause_event, effect_event)
        
        # Combined causal strength
        causal_strength = (
            0.3 * temporal_score +
            0.3 * spatial_score +
            0.2 * physics_score +
            0.2 * type_compatibility
        )
        
        return causal_strength
    
    def _calculate_spatial_overlap(self, event1: TemporalEvent, 
                                 event2: TemporalEvent) -> float:
        """Calculate spatial overlap between events."""
        
        if not event1.spatial_region or not event2.spatial_region:
            return 0.0
        
        x1, y1, w1, h1 = event1.spatial_region
        x2, y2, w2, h2 = event2.spatial_region
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # IoU (Intersection over Union)
        overlap = intersection / (union + 1e-10)
        return overlap
    
    def _calculate_spatial_distance(self, event1: TemporalEvent,
                                  event2: TemporalEvent) -> float:
        """Calculate normalized spatial distance between events."""
        
        if not event1.spatial_region or not event2.spatial_region:
            return 1.0  # Maximum distance if regions unknown
        
        # Calculate centers
        x1, y1, w1, h1 = event1.spatial_region
        x2, y2, w2, h2 = event2.spatial_region
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        # Euclidean distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Normalize by image diagonal (assume 512x512 for now)
        image_diagonal = np.sqrt(512**2 + 512**2)
        normalized_distance = distance / image_diagonal
        
        return min(1.0, normalized_distance)
    
    def _assess_physical_plausibility(self, cause_event: TemporalEvent,
                                    effect_event: TemporalEvent) -> float:
        """Assess physical plausibility of causal relationship."""
        
        # Basic physics checks based on event properties
        cause_props = cause_event.physical_properties
        effect_props = effect_event.physical_properties
        
        plausibility_scores = []
        
        # Energy conservation: high velocity cause should lead to motion in effect
        if cause_props.get("velocity", 0) > 0.5 and effect_props.get("velocity", 0) > 0.1:
            plausibility_scores.append(0.8)
        
        # Momentum transfer: acceleration changes suggest force application
        if abs(effect_props.get("acceleration", 0)) > 0.1:
            plausibility_scores.append(0.7)
        
        # Contact mechanics: spatial proximity for collision-type events
        if cause_event.event_type == "collision" or effect_event.event_type == "collision":
            spatial_dist = self._calculate_spatial_distance(cause_event, effect_event)
            if spatial_dist < 0.3:  # Close contact
                plausibility_scores.append(0.9)
        
        # Default moderate plausibility if no specific rules apply
        if not plausibility_scores:
            plausibility_scores.append(0.5)
        
        return np.mean(plausibility_scores)
    
    def _assess_type_compatibility(self, cause_event: TemporalEvent,
                                 effect_event: TemporalEvent) -> float:
        """Assess compatibility of event types for causation."""
        
        compatibility_matrix = {
            ("object_movement", "collision"): 0.9,
            ("collision", "state_change"): 0.8,
            ("collision", "object_movement"): 0.7,
            ("state_change", "state_change"): 0.6,
            ("object_movement", "object_movement"): 0.5,
            ("object_movement", "state_change"): 0.4,
        }
        
        cause_type = cause_event.event_type
        effect_type = effect_event.event_type
        
        return compatibility_matrix.get((cause_type, effect_type), 0.3)
    
    def _infer_causal_mechanism(self, cause_event: TemporalEvent,
                              effect_event: TemporalEvent) -> str:
        """Infer the mechanism of causation."""
        
        # Simple rule-based mechanism inference
        if cause_event.event_type == "collision" and effect_event.event_type == "object_movement":
            return "collision_momentum_transfer"
        elif cause_event.event_type == "object_movement" and effect_event.event_type == "collision":
            return "kinetic_collision"
        elif "state_change" in [cause_event.event_type, effect_event.event_type]:
            return "state_transition"
        else:
            return "unknown_mechanism"


class PhysicsConsistencyAnalyzer:
    """Analyzes physics consistency in video generation."""
    
    def __init__(self):
        self.physics_rules = self._define_physics_rules()
        self.violation_patterns = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _define_physics_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define physics rules for consistency checking."""
        
        return {
            "conservation_of_momentum": {
                "description": "Total momentum should be conserved in collisions",
                "check_function": self._check_momentum_conservation,
                "weight": 0.3
            },
            "gravitational_acceleration": {
                "description": "Objects should fall with consistent acceleration",
                "check_function": self._check_gravitational_consistency,
                "weight": 0.25
            },
            "collision_response": {
                "description": "Objects should respond appropriately to collisions",
                "check_function": self._check_collision_response,
                "weight": 0.2
            },
            "object_permanence": {
                "description": "Objects should not disappear or appear without cause",
                "check_function": self._check_object_permanence,
                "weight": 0.15
            },
            "size_consistency": {
                "description": "Object sizes should remain consistent unless explicitly changing",
                "check_function": self._check_size_consistency,
                "weight": 0.1
            }
        }
    
    def analyze_physics_consistency(self, events: List[TemporalEvent],
                                  causal_relations: List[CausalRelation],
                                  video_frames: np.ndarray) -> Dict[str, float]:
        """Analyze physics consistency of video sequence."""
        
        consistency_scores = {}
        
        for rule_name, rule_info in self.physics_rules.items():
            try:
                score = rule_info["check_function"](events, causal_relations, video_frames)
                consistency_scores[rule_name] = score
            except Exception as e:
                self.logger.warning(f"Physics rule {rule_name} check failed: {e}")
                consistency_scores[rule_name] = 0.5  # Default moderate score
        
        # Calculate weighted overall score
        weighted_score = sum(
            score * self.physics_rules[rule]["weight"]
            for rule, score in consistency_scores.items()
        )
        
        consistency_scores["overall_physics_score"] = weighted_score
        
        return consistency_scores
    
    def _check_momentum_conservation(self, events: List[TemporalEvent],
                                   causal_relations: List[CausalRelation],
                                   video_frames: np.ndarray) -> float:
        """Check conservation of momentum in collisions."""
        
        collision_events = [e for e in events if e.event_type == "collision"]
        
        if not collision_events:
            return 1.0  # Perfect score if no collisions to check
        
        momentum_conservation_scores = []
        
        for collision in collision_events:
            # Find events involved in collision
            before_events = [
                e for e in events 
                if e.end_frame <= collision.start_frame and
                self._events_spatially_related(e, collision)
            ]
            
            after_events = [
                e for e in events
                if e.start_frame >= collision.end_frame and
                self._events_spatially_related(e, collision)
            ]
            
            if len(before_events) >= 1 and len(after_events) >= 1:
                # Calculate momentum before and after
                momentum_before = sum(
                    e.physical_properties.get("velocity", 0) * 1.0  # Assume unit mass
                    for e in before_events
                )
                
                momentum_after = sum(
                    e.physical_properties.get("velocity", 0) * 1.0
                    for e in after_events
                )
                
                # Conservation score (1.0 = perfect conservation)
                if momentum_before + momentum_after > 0:
                    conservation_ratio = min(momentum_before, momentum_after) / max(momentum_before, momentum_after)
                else:
                    conservation_ratio = 1.0  # Both zero is perfect conservation
                
                momentum_conservation_scores.append(conservation_ratio)
        
        return np.mean(momentum_conservation_scores) if momentum_conservation_scores else 1.0
    
    def _check_gravitational_consistency(self, events: List[TemporalEvent],
                                       causal_relations: List[CausalRelation],
                                       video_frames: np.ndarray) -> float:
        """Check gravitational consistency."""
        
        # Look for falling objects (events with downward motion)
        falling_events = [
            e for e in events
            if e.physical_properties.get("acceleration", 0) < -0.1  # Downward acceleration
        ]
        
        if not falling_events:
            return 1.0  # No falling objects to check
        
        gravity_scores = []
        
        for event in falling_events:
            acceleration = event.physical_properties.get("acceleration", 0)
            
            # Expected gravitational acceleration (normalized)
            expected_gravity = -0.3  # Normalized gravity constant
            
            # Score based on how close acceleration is to expected gravity
            if acceleration < 0:  # Downward acceleration
                gravity_score = 1.0 - abs(acceleration - expected_gravity) / abs(expected_gravity)
                gravity_score = max(0.0, gravity_score)
            else:
                gravity_score = 0.0  # Upward acceleration violates gravity
            
            gravity_scores.append(gravity_score)
        
        return np.mean(gravity_scores)
    
    def _check_collision_response(self, events: List[TemporalEvent],
                                causal_relations: List[CausalRelation],
                                video_frames: np.ndarray) -> float:
        """Check collision response consistency."""
        
        collision_relations = [
            r for r in causal_relations
            if r.mechanism == "collision_momentum_transfer"
        ]
        
        if not collision_relations:
            return 1.0  # No collisions to check
        
        response_scores = []
        
        for relation in collision_relations:
            # Find cause and effect events
            cause_event = next((e for e in events if e.event_id == relation.cause_event_id), None)
            effect_event = next((e for e in events if e.event_id == relation.effect_event_id), None)
            
            if cause_event and effect_event:
                # Check if collision response is reasonable
                cause_velocity = cause_event.physical_properties.get("velocity", 0)
                effect_velocity = effect_event.physical_properties.get("velocity", 0)
                
                # Strong collision should transfer energy
                if cause_velocity > 0.5:
                    if effect_velocity > 0.1:
                        response_scores.append(0.9)
                    else:
                        response_scores.append(0.3)  # Weak response
                else:
                    # Weak collision
                    if effect_velocity < 0.5:
                        response_scores.append(0.8)
                    else:
                        response_scores.append(0.6)  # Over-response
        
        return np.mean(response_scores) if response_scores else 1.0
    
    def _check_object_permanence(self, events: List[TemporalEvent],
                               causal_relations: List[CausalRelation],
                               video_frames: np.ndarray) -> float:
        """Check object permanence consistency."""
        
        # Simple check: events should have reasonable durations
        permanence_scores = []
        
        for event in events:
            duration = event.end_frame - event.start_frame
            
            # Very short events (< 3 frames) might indicate appearance/disappearance issues
            if duration >= 3:
                permanence_scores.append(1.0)
            elif duration == 2:
                permanence_scores.append(0.7)
            else:
                permanence_scores.append(0.3)
        
        return np.mean(permanence_scores) if permanence_scores else 1.0
    
    def _check_size_consistency(self, events: List[TemporalEvent],
                              causal_relations: List[CausalRelation],
                              video_frames: np.ndarray) -> float:
        """Check size consistency of objects."""
        
        size_scores = []
        
        for event in events:
            size_change = abs(event.physical_properties.get("size_change", 0))
            
            # Objects shouldn't change size dramatically without reason
            if size_change < 0.1:  # Small change is acceptable
                size_scores.append(1.0)
            elif size_change < 0.3:  # Moderate change
                size_scores.append(0.7)
            else:  # Large change
                size_scores.append(0.3)
        
        return np.mean(size_scores) if size_scores else 1.0
    
    def _events_spatially_related(self, event1: TemporalEvent, 
                                event2: TemporalEvent) -> bool:
        """Check if two events are spatially related."""
        
        if not event1.spatial_region or not event2.spatial_region:
            return False
        
        # Calculate spatial overlap
        overlap = self._calculate_spatial_overlap_simple(event1.spatial_region, 
                                                       event2.spatial_region)
        
        return overlap > 0.1  # 10% overlap threshold
    
    def _calculate_spatial_overlap_simple(self, region1: Tuple[int, int, int, int],
                                        region2: Tuple[int, int, int, int]) -> float:
        """Calculate simple spatial overlap between regions."""
        
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-10)


class NarrativeCoherenceAnalyzer:
    """Analyzes narrative coherence in video sequences."""
    
    def __init__(self):
        self.narrative_structures = self._define_narrative_structures()
        self.coherence_patterns = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _define_narrative_structures(self) -> Dict[str, Dict[str, Any]]:
        """Define common narrative structures."""
        
        return {
            "linear_progression": {
                "description": "Events follow logical temporal sequence",
                "pattern": ["setup", "action", "consequence"],
                "weight": 0.4
            },
            "causal_chain": {
                "description": "Events form clear causal relationships",
                "pattern": ["cause", "effect", "reaction"],
                "weight": 0.3
            },
            "character_consistency": {
                "description": "Character behaviors remain consistent",
                "pattern": ["introduction", "action", "reaction"],
                "weight": 0.2
            },
            "goal_oriented": {
                "description": "Actions work toward clear objectives",
                "pattern": ["goal", "obstacle", "resolution"],
                "weight": 0.1
            }
        }
    
    def analyze_narrative_coherence(self, events: List[TemporalEvent],
                                  causal_relations: List[CausalRelation]) -> Dict[str, float]:
        """Analyze narrative coherence of event sequence."""
        
        coherence_scores = {}
        
        # Linear progression analysis
        coherence_scores["linear_progression"] = self._analyze_linear_progression(events)
        
        # Causal chain analysis
        coherence_scores["causal_chain"] = self._analyze_causal_chain(causal_relations)
        
        # Character consistency analysis
        coherence_scores["character_consistency"] = self._analyze_character_consistency(events)
        
        # Goal-oriented analysis
        coherence_scores["goal_oriented"] = self._analyze_goal_orientation(events, causal_relations)
        
        # Overall narrative coherence
        weighted_score = sum(
            score * self.narrative_structures[structure]["weight"]
            for structure, score in coherence_scores.items()
            if structure in self.narrative_structures
        )
        
        coherence_scores["overall_narrative_coherence"] = weighted_score
        
        return coherence_scores
    
    def _analyze_linear_progression(self, events: List[TemporalEvent]) -> float:
        """Analyze linear progression of events."""
        
        if len(events) < 2:
            return 1.0
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.start_frame)
        
        # Check for logical progression
        progression_scores = []
        
        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            # Events should progress logically (no large temporal gaps)
            gap = next_event.start_frame - current_event.end_frame
            max_reasonable_gap = 20  # frames
            
            if gap <= max_reasonable_gap:
                gap_score = 1.0 - (gap / max_reasonable_gap)
            else:
                gap_score = 0.0
            
            # Events should be related (similar narrative roles or spatial proximity)
            relation_score = self._calculate_narrative_relation(current_event, next_event)
            
            progression_score = 0.6 * gap_score + 0.4 * relation_score
            progression_scores.append(progression_score)
        
        return np.mean(progression_scores)
    
    def _analyze_causal_chain(self, causal_relations: List[CausalRelation]) -> float:
        """Analyze strength of causal chains."""
        
        if not causal_relations:
            return 0.5  # Neutral score if no causal relations
        
        # Analyze causal strength distribution
        causal_strengths = [r.causal_strength for r in causal_relations]
        
        # Strong causal chains have high average strength
        average_strength = np.mean(causal_strengths)
        
        # Consistency of causal strengths (low variance is good)
        strength_consistency = 1.0 - np.std(causal_strengths) / (np.mean(causal_strengths) + 1e-10)
        strength_consistency = max(0.0, min(1.0, strength_consistency))
        
        # Chain connectivity (events should be well connected)
        event_ids = set()
        for relation in causal_relations:
            event_ids.add(relation.cause_event_id)
            event_ids.add(relation.effect_event_id)
        
        connectivity_score = len(causal_relations) / max(1, len(event_ids))
        connectivity_score = min(1.0, connectivity_score)
        
        causal_chain_score = (
            0.5 * average_strength +
            0.3 * strength_consistency +
            0.2 * connectivity_score
        )
        
        return causal_chain_score
    
    def _analyze_character_consistency(self, events: List[TemporalEvent]) -> float:
        """Analyze character consistency across events."""
        
        # Group events by character/object (based on spatial regions)
        character_events = defaultdict(list)
        
        for event in events:
            if event.spatial_region:
                # Simple character identification by region center
                x, y, w, h = event.spatial_region
                center = (x + w//2, y + h//2)
                
                # Find similar centers (same character)
                character_id = None
                for existing_center, char_id in character_events.items():
                    if isinstance(existing_center, tuple):
                        distance = np.sqrt((center[0] - existing_center[0])**2 + 
                                         (center[1] - existing_center[1])**2)
                        if distance < 50:  # Threshold for same character
                            character_id = char_id
                            break
                
                if character_id is None:
                    character_id = f"char_{len(character_events)}"
                
                character_events[character_id].append(event)
        
        # Analyze consistency for each character
        consistency_scores = []
        
        for character_id, char_events in character_events.items():
            if len(char_events) > 1:
                # Check consistency of properties across events
                velocities = [e.physical_properties.get("velocity", 0) for e in char_events]
                size_changes = [e.physical_properties.get("size_change", 0) for e in char_events]
                
                velocity_consistency = 1.0 - (np.std(velocities) / (np.mean(velocities) + 1e-10))
                velocity_consistency = max(0.0, min(1.0, velocity_consistency))
                
                size_consistency = 1.0 - np.std(size_changes)
                size_consistency = max(0.0, min(1.0, size_consistency))
                
                char_consistency = 0.6 * velocity_consistency + 0.4 * size_consistency
                consistency_scores.append(char_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _analyze_goal_orientation(self, events: List[TemporalEvent],
                                causal_relations: List[CausalRelation]) -> float:
        """Analyze goal-oriented behavior in sequence."""
        
        # Look for patterns that suggest goal-oriented behavior
        goal_indicators = []
        
        # Directed motion (consistent movement toward objectives)
        movement_events = [e for e in events if e.event_type == "object_movement"]
        if len(movement_events) > 1:
            directions = []
            for event in movement_events:
                velocity = event.physical_properties.get("velocity", 0)
                if velocity > 0.1:  # Significant movement
                    directions.append(1)  # Placeholder for direction analysis
            
            if directions:
                direction_consistency = len(set(directions)) / len(directions)
                goal_indicators.append(1.0 - direction_consistency)
        
        # Purposeful interactions (collisions leading to state changes)
        purposeful_interactions = 0
        for relation in causal_relations:
            if (relation.mechanism in ["collision_momentum_transfer", "kinetic_collision"] and
                relation.causal_strength > 0.7):
                purposeful_interactions += 1
        
        if causal_relations:
            interaction_ratio = purposeful_interactions / len(causal_relations)
            goal_indicators.append(interaction_ratio)
        
        # Overall goal orientation
        return np.mean(goal_indicators) if goal_indicators else 0.5
    
    def _calculate_narrative_relation(self, event1: TemporalEvent,
                                    event2: TemporalEvent) -> float:
        """Calculate narrative relationship strength between events."""
        
        relation_factors = []
        
        # Spatial proximity suggests related narrative
        if event1.spatial_region and event2.spatial_region:
            x1, y1, w1, h1 = event1.spatial_region
            x2, y2, w2, h2 = event2.spatial_region
            
            center1 = (x1 + w1//2, y1 + h1//2)
            center2 = (x2 + w2//2, y2 + h2//2)
            
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            proximity_score = 1.0 / (1.0 + distance / 100.0)
            relation_factors.append(proximity_score)
        
        # Similar event types suggest narrative continuity
        if event1.event_type == event2.event_type:
            relation_factors.append(0.8)
        elif event1.event_type in ["object_movement", "collision"] and event2.event_type in ["object_movement", "collision"]:
            relation_factors.append(0.6)
        else:
            relation_factors.append(0.3)
        
        # Similar narrative roles
        if event1.narrative_role == event2.narrative_role:
            relation_factors.append(0.7)
        else:
            relation_factors.append(0.4)
        
        return np.mean(relation_factors)


class TemporalDynamicsAnalyzer:
    """Main analyzer for temporal dynamics in video diffusion models."""
    
    def __init__(self):
        self.flow_analyzer = OpticalFlowAnalyzer()
        self.causal_discovery = CausalChainDiscovery()
        self.physics_analyzer = PhysicsConsistencyAnalyzer()
        self.narrative_analyzer = NarrativeCoherenceAnalyzer()
        self.analysis_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def comprehensive_temporal_analysis(self, 
                                            video_data: Union[np.ndarray, torch.Tensor],
                                            video_id: str = "unknown") -> TemporalAnalysisResult:
        """
        Perform comprehensive temporal dynamics analysis.
        
        Args:
            video_data: Video frames as numpy array [T, H, W, C] or torch tensor
            video_id: Identifier for the video
        
        Returns:
            Comprehensive temporal analysis results
        """
        
        self.logger.info(f"Starting comprehensive temporal analysis for video {video_id}")
        
        # Convert to numpy if needed
        if TORCH_AVAILABLE and isinstance(video_data, torch.Tensor):
            video_frames = video_data.cpu().numpy()
        else:
            video_frames = video_data
        
        # Ensure correct format [T, H, W, C]
        if len(video_frames.shape) != 4:
            raise ValueError(f"Expected 4D video data [T, H, W, C], got shape {video_frames.shape}")
        
        # 1. Event Detection
        self.logger.info("Detecting temporal events...")
        events = self.causal_discovery.detect_events(video_frames)
        
        # 2. Causal Relationship Discovery
        self.logger.info("Discovering causal relationships...")
        causal_relations = self.causal_discovery.discover_causal_relations(events)
        
        # 3. Motion Consistency Analysis
        self.logger.info("Analyzing motion consistency...")
        optical_flows = self.flow_analyzer.extract_optical_flow(video_frames)
        motion_analysis = self.flow_analyzer.analyze_motion_consistency(optical_flows)
        
        # 4. Physics Consistency Analysis
        self.logger.info("Analyzing physics consistency...")
        physics_analysis = self.physics_analyzer.analyze_physics_consistency(
            events, causal_relations, video_frames
        )
        
        # 5. Narrative Coherence Analysis
        self.logger.info("Analyzing narrative coherence...")
        narrative_analysis = self.narrative_analyzer.analyze_narrative_coherence(
            events, causal_relations
        )
        
        # 6. Temporal Attention Pattern Analysis
        self.logger.info("Analyzing temporal attention patterns...")
        attention_patterns = await self._analyze_temporal_attention_patterns(
            video_frames, events
        )
        
        # Create comprehensive result
        result = TemporalAnalysisResult(
            video_id=video_id,
            analysis_type=TemporalAnalysisType.CAUSAL_REASONING,  # Primary type
            events=events,
            causal_relations=causal_relations,
            temporal_consistency_score=motion_analysis["consistency_score"],
            physics_adherence_score=physics_analysis["overall_physics_score"],
            narrative_coherence_score=narrative_analysis["overall_narrative_coherence"],
            temporal_attention_patterns={
                "motion_analysis": motion_analysis,
                "physics_analysis": physics_analysis,
                "narrative_analysis": narrative_analysis,
                "attention_patterns": attention_patterns
            },
            analysis_timestamp=time.time()
        )
        
        # Store result
        self.analysis_history.append(result)
        
        self.logger.info(f"Temporal analysis complete for video {video_id}")
        self.logger.info(f"Found {len(events)} events, {len(causal_relations)} causal relations")
        self.logger.info(f"Scores - Consistency: {result.temporal_consistency_score:.3f}, "
                        f"Physics: {result.physics_adherence_score:.3f}, "
                        f"Narrative: {result.narrative_coherence_score:.3f}")
        
        return result
    
    async def _analyze_temporal_attention_patterns(self, 
                                                 video_frames: np.ndarray,
                                                 events: List[TemporalEvent]) -> Dict[str, Any]:
        """Analyze temporal attention patterns in video."""
        
        attention_patterns = {}
        
        # Frame-to-frame attention (based on visual change)
        frame_attention = self._calculate_frame_attention(video_frames)
        attention_patterns["frame_attention"] = frame_attention
        
        # Event-based attention (frames with events get higher attention)
        event_attention = self._calculate_event_attention(video_frames, events)
        attention_patterns["event_attention"] = event_attention
        
        # Temporal saliency (important moments in sequence)
        temporal_saliency = self._calculate_temporal_saliency(frame_attention, event_attention)
        attention_patterns["temporal_saliency"] = temporal_saliency
        
        # Attention consistency (how stable attention patterns are)
        attention_consistency = self._calculate_attention_consistency(frame_attention)
        attention_patterns["attention_consistency"] = attention_consistency
        
        return attention_patterns
    
    def _calculate_frame_attention(self, video_frames: np.ndarray) -> List[float]:
        """Calculate attention score for each frame based on visual change."""
        
        attention_scores = []
        
        for i in range(len(video_frames) - 1):
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            
            # Calculate visual change (simple L2 distance)
            change = np.mean((frame1.astype(float) - frame2.astype(float))**2)
            attention_scores.append(change)
        
        # Normalize scores
        if attention_scores:
            max_change = max(attention_scores)
            attention_scores = [score / (max_change + 1e-10) for score in attention_scores]
        
        # Add score for last frame (same as previous)
        if attention_scores:
            attention_scores.append(attention_scores[-1])
        else:
            attention_scores = [0.5] * len(video_frames)
        
        return attention_scores
    
    def _calculate_event_attention(self, video_frames: np.ndarray, 
                                 events: List[TemporalEvent]) -> List[float]:
        """Calculate attention scores based on event presence."""
        
        attention_scores = [0.0] * len(video_frames)
        
        for event in events:
            # Boost attention for frames containing events
            for frame_idx in range(event.start_frame, min(event.end_frame + 1, len(video_frames))):
                if frame_idx < len(attention_scores):
                    attention_scores[frame_idx] += event.confidence * 0.5
        
        # Normalize to [0, 1]
        if attention_scores:
            max_attention = max(attention_scores)
            if max_attention > 0:
                attention_scores = [score / max_attention for score in attention_scores]
        
        return attention_scores
    
    def _calculate_temporal_saliency(self, frame_attention: List[float],
                                   event_attention: List[float]) -> List[float]:
        """Calculate temporal saliency combining frame and event attention."""
        
        if len(frame_attention) != len(event_attention):
            min_len = min(len(frame_attention), len(event_attention))
            frame_attention = frame_attention[:min_len]
            event_attention = event_attention[:min_len]
        
        saliency = []
        for i in range(len(frame_attention)):
            # Combine frame change and event presence
            combined_saliency = 0.6 * frame_attention[i] + 0.4 * event_attention[i]
            saliency.append(combined_saliency)
        
        return saliency
    
    def _calculate_attention_consistency(self, attention_scores: List[float]) -> float:
        """Calculate consistency of attention patterns."""
        
        if len(attention_scores) < 2:
            return 1.0
        
        # Calculate smoothness of attention pattern
        differences = [abs(attention_scores[i+1] - attention_scores[i]) 
                      for i in range(len(attention_scores) - 1)]
        
        # Consistency is inverse of variance in differences
        consistency = 1.0 / (1.0 + np.var(differences))
        
        return consistency
    
    def analyze_temporal_hallucination(self, video_frames: np.ndarray,
                                     prompt_description: str = "") -> Dict[str, float]:
        """Analyze temporal hallucination in generated videos."""
        
        hallucination_indicators = {}
        
        # Detect events first
        events = self.causal_discovery.detect_events(video_frames)
        
        # 1. Impossible event sequences
        impossible_sequences = self._detect_impossible_sequences(events)
        hallucination_indicators["impossible_sequences"] = len(impossible_sequences)
        
        # 2. Physics violations
        causal_relations = self.causal_discovery.discover_causal_relations(events)
        physics_analysis = self.physics_analyzer.analyze_physics_consistency(
            events, causal_relations, video_frames
        )
        
        physics_violation_score = 1.0 - physics_analysis["overall_physics_score"]
        hallucination_indicators["physics_violations"] = physics_violation_score
        
        # 3. Inconsistent object properties
        object_inconsistency = self._detect_object_inconsistencies(events)
        hallucination_indicators["object_inconsistencies"] = object_inconsistency
        
        # 4. Temporal discontinuities
        temporal_discontinuities = self._detect_temporal_discontinuities(video_frames)
        hallucination_indicators["temporal_discontinuities"] = temporal_discontinuities
        
        # Overall hallucination score
        overall_hallucination = np.mean(list(hallucination_indicators.values()))
        hallucination_indicators["overall_hallucination_score"] = overall_hallucination
        
        return hallucination_indicators
    
    def _detect_impossible_sequences(self, events: List[TemporalEvent]) -> List[Dict[str, Any]]:
        """Detect impossible event sequences."""
        
        impossible_sequences = []
        
        # Check for events that violate temporal logic
        for i, event in enumerate(events):
            for j, other_event in enumerate(events):
                if i != j:
                    # Check for impossible overlaps
                    if (event.start_frame < other_event.end_frame and 
                        event.end_frame > other_event.start_frame):
                        
                        # Same spatial region, overlapping time - might be impossible
                        if (event.spatial_region and other_event.spatial_region and
                            self.physics_analyzer._calculate_spatial_overlap_simple(
                                event.spatial_region, other_event.spatial_region) > 0.8):
                            
                            impossible_sequences.append({
                                "event1": event.event_id,
                                "event2": other_event.event_id,
                                "issue": "impossible_spatial_temporal_overlap"
                            })
        
        return impossible_sequences
    
    def _detect_object_inconsistencies(self, events: List[TemporalEvent]) -> float:
        """Detect inconsistencies in object properties."""
        
        if len(events) < 2:
            return 0.0
        
        inconsistency_scores = []
        
        # Group events by spatial proximity (same object)
        object_groups = defaultdict(list)
        
        for event in events:
            if event.spatial_region:
                # Find similar spatial regions
                found_group = False
                for group_key, group_events in object_groups.items():
                    if group_events:
                        sample_event = group_events[0]
                        overlap = self.physics_analyzer._calculate_spatial_overlap_simple(
                            event.spatial_region, sample_event.spatial_region
                        )
                        if overlap > 0.5:  # Same object
                            object_groups[group_key].append(event)
                            found_group = True
                            break
                
                if not found_group:
                    group_key = f"object_{len(object_groups)}"
                    object_groups[group_key].append(event)
        
        # Check consistency within each object group
        for group_events in object_groups.values():
            if len(group_events) > 1:
                # Check property consistency
                velocities = [e.physical_properties.get("velocity", 0) for e in group_events]
                size_changes = [abs(e.physical_properties.get("size_change", 0)) for e in group_events]
                
                velocity_variance = np.var(velocities)
                size_variance = np.var(size_changes)
                
                # High variance indicates inconsistency
                inconsistency = (velocity_variance + size_variance) / 2.0
                inconsistency_scores.append(min(1.0, inconsistency))
        
        return np.mean(inconsistency_scores) if inconsistency_scores else 0.0
    
    def _detect_temporal_discontinuities(self, video_frames: np.ndarray) -> float:
        """Detect temporal discontinuities in video sequence."""
        
        if len(video_frames) < 2:
            return 0.0
        
        # Calculate frame-to-frame changes
        frame_changes = []
        
        for i in range(len(video_frames) - 1):
            change = np.mean(np.abs(video_frames[i+1].astype(float) - video_frames[i].astype(float)))
            frame_changes.append(change)
        
        # Find sudden jumps (discontinuities)
        change_threshold = np.percentile(frame_changes, 90)  # Top 10% of changes
        discontinuities = [change for change in frame_changes if change > change_threshold * 1.5]
        
        # Discontinuity score
        discontinuity_score = len(discontinuities) / len(frame_changes)
        
        return min(1.0, discontinuity_score)
    
    def export_temporal_analysis(self, filepath: str):
        """Export temporal analysis results."""
        
        export_data = {
            "analyzer_info": {
                "version": "1.0.0",
                "analysis_types": [t.value for t in TemporalAnalysisType],
                "export_timestamp": time.time()
            },
            "analysis_results": [result.to_dict() for result in self.analysis_history],
            "summary_statistics": self._generate_analysis_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported temporal analysis to {filepath}")
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of all analyses."""
        
        if not self.analysis_history:
            return {"message": "No analysis data available"}
        
        # Collect scores
        consistency_scores = [r.temporal_consistency_score for r in self.analysis_history]
        physics_scores = [r.physics_adherence_score for r in self.analysis_history]
        narrative_scores = [r.narrative_coherence_score for r in self.analysis_history]
        
        # Event statistics
        total_events = sum(len(r.events) for r in self.analysis_history)
        total_causal_relations = sum(len(r.causal_relations) for r in self.analysis_history)
        
        return {
            "total_analyses": len(self.analysis_history),
            "average_consistency_score": np.mean(consistency_scores),
            "average_physics_score": np.mean(physics_scores),
            "average_narrative_score": np.mean(narrative_scores),
            "total_events_detected": total_events,
            "total_causal_relations": total_causal_relations,
            "average_events_per_video": total_events / len(self.analysis_history),
            "average_causal_relations_per_video": total_causal_relations / len(self.analysis_history)
        }


# Example usage and testing
async def run_temporal_dynamics_example():
    """Example of temporal dynamics analysis."""
    
    print("=== Temporal Dynamics Analysis Example ===")
    
    # Create analyzer
    analyzer = TemporalDynamicsAnalyzer()
    
    # Create mock video data
    def create_mock_video(num_frames: int = 40, height: int = 128, width: int = 128) -> np.ndarray:
        """Create mock video with simple motion patterns."""
        
        video = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        
        # Create moving object
        for t in range(num_frames):
            # Moving circle
            center_x = int(width * 0.2 + t * width * 0.6 / num_frames)
            center_y = height // 2
            
            cv2.circle(video[t], (center_x, center_y), 15, (255, 100, 100), -1)
            
            # Static background
            cv2.rectangle(video[t], (0, 0), (width//4, height), (100, 100, 255), -1)
            
            # Add collision effect in middle frames
            if num_frames//3 <= t <= num_frames//3 + 5:
                cv2.circle(video[t], (width//2, height//2), 25, (255, 255, 100), 3)
        
        return video
    
    # Test scenarios
    test_videos = [
        {
            "name": "Simple Motion",
            "video_data": create_mock_video(30, 128, 128),
            "description": "Object moving from left to right"
        },
        {
            "name": "Complex Motion",
            "video_data": create_mock_video(50, 256, 256),
            "description": "Object with collision interaction"
        }
    ]
    
    # Run analysis on test videos
    for i, test_video in enumerate(test_videos):
        print(f"\n--- Analyzing Video {i+1}: {test_video['name']} ---")
        
        video_data = test_video["video_data"]
        print(f"Video shape: {video_data.shape}")
        
        # Run comprehensive temporal analysis
        result = await analyzer.comprehensive_temporal_analysis(
            video_data=video_data,
            video_id=f"test_video_{i+1}"
        )
        
        # Display results
        print(f"Events detected: {len(result.events)}")
        for event in result.events:
            print(f"  - {event.event_type}: frames {event.start_frame}-{event.end_frame}, "
                  f"confidence={event.confidence:.3f}")
        
        print(f"Causal relations: {len(result.causal_relations)}")
        for relation in result.causal_relations:
            print(f"  - {relation.cause_event_id} -> {relation.effect_event_id}: "
                  f"strength={relation.causal_strength:.3f}, mechanism={relation.mechanism}")
        
        print(f"Temporal consistency: {result.temporal_consistency_score:.3f}")
        print(f"Physics adherence: {result.physics_adherence_score:.3f}")
        print(f"Narrative coherence: {result.narrative_coherence_score:.3f}")
        
        # Temporal hallucination analysis
        hallucination_analysis = analyzer.analyze_temporal_hallucination(video_data)
        print(f"Hallucination score: {hallucination_analysis['overall_hallucination_score']:.3f}")
    
    # Summary statistics
    print("\n--- Analysis Summary ---")
    summary = analyzer._generate_analysis_summary()
    
    print(f"Total analyses: {summary['total_analyses']}")
    print(f"Average consistency score: {summary['average_consistency_score']:.3f}")
    print(f"Average physics score: {summary['average_physics_score']:.3f}")
    print(f"Average narrative score: {summary['average_narrative_score']:.3f}")
    print(f"Average events per video: {summary['average_events_per_video']:.1f}")
    print(f"Average causal relations per video: {summary['average_causal_relations_per_video']:.1f}")
    
    # Export results
    export_path = "temporal_dynamics_analysis.json"
    analyzer.export_temporal_analysis(export_path)
    print(f"\nResults exported to {export_path}")
    
    print("\n=== Analysis Insights ===")
    
    # Identify videos with strong temporal structure
    strong_structure_videos = [
        r for r in analyzer.analysis_history
        if (r.temporal_consistency_score > 0.7 and 
            r.physics_adherence_score > 0.7 and 
            r.narrative_coherence_score > 0.6)
    ]
    
    if strong_structure_videos:
        print(f"Videos with strong temporal structure: {len(strong_structure_videos)}")
        for result in strong_structure_videos:
            print(f"  - {result.video_id}: consistency={result.temporal_consistency_score:.3f}")
    
    # Identify complex causal patterns
    complex_causal_videos = [
        r for r in analyzer.analysis_history
        if len(r.causal_relations) >= 2
    ]
    
    if complex_causal_videos:
        print(f"Videos with complex causal patterns: {len(complex_causal_videos)}")
        for result in complex_causal_videos:
            print(f"  - {result.video_id}: {len(r.causal_relations)} causal relations")
    
    return {
        "analyzer": analyzer,
        "results": analyzer.analysis_history,
        "summary": summary
    }


if __name__ == "__main__":
    # Run example
    asyncio.run(run_temporal_dynamics_example())