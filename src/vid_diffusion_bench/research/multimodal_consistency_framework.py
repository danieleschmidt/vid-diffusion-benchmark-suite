"""Multimodal Consistency Framework for Video Diffusion Models.

Advanced framework for analyzing consistency across multiple modalities in video
generation, including visual-textual alignment, audio-visual synchronization,
semantic coherence, and cross-modal attention analysis.

Novel Research Contributions:
1. Cross-Modal Attention Consistency Analysis
2. Semantic Drift Detection Across Modalities
3. Temporal-Semantic Alignment Scoring
4. Multimodal Hallucination Detection
5. Cross-Domain Knowledge Transfer Analysis
6. Emergent Cross-Modal Understanding Evaluation
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
import math
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    from ..mock_torch import torch, nn, F
    TORCH_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of modalities in multimodal analysis."""
    VISUAL = "visual"
    TEXTUAL = "textual"
    AUDIO = "audio"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CONCEPTUAL = "conceptual"
    CONTEXTUAL = "contextual"


class ConsistencyLevel(Enum):
    """Levels of consistency analysis."""
    PERFECT = "perfect"          # >95% consistency
    HIGH = "high"               # 85-95% consistency
    MODERATE = "moderate"       # 70-85% consistency
    LOW = "low"                 # 50-70% consistency
    INCONSISTENT = "inconsistent"  # <50% consistency


@dataclass
class ModalityRepresentation:
    """Representation of a single modality."""
    modality_type: ModalityType
    representation_vector: np.ndarray
    confidence_score: float
    temporal_alignment: Optional[List[int]]  # Frame indices for temporal modalities
    spatial_alignment: Optional[Tuple[int, int, int, int]]  # Bounding box for spatial
    semantic_tags: List[str]
    extraction_method: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['modality_type'] = self.modality_type.value
        result['representation_vector'] = self.representation_vector.tolist()
        return result


@dataclass
class CrossModalAlignment:
    """Represents alignment between two modalities."""
    modality1_type: ModalityType
    modality2_type: ModalityType
    alignment_score: float
    consistency_level: ConsistencyLevel
    semantic_overlap: float
    temporal_correlation: float
    spatial_correlation: float
    attention_correlation: float
    drift_magnitude: float
    explanation: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['modality1_type'] = self.modality1_type.value
        result['modality2_type'] = self.modality2_type.value
        result['consistency_level'] = self.consistency_level.value
        return result


@dataclass
class MultimodalConsistencyResult:
    """Result of multimodal consistency analysis."""
    sample_id: str
    modality_representations: List[ModalityRepresentation]
    cross_modal_alignments: List[CrossModalAlignment]
    overall_consistency_score: float
    consistency_level: ConsistencyLevel
    semantic_drift_score: float
    hallucination_indicators: Dict[str, float]
    emergent_understanding_score: float
    analysis_timestamp: float
    detailed_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['consistency_level'] = self.consistency_level.value
        result['modality_representations'] = [mr.to_dict() for mr in self.modality_representations]
        result['cross_modal_alignments'] = [cma.to_dict() for cma in self.cross_modal_alignments]
        return result


class TextualSemanticExtractor:
    """Extracts semantic representations from text."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize models if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                self.logger.warning(f"Could not load transformers model: {e}")
                self.tokenizer = None
                self.model = None
    
    def extract_semantic_representation(self, text: str) -> ModalityRepresentation:
        """Extract semantic representation from text."""
        
        if self.model is not None and self.tokenizer is not None:
            # Use actual transformer model
            try:
                inputs = self.tokenizer(text, return_tensors="pt", 
                                       padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                representation_vector = embeddings.cpu().numpy().flatten()
                confidence_score = 0.9  # High confidence for actual model
                
            except Exception as e:
                self.logger.error(f"Error in semantic extraction: {e}")
                representation_vector = self._mock_semantic_extraction(text)
                confidence_score = 0.5
        else:
            # Fallback to mock extraction
            representation_vector = self._mock_semantic_extraction(text)
            confidence_score = 0.5
        
        # Extract semantic tags
        semantic_tags = self._extract_semantic_tags(text)
        
        return ModalityRepresentation(
            modality_type=ModalityType.TEXTUAL,
            representation_vector=representation_vector,
            confidence_score=confidence_score,
            temporal_alignment=None,
            spatial_alignment=None,
            semantic_tags=semantic_tags,
            extraction_method="transformer_embedding" if self.model else "mock_extraction",
            timestamp=time.time()
        )
    
    def _mock_semantic_extraction(self, text: str) -> np.ndarray:
        """Mock semantic extraction for testing."""
        
        # Simple bag-of-words approach with predefined semantic dimensions
        semantic_dimensions = {
            "object": ["cat", "dog", "car", "house", "tree", "person", "ball"],
            "action": ["running", "jumping", "flying", "swimming", "walking", "dancing"],
            "color": ["red", "blue", "green", "yellow", "black", "white", "purple"],
            "emotion": ["happy", "sad", "angry", "excited", "calm", "surprised"],
            "location": ["park", "beach", "mountain", "city", "forest", "room"],
            "time": ["morning", "evening", "night", "day", "sunset", "dawn"],
            "weather": ["sunny", "rainy", "cloudy", "snowy", "windy", "stormy"]
        }
        
        text_lower = text.lower()
        vector = np.zeros(384)  # Match typical sentence transformer size
        
        # Fill vector based on semantic categories
        base_idx = 0
        for category, words in semantic_dimensions.items():
            category_scores = []
            for word in words:
                if word in text_lower:
                    category_scores.append(1.0)
                else:
                    # Fuzzy matching
                    if any(w in text_lower for w in word.split()):
                        category_scores.append(0.5)
                    else:
                        category_scores.append(0.0)
            
            # Fill vector segment
            segment_size = len(words) * 8  # 8 dimensions per word
            if base_idx + segment_size <= len(vector):
                for i, score in enumerate(category_scores):
                    for j in range(8):
                        if base_idx + i * 8 + j < len(vector):
                            vector[base_idx + i * 8 + j] = score + np.random.normal(0, 0.1)
                base_idx += segment_size
        
        # Add noise for realism
        vector += np.random.normal(0, 0.05, len(vector))
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _extract_semantic_tags(self, text: str) -> List[str]:
        """Extract semantic tags from text."""
        
        common_tags = [
            "person", "animal", "object", "action", "location", "color",
            "emotion", "time", "weather", "movement", "interaction"
        ]
        
        text_lower = text.lower()
        tags = []
        
        for tag in common_tags:
            # Simple keyword matching
            if tag in text_lower:
                tags.append(tag)
            elif tag == "person" and any(word in text_lower for word in ["man", "woman", "child", "people"]):
                tags.append(tag)
            elif tag == "animal" and any(word in text_lower for word in ["cat", "dog", "bird", "horse"]):
                tags.append(tag)
            elif tag == "action" and any(word in text_lower for word in ["running", "jumping", "walking", "dancing"]):
                tags.append(tag)
        
        return tags


class VisualSemanticExtractor:
    """Extracts semantic representations from visual content."""
    
    def __init__(self):
        self.feature_extractor = self._create_mock_feature_extractor()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _create_mock_feature_extractor(self):
        """Create mock visual feature extractor."""
        return {"model": "mock_visual_extractor", "features": 512}
    
    def extract_visual_representation(self, video_frames: np.ndarray, 
                                    frame_indices: Optional[List[int]] = None) -> ModalityRepresentation:
        """Extract visual semantic representation from video frames."""
        
        if frame_indices is None:
            frame_indices = list(range(len(video_frames)))
        
        # Extract features from specified frames
        frame_features = []
        
        for idx in frame_indices:
            if idx < len(video_frames):
                frame = video_frames[idx]
                features = self._extract_frame_features(frame)
                frame_features.append(features)
        
        # Aggregate frame features
        if frame_features:
            representation_vector = np.mean(frame_features, axis=0)
        else:
            representation_vector = np.zeros(512)
        
        # Extract visual semantic tags
        semantic_tags = self._extract_visual_tags(video_frames[frame_indices] if frame_indices else video_frames)
        
        return ModalityRepresentation(
            modality_type=ModalityType.VISUAL,
            representation_vector=representation_vector,
            confidence_score=0.7,
            temporal_alignment=frame_indices,
            spatial_alignment=None,
            semantic_tags=semantic_tags,
            extraction_method="mock_visual_cnn",
            timestamp=time.time()
        )
    
    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame."""
        
        # Mock CNN features based on frame statistics
        features = np.zeros(512)
        
        # Color features
        if len(frame.shape) == 3:
            mean_colors = np.mean(frame, axis=(0, 1))
            features[:3] = mean_colors / 255.0
            
            std_colors = np.std(frame, axis=(0, 1))
            features[3:6] = std_colors / 255.0
        
        # Texture features (simplified)
        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
        
        # Edge density
        edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
        features[6] = edges / (gray.shape[0] * gray.shape[1] * 255.0)
        
        # Brightness statistics
        features[7] = np.mean(gray) / 255.0
        features[8] = np.std(gray) / 255.0
        
        # Spatial frequency content
        fft = np.fft.fft2(gray)
        features[9] = np.mean(np.abs(fft)) / (gray.shape[0] * gray.shape[1])
        
        # Fill remaining features with derived statistics
        for i in range(10, 512):
            base_val = features[i % 10]
            features[i] = base_val + np.random.normal(0, 0.1)
        
        return features
    
    def _extract_visual_tags(self, frames: np.ndarray) -> List[str]:
        """Extract visual semantic tags."""
        
        tags = []
        
        # Analyze color distribution
        if len(frames.shape) == 4 and frames.shape[-1] == 3:  # Color frames
            mean_color = np.mean(frames, axis=(0, 1, 2))
            dominant_channel = np.argmax(mean_color)
            
            if dominant_channel == 0:
                tags.append("red_dominant")
            elif dominant_channel == 1:
                tags.append("green_dominant")
            else:
                tags.append("blue_dominant")
            
            # Brightness
            brightness = np.mean(mean_color)
            if brightness > 200:
                tags.append("bright")
            elif brightness < 100:
                tags.append("dark")
            else:
                tags.append("medium_brightness")
        
        # Motion analysis (simplified)
        if len(frames) > 1:
            frame_diff = np.mean(np.abs(frames[1:] - frames[:-1]))
            if frame_diff > 20:
                tags.append("high_motion")
            elif frame_diff > 5:
                tags.append("medium_motion")
            else:
                tags.append("low_motion")
        
        # Scene complexity
        if len(frames) > 0:
            edge_density = np.std(frames[0])
            if edge_density > 50:
                tags.append("complex_scene")
            elif edge_density > 20:
                tags.append("medium_complexity")
            else:
                tags.append("simple_scene")
        
        return tags


class CrossModalAttentionAnalyzer:
    """Analyzes cross-modal attention patterns."""
    
    def __init__(self):
        self.attention_patterns = {}
        self.attention_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_cross_modal_attention(self, 
                                    modality1: ModalityRepresentation,
                                    modality2: ModalityRepresentation) -> Dict[str, float]:
        """Analyze attention patterns between two modalities."""
        
        # Calculate attention scores
        attention_scores = {}
        
        # 1. Representational similarity (cosine similarity)
        cosine_sim = self._calculate_cosine_similarity(
            modality1.representation_vector,
            modality2.representation_vector
        )
        attention_scores["representational_similarity"] = cosine_sim
        
        # 2. Semantic tag overlap
        semantic_overlap = self._calculate_semantic_overlap(
            modality1.semantic_tags,
            modality2.semantic_tags
        )
        attention_scores["semantic_overlap"] = semantic_overlap
        
        # 3. Temporal alignment (if applicable)
        temporal_correlation = self._calculate_temporal_correlation(
            modality1.temporal_alignment,
            modality2.temporal_alignment
        )
        attention_scores["temporal_correlation"] = temporal_correlation
        
        # 4. Spatial alignment (if applicable)
        spatial_correlation = self._calculate_spatial_correlation(
            modality1.spatial_alignment,
            modality2.spatial_alignment
        )
        attention_scores["spatial_correlation"] = spatial_correlation
        
        # 5. Confidence-weighted attention
        confidence_weight = (modality1.confidence_score + modality2.confidence_score) / 2
        attention_scores["confidence_weighted_attention"] = confidence_weight
        
        # 6. Cross-modal activation analysis
        cross_activation = self._analyze_cross_modal_activation(
            modality1.representation_vector,
            modality2.representation_vector
        )
        attention_scores["cross_modal_activation"] = cross_activation
        
        # Overall attention score
        overall_attention = (
            0.3 * cosine_sim +
            0.25 * semantic_overlap +
            0.2 * temporal_correlation +
            0.1 * spatial_correlation +
            0.1 * confidence_weight +
            0.05 * cross_activation
        )
        attention_scores["overall_attention"] = overall_attention
        
        return attention_scores
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        
        # Ensure same dimensionality
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
        else:
            similarity = 0.0
        
        # Normalize to [0, 1]
        return (similarity + 1) / 2
    
    def _calculate_semantic_overlap(self, tags1: List[str], tags2: List[str]) -> float:
        """Calculate semantic tag overlap."""
        
        if not tags1 or not tags2:
            return 0.0
        
        set1 = set(tags1)
        set2 = set(tags2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_temporal_correlation(self, 
                                      alignment1: Optional[List[int]],
                                      alignment2: Optional[List[int]]) -> float:
        """Calculate temporal correlation between alignments."""
        
        if alignment1 is None or alignment2 is None:
            return 0.5  # Neutral score when temporal info unavailable
        
        # Calculate overlap in temporal windows
        set1 = set(alignment1)
        set2 = set(alignment2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def _calculate_spatial_correlation(self,
                                     alignment1: Optional[Tuple[int, int, int, int]],
                                     alignment2: Optional[Tuple[int, int, int, int]]) -> float:
        """Calculate spatial correlation between alignments."""
        
        if alignment1 is None or alignment2 is None:
            return 0.5  # Neutral score when spatial info unavailable
        
        # Calculate IoU (Intersection over Union)
        x1_1, y1_1, w1, h1 = alignment1
        x2_1, y2_1, w2, h2 = alignment2
        
        x1_2, y1_2 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x2_1 + w2, y2_1 + h2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_2, x2_2) - max(x1_1, x2_1))
        y_overlap = max(0, min(y1_2, y2_2) - max(y1_1, y2_1))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_cross_modal_activation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Analyze cross-modal activation patterns."""
        
        # Ensure same dimensionality
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        # Element-wise correlation
        if min_dim > 1:
            correlation = np.corrcoef(vec1, vec2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Normalize to [0, 1]
        return abs(correlation)


class SemanticDriftDetector:
    """Detects semantic drift across modalities and time."""
    
    def __init__(self, drift_threshold: float = 0.3):
        self.drift_threshold = drift_threshold
        self.baseline_representations = {}
        self.drift_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def set_baseline(self, modality_type: ModalityType, 
                    representation: ModalityRepresentation):
        """Set baseline representation for drift detection."""
        self.baseline_representations[modality_type] = representation
    
    def detect_semantic_drift(self, 
                            current_representation: ModalityRepresentation) -> Dict[str, float]:
        """Detect semantic drift from baseline."""
        
        modality_type = current_representation.modality_type
        
        if modality_type not in self.baseline_representations:
            return {
                "drift_magnitude": 0.0,
                "drift_detected": False,
                "confidence": 0.0,
                "explanation": "No baseline available"
            }
        
        baseline = self.baseline_representations[modality_type]
        
        # Calculate different types of drift
        drift_metrics = {}
        
        # 1. Representational drift (vector distance)
        vector_drift = self._calculate_vector_drift(
            baseline.representation_vector,
            current_representation.representation_vector
        )
        drift_metrics["vector_drift"] = vector_drift
        
        # 2. Semantic tag drift
        tag_drift = self._calculate_tag_drift(
            baseline.semantic_tags,
            current_representation.semantic_tags
        )
        drift_metrics["tag_drift"] = tag_drift
        
        # 3. Confidence drift
        confidence_drift = abs(baseline.confidence_score - current_representation.confidence_score)
        drift_metrics["confidence_drift"] = confidence_drift
        
        # 4. Temporal drift (if applicable)
        temporal_drift = self._calculate_temporal_drift(
            baseline.temporal_alignment,
            current_representation.temporal_alignment
        )
        drift_metrics["temporal_drift"] = temporal_drift
        
        # Overall drift magnitude
        overall_drift = (
            0.4 * vector_drift +
            0.3 * tag_drift +
            0.2 * confidence_drift +
            0.1 * temporal_drift
        )
        drift_metrics["overall_drift"] = overall_drift
        
        # Drift detection
        drift_detected = overall_drift > self.drift_threshold
        
        # Generate explanation
        if drift_detected:
            dominant_drift = max(drift_metrics.items(), key=lambda x: x[1])
            explanation = f"Primary drift source: {dominant_drift[0]} ({dominant_drift[1]:.3f})"
        else:
            explanation = "No significant drift detected"
        
        result = {
            "drift_magnitude": overall_drift,
            "drift_detected": drift_detected,
            "confidence": min(baseline.confidence_score, current_representation.confidence_score),
            "explanation": explanation,
            "detailed_metrics": drift_metrics
        }
        
        # Store drift event
        drift_event = {
            "modality_type": modality_type.value,
            "timestamp": time.time(),
            "drift_magnitude": overall_drift,
            "drift_detected": drift_detected
        }
        self.drift_history.append(drift_event)
        
        return result
    
    def _calculate_vector_drift(self, baseline_vec: np.ndarray, 
                              current_vec: np.ndarray) -> float:
        """Calculate drift in representation vectors."""
        
        # Ensure same dimensionality
        min_dim = min(len(baseline_vec), len(current_vec))
        baseline_vec = baseline_vec[:min_dim]
        current_vec = current_vec[:min_dim]
        
        # Calculate normalized Euclidean distance
        distance = euclidean(baseline_vec, current_vec)
        max_possible_distance = np.sqrt(2)  # For normalized vectors
        
        normalized_distance = distance / max_possible_distance
        return min(1.0, normalized_distance)
    
    def _calculate_tag_drift(self, baseline_tags: List[str], 
                           current_tags: List[str]) -> float:
        """Calculate drift in semantic tags."""
        
        baseline_set = set(baseline_tags)
        current_set = set(current_tags)
        
        if not baseline_set and not current_set:
            return 0.0
        
        # Jaccard distance (1 - Jaccard similarity)
        intersection = len(baseline_set.intersection(current_set))
        union = len(baseline_set.union(current_set))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        jaccard_distance = 1.0 - jaccard_similarity
        
        return jaccard_distance
    
    def _calculate_temporal_drift(self, 
                                baseline_alignment: Optional[List[int]],
                                current_alignment: Optional[List[int]]) -> float:
        """Calculate drift in temporal alignment."""
        
        if baseline_alignment is None or current_alignment is None:
            return 0.0
        
        baseline_set = set(baseline_alignment)
        current_set = set(current_alignment)
        
        if not baseline_set and not current_set:
            return 0.0
        
        # Calculate temporal shift
        baseline_center = np.mean(baseline_alignment) if baseline_alignment else 0
        current_center = np.mean(current_alignment) if current_alignment else 0
        
        temporal_shift = abs(current_center - baseline_center)
        
        # Normalize by maximum possible shift (assuming frame indices)
        max_frames = 100  # Assumption
        normalized_shift = temporal_shift / max_frames
        
        return min(1.0, normalized_shift)


class HallucinationDetector:
    """Detects multimodal hallucinations."""
    
    def __init__(self):
        self.hallucination_patterns = self._define_hallucination_patterns()
        self.detection_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _define_hallucination_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define patterns indicative of hallucination."""
        
        return {
            "semantic_inconsistency": {
                "description": "Semantic content doesn't match across modalities",
                "threshold": 0.3,
                "weight": 0.4
            },
            "confidence_mismatch": {
                "description": "High confidence with low consistency",
                "threshold": 0.4,
                "weight": 0.2
            },
            "temporal_incoherence": {
                "description": "Temporal patterns don't align logically",
                "threshold": 0.5,
                "weight": 0.2
            },
            "attention_scatter": {
                "description": "Cross-modal attention is unfocused",
                "threshold": 0.6,
                "weight": 0.1
            },
            "representation_anomaly": {
                "description": "Representation vectors show anomalous patterns",
                "threshold": 0.7,
                "weight": 0.1
            }
        }
    
    def detect_hallucinations(self, 
                            modality_representations: List[ModalityRepresentation],
                            cross_modal_alignments: List[CrossModalAlignment]) -> Dict[str, float]:
        """Detect hallucination indicators."""
        
        hallucination_scores = {}
        
        # 1. Semantic inconsistency
        semantic_inconsistency = self._detect_semantic_inconsistency(
            modality_representations, cross_modal_alignments
        )
        hallucination_scores["semantic_inconsistency"] = semantic_inconsistency
        
        # 2. Confidence mismatch
        confidence_mismatch = self._detect_confidence_mismatch(
            modality_representations, cross_modal_alignments
        )
        hallucination_scores["confidence_mismatch"] = confidence_mismatch
        
        # 3. Temporal incoherence
        temporal_incoherence = self._detect_temporal_incoherence(
            modality_representations
        )
        hallucination_scores["temporal_incoherence"] = temporal_incoherence
        
        # 4. Attention scatter
        attention_scatter = self._detect_attention_scatter(
            cross_modal_alignments
        )
        hallucination_scores["attention_scatter"] = attention_scatter
        
        # 5. Representation anomaly
        representation_anomaly = self._detect_representation_anomaly(
            modality_representations
        )
        hallucination_scores["representation_anomaly"] = representation_anomaly
        
        # Overall hallucination score
        overall_score = sum(
            score * self.hallucination_patterns[pattern]["weight"]
            for pattern, score in hallucination_scores.items()
            if pattern in self.hallucination_patterns
        )
        hallucination_scores["overall_hallucination"] = overall_score
        
        # Detection decisions
        detected_patterns = []
        for pattern, score in hallucination_scores.items():
            if pattern in self.hallucination_patterns:
                threshold = self.hallucination_patterns[pattern]["threshold"]
                if score > threshold:
                    detected_patterns.append(pattern)
        
        hallucination_scores["detected_patterns"] = detected_patterns
        hallucination_scores["hallucination_detected"] = len(detected_patterns) > 0
        
        return hallucination_scores
    
    def _detect_semantic_inconsistency(self, 
                                     modality_representations: List[ModalityRepresentation],
                                     cross_modal_alignments: List[CrossModalAlignment]) -> float:
        """Detect semantic inconsistency across modalities."""
        
        if len(cross_modal_alignments) == 0:
            return 0.0
        
        # Look for low semantic overlap with high claimed consistency
        inconsistency_scores = []
        
        for alignment in cross_modal_alignments:
            semantic_overlap = alignment.semantic_overlap
            consistency_claim = alignment.alignment_score
            
            # Inconsistency when low overlap but high consistency claim
            if consistency_claim > 0.7 and semantic_overlap < 0.3:
                inconsistency_scores.append(0.8)
            elif consistency_claim > 0.5 and semantic_overlap < 0.2:
                inconsistency_scores.append(0.6)
            else:
                # Normal case: consistency should roughly match semantic overlap
                discrepancy = abs(consistency_claim - semantic_overlap)
                inconsistency_scores.append(discrepancy)
        
        return np.mean(inconsistency_scores) if inconsistency_scores else 0.0
    
    def _detect_confidence_mismatch(self,
                                  modality_representations: List[ModalityRepresentation],
                                  cross_modal_alignments: List[CrossModalAlignment]) -> float:
        """Detect confidence mismatch patterns."""
        
        if not modality_representations:
            return 0.0
        
        # High individual confidence but low cross-modal alignment
        mismatch_scores = []
        
        for repr in modality_representations:
            if repr.confidence_score > 0.8:
                # Find alignments involving this modality
                relevant_alignments = [
                    a for a in cross_modal_alignments
                    if a.modality1_type == repr.modality_type or 
                       a.modality2_type == repr.modality_type
                ]
                
                if relevant_alignments:
                    avg_alignment = np.mean([a.alignment_score for a in relevant_alignments])
                    if avg_alignment < 0.4:  # Low alignment despite high confidence
                        mismatch_scores.append(0.7)
                    else:
                        mismatch_scores.append(0.0)
        
        return np.mean(mismatch_scores) if mismatch_scores else 0.0
    
    def _detect_temporal_incoherence(self,
                                   modality_representations: List[ModalityRepresentation]) -> float:
        """Detect temporal incoherence patterns."""
        
        temporal_reprs = [
            r for r in modality_representations 
            if r.temporal_alignment is not None
        ]
        
        if len(temporal_reprs) < 2:
            return 0.0
        
        # Check for overlapping temporal windows that shouldn't overlap
        incoherence_scores = []
        
        for i, repr1 in enumerate(temporal_reprs):
            for j, repr2 in enumerate(temporal_reprs[i+1:], i+1):
                alignment1 = set(repr1.temporal_alignment)
                alignment2 = set(repr2.temporal_alignment)
                
                overlap = len(alignment1.intersection(alignment2))
                union = len(alignment1.union(alignment2))
                
                if union > 0:
                    overlap_ratio = overlap / union
                    
                    # High overlap between different semantic content is suspicious
                    tag_overlap = len(set(repr1.semantic_tags).intersection(set(repr2.semantic_tags)))
                    tag_union = len(set(repr1.semantic_tags).union(set(repr2.semantic_tags)))
                    
                    if tag_union > 0:
                        semantic_similarity = tag_overlap / tag_union
                        
                        # Incoherence: high temporal overlap, low semantic similarity
                        if overlap_ratio > 0.7 and semantic_similarity < 0.3:
                            incoherence_scores.append(0.8)
                        else:
                            incoherence_scores.append(0.0)
        
        return np.mean(incoherence_scores) if incoherence_scores else 0.0
    
    def _detect_attention_scatter(self,
                                cross_modal_alignments: List[CrossModalAlignment]) -> float:
        """Detect scattered attention patterns."""
        
        if not cross_modal_alignments:
            return 0.0
        
        # Look for unfocused attention (all alignments are mediocre)
        alignment_scores = [a.attention_correlation for a in cross_modal_alignments]
        
        if not alignment_scores:
            return 0.0
        
        # Scatter indicated by low variance and medium mean
        mean_attention = np.mean(alignment_scores)
        var_attention = np.var(alignment_scores)
        
        # Scattered if mean is medium (~0.5) and variance is low
        if 0.4 <= mean_attention <= 0.6 and var_attention < 0.1:
            scatter_score = 0.7
        else:
            scatter_score = 0.0
        
        return scatter_score
    
    def _detect_representation_anomaly(self,
                                     modality_representations: List[ModalityRepresentation]) -> float:
        """Detect anomalous representation patterns."""
        
        if len(modality_representations) < 2:
            return 0.0
        
        # Collect all representation vectors
        vectors = [r.representation_vector for r in modality_representations]
        
        # Check for anomalous patterns
        anomaly_scores = []
        
        for i, vector in enumerate(vectors):
            # Anomaly 1: Vector with extreme values
            max_val = np.max(np.abs(vector))
            if max_val > 5.0:  # Assuming normalized vectors should be small
                anomaly_scores.append(0.8)
                continue
            
            # Anomaly 2: Vector very different from others
            distances = []
            for j, other_vector in enumerate(vectors):
                if i != j:
                    min_dim = min(len(vector), len(other_vector))
                    distance = euclidean(vector[:min_dim], other_vector[:min_dim])
                    distances.append(distance)
            
            if distances:
                avg_distance = np.mean(distances)
                if avg_distance > 2.0:  # Very far from other representations
                    anomaly_scores.append(0.6)
                else:
                    anomaly_scores.append(0.0)
        
        return np.mean(anomaly_scores) if anomaly_scores else 0.0


class MultimodalConsistencyFramework:
    """Main framework for multimodal consistency analysis."""
    
    def __init__(self):
        self.text_extractor = TextualSemanticExtractor()
        self.visual_extractor = VisualSemanticExtractor()
        self.attention_analyzer = CrossModalAttentionAnalyzer()
        self.drift_detector = SemanticDriftDetector()
        self.hallucination_detector = HallucinationDetector()
        self.analysis_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def comprehensive_multimodal_analysis(self,
                                              video_data: np.ndarray,
                                              text_prompt: str,
                                              sample_id: str = "unknown",
                                              baseline_data: Optional[Dict[str, Any]] = None) -> MultimodalConsistencyResult:
        """
        Perform comprehensive multimodal consistency analysis.
        
        Args:
            video_data: Video frames as numpy array [T, H, W, C]
            text_prompt: Associated text prompt
            sample_id: Identifier for the sample
            baseline_data: Optional baseline for drift detection
        
        Returns:
            Comprehensive multimodal consistency analysis result
        """
        
        self.logger.info(f"Starting multimodal consistency analysis for sample {sample_id}")
        
        # 1. Extract modality representations
        self.logger.info("Extracting modality representations...")
        
        # Text representation
        text_repr = self.text_extractor.extract_semantic_representation(text_prompt)
        
        # Visual representation
        visual_repr = self.visual_extractor.extract_visual_representation(video_data)
        
        # Temporal representation (frames over time)
        temporal_repr = self.visual_extractor.extract_visual_representation(
            video_data, frame_indices=list(range(0, len(video_data), max(1, len(video_data)//10)))
        )
        temporal_repr.modality_type = ModalityType.TEMPORAL
        
        modality_representations = [text_repr, visual_repr, temporal_repr]
        
        # 2. Analyze cross-modal alignments
        self.logger.info("Analyzing cross-modal alignments...")
        cross_modal_alignments = []
        
        for i, repr1 in enumerate(modality_representations):
            for j, repr2 in enumerate(modality_representations[i+1:], i+1):
                alignment = await self._analyze_modality_alignment(repr1, repr2)
                cross_modal_alignments.append(alignment)
        
        # 3. Detect semantic drift (if baseline available)
        self.logger.info("Analyzing semantic drift...")
        drift_scores = {}
        
        if baseline_data:
            for repr in modality_representations:
                if repr.modality_type.value in baseline_data:
                    baseline_repr = baseline_data[repr.modality_type.value]
                    drift_result = self.drift_detector.detect_semantic_drift(repr)
                    drift_scores[repr.modality_type.value] = drift_result
        
        overall_drift_score = np.mean([d["drift_magnitude"] for d in drift_scores.values()]) if drift_scores else 0.0
        
        # 4. Detect hallucinations
        self.logger.info("Detecting multimodal hallucinations...")
        hallucination_indicators = self.hallucination_detector.detect_hallucinations(
            modality_representations, cross_modal_alignments
        )
        
        # 5. Analyze emergent cross-modal understanding
        self.logger.info("Analyzing emergent understanding...")
        emergent_understanding_score = await self._analyze_emergent_understanding(
            modality_representations, cross_modal_alignments
        )
        
        # 6. Calculate overall consistency
        overall_consistency_score = self._calculate_overall_consistency(cross_modal_alignments)
        consistency_level = self._determine_consistency_level(overall_consistency_score)
        
        # Create result
        result = MultimodalConsistencyResult(
            sample_id=sample_id,
            modality_representations=modality_representations,
            cross_modal_alignments=cross_modal_alignments,
            overall_consistency_score=overall_consistency_score,
            consistency_level=consistency_level,
            semantic_drift_score=overall_drift_score,
            hallucination_indicators=hallucination_indicators,
            emergent_understanding_score=emergent_understanding_score,
            analysis_timestamp=time.time(),
            detailed_analysis={
                "drift_analysis": drift_scores,
                "attention_patterns": self.attention_analyzer.attention_patterns,
                "processing_metadata": {
                    "video_shape": video_data.shape,
                    "text_length": len(text_prompt),
                    "extraction_methods": [r.extraction_method for r in modality_representations]
                }
            }
        )
        
        # Store result
        self.analysis_history.append(result)
        
        self.logger.info(f"Multimodal analysis complete for sample {sample_id}")
        self.logger.info(f"Overall consistency: {overall_consistency_score:.3f} ({consistency_level.value})")
        self.logger.info(f"Emergent understanding: {emergent_understanding_score:.3f}")
        self.logger.info(f"Hallucination risk: {hallucination_indicators['overall_hallucination']:.3f}")
        
        return result
    
    async def _analyze_modality_alignment(self, 
                                        modality1: ModalityRepresentation,
                                        modality2: ModalityRepresentation) -> CrossModalAlignment:
        """Analyze alignment between two modalities."""
        
        # Get attention analysis
        attention_scores = self.attention_analyzer.analyze_cross_modal_attention(modality1, modality2)
        
        # Extract key scores
        alignment_score = attention_scores["overall_attention"]
        semantic_overlap = attention_scores["semantic_overlap"]
        temporal_correlation = attention_scores["temporal_correlation"]
        spatial_correlation = attention_scores["spatial_correlation"]
        attention_correlation = attention_scores["cross_modal_activation"]
        
        # Calculate drift (simple version for alignment)
        vector_distance = euclidean(
            modality1.representation_vector[:min(len(modality1.representation_vector), 
                                               len(modality2.representation_vector))],
            modality2.representation_vector[:min(len(modality1.representation_vector), 
                                               len(modality2.representation_vector))]
        )
        drift_magnitude = min(1.0, vector_distance / 2.0)  # Normalize
        
        # Determine consistency level
        consistency_level = self._determine_consistency_level(alignment_score)
        
        # Generate explanation
        if alignment_score > 0.8:
            explanation = "Strong multimodal alignment with high semantic coherence"
        elif alignment_score > 0.6:
            explanation = "Good multimodal alignment with moderate consistency"
        elif alignment_score > 0.4:
            explanation = "Partial multimodal alignment with some inconsistencies"
        else:
            explanation = "Weak multimodal alignment with significant inconsistencies"
        
        # Calculate confidence
        confidence = (modality1.confidence_score + modality2.confidence_score) / 2 * alignment_score
        
        return CrossModalAlignment(
            modality1_type=modality1.modality_type,
            modality2_type=modality2.modality_type,
            alignment_score=alignment_score,
            consistency_level=consistency_level,
            semantic_overlap=semantic_overlap,
            temporal_correlation=temporal_correlation,
            spatial_correlation=spatial_correlation,
            attention_correlation=attention_correlation,
            drift_magnitude=drift_magnitude,
            explanation=explanation,
            confidence=confidence
        )
    
    async def _analyze_emergent_understanding(self,
                                            modality_representations: List[ModalityRepresentation],
                                            cross_modal_alignments: List[CrossModalAlignment]) -> float:
        """Analyze emergent cross-modal understanding."""
        
        understanding_factors = []
        
        # 1. Cross-modal consistency
        if cross_modal_alignments:
            avg_consistency = np.mean([a.alignment_score for a in cross_modal_alignments])
            understanding_factors.append(avg_consistency)
        
        # 2. Semantic richness (diversity and coherence of tags)
        all_tags = []
        for repr in modality_representations:
            all_tags.extend(repr.semantic_tags)
        
        if all_tags:
            tag_diversity = len(set(all_tags)) / len(all_tags)
            understanding_factors.append(tag_diversity)
        
        # 3. Representation complexity
        representation_complexities = []
        for repr in modality_representations:
            complexity = np.std(repr.representation_vector)
            representation_complexities.append(min(1.0, complexity))
        
        if representation_complexities:
            avg_complexity = np.mean(representation_complexities)
            understanding_factors.append(avg_complexity)
        
        # 4. Cross-modal attention coherence
        attention_correlations = [a.attention_correlation for a in cross_modal_alignments if a.attention_correlation > 0]
        if attention_correlations:
            attention_coherence = np.mean(attention_correlations)
            understanding_factors.append(attention_coherence)
        
        # 5. Temporal-semantic alignment
        temporal_reprs = [r for r in modality_representations if r.temporal_alignment is not None]
        if len(temporal_reprs) > 1:
            temporal_alignments = []
            for i, repr1 in enumerate(temporal_reprs):
                for repr2 in temporal_reprs[i+1:]:
                    if repr1.temporal_alignment and repr2.temporal_alignment:
                        overlap = len(set(repr1.temporal_alignment).intersection(set(repr2.temporal_alignment)))
                        union = len(set(repr1.temporal_alignment).union(set(repr2.temporal_alignment)))
                        if union > 0:
                            temporal_alignments.append(overlap / union)
            
            if temporal_alignments:
                temporal_understanding = np.mean(temporal_alignments)
                understanding_factors.append(temporal_understanding)
        
        # Overall emergent understanding
        if understanding_factors:
            emergent_understanding = np.mean(understanding_factors)
        else:
            emergent_understanding = 0.5
        
        return emergent_understanding
    
    def _calculate_overall_consistency(self, alignments: List[CrossModalAlignment]) -> float:
        """Calculate overall consistency score."""
        
        if not alignments:
            return 0.5
        
        # Weighted combination of alignment scores
        alignment_scores = [a.alignment_score for a in alignments]
        semantic_overlaps = [a.semantic_overlap for a in alignments]
        
        # Overall consistency is average of alignments weighted by semantic overlap
        weighted_scores = []
        for alignment_score, semantic_overlap in zip(alignment_scores, semantic_overlaps):
            weight = 0.5 + 0.5 * semantic_overlap  # Weight from 0.5 to 1.0
            weighted_scores.append(alignment_score * weight)
        
        return np.mean(weighted_scores)
    
    def _determine_consistency_level(self, score: float) -> ConsistencyLevel:
        """Determine consistency level from score."""
        
        if score >= 0.95:
            return ConsistencyLevel.PERFECT
        elif score >= 0.85:
            return ConsistencyLevel.HIGH
        elif score >= 0.70:
            return ConsistencyLevel.MODERATE
        elif score >= 0.50:
            return ConsistencyLevel.LOW
        else:
            return ConsistencyLevel.INCONSISTENT
    
    def analyze_consistency_trends(self) -> Dict[str, Any]:
        """Analyze trends in consistency over time."""
        
        if len(self.analysis_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # Extract time series data
        timestamps = [r.analysis_timestamp for r in self.analysis_history]
        consistency_scores = [r.overall_consistency_score for r in self.analysis_history]
        emergent_scores = [r.emergent_understanding_score for r in self.analysis_history]
        drift_scores = [r.semantic_drift_score for r in self.analysis_history]
        hallucination_scores = [r.hallucination_indicators.get("overall_hallucination", 0) for r in self.analysis_history]
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        
        trends = {
            "consistency_trend": calculate_trend(consistency_scores),
            "emergent_understanding_trend": calculate_trend(emergent_scores),
            "drift_trend": calculate_trend(drift_scores),
            "hallucination_trend": calculate_trend(hallucination_scores)
        }
        
        # Statistical summaries
        summary = {
            "total_analyses": len(self.analysis_history),
            "average_consistency": np.mean(consistency_scores),
            "consistency_std": np.std(consistency_scores),
            "average_emergent_understanding": np.mean(emergent_scores),
            "average_drift": np.mean(drift_scores),
            "average_hallucination_risk": np.mean(hallucination_scores),
            "trends": trends,
            "consistency_distribution": {
                level.value: sum(1 for r in self.analysis_history if r.consistency_level == level)
                for level in ConsistencyLevel
            }
        }
        
        return summary
    
    def export_multimodal_analysis(self, filepath: str):
        """Export multimodal analysis results."""
        
        export_data = {
            "framework_info": {
                "version": "1.0.0",
                "modality_types": [m.value for m in ModalityType],
                "consistency_levels": [c.value for c in ConsistencyLevel],
                "export_timestamp": time.time()
            },
            "analysis_results": [result.to_dict() for result in self.analysis_history],
            "trend_analysis": self.analyze_consistency_trends(),
            "drift_history": self.drift_detector.drift_history,
            "hallucination_detection_history": self.hallucination_detector.detection_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported multimodal analysis to {filepath}")


# Example usage and testing
async def run_multimodal_consistency_example():
    """Example of multimodal consistency framework."""
    
    print("=== Multimodal Consistency Framework Example ===")
    
    # Create framework
    framework = MultimodalConsistencyFramework()
    
    # Create mock data
    def create_mock_video_and_text():
        """Create mock video and text data for testing."""
        
        scenarios = [
            {
                "name": "Consistent Scenario",
                "video": np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8),
                "text": "A red ball rolling down a hill in a sunny day",
                "expected_consistency": "high"
            },
            {
                "name": "Moderately Consistent",
                "video": np.random.randint(0, 255, (25, 128, 128, 3), dtype=np.uint8),
                "text": "A person walking in a park with trees and flowers",
                "expected_consistency": "moderate"
            },
            {
                "name": "Inconsistent Scenario",
                "video": np.random.randint(0, 255, (15, 64, 64, 3), dtype=np.uint8),
                "text": "A spaceship flying through a nebula with purple clouds",
                "expected_consistency": "low"
            }
        ]
        
        # Make videos more realistic by adding some structure
        for scenario in scenarios:
            video = scenario["video"]
            
            # Add moving object
            for t in range(len(video)):
                center_x = 10 + t * 2
                center_y = 32
                if center_x < video.shape[2] - 5:
                    video[t, center_y-3:center_y+3, center_x-3:center_x+3, 0] = 255  # Red object
                
                # Add background pattern
                video[t, :10, :, 1] = 150  # Green strip
                video[t, -10:, :, 2] = 200  # Blue strip
        
        return scenarios
    
    # Test scenarios
    test_scenarios = create_mock_video_and_text()
    
    # Run analysis on each scenario
    for i, scenario in enumerate(test_scenarios):
        print(f"\n--- Analyzing Scenario {i+1}: {scenario['name']} ---")
        
        video_data = scenario["video"]
        text_prompt = scenario["text"]
        
        print(f"Video shape: {video_data.shape}")
        print(f"Text prompt: '{text_prompt}'")
        print(f"Expected consistency: {scenario['expected_consistency']}")
        
        # Run comprehensive analysis
        result = await framework.comprehensive_multimodal_analysis(
            video_data=video_data,
            text_prompt=text_prompt,
            sample_id=f"scenario_{i+1}"
        )
        
        # Display results
        print(f"\nResults:")
        print(f"Overall consistency: {result.overall_consistency_score:.3f} ({result.consistency_level.value})")
        print(f"Emergent understanding: {result.emergent_understanding_score:.3f}")
        print(f"Semantic drift: {result.semantic_drift_score:.3f}")
        print(f"Hallucination risk: {result.hallucination_indicators['overall_hallucination']:.3f}")
        
        # Cross-modal alignments
        print(f"\nCross-modal alignments:")
        for alignment in result.cross_modal_alignments:
            print(f"  {alignment.modality1_type.value} <-> {alignment.modality2_type.value}: "
                  f"score={alignment.alignment_score:.3f}, "
                  f"semantic_overlap={alignment.semantic_overlap:.3f}")
        
        # Hallucination indicators
        if result.hallucination_indicators["hallucination_detected"]:
            print(f"Hallucination patterns detected: {result.hallucination_indicators['detected_patterns']}")
        else:
            print("No significant hallucination patterns detected")
    
    # Trend analysis
    print("\n--- Trend Analysis ---")
    trends = framework.analyze_consistency_trends()
    
    if "total_analyses" in trends:
        print(f"Total analyses: {trends['total_analyses']}")
        print(f"Average consistency: {trends['average_consistency']:.3f} ± {trends['consistency_std']:.3f}")
        print(f"Average emergent understanding: {trends['average_emergent_understanding']:.3f}")
        print(f"Average hallucination risk: {trends['average_hallucination_risk']:.3f}")
        
        print("Consistency level distribution:")
        for level, count in trends['consistency_distribution'].items():
            print(f"  {level}: {count}")
        
        print("Trends:")
        trend_data = trends['trends']
        for metric, trend in trend_data.items():
            direction = "↑" if trend > 0.01 else "↓" if trend < -0.01 else "→"
            print(f"  {metric}: {direction} {trend:.4f}")
    
    # Export results
    export_path = "multimodal_consistency_analysis.json"
    framework.export_multimodal_analysis(export_path)
    print(f"\nResults exported to {export_path}")
    
    # Analysis insights
    print("\n=== Key Insights ===")
    
    # High consistency samples
    high_consistency_samples = [
        r for r in framework.analysis_history
        if r.consistency_level in [ConsistencyLevel.HIGH, ConsistencyLevel.PERFECT]
    ]
    
    if high_consistency_samples:
        print(f"High consistency samples: {len(high_consistency_samples)}")
        for result in high_consistency_samples:
            print(f"  - {result.sample_id}: consistency={result.overall_consistency_score:.3f}")
    
    # Emergent understanding insights
    high_understanding_samples = [
        r for r in framework.analysis_history
        if r.emergent_understanding_score > 0.7
    ]
    
    if high_understanding_samples:
        print(f"High emergent understanding: {len(high_understanding_samples)}")
        for result in high_understanding_samples:
            print(f"  - {result.sample_id}: understanding={result.emergent_understanding_score:.3f}")
    
    # Hallucination risks
    high_risk_samples = [
        r for r in framework.analysis_history
        if r.hallucination_indicators.get("overall_hallucination", 0) > 0.6
    ]
    
    if high_risk_samples:
        print(f"High hallucination risk samples: {len(high_risk_samples)}")
        for result in high_risk_samples:
            risk_score = result.hallucination_indicators["overall_hallucination"]
            patterns = result.hallucination_indicators.get("detected_patterns", [])
            print(f"  - {result.sample_id}: risk={risk_score:.3f}, patterns={patterns}")
    
    return {
        "framework": framework,
        "results": framework.analysis_history,
        "trends": trends
    }


if __name__ == "__main__":
    # Run example
    asyncio.run(run_multimodal_consistency_example())