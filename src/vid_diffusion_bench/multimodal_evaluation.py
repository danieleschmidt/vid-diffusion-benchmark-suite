"""Multi-modal evaluation framework for video diffusion models.

Comprehensive evaluation including audio-visual synchronization,
cross-modal consistency, and perceptual quality assessment.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import cv2
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MultiModalResult:
    """Container for multi-modal evaluation results."""
    visual_quality: float
    audio_quality: Optional[float]
    audio_visual_sync: Optional[float]
    text_video_alignment: float
    perceptual_quality: float
    cross_modal_consistency: float
    overall_score: float
    detailed_metrics: Dict[str, Any]


class AudioVisualAnalyzer:
    """Analyzes audio-visual synchronization and quality."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.sample_rate = 44100
        self.sync_window_ms = 40  # 40ms sync tolerance
        
    def extract_audio_features(self, audio_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract audio features for synchronization analysis."""
        # Compute spectral features
        stft = torch.stft(
            audio_tensor.squeeze(),
            n_fft=2048,
            hop_length=512,
            window=torch.hann_window(2048).to(self.device),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        spectral_centroid = self._compute_spectral_centroid(magnitude)
        spectral_rolloff = self._compute_spectral_rolloff(magnitude)
        spectral_flux = self._compute_spectral_flux(magnitude)
        onset_strength = self._compute_onset_strength(magnitude)
        
        return {
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "spectral_flux": spectral_flux,
            "onset_strength": onset_strength,
            "rms_energy": torch.sqrt(torch.mean(magnitude**2, dim=0))
        }
    
    def extract_visual_motion_features(self, video_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract visual motion features for synchronization."""
        # Convert to grayscale for optical flow
        if video_tensor.shape[1] == 3:  # RGB
            gray_video = torch.mean(video_tensor, dim=1, keepdim=True)
        else:
            gray_video = video_tensor
        
        # Compute frame differences (motion approximation)
        frame_diff = torch.diff(gray_video, dim=0)
        motion_intensity = torch.mean(torch.abs(frame_diff), dim=[1, 2, 3])
        
        # Compute motion vectors (simplified optical flow)
        motion_vectors = self._compute_motion_vectors(gray_video)
        
        # Edge-based motion detection
        edge_motion = self._compute_edge_motion(gray_video)
        
        return {
            "motion_intensity": motion_intensity,
            "motion_vectors": motion_vectors,
            "edge_motion": edge_motion,
            "frame_variance": torch.var(gray_video, dim=[1, 2, 3])
        }
    
    def compute_audio_visual_sync(
        self, 
        video_tensor: torch.Tensor, 
        audio_tensor: torch.Tensor,
        fps: float = 24.0
    ) -> Dict[str, float]:
        """Compute audio-visual synchronization metrics."""
        
        audio_features = self.extract_audio_features(audio_tensor)
        visual_features = self.extract_visual_motion_features(video_tensor)
        
        # Align temporal dimensions
        video_frames = video_tensor.shape[0]
        audio_frames = audio_features["onset_strength"].shape[0]
        
        # Resample to match frame rates
        target_frames = min(video_frames, int(audio_frames * fps / (self.sample_rate / 512)))
        
        visual_motion = F.interpolate(
            visual_features["motion_intensity"].unsqueeze(0).unsqueeze(0),
            size=target_frames,
            mode='linear',
            align_corners=False
        ).squeeze()
        
        audio_onset = F.interpolate(
            audio_features["onset_strength"].unsqueeze(0).unsqueeze(0),
            size=target_frames,
            mode='linear',
            align_corners=False
        ).squeeze()
        
        # Compute cross-correlation for sync detection
        sync_correlation = self._compute_cross_correlation(visual_motion, audio_onset)
        
        # Compute rhythmic similarity
        rhythmic_similarity = self._compute_rhythmic_similarity(visual_motion, audio_onset)
        
        # Compute onset alignment
        onset_alignment = self._compute_onset_alignment(visual_motion, audio_onset)
        
        return {
            "sync_correlation": float(sync_correlation.max()),
            "rhythmic_similarity": float(rhythmic_similarity),
            "onset_alignment": float(onset_alignment),
            "overall_sync_score": float((sync_correlation.max() + rhythmic_similarity + onset_alignment) / 3)
        }
    
    def _compute_spectral_centroid(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid of audio signal."""
        frequencies = torch.linspace(0, self.sample_rate/2, magnitude.shape[0]).to(self.device)
        centroid = torch.sum(magnitude * frequencies.unsqueeze(1), dim=0) / torch.sum(magnitude, dim=0)
        return centroid
    
    def _compute_spectral_rolloff(self, magnitude: torch.Tensor, rolloff_percent: float = 0.95) -> torch.Tensor:
        """Compute spectral rolloff frequency."""
        cumsum = torch.cumsum(magnitude, dim=0)
        total = cumsum[-1:]
        rolloff_indices = torch.argmax((cumsum / total) > rolloff_percent, dim=0)
        
        frequencies = torch.linspace(0, self.sample_rate/2, magnitude.shape[0]).to(self.device)
        return frequencies[rolloff_indices]
    
    def _compute_spectral_flux(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute spectral flux (measure of spectral change)."""
        diff = torch.diff(magnitude, dim=1)
        flux = torch.sum(torch.relu(diff), dim=0)  # Only positive changes
        return torch.cat([torch.zeros(1).to(self.device), flux])
    
    def _compute_onset_strength(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute onset strength function."""
        # High-frequency emphasis for onset detection
        freq_weights = torch.linspace(0.1, 1.0, magnitude.shape[0]).to(self.device)
        weighted_magnitude = magnitude * freq_weights.unsqueeze(1)
        
        # Spectral flux with adaptive threshold
        flux = self._compute_spectral_flux(weighted_magnitude)
        
        # Apply median filtering to reduce noise
        kernel_size = min(5, flux.shape[0])
        if kernel_size >= 3:
            onset_strength = torch.tensor([
                torch.median(flux[max(0, i-kernel_size//2):min(len(flux), i+kernel_size//2+1)])
                for i in range(len(flux))
            ]).to(self.device)
        else:
            onset_strength = flux
            
        return onset_strength
    
    def _compute_motion_vectors(self, gray_video: torch.Tensor) -> torch.Tensor:
        """Compute simplified motion vectors between frames."""
        # Use gradient-based motion estimation
        motion_vectors = []
        
        for i in range(1, gray_video.shape[0]):
            prev_frame = gray_video[i-1].squeeze()
            curr_frame = gray_video[i].squeeze()
            
            # Compute gradients
            grad_x = torch.diff(curr_frame, dim=1)
            grad_y = torch.diff(curr_frame, dim=0)
            grad_t = curr_frame[:-1, :-1] - prev_frame[:-1, :-1]
            
            # Optical flow approximation (Lucas-Kanade inspired)
            eps = 1e-8
            motion_x = -grad_t * grad_x[:-1] / (grad_x[:-1]**2 + grad_y[:, :-1]**2 + eps)
            motion_y = -grad_t * grad_y[:, :-1] / (grad_x[:-1]**2 + grad_y[:, :-1]**2 + eps)
            
            motion_magnitude = torch.sqrt(motion_x**2 + motion_y**2)
            motion_vectors.append(torch.mean(motion_magnitude))
        
        return torch.stack(motion_vectors)
    
    def _compute_edge_motion(self, gray_video: torch.Tensor) -> torch.Tensor:
        """Compute motion based on edge changes."""
        edge_motion = []
        
        for i in range(1, gray_video.shape[0]):
            prev_frame = gray_video[i-1].squeeze()
            curr_frame = gray_video[i].squeeze()
            
            # Sobel edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device)
            
            edges_prev = torch.sqrt(
                F.conv2d(prev_frame.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)**2 +
                F.conv2d(prev_frame.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)**2
            ).squeeze()
            
            edges_curr = torch.sqrt(
                F.conv2d(curr_frame.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)**2 +
                F.conv2d(curr_frame.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)**2
            ).squeeze()
            
            edge_diff = torch.mean(torch.abs(edges_curr - edges_prev))
            edge_motion.append(edge_diff)
        
        return torch.stack(edge_motion)
    
    def _compute_cross_correlation(self, signal1: torch.Tensor, signal2: torch.Tensor) -> torch.Tensor:
        """Compute normalized cross-correlation between two signals."""
        # Normalize signals
        signal1_norm = (signal1 - torch.mean(signal1)) / (torch.std(signal1) + 1e-8)
        signal2_norm = (signal2 - torch.mean(signal2)) / (torch.std(signal2) + 1e-8)
        
        # Compute cross-correlation using convolution
        cross_corr = F.conv1d(
            signal1_norm.unsqueeze(0).unsqueeze(0),
            signal2_norm.flip(0).unsqueeze(0).unsqueeze(0),
            padding=len(signal2_norm)-1
        ).squeeze()
        
        # Normalize by signal lengths
        cross_corr = cross_corr / len(signal1_norm)
        
        return cross_corr
    
    def _compute_rhythmic_similarity(self, visual_signal: torch.Tensor, audio_signal: torch.Tensor) -> torch.Tensor:
        """Compute rhythmic similarity between visual and audio signals."""
        # Extract rhythm patterns using autocorrelation
        visual_rhythm = self._extract_rhythm_pattern(visual_signal)
        audio_rhythm = self._extract_rhythm_pattern(audio_signal)
        
        # Compute similarity between rhythm patterns
        min_len = min(len(visual_rhythm), len(audio_rhythm))
        visual_rhythm = visual_rhythm[:min_len]
        audio_rhythm = audio_rhythm[:min_len]
        
        # Normalized correlation
        correlation = torch.corrcoef(torch.stack([visual_rhythm, audio_rhythm]))[0, 1]
        
        return torch.abs(correlation) if not torch.isnan(correlation) else torch.tensor(0.0)
    
    def _extract_rhythm_pattern(self, signal: torch.Tensor, max_lag: int = 50) -> torch.Tensor:
        """Extract rhythm pattern using autocorrelation."""
        signal_norm = (signal - torch.mean(signal)) / (torch.std(signal) + 1e-8)
        
        autocorr = []
        for lag in range(1, min(max_lag, len(signal)//2)):
            if len(signal_norm) > lag:
                corr = torch.corrcoef(torch.stack([signal_norm[:-lag], signal_norm[lag:]]))[0, 1]
                autocorr.append(corr if not torch.isnan(corr) else torch.tensor(0.0))
        
        return torch.stack(autocorr) if autocorr else torch.zeros(1)
    
    def _compute_onset_alignment(self, visual_motion: torch.Tensor, audio_onset: torch.Tensor) -> torch.Tensor:
        """Compute alignment between visual motion onsets and audio onsets."""
        # Detect peaks (onsets) in both signals
        visual_onsets = self._detect_onsets(visual_motion)
        audio_onsets = self._detect_onsets(audio_onset)
        
        if len(visual_onsets) == 0 or len(audio_onsets) == 0:
            return torch.tensor(0.0)
        
        # Find closest matches between onset times
        alignment_scores = []
        for v_onset in visual_onsets:
            distances = torch.abs(audio_onsets - v_onset)
            min_distance = torch.min(distances)
            alignment_scores.append(torch.exp(-min_distance / 5.0))  # Gaussian-like decay
        
        return torch.mean(torch.stack(alignment_scores))
    
    def _detect_onsets(self, signal: torch.Tensor, threshold_factor: float = 1.5) -> torch.Tensor:
        """Detect onset peaks in a signal."""
        # Compute moving average for threshold
        window_size = max(3, len(signal) // 20)
        threshold = torch.tensor([
            torch.mean(signal[max(0, i-window_size):min(len(signal), i+window_size+1)]) * threshold_factor
            for i in range(len(signal))
        ])
        
        # Find peaks above threshold
        onsets = []
        for i in range(1, len(signal)-1):
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and 
                signal[i] > threshold[i]):
                onsets.append(i)
        
        return torch.tensor(onsets, dtype=torch.float32) if onsets else torch.tensor([])


class TextVideoAlignmentEvaluator:
    """Evaluates alignment between text prompts and generated videos."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        # In practice, would load pre-trained CLIP or similar model
        self.text_encoder = None  # Placeholder for text encoder
        self.video_encoder = None  # Placeholder for video encoder
    
    def compute_text_video_alignment(
        self, 
        text_prompt: str, 
        video_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Compute alignment between text and video content."""
        
        # Mock implementation - in practice would use CLIP-like models
        alignment_scores = {
            "semantic_alignment": self._compute_semantic_alignment(text_prompt, video_tensor),
            "visual_concept_match": self._compute_visual_concept_match(text_prompt, video_tensor),
            "temporal_coherence": self._compute_temporal_coherence(text_prompt, video_tensor),
            "style_consistency": self._compute_style_consistency(text_prompt, video_tensor)
        }
        
        overall_alignment = np.mean(list(alignment_scores.values()))
        alignment_scores["overall_alignment"] = overall_alignment
        
        return alignment_scores
    
    def _compute_semantic_alignment(self, text: str, video: torch.Tensor) -> float:
        """Compute semantic alignment between text and video."""
        # Mock implementation - would use CLIP embeddings
        # Extract key concepts from text
        concepts = self._extract_text_concepts(text)
        
        # Analyze video for matching concepts
        video_concepts = self._extract_video_concepts(video)
        
        # Compute concept overlap
        if not concepts or not video_concepts:
            return 0.5  # Neutral score
        
        overlap = len(set(concepts) & set(video_concepts))
        union = len(set(concepts) | set(video_concepts))
        
        return overlap / union if union > 0 else 0.0
    
    def _compute_visual_concept_match(self, text: str, video: torch.Tensor) -> float:
        """Compute visual concept matching score."""
        # Mock implementation based on color, texture, shape analysis
        text_features = self._extract_text_visual_features(text)
        video_features = self._extract_video_visual_features(video)
        
        # Compute feature similarity
        similarity_scores = []
        for feature_type in text_features:
            if feature_type in video_features:
                text_feat = text_features[feature_type]
                video_feat = video_features[feature_type]
                similarity = self._compute_feature_similarity(text_feat, video_feat)
                similarity_scores.append(similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.5
    
    def _compute_temporal_coherence(self, text: str, video: torch.Tensor) -> float:
        """Compute temporal coherence with text description."""
        # Analyze text for temporal elements
        temporal_words = ["slowly", "quickly", "gradually", "suddenly", "smooth", "rapid"]
        
        text_lower = text.lower()
        temporal_indicators = [word for word in temporal_words if word in text_lower]
        
        if not temporal_indicators:
            return 0.8  # High score if no specific temporal requirements
        
        # Analyze video motion characteristics
        motion_analysis = self._analyze_video_motion(video)
        
        # Match temporal indicators with motion
        coherence_scores = []
        for indicator in temporal_indicators:
            if indicator in ["slowly", "gradually", "smooth"]:
                score = 1.0 - motion_analysis["motion_variance"]  # Low variance = smooth
            elif indicator in ["quickly", "rapidly", "sudden"]:
                score = motion_analysis["motion_intensity"]  # High intensity = quick
            else:
                score = 0.5
            
            coherence_scores.append(score)
        
        return np.mean(coherence_scores)
    
    def _compute_style_consistency(self, text: str, video: torch.Tensor) -> float:
        """Compute style consistency between text and video."""
        # Extract style keywords from text
        style_words = [
            "realistic", "cartoon", "anime", "painting", "sketch", "photography",
            "cinematic", "artistic", "abstract", "vintage", "modern", "futuristic"
        ]
        
        text_lower = text.lower()
        text_styles = [word for word in style_words if word in text_lower]
        
        if not text_styles:
            return 0.7  # Neutral score if no style specified
        
        # Analyze video for style characteristics
        video_style_analysis = self._analyze_video_style(video)
        
        # Match styles (simplified)
        style_matches = 0
        for style in text_styles:
            if style in ["realistic", "photography"]:
                if video_style_analysis["realism_score"] > 0.7:
                    style_matches += 1
            elif style in ["cartoon", "anime"]:
                if video_style_analysis["stylization_score"] > 0.7:
                    style_matches += 1
            elif style == "cinematic":
                if video_style_analysis["cinematic_score"] > 0.6:
                    style_matches += 1
        
        return style_matches / len(text_styles) if text_styles else 0.7
    
    def _extract_text_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simplified concept extraction
        common_objects = [
            "cat", "dog", "car", "tree", "house", "person", "water", "fire",
            "mountain", "sky", "cloud", "flower", "bird", "ocean", "forest"
        ]
        
        text_lower = text.lower()
        return [obj for obj in common_objects if obj in text_lower]
    
    def _extract_video_concepts(self, video: torch.Tensor) -> List[str]:
        """Extract concepts from video (mock implementation)."""
        # In practice, would use object detection/segmentation models
        # Mock based on color distribution and motion patterns
        
        concepts = []
        
        # Analyze color distribution
        mean_color = torch.mean(video, dim=[0, 2, 3])  # Average across frames, height, width
        
        # Simple color-based concept detection
        if mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:  # Green dominant
            concepts.extend(["tree", "forest", "grass"])
        if mean_color[2] > 0.7:  # High blue
            concepts.extend(["sky", "water", "ocean"])
        if torch.std(video) < 0.1:  # Low variance might indicate static objects
            concepts.extend(["house", "mountain"])
        
        return concepts
    
    def _extract_text_visual_features(self, text: str) -> Dict[str, List[float]]:
        """Extract visual features implied by text."""
        # Mock implementation - would use more sophisticated NLP
        features = {
            "color": [],
            "texture": [],
            "shape": []
        }
        
        color_words = {
            "red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0],
            "yellow": [1.0, 1.0, 0.0], "purple": [1.0, 0.0, 1.0], "orange": [1.0, 0.5, 0.0]
        }
        
        text_lower = text.lower()
        for color, rgb in color_words.items():
            if color in text_lower:
                features["color"].extend(rgb)
        
        if not features["color"]:
            features["color"] = [0.5, 0.5, 0.5]  # Default gray
        
        return features
    
    def _extract_video_visual_features(self, video: torch.Tensor) -> Dict[str, List[float]]:
        """Extract visual features from video."""
        features = {
            "color": torch.mean(video, dim=[0, 2, 3]).tolist(),  # Average color
            "texture": [float(torch.std(video))],  # Texture variance approximation
            "shape": [float(torch.mean(torch.abs(torch.diff(video, dim=2))))]  # Edge strength
        }
        
        return features
    
    def _compute_feature_similarity(self, feat1: List[float], feat2: List[float]) -> float:
        """Compute similarity between feature vectors."""
        if not feat1 or not feat2:
            return 0.0
        
        # Normalize and compute cosine similarity
        f1 = np.array(feat1)
        f2 = np.array(feat2)
        
        if len(f1) != len(f2):
            # Pad shorter vector
            max_len = max(len(f1), len(f2))
            f1 = np.pad(f1, (0, max_len - len(f1)))
            f2 = np.pad(f2, (0, max_len - len(f2)))
        
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(f1, f2) / (norm1 * norm2)
        return float(similarity)
    
    def _analyze_video_motion(self, video: torch.Tensor) -> Dict[str, float]:
        """Analyze motion characteristics of video."""
        # Compute frame differences
        if video.shape[0] > 1:
            frame_diffs = torch.diff(video, dim=0)
            motion_intensity = torch.mean(torch.abs(frame_diffs)).item()
            motion_variance = torch.var(torch.mean(torch.abs(frame_diffs), dim=[1, 2, 3])).item()
        else:
            motion_intensity = 0.0
            motion_variance = 0.0
        
        return {
            "motion_intensity": motion_intensity,
            "motion_variance": motion_variance
        }
    
    def _analyze_video_style(self, video: torch.Tensor) -> Dict[str, float]:
        """Analyze style characteristics of video."""
        # Compute style-related metrics
        color_saturation = torch.std(video, dim=0).mean().item()
        edge_strength = self._compute_edge_strength(video)
        color_diversity = self._compute_color_diversity(video)
        
        # Style scores (simplified heuristics)
        realism_score = min(1.0, color_diversity * 0.5 + (1 - color_saturation) * 0.5)
        stylization_score = min(1.0, color_saturation * 0.7 + (1 - edge_strength) * 0.3)
        cinematic_score = min(1.0, edge_strength * 0.6 + color_diversity * 0.4)
        
        return {
            "realism_score": realism_score,
            "stylization_score": stylization_score,
            "cinematic_score": cinematic_score
        }
    
    def _compute_edge_strength(self, video: torch.Tensor) -> float:
        """Compute average edge strength in video."""
        if len(video.shape) == 4:  # (T, C, H, W)
            # Convert to grayscale
            gray = torch.mean(video, dim=1)
        else:
            gray = video
        
        # Simple edge detection using gradients
        grad_x = torch.abs(torch.diff(gray, dim=2))
        grad_y = torch.abs(torch.diff(gray, dim=1))
        
        edge_strength = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
        return float(edge_strength)
    
    def _compute_color_diversity(self, video: torch.Tensor) -> float:
        """Compute color diversity in video."""
        if video.shape[1] == 3:  # RGB
            # Compute color distribution
            colors = video.reshape(-1, 3)
            unique_colors = torch.unique(colors, dim=0)
            diversity = len(unique_colors) / len(colors)
            return float(diversity)
        else:
            return 0.5  # Neutral score for non-RGB videos


class MultiModalEvaluator:
    """Comprehensive multi-modal evaluation system."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.audio_visual_analyzer = AudioVisualAnalyzer(device)
        self.text_video_evaluator = TextVideoAlignmentEvaluator(device)
        
    def evaluate_multimodal(
        self,
        video_tensor: torch.Tensor,
        text_prompt: str,
        audio_tensor: Optional[torch.Tensor] = None,
        fps: float = 24.0
    ) -> MultiModalResult:
        """Perform comprehensive multi-modal evaluation."""
        
        detailed_metrics = {}
        
        # Visual quality assessment
        visual_quality = self._assess_visual_quality(video_tensor)
        detailed_metrics["visual_metrics"] = visual_quality
        
        # Audio quality and sync (if audio provided)
        audio_quality = None
        audio_visual_sync = None
        if audio_tensor is not None:
            audio_quality = self._assess_audio_quality(audio_tensor)
            sync_metrics = self.audio_visual_analyzer.compute_audio_visual_sync(
                video_tensor, audio_tensor, fps
            )
            audio_visual_sync = sync_metrics["overall_sync_score"]
            detailed_metrics["audio_metrics"] = {"quality": audio_quality, "sync": sync_metrics}
        
        # Text-video alignment
        text_alignment_metrics = self.text_video_evaluator.compute_text_video_alignment(
            text_prompt, video_tensor
        )
        text_video_alignment = text_alignment_metrics["overall_alignment"]
        detailed_metrics["text_alignment_metrics"] = text_alignment_metrics
        
        # Perceptual quality
        perceptual_quality = self._assess_perceptual_quality(video_tensor, text_prompt)
        detailed_metrics["perceptual_metrics"] = perceptual_quality
        
        # Cross-modal consistency
        cross_modal_consistency = self._assess_cross_modal_consistency(
            video_tensor, text_prompt, audio_tensor
        )
        detailed_metrics["cross_modal_metrics"] = cross_modal_consistency
        
        # Compute overall score
        overall_score = self._compute_overall_score(
            visual_quality["overall_score"],
            audio_quality,
            audio_visual_sync,
            text_video_alignment,
            perceptual_quality["overall_score"],
            cross_modal_consistency["overall_score"]
        )
        
        return MultiModalResult(
            visual_quality=visual_quality["overall_score"],
            audio_quality=audio_quality,
            audio_visual_sync=audio_visual_sync,
            text_video_alignment=text_video_alignment,
            perceptual_quality=perceptual_quality["overall_score"],
            cross_modal_consistency=cross_modal_consistency["overall_score"],
            overall_score=overall_score,
            detailed_metrics=detailed_metrics
        )
    
    def _assess_visual_quality(self, video_tensor: torch.Tensor) -> Dict[str, float]:
        """Assess visual quality of video."""
        metrics = {}
        
        # Spatial quality
        metrics["sharpness"] = self._compute_sharpness(video_tensor)
        metrics["contrast"] = self._compute_contrast(video_tensor)
        metrics["color_richness"] = self._compute_color_richness(video_tensor)
        metrics["noise_level"] = self._compute_noise_level(video_tensor)
        
        # Temporal quality
        metrics["temporal_consistency"] = self._compute_temporal_consistency(video_tensor)
        metrics["motion_smoothness"] = self._compute_motion_smoothness(video_tensor)
        metrics["flicker_detection"] = self._detect_flicker(video_tensor)
        
        # Overall visual quality score
        spatial_score = (metrics["sharpness"] + metrics["contrast"] + 
                        metrics["color_richness"] + (1 - metrics["noise_level"])) / 4
        temporal_score = (metrics["temporal_consistency"] + metrics["motion_smoothness"] + 
                         (1 - metrics["flicker_detection"])) / 3
        
        metrics["spatial_score"] = spatial_score
        metrics["temporal_score"] = temporal_score
        metrics["overall_score"] = (spatial_score + temporal_score) / 2
        
        return metrics
    
    def _assess_audio_quality(self, audio_tensor: torch.Tensor) -> float:
        """Assess audio quality."""
        # Mock implementation - would use more sophisticated audio analysis
        
        # Signal-to-noise ratio approximation
        signal_power = torch.mean(audio_tensor ** 2)
        noise_estimate = torch.var(torch.diff(audio_tensor))  # High-frequency content as noise proxy
        snr = signal_power / (noise_estimate + 1e-8)
        snr_score = torch.clamp(torch.log10(snr) / 2, 0, 1)  # Normalize
        
        # Dynamic range
        dynamic_range = torch.max(audio_tensor) - torch.min(audio_tensor)
        dr_score = torch.clamp(dynamic_range / 2, 0, 1)
        
        # Spectral balance (avoid too much energy in any one frequency band)
        stft = torch.stft(
            audio_tensor.squeeze(),
            n_fft=2048,
            hop_length=512,
            window=torch.hann_window(2048),
            return_complex=True
        )
        magnitude = torch.abs(stft)
        spectral_balance = 1 - torch.std(torch.mean(magnitude, dim=1)) / torch.mean(magnitude)
        
        overall_quality = (float(snr_score) + float(dr_score) + float(spectral_balance)) / 3
        return overall_quality
    
    def _assess_perceptual_quality(self, video_tensor: torch.Tensor, text_prompt: str) -> Dict[str, float]:
        """Assess perceptual quality factors."""
        
        metrics = {}
        
        # Aesthetic appeal (simplified)
        metrics["composition"] = self._assess_composition(video_tensor)
        metrics["color_harmony"] = self._assess_color_harmony(video_tensor)
        metrics["visual_interest"] = self._assess_visual_interest(video_tensor)
        
        # Content relevance
        metrics["subject_clarity"] = self._assess_subject_clarity(video_tensor, text_prompt)
        metrics["scene_coherence"] = self._assess_scene_coherence(video_tensor)
        
        # Technical perception
        metrics["perceived_resolution"] = self._assess_perceived_resolution(video_tensor)
        metrics["motion_naturalness"] = self._assess_motion_naturalness(video_tensor)
        
        # Compute overall perceptual quality
        aesthetic_score = (metrics["composition"] + metrics["color_harmony"] + 
                          metrics["visual_interest"]) / 3
        content_score = (metrics["subject_clarity"] + metrics["scene_coherence"]) / 2
        technical_score = (metrics["perceived_resolution"] + metrics["motion_naturalness"]) / 2
        
        metrics["aesthetic_score"] = aesthetic_score
        metrics["content_score"] = content_score
        metrics["technical_score"] = technical_score
        metrics["overall_score"] = (aesthetic_score + content_score + technical_score) / 3
        
        return metrics
    
    def _assess_cross_modal_consistency(
        self, 
        video_tensor: torch.Tensor, 
        text_prompt: str, 
        audio_tensor: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """Assess consistency across different modalities."""
        
        metrics = {}
        
        # Text-visual consistency (already computed but include here)
        text_visual = self.text_video_evaluator.compute_text_video_alignment(text_prompt, video_tensor)
        metrics["text_visual_consistency"] = text_visual["overall_alignment"]
        
        # Audio-visual consistency (if audio available)
        if audio_tensor is not None:
            av_sync = self.audio_visual_analyzer.compute_audio_visual_sync(video_tensor, audio_tensor)
            metrics["audio_visual_consistency"] = av_sync["overall_sync_score"]
            
            # Text-audio semantic consistency (mock)
            metrics["text_audio_consistency"] = self._assess_text_audio_consistency(text_prompt, audio_tensor)
        else:
            metrics["audio_visual_consistency"] = None
            metrics["text_audio_consistency"] = None
        
        # Overall consistency score
        consistency_scores = [score for score in metrics.values() if score is not None]
        metrics["overall_score"] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return metrics
    
    def _compute_overall_score(
        self,
        visual_quality: float,
        audio_quality: Optional[float],
        audio_visual_sync: Optional[float],
        text_video_alignment: float,
        perceptual_quality: float,
        cross_modal_consistency: float
    ) -> float:
        """Compute weighted overall multi-modal score."""
        
        # Weights for different aspects
        weights = {
            "visual": 0.25,
            "audio": 0.15,
            "sync": 0.15,
            "text_alignment": 0.20,
            "perceptual": 0.15,
            "cross_modal": 0.10
        }
        
        scores = []
        total_weight = 0
        
        scores.append(visual_quality * weights["visual"])
        total_weight += weights["visual"]
        
        if audio_quality is not None:
            scores.append(audio_quality * weights["audio"])
            total_weight += weights["audio"]
        
        if audio_visual_sync is not None:
            scores.append(audio_visual_sync * weights["sync"])
            total_weight += weights["sync"]
        
        scores.append(text_video_alignment * weights["text_alignment"])
        total_weight += weights["text_alignment"]
        
        scores.append(perceptual_quality * weights["perceptual"])
        total_weight += weights["perceptual"]
        
        scores.append(cross_modal_consistency * weights["cross_modal"])
        total_weight += weights["cross_modal"]
        
        return sum(scores) / total_weight if total_weight > 0 else 0.0
    
    # Visual quality helper methods
    def _compute_sharpness(self, video: torch.Tensor) -> float:
        """Compute sharpness using Laplacian variance."""
        if len(video.shape) == 4:
            gray = torch.mean(video, dim=1)
        else:
            gray = video
        
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).to(self.device)
        laplacian = F.conv2d(
            gray.unsqueeze(1), 
            laplacian_kernel.unsqueeze(0).unsqueeze(0), 
            padding=1
        )
        
        sharpness = torch.var(laplacian).item()
        return min(1.0, sharpness / 100)  # Normalize
    
    def _compute_contrast(self, video: torch.Tensor) -> float:
        """Compute contrast using standard deviation."""
        contrast = torch.std(video).item()
        return min(1.0, contrast / 0.3)  # Normalize
    
    def _compute_color_richness(self, video: torch.Tensor) -> float:
        """Compute color richness."""
        if video.shape[1] == 3:  # RGB
            # Compute color distribution in HSV space (approximated)
            max_vals = torch.max(video, dim=1)[0]
            min_vals = torch.min(video, dim=1)[0]
            saturation = (max_vals - min_vals) / (max_vals + 1e-8)
            color_richness = torch.mean(saturation).item()
            return color_richness
        return 0.5
    
    def _compute_noise_level(self, video: torch.Tensor) -> float:
        """Estimate noise level in video."""
        # High-frequency content as noise proxy
        if video.shape[0] > 1:
            temporal_noise = torch.std(torch.diff(video, dim=0)).item()
        else:
            temporal_noise = 0.0
        
        spatial_noise = torch.std(torch.diff(torch.diff(video, dim=2), dim=3)).item()
        
        noise_level = (temporal_noise + spatial_noise) / 2
        return min(1.0, noise_level / 0.1)  # Normalize
    
    def _compute_temporal_consistency(self, video: torch.Tensor) -> float:
        """Compute temporal consistency across frames."""
        if video.shape[0] <= 1:
            return 1.0
        
        frame_similarities = []
        for i in range(1, video.shape[0]):
            similarity = F.cosine_similarity(
                video[i-1].flatten(),
                video[i].flatten(),
                dim=0
            )
            frame_similarities.append(similarity)
        
        consistency = torch.mean(torch.stack(frame_similarities)).item()
        return consistency
    
    def _compute_motion_smoothness(self, video: torch.Tensor) -> float:
        """Compute motion smoothness."""
        if video.shape[0] <= 2:
            return 1.0
        
        # Compute second-order differences (acceleration)
        motion = torch.diff(video, dim=0)
        acceleration = torch.diff(motion, dim=0)
        
        smoothness = 1.0 - torch.std(acceleration).item()
        return max(0.0, smoothness)
    
    def _detect_flicker(self, video: torch.Tensor) -> float:
        """Detect flicker in video."""
        if video.shape[0] <= 2:
            return 0.0
        
        # Compute frame-to-frame brightness changes
        brightness = torch.mean(video, dim=[1, 2, 3])
        brightness_changes = torch.diff(brightness)
        
        # High variance in brightness changes indicates flicker
        flicker_score = torch.std(brightness_changes).item()
        return min(1.0, flicker_score / 0.1)  # Normalize
    
    # Perceptual quality helper methods (simplified implementations)
    def _assess_composition(self, video: torch.Tensor) -> float:
        """Assess composition quality."""
        # Rule of thirds approximation
        h, w = video.shape[-2:]
        thirds_h = [h//3, 2*h//3]
        thirds_w = [w//3, 2*w//3]
        
        # Check if there's visual interest at intersection points
        interest_score = 0.0
        for th in thirds_h:
            for tw in thirds_w:
                region = video[:, :, max(0, th-10):min(h, th+10), max(0, tw-10):min(w, tw+10)]
                interest_score += torch.std(region).item()
        
        return min(1.0, interest_score / 4.0)
    
    def _assess_color_harmony(self, video: torch.Tensor) -> float:
        """Assess color harmony."""
        if video.shape[1] != 3:
            return 0.5
        
        # Compute color distribution
        colors = video.reshape(-1, 3)
        color_std = torch.std(colors, dim=0)
        
        # Balanced color distribution indicates harmony
        harmony = 1.0 - torch.std(color_std).item()
        return max(0.0, harmony)
    
    def _assess_visual_interest(self, video: torch.Tensor) -> float:
        """Assess visual interest/complexity."""
        # Edge density as proxy for visual interest
        if len(video.shape) == 4:
            gray = torch.mean(video, dim=1)
        else:
            gray = video
        
        edges = torch.abs(torch.diff(gray, dim=2)) + torch.abs(torch.diff(gray, dim=1))
        interest = torch.mean(edges).item()
        
        return min(1.0, interest / 0.5)
    
    def _assess_subject_clarity(self, video: torch.Tensor, text_prompt: str) -> float:
        """Assess how clearly the main subject is visible."""
        # Mock implementation - would use object detection
        concepts = self.text_video_evaluator._extract_text_concepts(text_prompt)
        
        if not concepts:
            return 0.7  # Neutral if no clear subject
        
        # Assume higher contrast indicates clearer subjects
        contrast = torch.std(video).item()
        clarity = min(1.0, contrast / 0.3)
        
        return clarity
    
    def _assess_scene_coherence(self, video: torch.Tensor) -> float:
        """Assess coherence of the scene."""
        # Temporal consistency as proxy for scene coherence
        return self._compute_temporal_consistency(video)
    
    def _assess_perceived_resolution(self, video: torch.Tensor) -> float:
        """Assess perceived resolution quality."""
        # Based on high-frequency content
        return self._compute_sharpness(video)
    
    def _assess_motion_naturalness(self, video: torch.Tensor) -> float:
        """Assess naturalness of motion."""
        return self._compute_motion_smoothness(video)
    
    def _assess_text_audio_consistency(self, text_prompt: str, audio_tensor: torch.Tensor) -> float:
        """Assess semantic consistency between text and audio."""
        # Mock implementation - would analyze audio content vs text description
        
        # Check for audio-related terms in text
        audio_terms = ["music", "sound", "loud", "quiet", "noise", "melody", "rhythm"]
        text_lower = text_prompt.lower()
        
        has_audio_description = any(term in text_lower for term in audio_terms)
        
        if not has_audio_description:
            return 0.8  # High score if no specific audio requirements
        
        # Analyze audio characteristics
        audio_energy = torch.mean(audio_tensor ** 2).item()
        audio_variance = torch.var(audio_tensor).item()
        
        # Match audio characteristics with text description
        consistency_score = 0.5  # Baseline
        
        if "loud" in text_lower and audio_energy > 0.5:
            consistency_score += 0.3
        elif "quiet" in text_lower and audio_energy < 0.1:
            consistency_score += 0.3
        
        if "music" in text_lower and audio_variance > 0.1:  # Music typically has more variation
            consistency_score += 0.2
        
        return min(1.0, consistency_score)


# Convenience functions
def evaluate_video_multimodal(
    video_tensor: torch.Tensor,
    text_prompt: str,
    audio_tensor: Optional[torch.Tensor] = None,
    device: str = "cuda"
) -> MultiModalResult:
    """Convenience function for multi-modal evaluation."""
    evaluator = MultiModalEvaluator(device)
    return evaluator.evaluate_multimodal(video_tensor, text_prompt, audio_tensor)


def benchmark_multimodal_quality(
    model_results: Dict[str, Tuple[torch.Tensor, str, Optional[torch.Tensor]]],
    device: str = "cuda"
) -> Dict[str, MultiModalResult]:
    """Benchmark multiple models using multi-modal evaluation."""
    evaluator = MultiModalEvaluator(device)
    results = {}
    
    for model_name, (video, prompt, audio) in model_results.items():
        logger.info(f"Evaluating multi-modal quality for {model_name}")
        results[model_name] = evaluator.evaluate_multimodal(video, prompt, audio)
    
    return results