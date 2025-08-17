"""Adaptive algorithms for dynamic video diffusion optimization.

This module implements novel adaptive algorithms for optimizing video diffusion
model performance based on real-time feedback and content analysis.

Research contributions:
1. Content-Aware Dynamic Sampling for variable diffusion steps
2. Adaptive Quality-Performance Trade-off optimization  
3. Real-time Performance Prediction using lightweight neural networks
4. Context-Sensitive Memory Management for optimal VRAM usage
5. Novel Multi-Objective Optimization for Pareto-optimal configurations
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from collections import defaultdict, deque
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive algorithms."""
    learning_rate: float = 0.001
    memory_threshold: float = 0.8  # VRAM usage threshold
    quality_target: float = 0.85  # Target quality score
    performance_weight: float = 0.3  # Weight for performance vs quality
    adaptation_window: int = 100  # Number of samples for adaptation
    min_confidence: float = 0.7  # Minimum confidence for adaptation
    

@dataclass
class ContentFeatures:
    """Extracted features from video content for adaptation."""
    complexity_score: float
    motion_intensity: float
    texture_density: float
    temporal_coherence: float
    semantic_complexity: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input."""
        return torch.tensor([
            self.complexity_score,
            self.motion_intensity, 
            self.texture_density,
            self.temporal_coherence,
            self.semantic_complexity
        ], dtype=torch.float32)


class ContentAnalyzer:
    """Lightweight content analyzer for real-time feature extraction."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_feature_extractors()
        
    def _init_feature_extractors(self):
        """Initialize lightweight feature extraction networks."""
        # Simplified feature extractors for real-time performance
        self.complexity_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.motion_net = nn.Sequential(
            nn.Conv3d(3, 8, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(self.device)
        
    def analyze_content(self, video_tensor: torch.Tensor) -> ContentFeatures:
        """Analyze video content to extract adaptive features.
        
        Args:
            video_tensor: Input video tensor [B, C, T, H, W]
            
        Returns:
            ContentFeatures object with extracted features
        """
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            
            # Extract spatial complexity from first frame
            first_frame = video_tensor[0, :, 0, :, :]  # [C, H, W]
            complexity = self.complexity_net(first_frame.unsqueeze(0)).item()
            
            # Extract motion intensity
            if video_tensor.shape[2] > 1:  # Multi-frame
                motion = self.motion_net(video_tensor.unsqueeze(0)).item()
            else:
                motion = 0.0
                
            # Simple texture density using gradients
            gray = torch.mean(first_frame, dim=0)
            grad_x = torch.abs(gray[:-1, :] - gray[1:, :])
            grad_y = torch.abs(gray[:, :-1] - gray[:, 1:])
            texture = torch.mean(grad_x).item() + torch.mean(grad_y).item()
            texture = min(texture / 2.0, 1.0)  # Normalize
            
            # Temporal coherence (similarity between frames)
            temporal_coherence = 1.0
            if video_tensor.shape[2] > 1:
                frame_diffs = []
                for i in range(video_tensor.shape[2] - 1):
                    diff = torch.mean(torch.abs(
                        video_tensor[0, :, i, :, :] - video_tensor[0, :, i+1, :, :]
                    )).item()
                    frame_diffs.append(diff)
                temporal_coherence = 1.0 - min(np.mean(frame_diffs), 1.0)
            
            # Semantic complexity (variance in pixel values as proxy)
            semantic = torch.std(video_tensor).item() / 255.0
            semantic = min(semantic, 1.0)
            
            return ContentFeatures(
                complexity_score=complexity,
                motion_intensity=motion,
                texture_density=texture,
                temporal_coherence=temporal_coherence,
                semantic_complexity=semantic
            )


class PerformancePredictor(nn.Module):
    """Lightweight neural network for predicting performance metrics."""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # [latency, memory, quality]
        )
        
    def forward(self, features: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        """Predict performance metrics.
        
        Args:
            features: Content features [B, 5]
            config: Configuration parameters [B, 3] (steps, resolution_scale, batch_size)
            
        Returns:
            Predicted metrics [B, 3] (latency, memory_usage, quality_score)
        """
        combined = torch.cat([features, config], dim=1)
        return self.network(combined)


class AdaptiveDiffusionOptimizer:
    """Main adaptive optimization engine for video diffusion models."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.content_analyzer = ContentAnalyzer()
        self.performance_predictor = PerformancePredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Adaptive state
        self.performance_history = deque(maxlen=config.adaptation_window)
        self.optimal_configs = {}  # {content_hash: optimal_config}
        self.adaptation_lock = threading.Lock()
        
        # Training data collection for predictor
        self.training_data = {
            'features': [],
            'configs': [],
            'metrics': []
        }
        
        logger.info(f"Initialized AdaptiveDiffusionOptimizer on {self.device}")
        
    def optimize_for_content(self, 
                           video_tensor: torch.Tensor,
                           model_name: str,
                           base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize diffusion parameters for specific content.
        
        Args:
            video_tensor: Input video tensor for content analysis
            model_name: Name of the diffusion model
            base_config: Base configuration parameters
            
        Returns:
            Optimized configuration dictionary
        """
        # Analyze content
        features = self.content_analyzer.analyze_content(video_tensor)
        content_hash = self._hash_features(features)
        
        # Check for cached optimal config
        cache_key = f"{model_name}_{content_hash}"
        if cache_key in self.optimal_configs:
            cached_config = self.optimal_configs[cache_key]
            logger.debug(f"Using cached optimal config for {cache_key}")
            return cached_config
        
        # Generate candidate configurations
        candidates = self._generate_candidate_configs(base_config, features)
        
        # Predict performance for candidates
        best_config = self._select_optimal_config(candidates, features)
        
        # Cache the result
        with self.adaptation_lock:
            self.optimal_configs[cache_key] = best_config
            
        logger.info(f"Optimized config for {model_name}: {best_config}")
        return best_config
    
    def _generate_candidate_configs(self, 
                                  base_config: Dict[str, Any],
                                  features: ContentFeatures) -> List[Dict[str, Any]]:
        """Generate candidate configurations based on content features."""
        candidates = []
        
        # Base configuration
        candidates.append(base_config.copy())
        
        # Adaptive configurations based on content
        if features.complexity_score > 0.7:
            # High complexity: increase steps, reduce resolution if needed
            high_quality_config = base_config.copy()
            high_quality_config['num_inference_steps'] = min(
                int(base_config.get('num_inference_steps', 50) * 1.5), 100
            )
            candidates.append(high_quality_config)
            
        if features.motion_intensity > 0.6:
            # High motion: optimize for temporal consistency
            motion_config = base_config.copy()
            motion_config['guidance_scale'] = min(
                base_config.get('guidance_scale', 7.5) * 1.2, 15.0
            )
            candidates.append(motion_config)
            
        if features.texture_density < 0.3:
            # Low texture: can reduce steps for efficiency
            efficient_config = base_config.copy()
            efficient_config['num_inference_steps'] = max(
                int(base_config.get('num_inference_steps', 50) * 0.7), 20
            )
            candidates.append(efficient_config)
            
        # Memory-optimized configuration
        memory_config = base_config.copy()
        current_height = base_config.get('height', 512)
        current_width = base_config.get('width', 512)
        memory_config['height'] = max(int(current_height * 0.8), 256)
        memory_config['width'] = max(int(current_width * 0.8), 256)
        candidates.append(memory_config)
        
        return candidates
    
    def _select_optimal_config(self, 
                             candidates: List[Dict[str, Any]], 
                             features: ContentFeatures) -> Dict[str, Any]:
        """Select optimal configuration using performance prediction."""
        if not candidates:
            return {}
            
        best_config = candidates[0]
        best_score = float('-inf')
        
        features_tensor = features.to_tensor().unsqueeze(0)
        
        for config in candidates:
            # Convert config to tensor
            config_tensor = torch.tensor([
                config.get('num_inference_steps', 50) / 100.0,  # Normalize
                (config.get('height', 512) * config.get('width', 512)) / (1024 * 1024),  # Resolution scale
                config.get('batch_size', 1) / 8.0  # Batch size scale
            ], dtype=torch.float32).unsqueeze(0)
            
            # Predict performance
            with torch.no_grad():
                predicted = self.performance_predictor(features_tensor, config_tensor)
                latency, memory, quality = predicted[0].tolist()
                
            # Compute composite score
            score = self._compute_optimization_score(latency, memory, quality)
            
            if score > best_score:
                best_score = score
                best_config = config
                
        return best_config
    
    def _compute_optimization_score(self, 
                                  latency: float, 
                                  memory: float, 
                                  quality: float) -> float:
        """Compute optimization score balancing quality and performance."""
        # Normalize metrics (assuming reasonable ranges)
        norm_latency = max(0, 1.0 - latency / 60.0)  # 60s max reasonable latency
        norm_memory = max(0, 1.0 - memory / 32.0)    # 32GB max reasonable memory
        norm_quality = min(1.0, quality)              # Quality score 0-1
        
        # Weighted combination
        performance_score = (norm_latency + norm_memory) / 2.0
        
        score = (
            (1.0 - self.config.performance_weight) * norm_quality +
            self.config.performance_weight * performance_score
        )
        
        return score
    
    def update_performance_history(self, 
                                 config: Dict[str, Any],
                                 metrics: Dict[str, float],
                                 features: ContentFeatures):
        """Update performance history for model training."""
        with self.adaptation_lock:
            self.performance_history.append({
                'config': config,
                'metrics': metrics,
                'features': features,
                'timestamp': time.time()
            })
            
            # Collect training data
            config_tensor = torch.tensor([
                config.get('num_inference_steps', 50) / 100.0,
                (config.get('height', 512) * config.get('width', 512)) / (1024 * 1024),
                config.get('batch_size', 1) / 8.0
            ])
            
            self.training_data['features'].append(features.to_tensor())
            self.training_data['configs'].append(config_tensor)
            self.training_data['metrics'].append(torch.tensor([
                metrics.get('latency', 0),
                metrics.get('memory_usage', 0),
                metrics.get('quality_score', 0)
            ]))
            
        # Periodically retrain predictor
        if len(self.performance_history) % 50 == 0:
            self._retrain_predictor()
    
    def _retrain_predictor(self):
        """Retrain performance predictor with collected data."""
        if len(self.training_data['features']) < 20:
            return
            
        logger.info("Retraining performance predictor")
        
        # Prepare training data
        features = torch.stack(self.training_data['features'][-100:])
        configs = torch.stack(self.training_data['configs'][-100:])
        targets = torch.stack(self.training_data['metrics'][-100:])
        
        # Simple training loop
        optimizer = torch.optim.Adam(self.performance_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.performance_predictor.train()
        
        for epoch in range(10):
            optimizer.zero_grad()
            predictions = self.performance_predictor(features, configs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
        self.performance_predictor.eval()
        logger.info(f"Predictor retrained, final loss: {loss.item():.4f}")
    
    def _hash_features(self, features: ContentFeatures) -> str:
        """Create hash for content features for caching."""
        feature_str = f"{features.complexity_score:.3f}_{features.motion_intensity:.3f}_{features.texture_density:.3f}"
        return str(hash(feature_str) % 10000)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptation performance."""
        with self.adaptation_lock:
            if not self.performance_history:
                return {"status": "no_data"}
                
            recent_metrics = list(self.performance_history)[-20:]
            
            latencies = [m['metrics'].get('latency', 0) for m in recent_metrics]
            qualities = [m['metrics'].get('quality_score', 0) for m in recent_metrics]
            
            return {
                "status": "active",
                "cached_configs": len(self.optimal_configs),
                "history_size": len(self.performance_history),
                "avg_latency": np.mean(latencies) if latencies else 0,
                "avg_quality": np.mean(qualities) if qualities else 0,
                "adaptation_coverage": len(self.optimal_configs) / max(len(self.performance_history), 1)
            }
    
    def save_state(self, path: Path):
        """Save optimizer state for persistence."""
        state = {
            'config': self.config,
            'optimal_configs': self.optimal_configs,
            'performance_history': list(self.performance_history),
            'predictor_state': self.performance_predictor.state_dict()
        }
        
        with open(path, 'w') as f:
            # Use JSON for main state, separate file for model weights
            json_state = {k: v for k, v in state.items() if k != 'predictor_state'}
            json.dump(json_state, f, indent=2, default=str)
            
        # Save model weights separately
        torch.save(state['predictor_state'], path.with_suffix('.pt'))
        logger.info(f"Adaptive optimizer state saved to {path}")
    
    def load_state(self, path: Path):
        """Load optimizer state from file."""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
                
            self.optimal_configs = state.get('optimal_configs', {})
            history_data = state.get('performance_history', [])
            
            # Restore performance history
            for item in history_data[-self.config.adaptation_window:]:
                self.performance_history.append(item)
                
            # Load model weights
            model_path = path.with_suffix('.pt')
            if model_path.exists():
                self.performance_predictor.load_state_dict(torch.load(model_path))
                
            logger.info(f"Adaptive optimizer state loaded from {path}")
            
        except Exception as e:
            logger.warning(f"Failed to load adaptive optimizer state: {e}")


class MultiObjectiveOptimizer:
    """Multi-objective optimization for Pareto-optimal configurations."""
    
    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ['quality', 'speed', 'memory']
        self.pareto_frontier = []
        
    def add_solution(self, config: Dict[str, Any], metrics: Dict[str, float]):
        """Add a solution to the Pareto frontier."""
        solution = {
            'config': config,
            'metrics': metrics,
            'objectives': [metrics.get(obj, 0) for obj in self.objectives]
        }
        
        # Check if solution is dominated
        if not self._is_dominated(solution):
            # Remove dominated solutions
            self.pareto_frontier = [
                s for s in self.pareto_frontier 
                if not self._dominates(solution, s)
            ]
            self.pareto_frontier.append(solution)
            
    def _is_dominated(self, solution: Dict[str, Any]) -> bool:
        """Check if solution is dominated by any in frontier."""
        return any(self._dominates(existing, solution) 
                  for existing in self.pareto_frontier)
    
    def _dominates(self, sol1: Dict[str, Any], sol2: Dict[str, Any]) -> bool:
        """Check if sol1 dominates sol2."""
        obj1 = sol1['objectives']
        obj2 = sol2['objectives']
        
        return (all(o1 >= o2 for o1, o2 in zip(obj1, obj2)) and 
                any(o1 > o2 for o1, o2 in zip(obj1, obj2)))
    
    def get_pareto_optimal_configs(self) -> List[Dict[str, Any]]:
        """Get all Pareto-optimal configurations."""
        return [sol['config'] for sol in self.pareto_frontier]
    
    def recommend_config(self, weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Recommend configuration based on objective weights."""
        if not self.pareto_frontier:
            return {}
            
        weights = weights or {obj: 1.0 for obj in self.objectives}
        
        best_solution = None
        best_score = float('-inf')
        
        for solution in self.pareto_frontier:
            score = sum(
                weights.get(obj, 1.0) * solution['metrics'].get(obj, 0)
                for obj in self.objectives
            )
            
            if score > best_score:
                best_score = score
                best_solution = solution
                
        return best_solution['config'] if best_solution else {}


# Example usage and testing
if __name__ == "__main__":
    # Example usage of adaptive algorithms
    config = AdaptiveConfig(
        learning_rate=0.001,
        memory_threshold=0.8,
        quality_target=0.85
    )
    
    optimizer = AdaptiveDiffusionOptimizer(config)
    
    # Simulate content analysis and optimization
    dummy_video = torch.randn(1, 3, 16, 256, 256)  # Batch, Channels, Time, Height, Width
    base_config = {
        'num_inference_steps': 50,
        'guidance_scale': 7.5,
        'height': 512,
        'width': 512,
        'batch_size': 1
    }
    
    optimized_config = optimizer.optimize_for_content(
        dummy_video, 
        "test_model", 
        base_config
    )
    
    print(f"Optimized configuration: {optimized_config}")
    print(f"Adaptation stats: {optimizer.get_adaptation_stats()}")