"""Core benchmark suite implementation."""

import time
import json
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import torch
import psutil
import numpy as np

from .models.registry import get_model, list_models
from .models.base import ModelAdapter
from .metrics import VideoQualityMetrics
from .profiler import EfficiencyProfiler
from .prompts import StandardPrompts

logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Container for benchmark evaluation results."""
    
    def __init__(self, model_name: str, prompts: List[str]):
        self.model_name = model_name
        self.prompts = prompts
        self.timestamp = datetime.now().isoformat()
        self.results = {}
        self.metrics = {}
        self.performance = {}
        self.errors = []
        
    def add_result(self, prompt_idx: int, video_tensor: torch.Tensor, 
                   generation_time: float, memory_usage: Dict[str, float]):
        """Add generation result for a prompt."""
        self.results[prompt_idx] = {
            "prompt": self.prompts[prompt_idx],
            "video_shape": video_tensor.shape,
            "generation_time": generation_time,
            "memory_usage": memory_usage,
            "status": "success"
        }
        
    def add_error(self, prompt_idx: int, error: Exception):
        """Add error for a failed prompt."""
        self.errors.append({
            "prompt_idx": prompt_idx,
            "prompt": self.prompts[prompt_idx],
            "error": str(error),
            "error_type": type(error).__name__
        })
        self.results[prompt_idx] = {
            "prompt": self.prompts[prompt_idx],
            "status": "failed",
            "error": str(error)
        }
        
    def set_metrics(self, fvd: float, is_score: float, clip_score: float, 
                    temporal_consistency: float):
        """Set computed quality metrics."""
        self.metrics = {
            "fvd": fvd,
            "inception_score": is_score,
            "clip_similarity": clip_score,
            "temporal_consistency": temporal_consistency,
            "overall_score": self._compute_overall_score(fvd, is_score, clip_score, temporal_consistency)
        }
        
    def set_performance(self, avg_latency: float, throughput: float, 
                       peak_vram_gb: float, avg_power_watts: float):
        """Set performance metrics."""
        self.performance = {
            "avg_latency_ms": avg_latency * 1000,
            "throughput_fps": throughput,
            "peak_vram_gb": peak_vram_gb,
            "avg_power_watts": avg_power_watts,
            "efficiency_score": self._compute_efficiency_score(avg_latency, peak_vram_gb)
        }
        
    def _compute_overall_score(self, fvd: float, is_score: float, 
                              clip_score: float, temporal_consistency: float) -> float:
        """Compute weighted overall quality score."""
        fvd_norm = max(0, (200 - fvd) / 200)  # Normalize FVD (lower is better)
        is_norm = min(1, is_score / 50)  # Normalize IS (higher is better)
        clip_norm = clip_score  # CLIP score already 0-1
        temporal_norm = temporal_consistency  # Temporal consistency 0-1
        
        return (fvd_norm * 0.4 + is_norm * 0.3 + clip_norm * 0.2 + temporal_norm * 0.1) * 100
        
    def _compute_efficiency_score(self, latency: float, vram_gb: float) -> float:
        """Compute efficiency score (higher is better)."""
        latency_score = max(0, (10 - latency) / 10)  # Penalize high latency
        memory_score = max(0, (32 - vram_gb) / 32)  # Penalize high memory usage
        return (latency_score * 0.6 + memory_score * 0.4) * 100
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate of evaluations."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results.values() if r["status"] == "success")
        return successful / len(self.results)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "success_rate": self.success_rate,
            "num_prompts": len(self.prompts),
            "metrics": self.metrics,
            "performance": self.performance,
            "results": self.results,
            "errors": self.errors
        }


class BenchmarkSuite:
    """Main benchmark suite orchestrator."""
    
    def __init__(self, device: str = "auto", output_dir: str = "./results"):
        """Initialize benchmark suite.
        
        Args:
            device: Device to run benchmarks on ('cpu', 'cuda', 'auto')
            output_dir: Directory to save benchmark results
        """
        self.device = self._resolve_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._models = {}
        self.metrics_engine = VideoQualityMetrics()
        self.profiler = EfficiencyProfiler()
        
        logger.info(f"BenchmarkSuite initialized with device: {self.device}")
        
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
        
    def list_available_models(self) -> List[str]:
        """List all available models in registry."""
        return list_models()
        
    def evaluate_model(
        self,
        model_name: str,
        prompts: Optional[List[str]] = None,
        num_frames: int = 16,
        fps: int = 8,
        resolution: tuple = (512, 512),
        batch_size: int = 1,
        save_videos: bool = False,
        **kwargs
    ) -> BenchmarkResult:
        """Evaluate a single model.
        
        Args:
            model_name: Name of registered model
            prompts: List of text prompts (uses StandardPrompts if None)
            num_frames: Number of frames to generate
            fps: Target frames per second
            resolution: Output resolution (width, height)
            batch_size: Number of prompts to process in parallel
            save_videos: Whether to save generated videos to disk
            
        Returns:
            BenchmarkResult containing evaluation results
        """
        if prompts is None:
            prompts = StandardPrompts.DIVERSE_SET_V2[:10]  # Use first 10 standard prompts
            
        logger.info(f"Evaluating model: {model_name} with {len(prompts)} prompts")
        
        # Initialize result container
        result = BenchmarkResult(model_name, prompts)
        
        try:
            # Load model
            model = self._load_model(model_name)
            
            # Generate videos for all prompts
            self._generate_videos(model, result, prompts, num_frames, fps, 
                                resolution, batch_size, save_videos, **kwargs)
            
            # Compute quality metrics if we have successful generations
            successful_videos = self._get_successful_videos(result)
            if successful_videos:
                self._compute_quality_metrics(result, successful_videos, prompts)
                
            # Compute performance metrics
            self._compute_performance_metrics(result)
            
            # Save results
            self._save_results(result)
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {e}")
            result.add_error(-1, e)
            
        logger.info(f"Evaluation complete. Success rate: {result.success_rate:.2%}")
        return result
        
    def evaluate_multiple_models(
        self,
        model_names: List[str],
        prompts: Optional[List[str]] = None,
        max_workers: int = 2,
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """Evaluate multiple models in parallel.
        
        Args:
            model_names: List of model names to evaluate
            prompts: List of text prompts (uses StandardPrompts if None)
            max_workers: Maximum number of models to evaluate in parallel
            **kwargs: Additional arguments passed to evaluate_model
            
        Returns:
            Dictionary mapping model names to BenchmarkResults
        """
        logger.info(f"Evaluating {len(model_names)} models in parallel")
        
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_model = {
                executor.submit(self.evaluate_model, name, prompts, **kwargs): name
                for name in model_names
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    results[model_name] = future.result()
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name}: {e}")
                    results[model_name] = BenchmarkResult(model_name, prompts or [])
                    results[model_name].add_error(-1, e)
                    
        return results
        
    def compare_models(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Compare benchmark results across models.
        
        Args:
            results: Dictionary of model results from evaluate_multiple_models
            
        Returns:
            Comparison analysis with rankings and insights
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models_compared": len(results),
            "rankings": {},
            "pareto_frontier": [],
            "summary": {}
        }
        
        # Extract metrics for comparison
        model_metrics = {}
        for name, result in results.items():
            if result.metrics and result.performance:
                model_metrics[name] = {
                    "quality_score": result.metrics.get("overall_score", 0),
                    "efficiency_score": result.performance.get("efficiency_score", 0),
                    "latency": result.performance.get("avg_latency_ms", float('inf')),
                    "vram": result.performance.get("peak_vram_gb", float('inf')),
                    "success_rate": result.success_rate
                }
                
        # Rank models by different criteria
        if model_metrics:
            comparison["rankings"] = {
                "quality": sorted(model_metrics.items(), 
                                key=lambda x: x[1]["quality_score"], reverse=True),
                "efficiency": sorted(model_metrics.items(), 
                                   key=lambda x: x[1]["efficiency_score"], reverse=True),
                "speed": sorted(model_metrics.items(), 
                              key=lambda x: x[1]["latency"]),
                "memory": sorted(model_metrics.items(), 
                               key=lambda x: x[1]["vram"])
            }
            
            # Compute Pareto frontier (quality vs efficiency)
            comparison["pareto_frontier"] = self._compute_pareto_frontier(model_metrics)
            
        return comparison
        
    def _load_model(self, model_name: str) -> ModelAdapter:
        """Load model from registry."""
        if model_name not in self._models:
            self._models[model_name] = get_model(model_name, device=self.device)
        return self._models[model_name]
        
    def _generate_videos(self, model: ModelAdapter, result: BenchmarkResult,
                        prompts: List[str], num_frames: int, fps: int,
                        resolution: tuple, batch_size: int, save_videos: bool,
                        **kwargs):
        """Generate videos for all prompts."""
        for i, prompt in enumerate(prompts):
            try:
                # Monitor memory before generation
                initial_memory = self._get_memory_stats()
                
                # Generate video with profiling
                start_time = time.time()
                with self.profiler.track(model.name):
                    video_tensor = model.generate(
                        prompt, 
                        num_frames=num_frames, 
                        fps=fps, 
                        **kwargs
                    )
                end_time = time.time()
                
                # Monitor memory after generation
                final_memory = self._get_memory_stats()
                memory_usage = {
                    "peak_gpu_mb": final_memory.get("gpu_mb", 0) - initial_memory.get("gpu_mb", 0),
                    "peak_cpu_mb": final_memory.get("cpu_mb", 0) - initial_memory.get("cpu_mb", 0)
                }
                
                # Save video if requested
                if save_videos:
                    self._save_video(video_tensor, result.model_name, i, prompt)
                    
                # Record successful result
                result.add_result(i, video_tensor, end_time - start_time, memory_usage)
                
            except Exception as e:
                logger.error(f"Failed to generate video for prompt {i}: {e}")
                result.add_error(i, e)
                
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            "cpu_mb": psutil.virtual_memory().used / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            stats["gpu_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            
        return stats
        
    def _save_video(self, video_tensor: torch.Tensor, model_name: str, 
                   prompt_idx: int, prompt: str):
        """Save video tensor to disk."""
        video_dir = self.output_dir / "videos" / model_name
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensor to numpy and save (simplified)
        video_path = video_dir / f"prompt_{prompt_idx:03d}.npy"
        np.save(video_path, video_tensor.cpu().numpy())
        
        # Save prompt text
        prompt_path = video_dir / f"prompt_{prompt_idx:03d}.txt"
        prompt_path.write_text(prompt)
        
    def _get_successful_videos(self, result: BenchmarkResult) -> List[torch.Tensor]:
        """Extract video tensors from successful generations."""
        videos = []
        for res in result.results.values():
            if res["status"] == "success" and "video_tensor" in res:
                videos.append(res["video_tensor"])
        return videos
        
    def _compute_quality_metrics(self, result: BenchmarkResult, 
                                videos: List[torch.Tensor], prompts: List[str]):
        """Compute quality metrics for generated videos."""
        try:
            # Compute FVD (simplified - would need reference dataset)
            fvd = self.metrics_engine.compute_fvd(videos, reference_dataset="mock")
            
            # Compute Inception Score
            is_mean, is_std = self.metrics_engine.compute_is(videos)
            
            # Compute CLIP similarity (simplified)
            clip_score = self.metrics_engine.compute_clipsim(prompts, videos)
            
            # Compute temporal consistency
            temporal_score = self.metrics_engine.compute_temporal_consistency(videos)
            
            result.set_metrics(fvd, is_mean, clip_score, temporal_score)
            
        except Exception as e:
            logger.error(f"Failed to compute quality metrics: {e}")
            result.set_metrics(0.0, 0.0, 0.0, 0.0)
            
    def _compute_performance_metrics(self, result: BenchmarkResult):
        """Compute performance metrics from generation results."""
        if not result.results:
            result.set_performance(0.0, 0.0, 0.0, 0.0)
            return
            
        successful_results = [r for r in result.results.values() if r["status"] == "success"]
        if not successful_results:
            result.set_performance(0.0, 0.0, 0.0, 0.0)
            return
            
        # Calculate averages
        avg_latency = np.mean([r["generation_time"] for r in successful_results])
        peak_vram = max([r["memory_usage"]["peak_gpu_mb"] for r in successful_results]) / 1024  # Convert to GB
        
        # Calculate throughput (videos per minute)
        throughput = 60.0 / avg_latency if avg_latency > 0 else 0.0
        
        # Get power consumption from profiler (mock for now)
        avg_power = 250.0  # Mock value
        
        result.set_performance(avg_latency, throughput, peak_vram, avg_power)
        
    def _compute_pareto_frontier(self, model_metrics: Dict[str, Dict]) -> List[str]:
        """Compute Pareto frontier for quality vs efficiency trade-off."""
        # Sort by quality score descending
        sorted_models = sorted(model_metrics.items(), 
                             key=lambda x: x[1]["quality_score"], reverse=True)
        
        pareto_frontier = []
        max_efficiency = -1
        
        for name, metrics in sorted_models:
            if metrics["efficiency_score"] > max_efficiency:
                pareto_frontier.append(name)
                max_efficiency = metrics["efficiency_score"]
                
        return pareto_frontier
        
    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results to disk."""
        results_file = self.output_dir / f"{result.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        logger.info(f"Results saved to: {results_file}")


# Convenience functions for common use cases
def quick_benchmark(model_name: str, num_prompts: int = 5) -> BenchmarkResult:
    """Quick benchmark with standard settings."""
    suite = BenchmarkSuite()
    prompts = StandardPrompts.DIVERSE_SET_V2[:num_prompts]
    return suite.evaluate_model(model_name, prompts)


def compare_top_models(num_prompts: int = 10) -> Dict[str, Any]:
    """Compare top performing models."""
    suite = BenchmarkSuite()
    
    # Define top models to compare
    top_models = ["svd-xt", "pika-lumiere", "cogvideo", "modelscope-v2"]
    available_models = [m for m in top_models if m in suite.list_available_models()]
    
    if not available_models:
        logger.warning("No top models available in registry")
        return {}
        
    prompts = StandardPrompts.DIVERSE_SET_V2[:num_prompts]
    results = suite.evaluate_multiple_models(available_models, prompts)
    
    return suite.compare_models(results)