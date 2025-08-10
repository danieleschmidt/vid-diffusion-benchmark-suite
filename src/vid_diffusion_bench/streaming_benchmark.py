"""Real-time streaming benchmark for video diffusion models.

Advanced benchmarking for streaming video generation scenarios with 
latency optimization and adaptive quality control.
"""

import time
import asyncio
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable, AsyncGenerator, Tuple
from datetime import datetime
import torch
import numpy as np
from dataclasses import dataclass

from .models.registry import get_model
from .models.base import ModelAdapter
from .metrics import VideoQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class StreamingMetrics:
    """Container for streaming-specific metrics."""
    avg_frame_latency_ms: float
    frame_drop_rate: float
    quality_consistency: float
    adaptive_quality_score: float
    buffer_utilization: float
    throughput_stability: float
    

class AdaptiveQualityController:
    """Dynamically adjusts generation parameters based on performance."""
    
    def __init__(self, target_latency_ms: float = 100):
        self.target_latency = target_latency_ms
        self.quality_levels = [
            {"resolution": (256, 256), "frames": 8, "steps": 10},
            {"resolution": (384, 384), "frames": 12, "steps": 15}, 
            {"resolution": (512, 512), "frames": 16, "steps": 20},
            {"resolution": (768, 768), "frames": 24, "steps": 30}
        ]
        self.current_level = 2  # Start at medium quality
        self.performance_window = deque(maxlen=10)
        
    def adjust_quality(self, latency_ms: float) -> Dict:
        """Adjust quality parameters based on current latency."""
        self.performance_window.append(latency_ms)
        
        if len(self.performance_window) < 5:
            return self.quality_levels[self.current_level]
            
        avg_latency = np.mean(list(self.performance_window))
        
        if avg_latency > self.target_latency * 1.5 and self.current_level > 0:
            self.current_level -= 1  # Reduce quality
        elif avg_latency < self.target_latency * 0.7 and self.current_level < len(self.quality_levels) - 1:
            self.current_level += 1  # Increase quality
            
        return self.quality_levels[self.current_level]


class StreamingBuffer:
    """Thread-safe buffer for streaming video frames."""
    
    def __init__(self, max_size: int = 30):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.total_frames = 0
        self.dropped_frames = 0
        
    def put_frame(self, frame: torch.Tensor, prompt: str, timestamp: float) -> bool:
        """Add frame to buffer. Returns False if frame was dropped."""
        with self.condition:
            if len(self.buffer) >= self.buffer.maxlen:
                self.dropped_frames += 1
                return False
                
            self.buffer.append({
                "frame": frame,
                "prompt": prompt, 
                "timestamp": timestamp,
                "buffer_time": time.time()
            })
            self.total_frames += 1
            self.condition.notify_all()
            return True
    
    def get_frame(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get frame from buffer with timeout."""
        with self.condition:
            if not self.buffer:
                if not self.condition.wait(timeout):
                    return None
                    
            if self.buffer:
                return self.buffer.popleft()
        return None
    
    @property
    def utilization(self) -> float:
        """Current buffer utilization (0-1)."""
        return len(self.buffer) / self.buffer.maxlen
        
    @property
    def drop_rate(self) -> float:
        """Frame drop rate (0-1)."""
        if self.total_frames == 0:
            return 0.0
        return self.dropped_frames / self.total_frames


class StreamingBenchmark:
    """Real-time streaming benchmark for video diffusion models."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.quality_controller = AdaptiveQualityController()
        self.buffer = StreamingBuffer()
        self.metrics_engine = VideoQualityMetrics()
        self.is_running = False
        self._generation_stats = []
        
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    async def stream_benchmark(
        self,
        model_name: str,
        prompt_stream: AsyncGenerator[str, None],
        duration_seconds: int = 60,
        target_fps: float = 24.0,
        adaptive_quality: bool = True
    ) -> StreamingMetrics:
        """Run streaming benchmark with continuous prompt feed.
        
        Args:
            model_name: Name of model to benchmark
            prompt_stream: Async generator yielding text prompts
            duration_seconds: How long to run benchmark
            target_fps: Target frames per second for output
            adaptive_quality: Whether to use adaptive quality control
            
        Returns:
            StreamingMetrics with performance analysis
        """
        logger.info(f"Starting streaming benchmark for {model_name}")
        
        model = get_model(model_name, device=self.device)
        self.is_running = True
        self._generation_stats = []
        
        # Reset buffer and controller
        self.buffer = StreamingBuffer()
        self.quality_controller = AdaptiveQualityController(target_latency_ms=1000/target_fps)
        
        # Start generation and consumption tasks
        tasks = [
            asyncio.create_task(self._generation_loop(model, prompt_stream, adaptive_quality)),
            asyncio.create_task(self._consumption_loop(target_fps)),
            asyncio.create_task(self._metrics_collection_loop())
        ]
        
        try:
            # Run for specified duration
            await asyncio.sleep(duration_seconds)
            self.is_running = False
            
            # Wait for tasks to complete
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Streaming benchmark error: {e}")
            self.is_running = False
            
        return self._compute_streaming_metrics()
    
    async def _generation_loop(
        self, 
        model: ModelAdapter, 
        prompt_stream: AsyncGenerator[str, None],
        adaptive_quality: bool
    ):
        """Main generation loop."""
        executor = ThreadPoolExecutor(max_workers=2)
        
        try:
            async for prompt in prompt_stream:
                if not self.is_running:
                    break
                    
                # Get current quality parameters
                if adaptive_quality:
                    quality_params = self.quality_controller.adjust_quality(
                        np.mean([s["latency_ms"] for s in self._generation_stats[-5:]] or [0])
                    )
                else:
                    quality_params = {"resolution": (512, 512), "frames": 16, "steps": 20}
                
                # Submit generation task
                future = executor.submit(self._generate_frame, model, prompt, quality_params)
                
                # Process result asynchronously
                try:
                    frame_data = await asyncio.get_event_loop().run_in_executor(None, future.result, 0.5)
                    if frame_data:
                        self.buffer.put_frame(frame_data["frame"], prompt, frame_data["timestamp"])
                        self._generation_stats.append(frame_data["stats"])
                except Exception as e:
                    logger.debug(f"Generation timeout or error: {e}")
                    
        except Exception as e:
            logger.error(f"Generation loop error: {e}")
        finally:
            executor.shutdown(wait=False)
    
    def _generate_frame(self, model: ModelAdapter, prompt: str, params: Dict) -> Optional[Dict]:
        """Generate single frame with timing."""
        try:
            start_time = time.time()
            
            # Generate video with specified parameters
            video_tensor = model.generate(
                prompt,
                num_frames=params["frames"],
                resolution=params["resolution"]
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Extract first frame for streaming
            frame = video_tensor[0] if len(video_tensor.shape) > 3 else video_tensor
            
            return {
                "frame": frame,
                "timestamp": start_time,
                "stats": {
                    "latency_ms": latency_ms,
                    "resolution": params["resolution"],
                    "frames_generated": params["frames"],
                    "quality_level": self.quality_controller.current_level
                }
            }
            
        except Exception as e:
            logger.debug(f"Frame generation failed: {e}")
            return None
    
    async def _consumption_loop(self, target_fps: float):
        """Frame consumption loop simulating real-time display."""
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_frame_time >= frame_interval:
                frame_data = self.buffer.get_frame(timeout=frame_interval * 0.5)
                
                if frame_data:
                    # Simulate frame processing/display
                    await asyncio.sleep(0.001)  # Minimal processing delay
                    last_frame_time = current_time
                else:
                    # No frame available - indicates underrun
                    await asyncio.sleep(frame_interval * 0.1)
            else:
                await asyncio.sleep(0.001)
    
    async def _metrics_collection_loop(self):
        """Collect streaming metrics continuously."""
        while self.is_running:
            await asyncio.sleep(1.0)  # Collect metrics every second
            
            # Log current state
            if self._generation_stats:
                recent_latency = np.mean([s["latency_ms"] for s in self._generation_stats[-10:]])
                logger.debug(f"Streaming metrics - Latency: {recent_latency:.1f}ms, "
                           f"Buffer: {self.buffer.utilization:.2%}, "
                           f"Drop rate: {self.buffer.drop_rate:.2%}")
    
    def _compute_streaming_metrics(self) -> StreamingMetrics:
        """Compute final streaming performance metrics."""
        if not self._generation_stats:
            return StreamingMetrics(0, 0, 0, 0, 0, 0)
        
        latencies = [s["latency_ms"] for s in self._generation_stats]
        quality_levels = [s["quality_level"] for s in self._generation_stats]
        
        # Compute metrics
        avg_frame_latency = np.mean(latencies)
        frame_drop_rate = self.buffer.drop_rate
        
        # Quality consistency (lower variance is better)
        quality_consistency = 1.0 - (np.std(quality_levels) / (len(self.quality_controller.quality_levels) - 1))
        quality_consistency = max(0, quality_consistency)
        
        # Adaptive quality score (rewards maintaining higher quality)
        adaptive_quality_score = np.mean(quality_levels) / (len(self.quality_controller.quality_levels) - 1)
        
        # Buffer utilization (target around 50%)
        avg_buffer_utilization = np.mean([0.5])  # Would collect this during run
        
        # Throughput stability (lower variance in latency is better)
        throughput_stability = 1.0 - min(1.0, np.std(latencies) / avg_frame_latency)
        
        return StreamingMetrics(
            avg_frame_latency_ms=avg_frame_latency,
            frame_drop_rate=frame_drop_rate,
            quality_consistency=quality_consistency,
            adaptive_quality_score=adaptive_quality_score,
            buffer_utilization=avg_buffer_utilization,
            throughput_stability=throughput_stability
        )


# Convenience functions for common streaming scenarios
async def benchmark_live_streaming(
    model_name: str,
    prompts: List[str],
    duration_seconds: int = 30
) -> StreamingMetrics:
    """Benchmark model for live streaming scenario."""
    
    async def prompt_generator():
        """Generate prompts in a loop."""
        while True:
            for prompt in prompts:
                yield prompt
                await asyncio.sleep(0.1)  # Small delay between prompts
    
    benchmark = StreamingBenchmark()
    return await benchmark.stream_benchmark(
        model_name=model_name,
        prompt_stream=prompt_generator(),
        duration_seconds=duration_seconds,
        target_fps=15.0,  # Realistic for real-time generation
        adaptive_quality=True
    )


async def benchmark_interactive_generation(
    model_name: str, 
    interactive_prompts: List[str],
    response_time_target_ms: float = 500
) -> StreamingMetrics:
    """Benchmark model for interactive generation (chat, games)."""
    
    async def interactive_generator():
        """Generate prompts simulating user interactions."""
        for prompt in interactive_prompts:
            yield prompt
            # Random delay simulating user interaction
            await asyncio.sleep(np.random.exponential(2.0))  # Average 2s between interactions
    
    benchmark = StreamingBenchmark()
    benchmark.quality_controller.target_latency = response_time_target_ms
    
    return await benchmark.stream_benchmark(
        model_name=model_name,
        prompt_stream=interactive_generator(),
        duration_seconds=len(interactive_prompts) * 3,  # Allow time for all prompts
        target_fps=2.0,  # Lower FPS for interactive use
        adaptive_quality=True
    )