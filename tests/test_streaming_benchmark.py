"""Comprehensive tests for streaming benchmark functionality."""

import pytest
import asyncio
import time
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock

from src.vid_diffusion_bench.streaming_benchmark import (
    StreamingBenchmark,
    AdaptiveQualityController,
    StreamingBuffer,
    StreamingMetrics,
    benchmark_live_streaming,
    benchmark_interactive_generation
)


class TestStreamingBuffer:
    """Test streaming buffer implementation."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = StreamingBuffer(max_size=10)
        assert buffer.buffer.maxlen == 10
        assert len(buffer.buffer) == 0
        assert buffer.total_frames == 0
        assert buffer.dropped_frames == 0
    
    def test_put_frame_success(self):
        """Test successful frame insertion."""
        buffer = StreamingBuffer(max_size=5)
        frame = torch.randn(3, 256, 256)
        
        result = buffer.put_frame(frame, "test prompt", time.time())
        assert result is True
        assert len(buffer.buffer) == 1
        assert buffer.total_frames == 1
        assert buffer.dropped_frames == 0
    
    def test_put_frame_drop_when_full(self):
        """Test frame dropping when buffer is full."""
        buffer = StreamingBuffer(max_size=2)
        frame = torch.randn(3, 256, 256)
        
        # Fill buffer
        buffer.put_frame(frame, "prompt1", time.time())
        buffer.put_frame(frame, "prompt2", time.time())
        
        # This should drop the frame
        result = buffer.put_frame(frame, "prompt3", time.time())
        assert result is False
        assert len(buffer.buffer) == 2
        assert buffer.total_frames == 3
        assert buffer.dropped_frames == 1
    
    def test_get_frame_success(self):
        """Test successful frame retrieval."""
        buffer = StreamingBuffer()
        frame = torch.randn(3, 256, 256)
        timestamp = time.time()
        
        buffer.put_frame(frame, "test prompt", timestamp)
        
        retrieved = buffer.get_frame(timeout=0.1)
        assert retrieved is not None
        assert retrieved["prompt"] == "test prompt"
        assert torch.equal(retrieved["frame"], frame)
        assert retrieved["timestamp"] == timestamp
    
    def test_get_frame_timeout(self):
        """Test frame retrieval timeout."""
        buffer = StreamingBuffer()
        
        retrieved = buffer.get_frame(timeout=0.1)
        assert retrieved is None
    
    def test_buffer_utilization(self):
        """Test buffer utilization calculation."""
        buffer = StreamingBuffer(max_size=10)
        
        assert buffer.utilization == 0.0
        
        frame = torch.randn(3, 256, 256)
        for i in range(5):
            buffer.put_frame(frame, f"prompt{i}", time.time())
        
        assert buffer.utilization == 0.5
    
    def test_drop_rate(self):
        """Test drop rate calculation."""
        buffer = StreamingBuffer(max_size=2)
        frame = torch.randn(3, 256, 256)
        
        # Add frames, some will be dropped
        results = []
        for i in range(5):
            result = buffer.put_frame(frame, f"prompt{i}", time.time())
            results.append(result)
        
        expected_drop_rate = 3 / 5  # 3 dropped out of 5 total
        assert buffer.drop_rate == expected_drop_rate


class TestAdaptiveQualityController:
    """Test adaptive quality controller."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = AdaptiveQualityController(target_latency_ms=100)
        assert controller.target_latency == 100
        assert len(controller.quality_levels) == 4
        assert controller.current_level == 2  # Start at medium
    
    def test_quality_adjustment_reduce(self):
        """Test quality reduction when latency is too high."""
        controller = AdaptiveQualityController(target_latency_ms=100)
        
        # Simulate high latency
        for _ in range(5):
            controller.performance_window.append(200)  # 2x target latency
        
        params = controller.adjust_quality(200)
        
        # Should reduce quality (lower level)
        assert controller.current_level < 2
        assert params["resolution"][0] < 512  # Lower resolution
    
    def test_quality_adjustment_increase(self):
        """Test quality increase when latency is low."""
        controller = AdaptiveQualityController(target_latency_ms=100)
        controller.current_level = 1  # Start at lower level
        
        # Simulate low latency
        for _ in range(5):
            controller.performance_window.append(50)  # 0.5x target latency
        
        params = controller.adjust_quality(50)
        
        # Should increase quality (higher level)
        assert controller.current_level > 1
    
    def test_quality_bounds(self):
        """Test quality level bounds."""
        controller = AdaptiveQualityController()
        
        # Force to minimum level
        controller.current_level = 0
        for _ in range(10):
            controller.adjust_quality(1000)  # Very high latency
        
        assert controller.current_level >= 0  # Should not go below 0
        
        # Force to maximum level
        controller.current_level = len(controller.quality_levels) - 1
        for _ in range(10):
            controller.adjust_quality(1)  # Very low latency
        
        assert controller.current_level < len(controller.quality_levels)


class TestStreamingBenchmark:
    """Test streaming benchmark functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.generate = Mock(return_value=torch.randn(16, 3, 256, 256))
        return model
    
    @pytest.fixture
    def streaming_benchmark(self):
        """Create streaming benchmark instance."""
        return StreamingBenchmark(device="cpu")  # Use CPU for testing
    
    def test_initialization(self, streaming_benchmark):
        """Test benchmark initialization."""
        assert streaming_benchmark.device == "cpu"
        assert streaming_benchmark.quality_controller is not None
        assert streaming_benchmark.buffer is not None
        assert not streaming_benchmark.is_running
    
    @patch('src.vid_diffusion_bench.streaming_benchmark.get_model')
    @pytest.mark.asyncio
    async def test_stream_benchmark_basic(self, mock_get_model, streaming_benchmark, mock_model):
        """Test basic streaming benchmark execution."""
        mock_get_model.return_value = mock_model
        
        # Create async prompt generator
        async def prompt_generator():
            prompts = ["prompt1", "prompt2", "prompt3"]
            for prompt in prompts:
                yield prompt
                await asyncio.sleep(0.1)
        
        # Run short benchmark
        metrics = await streaming_benchmark.stream_benchmark(
            model_name="test_model",
            prompt_stream=prompt_generator(),
            duration_seconds=1,
            target_fps=2.0
        )
        
        assert isinstance(metrics, StreamingMetrics)
        assert metrics.avg_frame_latency_ms >= 0
        assert 0 <= metrics.frame_drop_rate <= 1
        assert 0 <= metrics.quality_consistency <= 1
    
    def test_generate_frame_success(self, streaming_benchmark, mock_model):
        """Test successful frame generation."""
        quality_params = {"resolution": (256, 256), "frames": 8, "steps": 10}
        
        frame_data = streaming_benchmark._generate_frame(
            mock_model, "test prompt", quality_params
        )
        
        assert frame_data is not None
        assert "frame" in frame_data
        assert "timestamp" in frame_data
        assert "stats" in frame_data
        assert frame_data["stats"]["latency_ms"] > 0
    
    def test_generate_frame_failure(self, streaming_benchmark):
        """Test frame generation failure handling."""
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=Exception("Test error"))
        
        quality_params = {"resolution": (256, 256), "frames": 8, "steps": 10}
        
        frame_data = streaming_benchmark._generate_frame(
            mock_model, "test prompt", quality_params
        )
        
        assert frame_data is None
    
    def test_compute_streaming_metrics_empty(self, streaming_benchmark):
        """Test metrics computation with empty stats."""
        streaming_benchmark._generation_stats = []
        
        metrics = streaming_benchmark._compute_streaming_metrics()
        
        assert metrics.avg_frame_latency_ms == 0
        assert metrics.frame_drop_rate == 0
        assert metrics.quality_consistency == 0
    
    def test_compute_streaming_metrics_with_data(self, streaming_benchmark):
        """Test metrics computation with actual data."""
        # Simulate generation stats
        streaming_benchmark._generation_stats = [
            {"latency_ms": 100, "quality_level": 1},
            {"latency_ms": 120, "quality_level": 1},
            {"latency_ms": 110, "quality_level": 2},
        ]
        
        # Simulate buffer drops
        streaming_benchmark.buffer.total_frames = 10
        streaming_benchmark.buffer.dropped_frames = 2
        
        metrics = streaming_benchmark._compute_streaming_metrics()
        
        assert metrics.avg_frame_latency_ms == pytest.approx(110.0, rel=1e-2)
        assert metrics.frame_drop_rate == 0.2
        assert 0 <= metrics.quality_consistency <= 1


class TestStreamingBenchmarkIntegration:
    """Integration tests for streaming benchmark."""
    
    @patch('src.vid_diffusion_bench.streaming_benchmark.get_model')
    @pytest.mark.asyncio
    async def test_benchmark_live_streaming(self, mock_get_model):
        """Test live streaming benchmark function."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value=torch.randn(8, 3, 256, 256))
        mock_get_model.return_value = mock_model
        
        prompts = ["prompt1", "prompt2"]
        
        metrics = await benchmark_live_streaming(
            model_name="test_model",
            prompts=prompts,
            duration_seconds=2
        )
        
        assert isinstance(metrics, StreamingMetrics)
        assert metrics.avg_frame_latency_ms >= 0
    
    @patch('src.vid_diffusion_bench.streaming_benchmark.get_model')
    @pytest.mark.asyncio
    async def test_benchmark_interactive_generation(self, mock_get_model):
        """Test interactive generation benchmark function."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value=torch.randn(8, 3, 256, 256))
        mock_get_model.return_value = mock_model
        
        prompts = ["interactive prompt 1", "interactive prompt 2"]
        
        metrics = await benchmark_interactive_generation(
            model_name="test_model",
            interactive_prompts=prompts,
            response_time_target_ms=300
        )
        
        assert isinstance(metrics, StreamingMetrics)
        assert metrics.avg_frame_latency_ms >= 0


class TestStreamingMetrics:
    """Test streaming metrics data structure."""
    
    def test_metrics_creation(self):
        """Test metrics object creation."""
        metrics = StreamingMetrics(
            avg_frame_latency_ms=100.0,
            frame_drop_rate=0.05,
            quality_consistency=0.9,
            adaptive_quality_score=0.8,
            buffer_utilization=0.6,
            throughput_stability=0.85
        )
        
        assert metrics.avg_frame_latency_ms == 100.0
        assert metrics.frame_drop_rate == 0.05
        assert metrics.quality_consistency == 0.9
        assert metrics.adaptive_quality_score == 0.8
        assert metrics.buffer_utilization == 0.6
        assert metrics.throughput_stability == 0.85
    
    def test_metrics_bounds(self):
        """Test that metrics are within expected bounds."""
        metrics = StreamingMetrics(
            avg_frame_latency_ms=50.0,
            frame_drop_rate=0.1,
            quality_consistency=0.95,
            adaptive_quality_score=0.75,
            buffer_utilization=0.4,
            throughput_stability=0.9
        )
        
        # Latency should be positive
        assert metrics.avg_frame_latency_ms >= 0
        
        # Rates and scores should be between 0 and 1
        assert 0 <= metrics.frame_drop_rate <= 1
        assert 0 <= metrics.quality_consistency <= 1
        assert 0 <= metrics.adaptive_quality_score <= 1
        assert 0 <= metrics.buffer_utilization <= 1
        assert 0 <= metrics.throughput_stability <= 1


class TestStreamingBenchmarkPerformance:
    """Performance tests for streaming benchmark."""
    
    @patch('src.vid_diffusion_bench.streaming_benchmark.get_model')
    @pytest.mark.asyncio
    async def test_benchmark_performance_cpu(self, mock_get_model):
        """Test benchmark performance on CPU."""
        mock_model = Mock()
        # Simulate realistic generation time
        def mock_generate(*args, **kwargs):
            time.sleep(0.1)  # 100ms generation time
            return torch.randn(8, 3, 64, 64)  # Smaller tensor for speed
        
        mock_model.generate = Mock(side_effect=mock_generate)
        mock_get_model.return_value = mock_model
        
        async def fast_prompt_generator():
            for i in range(3):
                yield f"prompt {i}"
                await asyncio.sleep(0.05)
        
        benchmark = StreamingBenchmark(device="cpu")
        
        start_time = time.time()
        metrics = await benchmark.stream_benchmark(
            model_name="test_model",
            prompt_stream=fast_prompt_generator(),
            duration_seconds=1,
            target_fps=5.0
        )
        end_time = time.time()
        
        # Benchmark should complete within reasonable time
        assert end_time - start_time < 5  # Should finish within 5 seconds
        assert isinstance(metrics, StreamingMetrics)
    
    def test_buffer_performance(self):
        """Test buffer performance under load."""
        buffer = StreamingBuffer(max_size=1000)
        frame = torch.randn(3, 256, 256)
        
        # Test insertion performance
        start_time = time.time()
        for i in range(100):
            buffer.put_frame(frame, f"prompt{i}", time.time())
        insert_time = time.time() - start_time
        
        # Should be fast
        assert insert_time < 1.0  # Should complete within 1 second
        
        # Test retrieval performance
        start_time = time.time()
        retrieved_count = 0
        while buffer.get_frame(timeout=0.001):
            retrieved_count += 1
        retrieve_time = time.time() - start_time
        
        assert retrieved_count == 100
        assert retrieve_time < 1.0  # Should complete within 1 second


if __name__ == "__main__":
    pytest.main([__file__])