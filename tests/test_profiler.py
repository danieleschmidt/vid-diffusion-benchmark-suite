"""Tests for performance profiler."""

import pytest
import time
from vid_diffusion_bench.profiler import EfficiencyProfiler, ProfileStats


class TestEfficiencyProfiler:
    """Test cases for EfficiencyProfiler class."""
    
    def test_init(self):
        """Test profiler initialization."""
        profiler = EfficiencyProfiler()
        assert profiler._stats == {}
        
    def test_track_context_manager(self):
        """Test tracking context manager."""
        profiler = EfficiencyProfiler()
        
        with profiler.track("test_model"):
            time.sleep(0.01)  # Simulate work
            
        stats = profiler.get_stats("test_model")
        assert isinstance(stats, ProfileStats)
        assert stats.latency_ms > 0
        assert stats.throughput_fps > 0
        
    def test_get_stats_specific_model(self):
        """Test getting stats for specific model."""
        profiler = EfficiencyProfiler()
        
        with profiler.track("model_a"):
            time.sleep(0.005)
            
        with profiler.track("model_b"):
            time.sleep(0.01)
            
        stats_a = profiler.get_stats("model_a")
        stats_b = profiler.get_stats("model_b")
        
        assert stats_a.latency_ms < stats_b.latency_ms
        
    def test_get_stats_latest(self):
        """Test getting latest stats when no model specified."""
        profiler = EfficiencyProfiler()
        
        with profiler.track("model_1"):
            time.sleep(0.005)
            
        with profiler.track("model_2"):
            time.sleep(0.01)
            
        latest_stats = profiler.get_stats()
        model_2_stats = profiler.get_stats("model_2")
        
        assert latest_stats.latency_ms == model_2_stats.latency_ms
        
    def test_get_stats_empty_profiler(self):
        """Test getting stats from empty profiler."""
        profiler = EfficiencyProfiler()
        stats = profiler.get_stats()
        
        assert isinstance(stats, ProfileStats)
        assert stats.latency_ms == 0
        assert stats.throughput_fps == 0
        
    @pytest.mark.gpu
    def test_gpu_memory_tracking(self):
        """Test GPU memory tracking (requires CUDA)."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        profiler = EfficiencyProfiler()
        
        with profiler.track("gpu_model"):
            # Allocate some GPU memory
            tensor = torch.randn(1000, 1000).cuda()
            del tensor
            
        stats = profiler.get_stats("gpu_model")
        assert stats.vram_peak_gb >= 0