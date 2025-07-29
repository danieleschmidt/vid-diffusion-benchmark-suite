"""Tests for video quality metrics."""

import pytest
import torch
from vid_diffusion_bench.metrics import VideoQualityMetrics


class TestVideoQualityMetrics:
    """Test cases for VideoQualityMetrics class."""
    
    def test_init_default_device(self):
        """Test metrics initialization with default device."""
        metrics = VideoQualityMetrics()
        assert metrics.device == "cuda"
        
    def test_init_custom_device(self):
        """Test metrics initialization with custom device."""
        metrics = VideoQualityMetrics(device="cpu")
        assert metrics.device == "cpu"
        
    def test_compute_fvd(self, sample_video):
        """Test FVD computation."""
        metrics = VideoQualityMetrics(device="cpu")
        fvd_score = metrics.compute_fvd(sample_video)
        
        assert isinstance(fvd_score, float)
        assert fvd_score > 0
        
    def test_compute_is(self, sample_video):
        """Test Inception Score computation."""
        metrics = VideoQualityMetrics(device="cpu")
        is_mean, is_std = metrics.compute_is(sample_video)
        
        assert isinstance(is_mean, float)
        assert isinstance(is_std, float)
        assert is_mean > 0
        assert is_std >= 0
        
    def test_compute_clipsim(self, sample_prompts, sample_video):
        """Test CLIP similarity computation."""
        metrics = VideoQualityMetrics(device="cpu")
        clip_score = metrics.compute_clipsim(sample_prompts, sample_video)
        
        assert isinstance(clip_score, float)
        assert 0 <= clip_score <= 1
        
    def test_compute_temporal_consistency(self, sample_video):
        """Test temporal consistency computation."""
        metrics = VideoQualityMetrics(device="cpu")
        temp_score = metrics.compute_temporal_consistency(sample_video)
        
        assert isinstance(temp_score, float)
        assert 0 <= temp_score <= 1