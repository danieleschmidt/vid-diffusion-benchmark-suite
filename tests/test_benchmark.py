"""Tests for benchmark suite functionality."""

import pytest
from vid_diffusion_bench import BenchmarkSuite


class TestBenchmarkSuite:
    """Test cases for BenchmarkSuite class."""
    
    def test_init_default_device(self):
        """Test suite initialization with default device."""
        suite = BenchmarkSuite()
        assert suite.device == "auto"
        
    def test_init_custom_device(self):
        """Test suite initialization with custom device."""
        suite = BenchmarkSuite(device="cpu")
        assert suite.device == "cpu"
        
    def test_evaluate_model_basic(self, benchmark_suite, sample_prompts):
        """Test basic model evaluation."""
        results = benchmark_suite.evaluate_model(
            model_name="test_model",
            prompts=sample_prompts,
            num_frames=16
        )
        
        assert isinstance(results, dict)
        assert "model_name" in results
        assert "fvd" in results
        assert "latency" in results
        assert "peak_vram_gb" in results
        assert results["model_name"] == "test_model"
        assert results["num_prompts"] == len(sample_prompts)
        
    def test_evaluate_model_custom_params(self, benchmark_suite, sample_prompts):
        """Test model evaluation with custom parameters."""
        results = benchmark_suite.evaluate_model(
            model_name="test_model",
            prompts=sample_prompts,
            num_frames=32,
            fps=24,
            resolution=(1024, 1024)
        )
        
        assert results["num_prompts"] == len(sample_prompts)
        
    def test_evaluate_model_empty_prompts(self, benchmark_suite):
        """Test model evaluation with empty prompt list."""
        results = benchmark_suite.evaluate_model(
            model_name="test_model",
            prompts=[]
        )
        
        assert results["num_prompts"] == 0