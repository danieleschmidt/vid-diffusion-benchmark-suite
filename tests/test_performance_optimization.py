"""Comprehensive tests for performance optimization functionality."""

import pytest
import time
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from src.vid_diffusion_bench.performance_optimization import (
    MemoryManager,
    ModelOptimizer,
    BatchOptimizer,
    PerformanceProfiler,
    OptimizationProfile,
    MemoryProfile,
    create_high_performance_profile,
    create_memory_efficient_profile,
    create_balanced_profile,
    optimize_model_for_benchmarking,
    benchmark_optimization_impact
)
from src.vid_diffusion_bench.models.base import ModelAdapter


class MockModelAdapter(ModelAdapter):
    """Mock model adapter for testing."""
    
    def __init__(self, device="cpu", name="mock_model"):
        super().__init__(device=device)
        self._name = name
        # Create a simple PyTorch model
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
    
    def generate(self, prompt, num_frames=16, fps=8, **kwargs):
        """Mock generation method."""
        # Simulate generation time
        time.sleep(0.01)
        return torch.randn(num_frames, 3, 64, 64)
    
    @property
    def requirements(self):
        return {"vram_gb": 4, "precision": "fp16", "dependencies": ["torch"]}
    
    @property
    def name(self):
        return self._name


class TestMemoryManager:
    """Test memory management functionality."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create memory manager instance."""
        return MemoryManager(device="cpu")  # Use CPU for testing
    
    def test_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.device == "cpu"
        assert memory_manager.memory_pool == {}
        assert memory_manager.gc_threshold == 0.9
        assert isinstance(memory_manager.allocation_history, list)
    
    def test_managed_memory_context(self, memory_manager):
        """Test managed memory context manager."""
        initial_pool_size = len(memory_manager.memory_pool)
        
        with memory_manager.managed_memory(reserve_memory=10):
            # During context, reserved memory might be allocated
            pass
        
        # After context, memory should be cleaned up
        assert len(memory_manager.memory_pool) <= initial_pool_size
    
    def test_optimize_memory_layout(self, memory_manager):
        """Test memory layout optimization."""
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        
        optimized_model = memory_manager.optimize_memory_layout(model)
        
        # Model should be returned (possibly modified)
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
    
    def test_profile_memory_usage(self, memory_manager):
        """Test memory usage profiling."""
        def test_function(size):
            # Allocate some memory
            tensor = torch.randn(size, size)
            return tensor.sum()
        
        profile = memory_manager.profile_memory_usage(test_function, 100)
        
        assert isinstance(profile, MemoryProfile)
        assert profile.initial_memory >= 0
        assert profile.peak_memory >= profile.initial_memory
        assert profile.final_memory >= 0
        assert 0 <= profile.memory_efficiency <= 1
        assert 0 <= profile.fragmentation_ratio <= 1
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_get_memory_usage_cuda(self, mock_memory_allocated, mock_cuda_available, memory_manager):
        """Test GPU memory usage measurement."""
        memory_manager.device = "cuda"
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1024 * 1024 * 512  # 512 MB
        
        usage = memory_manager._get_memory_usage()
        assert usage == 512  # Should return MB
    
    def test_get_memory_usage_cpu(self, memory_manager):
        """Test CPU memory usage (should return 0)."""
        usage = memory_manager._get_memory_usage()
        assert usage == 0


class TestOptimizationProfile:
    """Test optimization profile configuration."""
    
    def test_default_profile(self):
        """Test default optimization profile."""
        profile = OptimizationProfile()
        
        assert profile.precision == "fp16"
        assert profile.compile_model is True
        assert profile.use_flash_attention is True
        assert profile.memory_efficient is True
        assert profile.gradient_checkpointing is False
        assert profile.quantization is None
    
    def test_custom_profile(self):
        """Test custom optimization profile."""
        profile = OptimizationProfile(
            precision="fp32",
            compile_model=False,
            quantization="dynamic",
            tensorrt_optimization=True
        )
        
        assert profile.precision == "fp32"
        assert profile.compile_model is False
        assert profile.quantization == "dynamic"
        assert profile.tensorrt_optimization is True
    
    def test_profile_creation_functions(self):
        """Test profile creation convenience functions."""
        hp_profile = create_high_performance_profile()
        assert hp_profile.precision == "fp16"
        assert hp_profile.compile_model is True
        
        mem_profile = create_memory_efficient_profile()
        assert mem_profile.memory_efficient is True
        assert mem_profile.gradient_checkpointing is True
        
        balanced_profile = create_balanced_profile()
        assert balanced_profile.precision == "fp16"
        assert balanced_profile.memory_efficient is True


class TestModelOptimizer:
    """Test model optimization functionality."""
    
    @pytest.fixture
    def optimizer(self):
        """Create model optimizer with test profile."""
        profile = OptimizationProfile(
            precision="fp16",
            compile_model=False,  # Disable for testing
            memory_efficient=True
        )
        return ModelOptimizer(profile)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockModelAdapter(device="cpu")
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.profile is not None
        assert isinstance(optimizer.compiled_models, dict)
        assert isinstance(optimizer.optimization_cache, dict)
    
    def test_optimize_model_basic(self, optimizer, mock_model):
        """Test basic model optimization."""
        optimized_model = optimizer.optimize_model(mock_model)
        
        assert optimized_model is not None
        assert isinstance(optimized_model, ModelAdapter)
    
    def test_optimize_precision(self, optimizer, mock_model):
        """Test precision optimization."""
        optimizer.profile.precision = "fp16"
        
        optimized_model = optimizer._optimize_precision(mock_model)
        
        # Should return a model (possibly modified)
        assert optimized_model is not None
    
    def test_quantize_model_dynamic(self, optimizer, mock_model):
        """Test dynamic quantization."""
        optimizer.profile.quantization = "dynamic"
        
        optimized_model = optimizer._quantize_model(mock_model)
        
        # Should not raise an error
        assert optimized_model is not None
    
    def test_optimization_caching(self, optimizer, mock_model):
        """Test optimization result caching."""
        # First optimization
        opt1 = optimizer.optimize_model(mock_model)
        cache_size_1 = len(optimizer.optimization_cache)
        
        # Second optimization with same model should use cache
        opt2 = optimizer.optimize_model(mock_model)
        cache_size_2 = len(optimizer.optimization_cache)
        
        assert cache_size_2 == cache_size_1  # No new cache entries
    
    def test_benchmark_optimization_impact(self, optimizer, mock_model):
        """Test optimization impact benchmarking."""
        sample_inputs = [torch.randn(1, 3, 64, 64) for _ in range(2)]
        
        results = optimizer.benchmark_optimization_impact(
            mock_model, sample_inputs, num_iterations=2
        )
        
        assert isinstance(results, dict)
        assert "baseline" in results
        assert "full_optimization" in results
        
        for config_name, metrics in results.items():
            if "error" not in metrics:
                assert "avg_latency" in metrics
                assert "peak_memory" in metrics
    
    def test_get_optimization_key(self, optimizer, mock_model):
        """Test optimization key generation."""
        key = optimizer._get_optimization_key(mock_model, optimizer.profile)
        
        assert isinstance(key, str)
        assert len(key) > 0
        assert mock_model.name in key
    
    def test_benchmark_model(self, optimizer, mock_model):
        """Test model benchmarking."""
        sample_inputs = [torch.randn(1, 3, 64, 64)]
        
        metrics = optimizer._benchmark_model(mock_model, sample_inputs, num_iterations=2)
        
        assert "avg_latency" in metrics
        assert "std_latency" in metrics
        assert "min_latency" in metrics
        assert "max_latency" in metrics
        assert "avg_memory" in metrics
        assert "peak_memory" in metrics
        
        # Latency should be positive
        assert metrics["avg_latency"] > 0
        assert metrics["min_latency"] >= 0


class TestBatchOptimizer:
    """Test batch processing optimization."""
    
    @pytest.fixture
    def batch_optimizer(self):
        """Create batch optimizer instance."""
        return BatchOptimizer()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockModelAdapter(device="cpu")
    
    def test_initialization(self, batch_optimizer):
        """Test batch optimizer initialization."""
        assert isinstance(batch_optimizer.optimal_batch_sizes, dict)
        assert isinstance(batch_optimizer.batch_profiles, dict)
    
    def test_find_optimal_batch_size(self, batch_optimizer, mock_model):
        """Test optimal batch size detection."""
        optimal_size = batch_optimizer.find_optimal_batch_size(
            mock_model,
            sample_prompt="test prompt",
            max_batch_size=4,
            memory_limit_mb=1000
        )
        
        assert isinstance(optimal_size, int)
        assert optimal_size >= 1
        assert optimal_size <= 4
        
        # Should cache the result
        assert mock_model.name in batch_optimizer.optimal_batch_sizes
    
    def test_optimize_batch_processing(self, batch_optimizer, mock_model):
        """Test batch processing optimization."""
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]
        
        # Set optimal batch size
        batch_optimizer.optimal_batch_sizes[mock_model.name] = 2
        
        results = batch_optimizer.optimize_batch_processing(mock_model, prompts)
        
        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, torch.Tensor)


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler instance."""
        return PerformanceProfiler()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockModelAdapter(device="cpu")
    
    def test_initialization(self, profiler):
        """Test profiler initialization."""
        assert isinstance(profiler.profiling_data, dict)
        assert profiler.memory_manager is not None
    
    def test_profile_model_performance(self, profiler, mock_model):
        """Test comprehensive model performance profiling."""
        test_prompts = ["prompt1", "prompt2"]
        profile = OptimizationProfile(compile_model=False)  # Disable compilation for testing
        
        profile_data = profiler.profile_model_performance(
            mock_model, test_prompts, profile, detailed=False
        )
        
        assert "model_name" in profile_data
        assert "optimization_profile" in profile_data
        assert "basic_metrics" in profile_data
        assert "memory_metrics" in profile_data
        assert "throughput_metrics" in profile_data
        
        # Check basic metrics structure
        basic_metrics = profile_data["basic_metrics"]
        assert "avg_latency" in basic_metrics
        assert "std_latency" in basic_metrics
        assert basic_metrics["avg_latency"] > 0
    
    def test_compare_optimization_strategies(self, profiler, mock_model):
        """Test optimization strategy comparison."""
        strategies = [
            OptimizationProfile(precision="fp32", compile_model=False),
            OptimizationProfile(precision="fp16", compile_model=False)
        ]
        test_prompts = ["prompt1"]
        
        comparison = profiler.compare_optimization_strategies(
            mock_model, strategies, test_prompts
        )
        
        assert len(comparison) == len(strategies)
        for strategy_name, data in comparison.items():
            if "error" not in data:
                assert "basic_metrics" in data
                assert "memory_metrics" in data
    
    def test_profile_basic_metrics(self, profiler, mock_model):
        """Test basic metrics profiling."""
        prompts = ["test prompt"]
        
        metrics = profiler._profile_basic_metrics(mock_model, prompts)
        
        assert "avg_latency" in metrics
        assert "std_latency" in metrics
        assert "min_latency" in metrics
        assert "max_latency" in metrics
        assert "p95_latency" in metrics
        assert "p99_latency" in metrics
        
        # All latencies should be positive
        assert metrics["avg_latency"] > 0
        assert metrics["min_latency"] >= 0
    
    def test_profile_memory_usage(self, profiler, mock_model):
        """Test memory usage profiling."""
        prompts = ["prompt1", "prompt2"]
        
        memory_metrics = profiler._profile_memory_usage(mock_model, prompts)
        
        assert "individual_profiles" in memory_metrics
        assert "aggregated_metrics" in memory_metrics
        assert len(memory_metrics["individual_profiles"]) > 0
    
    def test_profile_throughput(self, profiler, mock_model):
        """Test throughput profiling."""
        prompts = ["prompt1", "prompt2"]
        
        throughput_metrics = profiler._profile_throughput(mock_model, prompts)
        
        assert "single_sample_throughput" in throughput_metrics
        assert "batch_throughput" in throughput_metrics
        assert "throughput_improvement" in throughput_metrics
        
        # Throughput should be positive
        assert throughput_metrics["single_sample_throughput"] > 0
    
    def test_calculate_relative_improvement(self, profiler):
        """Test relative improvement calculation."""
        baseline = {
            "basic_metrics": {"avg_latency": 100.0},
            "memory_metrics": {"aggregated_metrics": {"avg_peak_memory": 1000.0}},
            "throughput_metrics": {"single_sample_throughput": 5.0}
        }
        
        comparison = {
            "basic_metrics": {"avg_latency": 80.0},
            "memory_metrics": {"aggregated_metrics": {"avg_peak_memory": 800.0}},
            "throughput_metrics": {"single_sample_throughput": 6.0}
        }
        
        improvements = profiler._calculate_relative_improvement(baseline, comparison)
        
        assert "latency_improvement" in improvements
        assert "memory_improvement" in improvements
        assert "throughput_improvement" in improvements
        
        # Should show improvement
        assert improvements["latency_improvement"] > 0  # Faster
        assert improvements["memory_improvement"] > 0  # Less memory
        assert improvements["throughput_improvement"] > 0  # Higher throughput


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockModelAdapter(device="cpu")
    
    def test_optimize_model_for_benchmarking(self, mock_model):
        """Test convenience function for model optimization."""
        optimized_model = optimize_model_for_benchmarking(mock_model, optimization_level="balanced")
        
        assert optimized_model is not None
        assert isinstance(optimized_model, ModelAdapter)
    
    def test_optimize_model_invalid_level(self, mock_model):
        """Test optimization with invalid level (should use balanced)."""
        optimized_model = optimize_model_for_benchmarking(mock_model, optimization_level="invalid")
        
        assert optimized_model is not None
    
    @patch('src.vid_diffusion_bench.performance_optimization.PerformanceProfiler')
    def test_benchmark_optimization_impact(self, mock_profiler_class, mock_model):
        """Test optimization impact benchmarking."""
        mock_profiler = Mock()
        mock_profiler.compare_optimization_strategies = Mock(return_value={"strategy_0": {"avg_latency": 100}})
        mock_profiler_class.return_value = mock_profiler
        
        results = benchmark_optimization_impact(mock_model)
        
        assert isinstance(results, dict)
        assert mock_profiler.compare_optimization_strategies.called


class TestPerformanceOptimizationIntegration:
    """Integration tests for performance optimization."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockModelAdapter(device="cpu", name="integration_test_model")
    
    def test_full_optimization_pipeline(self, mock_model):
        """Test complete optimization pipeline."""
        # Create optimizer with full profile
        profile = create_high_performance_profile()
        profile.compile_model = False  # Disable for testing
        profile.tensorrt_optimization = False  # Disable for testing
        
        optimizer = ModelOptimizer(profile)
        
        # Optimize model
        optimized_model = optimizer.optimize_model(mock_model)
        
        # Verify optimization
        assert optimized_model is not None
        assert optimized_model.name == mock_model.name
        
        # Test generation still works
        result = optimized_model.generate("test prompt", num_frames=8)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 8  # num_frames
    
    def test_memory_optimization_integration(self, mock_model):
        """Test memory optimization integration."""
        memory_manager = MemoryManager(device="cpu")
        
        with memory_manager.managed_memory(reserve_memory=50):
            # Profile memory usage during generation
            profile = memory_manager.profile_memory_usage(
                mock_model.generate, "test prompt", num_frames=8
            )
            
            assert isinstance(profile, MemoryProfile)
            assert profile.initial_memory >= 0
    
    def test_batch_optimization_integration(self, mock_model):
        """Test batch optimization integration."""
        batch_optimizer = BatchOptimizer()
        
        # Find optimal batch size
        optimal_size = batch_optimizer.find_optimal_batch_size(
            mock_model, max_batch_size=4, memory_limit_mb=500
        )
        
        # Use optimal batch size for processing
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]
        results = batch_optimizer.optimize_batch_processing(mock_model, prompts, optimal_size)
        
        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, torch.Tensor)


class TestPerformanceOptimizationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_memory_manager_cpu_device(self):
        """Test memory manager with CPU device."""
        manager = MemoryManager(device="cpu")
        
        # Should not crash with CPU device
        with manager.managed_memory():
            pass
        
        assert manager._get_memory_usage() == 0  # CPU should return 0
    
    def test_optimizer_with_none_model(self):
        """Test optimizer behavior with None model."""
        profile = OptimizationProfile()
        optimizer = ModelOptimizer(profile)
        
        # Should handle None gracefully
        try:
            result = optimizer._get_optimization_key(None, profile)
            # If it doesn't crash, check result
            assert isinstance(result, str)
        except AttributeError:
            # Expected behavior - accessing .name on None
            pass
    
    def test_profiler_with_empty_prompts(self):
        """Test profiler with empty prompt list."""
        profiler = PerformanceProfiler()
        mock_model = MockModelAdapter()
        
        metrics = profiler._profile_basic_metrics(mock_model, [])
        
        # Should return empty or default metrics
        assert isinstance(metrics, dict)
    
    def test_batch_optimizer_memory_limit(self):
        """Test batch optimizer with very low memory limit."""
        batch_optimizer = BatchOptimizer()
        mock_model = MockModelAdapter()
        
        # Very low memory limit should result in batch size 1
        optimal_size = batch_optimizer.find_optimal_batch_size(
            mock_model, memory_limit_mb=1  # Very low limit
        )
        
        assert optimal_size >= 1  # Should always return at least 1


if __name__ == "__main__":
    pytest.main([__file__])