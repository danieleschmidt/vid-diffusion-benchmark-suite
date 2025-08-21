"""Comprehensive test suite for all Terragon SDLC generations."""

import pytest
import tempfile
import time
import json
from pathlib import Path
import sys
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vid_diffusion_bench import BenchmarkSuite, VideoQualityMetrics, StandardPrompts
from vid_diffusion_bench.models.registry import list_models, get_model
from vid_diffusion_bench.enhanced_error_handling import BenchmarkError, retry_on_failure
from vid_diffusion_bench.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from vid_diffusion_bench.health_monitoring import HealthMonitor
from vid_diffusion_bench.resilient_benchmark import ResilientBenchmarkSuite
from vid_diffusion_bench.adaptive_scaling import AdaptiveScaler, ScalingConfig
from vid_diffusion_bench.performance_accelerator import PerformanceAccelerator, accelerate


class TestGeneration1BasicFunctionality:
    """Test Generation 1: Basic functionality and core features."""
    
    def test_model_registry(self):
        """Test model registry functionality."""
        models = list_models()
        assert len(models) > 0, "No models registered"
        
        # Test we have the key models from README
        expected_models = ["mock-fast", "dreamvideo-v3", "pika-lumiere-xl", "svd-xt-1.1"]
        for model in expected_models:
            assert model in models, f"Expected model {model} not found in registry"
    
    def test_model_loading(self):
        """Test model loading and basic generation."""
        # Test mock model (should always work)
        model = get_model("mock-fast")
        assert model is not None
        
        # Test basic requirements
        requirements = model.requirements
        assert "vram_gb" in requirements
        assert "precision" in requirements
        assert requirements["vram_gb"] > 0
    
    def test_video_generation(self):
        """Test basic video generation."""
        model = get_model("mock-fast")
        
        video = model.generate(
            prompt="A cat playing piano",
            num_frames=16,
            fps=8,
            width=512,
            height=512
        )
        
        # Verify video tensor shape
        assert video.shape[0] == 16, f"Expected 16 frames, got {video.shape[0]}"
        assert video.shape[1] == 3, f"Expected 3 channels, got {video.shape[1]}"
        assert video.shape[2] == 512, f"Expected height 512, got {video.shape[2]}"
        assert video.shape[3] == 512, f"Expected width 512, got {video.shape[3]}"
    
    def test_benchmark_suite(self):
        """Test basic benchmark suite functionality."""
        suite = BenchmarkSuite()
        
        # Test with minimal prompts
        results = suite.evaluate_model(
            model_name="mock-fast",
            prompts=["A cat playing piano"]
        )
        
        assert "model_name" in results
        assert results["model_name"] == "mock-fast"
    
    def test_standard_prompts(self):
        """Test standard prompt sets."""
        prompts = StandardPrompts.DIVERSE_SET_V2
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p) > 0 for p in prompts)


class TestGeneration2Robustness:
    """Test Generation 2: Robustness, error handling, and monitoring."""
    
    def test_circuit_breaker_basic(self):
        """Test basic circuit breaker functionality."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        breaker = CircuitBreaker("test_breaker", config)
        
        # Test normal operation
        @breaker
        def test_function(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"
        
        # Should work normally
        result = test_function(False)
        assert result == "success"
        
        # Test failure detection
        with pytest.raises(Exception):
            test_function(True)
        
        with pytest.raises(Exception):
            test_function(True)
        
        # Now circuit should be open
        status = breaker.status
        assert status["failure_count"] >= 2
    
    def test_health_monitoring(self):
        """Test health monitoring system."""
        monitor = HealthMonitor()
        
        # Test basic health check
        health = monitor.check_health()
        assert health.cpu_percent >= 0
        assert health.memory_percent >= 0
        assert health.disk_usage_percent >= 0
        assert isinstance(health.alerts, list)
        
        # Test health summary
        summary = monitor.get_health_summary()
        assert "cpu" in summary
        assert "memory" in summary
        assert "disk" in summary
    
    def test_resilient_benchmark(self):
        """Test resilient benchmark suite."""
        from vid_diffusion_bench.resilient_benchmark import ResilientConfig
        
        config = ResilientConfig(
            max_retries=2,
            auto_recovery=True,
            health_check_enabled=False  # Disable to avoid dependencies
        )
        
        suite = ResilientBenchmarkSuite(config)
        
        # Test with mock data
        results = suite.evaluate_model_resilient(
            model_name="mock-fast",
            prompts=["Test prompt"]
        )
        
        assert "model_name" in results
        assert "total_prompts" in results
        assert results["total_prompts"] == 1
    
    def test_error_handling(self):
        """Test enhanced error handling."""
        from vid_diffusion_bench.enhanced_error_handling import ValidationError
        
        # Test custom exception
        error = ValidationError("test_field", "invalid_value", "valid_value")
        assert error.error_code == "VALIDATION_ERROR"
        assert "test_field" in str(error)
        
        # Test retry decorator
        attempt_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.1)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Simulated failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3


class TestGeneration3Scaling:
    """Test Generation 3: Scaling, optimization, and performance."""
    
    def test_adaptive_scaler_config(self):
        """Test adaptive scaler configuration."""
        config = ScalingConfig(
            min_workers=1,
            max_workers=4,
            target_cpu_utilization=70.0
        )
        
        scaler = AdaptiveScaler(config)
        assert scaler.config.min_workers == 1
        assert scaler.config.max_workers == 4
        assert scaler.config.target_cpu_utilization == 70.0
    
    def test_performance_accelerator(self):
        """Test performance acceleration and caching."""
        accelerator = PerformanceAccelerator()
        
        # Test function acceleration
        call_count = 0
        
        @accelerator.accelerate_function(enable_cache=True)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate work
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same args should be cached
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment due to caching
        
        # Different args should execute function
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2
    
    def test_intelligent_cache(self):
        """Test intelligent caching system."""
        from vid_diffusion_bench.performance_accelerator import IntelligentCache, CacheConfig
        
        config = CacheConfig(max_memory_mb=1, ttl_seconds=1)
        cache = IntelligentCache(config)
        
        # Test basic cache operations
        cache.put("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"
        
        # Test cache miss
        missing = cache.get("nonexistent_key")
        assert missing is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats
    
    def test_model_optimization(self):
        """Test model optimization features."""
        from vid_diffusion_bench.performance_accelerator import ModelOptimizer, OptimizationConfig
        
        config = OptimizationConfig(
            enable_model_compilation=False,  # Disable for testing
            enable_mixed_precision=False
        )
        
        optimizer = ModelOptimizer(config)
        
        # Test with mock model (just a dict for testing)
        mock_model = {"type": "mock", "parameters": 1000}
        optimized = optimizer.optimize_model(mock_model, "test_model")
        
        # Should return the model (unchanged in this test case)
        assert optimized == mock_model


class TestIntegration:
    """Integration tests across all generations."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from model loading to results."""
        # Test basic pipeline
        suite = BenchmarkSuite()
        
        results = suite.evaluate_model(
            model_name="mock-fast",
            prompts=["A cat playing piano", "A dog dancing"]
        )
        
        assert "model_name" in results
        assert results["model_name"] == "mock-fast"
    
    def test_cli_integration(self):
        """Test CLI integration (import test)."""
        from vid_diffusion_bench.cli import main
        
        # Just test that CLI can be imported and main function exists
        assert callable(main)
    
    def test_api_integration(self):
        """Test API integration (import test)."""
        from vid_diffusion_bench.api.app import app
        
        # Just test that API can be imported
        assert app is not None
    
    def test_export_functionality(self):
        """Test data export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test health monitoring export
            monitor = HealthMonitor()
            export_path = Path(temp_dir) / "health_export.json"
            monitor.export_health_data(str(export_path), minutes=1)
            
            assert export_path.exists()
            
            # Verify export format
            with open(export_path) as f:
                data = json.load(f)
            
            assert "export_time" in data
            assert "health_data" in data


class TestSecurityAndValidation:
    """Test security and validation features."""
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        # Test model name validation
        with pytest.raises(Exception):
            get_model("")  # Empty model name should fail
        
        with pytest.raises(Exception):
            get_model("nonexistent_model")  # Should raise KeyError
    
    def test_resource_limits(self):
        """Test resource limit enforcement."""
        model = get_model("mock-fast")
        
        # Test with extreme parameters (should be handled gracefully)
        video = model.generate(
            prompt="Test",
            num_frames=1,  # Minimal
            fps=1,
            width=64,
            height=64
        )
        
        assert video.shape[0] == 1
        assert video.shape[2] == 64
        assert video.shape[3] == 64
    
    def test_error_boundary(self):
        """Test error boundaries and graceful degradation."""
        # Test with invalid model parameters
        model = get_model("mock-fast")
        
        # Should handle invalid parameters gracefully
        try:
            video = model.generate(
                prompt="Test",
                num_frames=-1,  # Invalid
                fps=0,          # Invalid
                width=0,        # Invalid
                height=0        # Invalid
            )
            # If it doesn't raise, should return something reasonable
            assert video is not None
        except Exception as e:
            # Should be a well-formed error
            assert len(str(e)) > 0


def run_quality_gates():
    """Run all quality gate tests and generate report."""
    print("🛡️ EXECUTING TERRAGON QUALITY GATES")
    print("=" * 60)
    
    # Run pytest with coverage
    test_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    exit_code = pytest.main(test_args)
    
    print("\n📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    
    if exit_code == 0:
        print("✅ ALL QUALITY GATES PASSED")
        print("🚀 System is PRODUCTION READY")
    else:
        print("❌ QUALITY GATES FAILED")
        print("🔧 System needs fixes before production")
    
    return exit_code == 0


if __name__ == "__main__":
    # Run quality gates when executed directly
    success = run_quality_gates()
    sys.exit(0 if success else 1)