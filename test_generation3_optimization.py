#!/usr/bin/env python3
"""Generation 3 optimization test - Advanced performance verification."""

import sys
import os
import logging
import time
import asyncio
import threading
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_intelligent_caching():
    """Test intelligent caching functionality."""
    try:
        from vid_diffusion_bench.generation3_optimization import IntelligentCaching
        
        cache = IntelligentCaching(max_size_gb=0.001)  # Very small cache for testing
        
        # Test basic operations
        cache.set("key1", {"data": "value1", "size": 100})
        cache.set("key2", {"data": "value2", "size": 200})
        
        result1 = cache.get("key1")
        assert result1 is not None
        assert result1["data"] == "value1"
        
        # Test cache eviction
        cache.set("key3", "large_value" * 1000)  # Should trigger eviction
        
        # Get cache stats
        stats = cache.get_stats()
        assert "hit_rate_percent" in stats
        assert "total_items" in stats
        assert stats["total_items"] >= 1
        
        print(f"✅ Intelligent caching: {stats['total_items']} items, {stats['hit_rate_percent']:.1f}% hit rate")
        return True
        
    except Exception as e:
        print(f"❌ Intelligent caching test failed: {e}")
        return False

def test_async_executor():
    """Test async benchmark executor."""
    try:
        from vid_diffusion_bench.generation3_optimization import AsyncBenchmarkExecutor
        
        executor = AsyncBenchmarkExecutor(max_concurrent=2)
        
        async def test_async():
            # Start scheduler
            await executor.start_scheduler()
            
            # Submit test tasks
            def test_task(task_id, duration=0.1):
                time.sleep(duration)
                return f"Task {task_id} completed"
            
            task_ids = []
            for i in range(3):
                task_id = await executor.submit_task(
                    test_task, i, duration=0.1, priority=0.5, estimated_duration=0.1
                )
                task_ids.append(task_id)
            
            # Get results
            results = []
            for task_id in task_ids:
                result = await executor.get_result(task_id, timeout=5.0)
                results.append(result)
            
            executor.stop_scheduler()
            return results
        
        results = asyncio.run(test_async())
        
        assert len(results) == 3
        for result in results:
            assert result['success']
            assert "Task" in result['result']
        
        print("✅ Async executor functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Async executor test failed: {e}")
        return False

def test_model_memory_pool():
    """Test model memory pooling."""
    try:
        from vid_diffusion_bench.generation3_optimization import ModelMemoryPool
        
        pool = ModelMemoryPool(max_pool_size_gb=0.001)  # Very small for testing
        
        # Mock model class
        class MockModel:
            def __init__(self, name):
                self.name = name
                self.data = [1] * 1000  # Some data to give it size
            
            def __sizeof__(self):
                return len(self.data) * 8  # Rough size estimation
        
        # Test model loading and pooling
        def load_model_a():
            return MockModel("model_a")
        
        def load_model_b():
            return MockModel("model_b")
        
        # Get model A (should load)
        model_a1 = pool.get_model("model_a", load_model_a)
        assert model_a1.name == "model_a"
        
        # Get model A again (should come from pool)
        model_a2 = pool.get_model("model_a", load_model_a)
        assert model_a1 is model_a2  # Same object reference
        
        # Get model B (may trigger eviction)
        model_b = pool.get_model("model_b", load_model_b)
        assert model_b.name == "model_b"
        
        # Get pool stats
        stats = pool.get_pool_stats()
        assert "total_models" in stats
        assert stats["total_models"] >= 1
        
        print(f"✅ Model memory pool: {stats['total_models']} models, {stats['total_size_gb']:.4f}GB")
        return True
        
    except Exception as e:
        print(f"❌ Model memory pool test failed: {e}")
        return False

def test_batch_optimizer():
    """Test batch optimization."""
    try:
        from vid_diffusion_bench.generation3_optimization import BatchOptimizer
        
        optimizer = BatchOptimizer(target_latency_ms=2000.0)
        
        # Test batch size calculation
        batch_size = optimizer.optimize_batch_size(
            model_name="test_model",
            data_size=100,
            available_memory_gb=8.0
        )
        
        assert batch_size > 0
        assert batch_size <= 100
        
        # Test performance updates
        optimizer.update_performance("test_model", batch_size, 1500.0, 0.95)
        optimizer.update_performance("test_model", batch_size + 1, 2500.0, 0.90)
        
        # Get optimized batch size (should learn from performance)
        new_batch_size = optimizer.optimize_batch_size("test_model", 100, 8.0)
        
        print(f"✅ Batch optimizer: initial={batch_size}, optimized={new_batch_size}")
        return True
        
    except Exception as e:
        print(f"❌ Batch optimizer test failed: {e}")
        return False

def test_performance_profiler():
    """Test performance profiler."""
    try:
        from vid_diffusion_bench.generation3_optimization import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        # Test profiling
        with profiler.profile("test_operation", {"param": "value"}):
            time.sleep(0.1)  # Simulate work
            data = [i for i in range(1000)]  # Use some memory
        
        with profiler.profile("test_operation"):
            time.sleep(0.05)  # Different timing
        
        # Get performance summary
        summary = profiler.get_performance_summary()
        
        assert "test_operation" in summary
        op_stats = summary["test_operation"]
        assert op_stats["call_count"] == 2
        assert op_stats["avg_duration_ms"] > 0
        assert "bottlenecks" in summary
        
        print(f"✅ Performance profiler: {op_stats['call_count']} calls, {op_stats['avg_duration_ms']:.1f}ms avg")
        return True
        
    except Exception as e:
        print(f"❌ Performance profiler test failed: {e}")
        return False

def test_gpu_optimization():
    """Test GPU optimization utilities."""
    try:
        from vid_diffusion_bench.generation3_optimization import optimize_gpu_memory, warm_up_gpu
        
        # These should not fail even without GPU
        optimize_gpu_memory()
        warm_up_gpu()
        
        print("✅ GPU optimization utilities executed")
        return True
        
    except Exception as e:
        print(f"❌ GPU optimization test failed: {e}")
        return False

def test_optimization_config():
    """Test optimization configuration."""
    try:
        from vid_diffusion_bench.generation3_optimization import OptimizationConfig, PerformanceMetrics
        
        # Test optimization config
        config = OptimizationConfig(
            enable_gpu_optimization=True,
            max_concurrent_models=4,
            optimization_level="aggressive"
        )
        
        assert config.enable_gpu_optimization
        assert config.max_concurrent_models == 4
        assert config.optimization_level == "aggressive"
        
        # Test performance metrics
        metrics = PerformanceMetrics(
            latency_p50=100.0,
            throughput_videos_per_second=2.5,
            gpu_utilization_percent=85.0
        )
        
        assert metrics.latency_p50 == 100.0
        assert metrics.throughput_videos_per_second == 2.5
        
        print("✅ Optimization configuration and metrics work")
        return True
        
    except Exception as e:
        print(f"❌ Optimization config test failed: {e}")
        return False

def test_generation3_integration():
    """Test Generation 3 integration smoke test."""
    try:
        # Test that we can import and initialize key components
        from vid_diffusion_bench.generation3_optimization import (
            IntelligentCaching, AsyncBenchmarkExecutor, ModelMemoryPool
        )
        
        # Test basic initialization
        cache = IntelligentCaching(max_size_gb=1.0)
        executor = AsyncBenchmarkExecutor(max_concurrent=2)
        pool = ModelMemoryPool(max_pool_size_gb=1.0)
        
        # Test they have expected methods
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'set')
        assert hasattr(executor, 'submit_task')
        assert hasattr(pool, 'get_model')
        
        print("✅ Generation 3 integration smoke test passed")
        return True
        
    except Exception as e:
        print(f"❌ Generation 3 integration test failed: {e}")
        return False

def main():
    """Run all Generation 3 optimization tests."""
    print("🚀 Running Generation 3 optimization tests...")
    print()
    
    tests = [
        test_intelligent_caching,
        test_async_executor, 
        test_model_memory_pool,
        test_batch_optimizer,
        test_performance_profiler,
        test_gpu_optimization,
        test_optimization_config,
        test_generation3_integration
    ]
    
    passed = 0
    for test in tests:
        print(f"Running {test.__name__}...")
        if test():
            passed += 1
        print()
    
    print(f"📊 Generation 3 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Generation 3: MAKE IT SCALE - COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("❌ Some Generation 3 tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)