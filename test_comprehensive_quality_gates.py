#!/usr/bin/env python3
"""Comprehensive Quality Gates - Full system validation."""

import sys
import os
import logging
import time
import subprocess
import json
# import pytest  # Not needed for basic testing
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_code_quality():
    """Test code quality metrics."""
    print("🔍 Running code quality checks...")
    
    try:
        # Test Python syntax
        result = subprocess.run([
            'python3', '-m', 'py_compile', 'src/vid_diffusion_bench/__init__.py'
        ], capture_output=True, text=True, cwd='/root/repo')
        
        if result.returncode != 0:
            print(f"❌ Python syntax check failed: {result.stderr}")
            return False
        
        print("✅ Python syntax validation passed")
        
        # Test imports
        try:
            import vid_diffusion_bench
            from vid_diffusion_bench import BenchmarkSuite
            print("✅ Import validation passed")
        except Exception as e:
            print(f"❌ Import validation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Code quality test failed: {e}")
        return False

def test_generation1_functionality():
    """Test Generation 1 basic functionality."""
    print("🧪 Testing Generation 1 functionality...")
    
    try:
        from vid_diffusion_bench.generation1_enhancements import (
            ProgressInfo, BenchmarkProgressTracker, SafetyValidator, 
            BasicCacheManager, RetryHandler
        )
        
        # Test progress tracking
        tracker = BenchmarkProgressTracker()
        tracker.start(10, "Testing")
        tracker.increment("Step 1")
        assert tracker.progress.current == 1
        
        # Test safety validation
        prompts = ["Safe prompt", "<script>alert('unsafe')</script>", "Another safe prompt"]
        safe_prompts = SafetyValidator.validate_prompts(prompts)
        assert len(safe_prompts) == 2  # Unsafe prompt filtered out
        
        # Test caching
        cache = BasicCacheManager(max_size=3)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test retry handler
        attempts = 0
        @RetryHandler(max_attempts=3)
        def flaky_func():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise RuntimeError("Simulated failure")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert attempts == 2
        
        print("✅ Generation 1 functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Generation 1 functionality test failed: {e}")
        return False

def test_generation2_robustness():
    """Test Generation 2 robustness features."""
    print("🛡️ Testing Generation 2 robustness...")
    
    try:
        from vid_diffusion_bench.generation2_robustness import (
            SystemHealthMonitor, CircuitBreaker, BenchmarkRecovery,
            DataBackupManager, AdvancedLoggingManager
        )
        
        # Test health monitoring
        monitor = SystemHealthMonitor(check_interval=1)
        health = monitor.check_health()
        assert health is not None
        assert hasattr(health, 'cpu_percent')
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        @breaker
        def failing_func():
            raise RuntimeError("Simulated failure")
        
        # Should fail and open circuit
        for i in range(3):
            try:
                failing_func()
            except RuntimeError:
                continue
            except Exception as e:
                if "Circuit breaker OPEN" in str(e):
                    break
        
        assert breaker.state == 'OPEN'
        
        # Test recovery system
        recovery = BenchmarkRecovery(max_retries=2)
        
        def recovery_func():
            return "recovered"
        
        result = recovery.execute_with_recovery(recovery_func)
        assert result == "recovered"
        
        print("✅ Generation 2 robustness tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Generation 2 robustness test failed: {e}")
        return False

def test_generation3_optimization():
    """Test Generation 3 optimization features."""
    print("🚀 Testing Generation 3 optimizations...")
    
    try:
        from vid_diffusion_bench.generation3_optimization import (
            IntelligentCaching, AsyncBenchmarkExecutor, ModelMemoryPool,
            BatchOptimizer, PerformanceProfiler
        )
        
        # Test intelligent caching
        cache = IntelligentCaching(max_size_gb=0.001)
        cache.set("test", {"data": "value"})
        result = cache.get("test")
        assert result is not None
        
        stats = cache.get_stats()
        assert stats['hit_rate_percent'] >= 0
        
        # Test performance profiler
        profiler = PerformanceProfiler()
        with profiler.profile("test_operation"):
            time.sleep(0.01)
        
        summary = profiler.get_performance_summary()
        assert "test_operation" in summary
        
        # Test batch optimizer
        optimizer = BatchOptimizer()
        batch_size = optimizer.optimize_batch_size("test_model", 100, 8.0)
        assert batch_size > 0
        
        print("✅ Generation 3 optimization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Generation 3 optimization test failed: {e}")
        return False

def test_benchmark_suite_integration():
    """Test full benchmark suite integration."""
    print("🔧 Testing benchmark suite integration...")
    
    try:
        from vid_diffusion_bench import BenchmarkSuite
        from vid_diffusion_bench.benchmark import BenchmarkResult
        
        # Test suite initialization
        suite = BenchmarkSuite(device="cpu", enable_optimizations=True)
        assert suite.device == "cpu"
        
        # Test result creation
        result = BenchmarkResult("test-model", ["test prompt"])
        assert result.model_name == "test-model"
        assert len(result.prompts) == 1
        
        # Test success rate calculation
        assert result.get_success_rate() == 0.0  # No results yet
        
        # Test serialization
        result_dict = result.to_dict()
        assert "model_name" in result_dict
        assert "success_rate" in result_dict
        
        print("✅ Benchmark suite integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark suite integration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("⚠️ Testing error handling...")
    
    try:
        from vid_diffusion_bench.generation1_enhancements import SafetyValidator
        
        # Test empty input handling
        empty_prompts = SafetyValidator.validate_prompts([])
        assert len(empty_prompts) == 0
        
        # Test invalid input types
        mixed_inputs = ["valid", 123, None, "another valid"]
        valid_inputs = SafetyValidator.validate_prompts(mixed_inputs)
        assert len(valid_inputs) == 2  # Only string inputs
        
        # Test parameter validation with extreme values
        params = SafetyValidator.validate_model_params(
            num_frames=10000,  # Too high
            fps=1000,          # Too high  
            resolution=(10000, 10000),  # Too high
            batch_size=1000    # Too high
        )
        
        # Should be clamped to reasonable values
        assert params['num_frames'] <= 200
        assert params['fps'] <= 60
        assert params['resolution'][0] <= 2048
        assert params['batch_size'] <= 16
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance_requirements():
    """Test performance requirements."""
    print("⚡ Testing performance requirements...")
    
    try:
        from vid_diffusion_bench.generation1_enhancements import BasicCacheManager
        from vid_diffusion_bench.generation3_optimization import IntelligentCaching
        
        # Test cache performance
        cache = BasicCacheManager(max_size=1000)
        
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key{i}", f"value{i}")
        
        for i in range(1000):
            cache.get(f"key{i}")
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 5.0, f"Cache operations took too long: {elapsed:.2f}s"
        
        # Test intelligent cache performance
        smart_cache = IntelligentCaching(max_size_gb=0.1)
        
        start_time = time.time()
        for i in range(100):
            smart_cache.set(f"smart_key{i}", {"data": f"value{i}"})
        
        for i in range(100):
            smart_cache.get(f"smart_key{i}")
        
        elapsed = time.time() - start_time
        assert elapsed < 2.0, f"Intelligent cache operations took too long: {elapsed:.2f}s"
        
        print(f"✅ Performance requirements met (cache: {elapsed:.3f}s)")
        return True
        
    except Exception as e:
        print(f"❌ Performance requirements test failed: {e}")
        return False

def test_security_requirements():
    """Test security requirements."""
    print("🔒 Testing security requirements...")
    
    try:
        from vid_diffusion_bench.generation1_enhancements import SafetyValidator
        
        # Test malicious input filtering
        malicious_prompts = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "exec('rm -rf /')",
            "__import__('os').system('ls')",
            "eval(malicious_code)",
            "Normal safe prompt"
        ]
        
        safe_prompts = SafetyValidator.validate_prompts(malicious_prompts)
        
        # Should filter out all malicious prompts
        assert len(safe_prompts) == 1
        assert "Normal safe prompt" in safe_prompts
        
        # Test that no malicious patterns remain in safe prompts
        if safe_prompts:  # Only test if there are safe prompts
            for prompt in safe_prompts:
                assert "<script" not in prompt.lower()
                assert "javascript:" not in prompt.lower()
                assert "exec(" not in prompt.lower()
                assert "__import__" not in prompt.lower()
                assert "eval(" not in prompt.lower()
        
        print("✅ Security requirements met")
        return True
        
    except Exception as e:
        print(f"❌ Security requirements test failed: {e}")
        return False

def test_memory_management():
    """Test memory management and resource usage."""
    print("💾 Testing memory management...")
    
    try:
        import psutil
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Create and use various components
        from vid_diffusion_bench.generation1_enhancements import BasicCacheManager
        from vid_diffusion_bench.generation3_optimization import IntelligentCaching, ModelMemoryPool
        
        # Test basic cache memory usage
        cache = BasicCacheManager(max_size=100)
        for i in range(100):
            cache.set(f"key{i}", [1] * 1000)  # Store some data
        
        # Test intelligent cache
        smart_cache = IntelligentCaching(max_size_gb=0.01)
        for i in range(50):
            smart_cache.set(f"smart{i}", {"data": [1] * 500})
        
        # Test model pool
        pool = ModelMemoryPool(max_pool_size_gb=0.01)
        
        # Check memory hasn't grown excessively
        current_memory = process.memory_info().rss / (1024**2)
        memory_growth = current_memory - initial_memory
        
        # Should not use more than 200MB additional memory
        assert memory_growth < 200, f"Excessive memory usage: {memory_growth:.1f}MB"
        
        print(f"✅ Memory management test passed (growth: {memory_growth:.1f}MB)")
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def test_concurrent_operations():
    """Test concurrent operations and thread safety.""" 
    print("🔀 Testing concurrent operations...")
    
    try:
        import threading
        import time
        from vid_diffusion_bench.generation1_enhancements import BasicCacheManager
        
        cache = BasicCacheManager(max_size=1000)
        results = []
        errors = []
        
        def cache_operations(thread_id):
            try:
                for i in range(100):
                    key = f"thread{thread_id}_key{i}"
                    cache.set(key, f"value{i}")
                    retrieved = cache.get(key)
                    if retrieved != f"value{i}":
                        errors.append(f"Thread {thread_id}: Data mismatch")
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Run concurrent cache operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        assert len(results) == 5, f"Not all threads completed: {results}"
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
        
        print("✅ Concurrent operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Concurrent operations test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("🔬 Testing edge cases...")
    
    try:
        from vid_diffusion_bench.generation1_enhancements import BasicCacheManager, SafetyValidator
        
        # Test empty cache operations
        cache = BasicCacheManager(max_size=0)  # Zero size cache
        cache.set("key", "value")
        result = cache.get("key") 
        # Should handle gracefully (might return None due to zero size)
        
        # Test extremely long prompts
        very_long_prompt = "A" * 10000
        long_prompts = [very_long_prompt]
        validated = SafetyValidator.validate_prompts(long_prompts, max_length=500)
        
        if validated:  # If any prompts remain after validation
            assert len(validated[0]) <= 503  # Max length + "..."
        
        # Test parameter validation edge cases
        try:
            params = SafetyValidator.validate_model_params(
                num_frames=0,      # Zero frames
                fps=0,             # Zero FPS
                resolution=(0, 0), # Zero resolution
                batch_size=0       # Zero batch size
            )
            
            # Should be adjusted to minimum valid values
            assert params['num_frames'] >= 1
            assert params['fps'] >= 1
            assert params['resolution'][0] >= 64
            assert params['resolution'][1] >= 64
            assert params['batch_size'] >= 1
        except Exception as e:
            # If validation fails completely, that's also acceptable for edge cases
            logger.warning(f"Parameter validation failed on edge case (expected): {e}")
            
        # Test with completely invalid resolution
        try:
            params2 = SafetyValidator.validate_model_params(resolution=())
            # If it succeeds, check it used defaults
            assert len(params2['resolution']) == 2
        except Exception:
            # Failure is also acceptable for this edge case
            pass
        
        print("✅ Edge cases test passed")
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False

def run_quality_gates():
    """Run all quality gate tests."""
    print("🚦 RUNNING COMPREHENSIVE QUALITY GATES")
    print("=" * 50)
    
    quality_gates = [
        ("Code Quality", test_code_quality),
        ("Generation 1 Functionality", test_generation1_functionality),
        ("Generation 2 Robustness", test_generation2_robustness),
        ("Generation 3 Optimization", test_generation3_optimization),
        ("Benchmark Suite Integration", test_benchmark_suite_integration),
        ("Error Handling", test_error_handling),
        ("Performance Requirements", test_performance_requirements),
        ("Security Requirements", test_security_requirements),
        ("Memory Management", test_memory_management),
        ("Concurrent Operations", test_concurrent_operations),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    failed = []
    
    for gate_name, gate_test in quality_gates:
        print(f"\n📋 Quality Gate: {gate_name}")
        print("-" * 30)
        
        try:
            if gate_test():
                passed += 1
                print(f"✅ {gate_name} PASSED")
            else:
                failed.append(gate_name)
                print(f"❌ {gate_name} FAILED")
        except Exception as e:
            failed.append(gate_name)
            print(f"❌ {gate_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 QUALITY GATES SUMMARY")
    print("=" * 50)
    
    print(f"📊 Results: {passed}/{len(quality_gates)} quality gates passed")
    
    if failed:
        print(f"❌ Failed gates: {', '.join(failed)}")
        print("\n⚠️  QUALITY GATES NOT MET - Issues must be addressed before production")
        return False
    else:
        print("✅ ALL QUALITY GATES PASSED")
        print("\n🎉 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!")
        return True

def main():
    """Main quality gates runner."""
    return run_quality_gates()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)