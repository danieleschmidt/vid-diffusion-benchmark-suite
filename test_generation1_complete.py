#!/usr/bin/env python3
"""Generation 1 completion test - Basic functionality verification."""

import sys
import os
import logging

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic package imports."""
    try:
        import vid_diffusion_bench
        from vid_diffusion_bench import BenchmarkSuite
        from vid_diffusion_bench.generation1_enhancements import (
            ProgressInfo, BenchmarkProgressTracker, RetryHandler, 
            SafetyValidator, BasicCacheManager
        )
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_benchmark_result():
    """Test BenchmarkResult functionality."""
    try:
        from vid_diffusion_bench.benchmark import BenchmarkResult
        
        prompts = ["A cat playing piano", "A sunset over mountains"]
        result = BenchmarkResult("test-model", prompts)
        
        # Test basic functionality
        assert result.model_name == "test-model"
        assert len(result.prompts) == 2
        assert result.get_success_rate() == 0.0  # No results yet
        
        # Test to_dict method
        result_dict = result.to_dict()
        assert "model_name" in result_dict
        assert "success_rate" in result_dict
        
        print("✅ BenchmarkResult functionality works")
        return True
    except Exception as e:
        print(f"❌ BenchmarkResult test failed: {e}")
        return False

def test_progress_tracker():
    """Test progress tracking functionality."""
    try:
        from vid_diffusion_bench.generation1_enhancements import BenchmarkProgressTracker
        
        tracker = BenchmarkProgressTracker()
        tracker.start(10, "Testing")
        
        assert tracker.progress.total == 10
        assert tracker.progress.current == 0
        assert tracker.progress.current_task == "Testing"
        
        tracker.increment("Step 1")
        assert tracker.progress.current == 1
        
        print("✅ Progress tracker functionality works")
        return True
    except Exception as e:
        print(f"❌ Progress tracker test failed: {e}")
        return False

def test_safety_validator():
    """Test safety validation functionality."""
    try:
        from vid_diffusion_bench.generation1_enhancements import SafetyValidator
        
        # Test prompt validation
        prompts = [
            "A cat playing piano",
            "<script>alert('hack')</script>",
            "A" * 2000,  # Very long prompt
            123,  # Non-string
            "Normal prompt"
        ]
        
        validated = SafetyValidator.validate_prompts(prompts, max_length=500, max_count=10)
        
        # Should filter out unsafe and invalid prompts
        assert len(validated) == 2  # Only safe prompts remain
        assert "A cat playing piano" in validated
        assert "Normal prompt" in validated
        
        # Test parameter validation
        params = SafetyValidator.validate_model_params(
            num_frames=300,  # Too high
            fps=100,  # Too high
            resolution=(5000, 5000),  # Too high
            batch_size=50  # Too high
        )
        
        assert params['num_frames'] <= 200
        assert params['fps'] <= 60
        assert params['resolution'][0] <= 2048
        assert params['batch_size'] <= 16
        
        print("✅ Safety validator functionality works")
        return True
    except Exception as e:
        print(f"❌ Safety validator test failed: {e}")
        return False

def test_cache_manager():
    """Test basic cache manager functionality."""
    try:
        from vid_diffusion_bench.generation1_enhancements import BasicCacheManager
        
        cache = BasicCacheManager(max_size=3)
        
        # Test basic operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        # Test LRU eviction
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
        
        print("✅ Cache manager functionality works")
        return True
    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        return False

def main():
    """Run all Generation 1 tests."""
    print("🚀 Running Generation 1 completion tests...")
    
    tests = [
        test_basic_imports,
        test_benchmark_result,
        test_progress_tracker,
        test_safety_validator,
        test_cache_manager
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Generation 1 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Generation 1: MAKE IT WORK - COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("❌ Some Generation 1 tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)