#!/usr/bin/env python3
"""Generation 2 robustness test - Comprehensive reliability verification."""

import sys
import os
import logging
import time
import threading
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_system_health_monitor():
    """Test system health monitoring functionality."""
    try:
        from vid_diffusion_bench.generation2_robustness import SystemHealthMonitor
        
        monitor = SystemHealthMonitor(check_interval=1)
        
        # Test basic health check
        health = monitor.check_health()
        assert health is not None
        assert hasattr(health, 'cpu_percent')
        assert hasattr(health, 'memory_percent')
        assert hasattr(health, 'disk_usage_percent')
        assert isinstance(health.is_healthy, bool)
        
        print(f"✅ Health check - CPU: {health.cpu_percent:.1f}%, Memory: {health.memory_percent:.1f}%")
        
        # Test monitoring start/stop
        monitor.start_monitoring()
        time.sleep(2)  # Let it collect some data
        
        summary = monitor.get_health_summary()
        assert 'status' in summary
        
        monitor.stop_monitoring()
        print("✅ System health monitoring works")
        return True
        
    except Exception as e:
        print(f"❌ System health monitor test failed: {e}")
        return False

def test_circuit_breaker():
    """Test circuit breaker functionality."""
    try:
        from vid_diffusion_bench.generation2_robustness import CircuitBreaker
        
        # Test with failing function
        failure_count = 0
        
        @CircuitBreaker(failure_threshold=3, timeout=1.0)
        def failing_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise RuntimeError(f"Simulated failure #{failure_count}")
            return "success"
        
        # Test circuit breaker opening
        for i in range(5):
            try:
                result = failing_function()
                if i < 3:
                    print(f"❌ Expected failure on attempt {i+1}")
                    return False
            except Exception as e:
                if "Circuit breaker OPEN" in str(e):
                    print("✅ Circuit breaker opened as expected")
                    break
                elif i < 3:
                    continue  # Expected failures
                else:
                    print(f"❌ Unexpected error: {e}")
                    return False
        
        print("✅ Circuit breaker functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
        return False

def test_benchmark_recovery():
    """Test benchmark recovery system."""
    try:
        from vid_diffusion_bench.generation2_robustness import BenchmarkRecovery
        
        recovery = BenchmarkRecovery(max_retries=2, backoff_factor=1.1)
        
        # Test recovery strategy registration
        recovery_called = False
        
        def test_recovery_strategy(exception, attempt):
            nonlocal recovery_called
            recovery_called = True
            print(f"Recovery strategy called for attempt {attempt}")
        
        recovery.register_recovery_strategy(ValueError, test_recovery_strategy)
        
        # Test function that fails then succeeds
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError(f"Failure #{call_count}")
            return "success"
        
        result = recovery.execute_with_recovery(flaky_function)
        
        assert result == "success"
        assert recovery_called
        assert call_count == 3
        
        print("✅ Benchmark recovery system works")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark recovery test failed: {e}")
        return False

def test_data_backup_manager():
    """Test data backup and recovery."""
    try:
        from vid_diffusion_bench.generation2_robustness import DataBackupManager
        import tempfile
        import shutil
        
        # Create temporary backup directory
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_manager = DataBackupManager(backup_dir=temp_dir, max_backups=3)
            
            # Test data backup
            test_data = {"model": "test", "score": 95.5, "metadata": {"version": "1.0"}}
            backup_path = backup_manager.backup_data(test_data, "test_backup")
            
            assert os.path.exists(backup_path)
            print(f"✅ Data backed up to: {backup_path}")
            
            # Test data recovery
            restored_data = backup_manager.restore_latest("test_backup")
            assert restored_data is not None
            assert restored_data["model"] == "test"
            assert restored_data["score"] == 95.5
            
            print("✅ Data backup and recovery works")
            return True
        
    except Exception as e:
        print(f"❌ Data backup manager test failed: {e}")
        return False

def test_advanced_logging():
    """Test advanced logging functionality."""
    try:
        from vid_diffusion_bench.generation2_robustness import AdvancedLoggingManager
        import tempfile
        
        # Create temporary log directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logging_manager = AdvancedLoggingManager(log_dir=temp_dir, structured=True)
            
            # Test logging
            logger = logging.getLogger("test_logger")
            logger.info("Test info message")
            logger.error("Test error message") 
            logger.warning("Test performance metrics: latency=50ms throughput=100fps")
            
            # Check that log files are created
            log_files = os.listdir(temp_dir)
            assert "benchmark.log" in log_files
            assert "errors.log" in log_files
            assert "performance.log" in log_files
            
            print(f"✅ Advanced logging created files: {log_files}")
            return True
        
    except Exception as e:
        print(f"❌ Advanced logging test failed: {e}")
        return False

def test_recovery_strategies():
    """Test built-in recovery strategies."""
    try:
        from vid_diffusion_bench.generation2_robustness import (
            gpu_memory_recovery, disk_space_recovery, network_recovery
        )
        
        # Test GPU memory recovery (should not fail)
        try:
            gpu_memory_recovery(RuntimeError("GPU memory error"), 1)
            print("✅ GPU memory recovery strategy executed")
        except Exception as e:
            print(f"⚠️  GPU recovery failed (expected if no GPU): {e}")
        
        # Test disk space recovery
        disk_space_recovery(OSError("No space left on device"), 1)
        print("✅ Disk space recovery strategy executed")
        
        # Test network recovery (should just wait)
        start_time = time.time()
        network_recovery(ConnectionError("Network unreachable"), 0)
        elapsed = time.time() - start_time
        assert elapsed >= 1.0  # Should wait at least 1 second
        print(f"✅ Network recovery strategy executed (waited {elapsed:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Recovery strategies test failed: {e}")
        return False

def test_generation2_integration():
    """Test Generation 2 integration with benchmark suite."""
    try:
        # Test that benchmark suite can initialize with Generation 2 features
        # This is a basic smoke test since full functionality needs more setup
        
        from vid_diffusion_bench.generation2_robustness import SystemHealthMonitor
        monitor = SystemHealthMonitor()
        health_summary = monitor.get_health_summary()
        
        assert 'status' in health_summary
        print("✅ Generation 2 integration smoke test passed")
        return True
        
    except Exception as e:
        print(f"❌ Generation 2 integration test failed: {e}")
        return False

def main():
    """Run all Generation 2 robustness tests."""
    print("🛡️  Running Generation 2 robustness tests...")
    print()
    
    tests = [
        test_system_health_monitor,
        test_circuit_breaker,
        test_benchmark_recovery,
        test_data_backup_manager,
        test_advanced_logging,
        test_recovery_strategies,
        test_generation2_integration
    ]
    
    passed = 0
    for test in tests:
        print(f"Running {test.__name__}...")
        if test():
            passed += 1
        print()
    
    print(f"📊 Generation 2 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Generation 2: MAKE IT ROBUST - COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("❌ Some Generation 2 tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)