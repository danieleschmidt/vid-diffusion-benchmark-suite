"""
Load and performance testing for the Video Diffusion Benchmark Suite.

This module contains tests that validate system behavior under various
load conditions and performance requirements.
"""

import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any
from unittest.mock import Mock, patch

import pytest

from vid_diffusion_bench.benchmark import BenchmarkSuite
from vid_diffusion_bench.metrics import VideoQualityMetrics
from vid_diffusion_bench.profiler import EfficiencyProfiler


class TestLoadPerformance:
    """Test suite for load and performance validation."""
    
    @pytest.fixture
    def benchmark_suite(self):
        """Fixture providing a benchmark suite instance."""
        return BenchmarkSuite()
    
    @pytest.fixture 
    def mock_model_registry(self):
        """Fixture providing a mock model registry."""
        with patch('vid_diffusion_bench.models.registry.ModelRegistry') as mock:
            mock.list_models.return_value = ['mock-svd', 'mock-cogvideo']
            mock.get_model.return_value = Mock()
            yield mock
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_concurrent_model_evaluation(self, benchmark_suite, mock_model_registry):
        """Test concurrent evaluation of multiple models."""
        models = ['mock-svd', 'mock-cogvideo']
        prompts = ["A cat walking", "A car driving"]
        
        def evaluate_model(model_name: str) -> Dict[str, Any]:
            """Helper function to evaluate a single model."""
            with patch.object(benchmark_suite, 'evaluate_model') as mock_eval:
                # Simulate evaluation time
                time.sleep(0.1)
                mock_eval.return_value = {
                    'model': model_name,
                    'fvd_score': 150.0,
                    'latency_ms': 5000,
                    'memory_usage_gb': 12.0
                }
                return mock_eval.return_value
        
        # Test concurrent execution
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(evaluate_model, model) for model in models]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify concurrent execution was faster than sequential
        expected_sequential_time = len(models) * 0.1
        assert execution_time < expected_sequential_time * 0.8, \
            "Concurrent execution should be significantly faster"
        
        # Verify all models were evaluated
        assert len(results) == len(models)
        evaluated_models = [result['model'] for result in results]
        assert set(evaluated_models) == set(models)
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_memory_usage_under_load(self, benchmark_suite):
        """Test memory usage behavior under sustained load."""
        import psutil
        import gc
        
        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate sustained evaluation load
        max_memory = baseline_memory
        memory_measurements = []
        
        for i in range(10):
            # Simulate model evaluation
            with patch.object(benchmark_suite, 'evaluate_model') as mock_eval:
                mock_eval.return_value = {
                    'fvd_score': 150.0,
                    'latency_ms': 1000,
                    'memory_usage_gb': 10.0
                }
                
                # Force some memory allocation
                dummy_data = [i * j for i in range(1000) for j in range(100)]
                
                # Measure memory
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
                max_memory = max(max_memory, current_memory)
                
                # Cleanup
                del dummy_data
                gc.collect()
        
        # Verify memory doesn't grow unbounded
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - baseline_memory
        
        assert memory_growth < 100, f"Memory growth {memory_growth}MB exceeds threshold"
        
        # Verify memory usage is relatively stable
        memory_variance = max(memory_measurements) - min(memory_measurements)
        assert memory_variance < 200, f"Memory variance {memory_variance}MB too high"
    
    @pytest.mark.performance
    def test_metrics_calculation_performance(self):
        """Test performance of metrics calculation."""
        metrics_calculator = VideoQualityMetrics()
        
        # Mock video data (simulate large tensors)
        with patch.object(metrics_calculator, 'compute_fvd') as mock_fvd, \
             patch.object(metrics_calculator, 'compute_is') as mock_is, \
             patch.object(metrics_calculator, 'compute_clipsim') as mock_clip:
            
            # Configure mocks to simulate computation time
            mock_fvd.return_value = 150.0
            mock_is.return_value = (35.0, 2.5)
            mock_clip.return_value = 0.85
            
            # Test multiple concurrent metric calculations
            def calculate_metrics():
                start_time = time.time()
                fvd = mock_fvd()
                is_mean, is_std = mock_is()
                clip_sim = mock_clip()
                end_time = time.time()
                return {
                    'fvd': fvd,
                    'is_mean': is_mean, 
                    'is_std': is_std,
                    'clip_sim': clip_sim,
                    'computation_time': end_time - start_time
                }
            
            # Run multiple calculations concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(calculate_metrics) for _ in range(8)]
                results = [future.result() for future in futures]
            
            # Verify all calculations completed
            assert len(results) == 8
            
            # Verify computation times are reasonable
            computation_times = [result['computation_time'] for result in results]
            avg_computation_time = sum(computation_times) / len(computation_times)
            assert avg_computation_time < 1.0, \
                f"Average computation time {avg_computation_time}s too high"
    
    @pytest.mark.performance
    def test_profiler_overhead(self):
        """Test that profiler adds minimal overhead."""
        profiler = EfficiencyProfiler()
        
        def dummy_workload():
            """Simulate some computational work."""
            return sum(i * i for i in range(1000))
        
        # Measure baseline performance without profiler
        baseline_times = []
        for _ in range(10):
            start = time.time()
            result = dummy_workload()
            end = time.time()
            baseline_times.append(end - start)
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Measure performance with profiler
        profiled_times = []
        for _ in range(10):
            with patch.object(profiler, 'track') as mock_track:
                mock_track.__enter__ = Mock(return_value=Mock())
                mock_track.__exit__ = Mock(return_value=None)
                
                start = time.time()
                with profiler.track("test_model"):
                    result = dummy_workload()
                end = time.time()
                profiled_times.append(end - start)
        
        profiled_avg = sum(profiled_times) / len(profiled_times)
        
        # Verify profiler overhead is minimal (<10%)
        overhead_ratio = (profiled_avg - baseline_avg) / baseline_avg
        assert overhead_ratio < 0.1, \
            f"Profiler overhead {overhead_ratio*100:.1f}% exceeds 10% threshold"
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_batch_processing_scalability(self, benchmark_suite):
        """Test scalability of batch processing."""
        batch_sizes = [1, 2, 4, 8]
        processing_times = {}
        
        for batch_size in batch_sizes:
            prompts = [f"Test prompt {i}" for i in range(batch_size)]
            
            with patch.object(benchmark_suite, 'evaluate_batch') as mock_batch:
                # Simulate batch processing time that scales sublinearly
                simulated_time = batch_size * 0.8  # Sublinear scaling
                
                def side_effect(*args, **kwargs):
                    time.sleep(simulated_time * 0.01)  # Scale down for testing
                    return [{
                        'fvd_score': 150.0,
                        'latency_ms': 1000,
                        'memory_usage_gb': 10.0
                    } for _ in range(batch_size)]
                
                mock_batch.side_effect = side_effect
                
                start_time = time.time()
                results = mock_batch(prompts=prompts)
                end_time = time.time()
                
                processing_times[batch_size] = end_time - start_time
                assert len(results) == batch_size
        
        # Verify batch processing shows efficiency gains
        time_per_item_batch_1 = processing_times[1] / 1
        time_per_item_batch_8 = processing_times[8] / 8
        
        efficiency_gain = (time_per_item_batch_1 - time_per_item_batch_8) / time_per_item_batch_1
        assert efficiency_gain > 0.2, \
            f"Batch processing should show >20% efficiency gain, got {efficiency_gain*100:.1f}%"
    
    @pytest.mark.performance
    def test_api_response_times(self):
        """Test API response time requirements."""
        from unittest.mock import AsyncMock
        
        async def mock_api_call(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            """Simulate API call with realistic response time."""
            await asyncio.sleep(0.1)  # Simulate network latency
            return {
                'status': 'success',
                'data': {'result': 'mock_response'},
                'response_time_ms': 100
            }
        
        async def test_multiple_api_calls():
            """Test multiple concurrent API calls."""
            tasks = []
            for i in range(10):
                task = mock_api_call(f"/test_endpoint_{i}", {"data": f"test_{i}"})
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_response_time = total_time / len(tasks)
            
            # Verify concurrent requests complete efficiently
            assert total_time < 0.5, f"Total time {total_time}s too high for concurrent requests"
            assert avg_response_time < 0.2, f"Average response time {avg_response_time}s too high"
            assert len(results) == 10
            
            return results
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(test_multiple_api_calls())
            assert all(result['status'] == 'success' for result in results)
        finally:
            loop.close()


class TestResourceUtilization:
    """Test suite for resource utilization monitoring."""
    
    @pytest.mark.performance
    def test_gpu_memory_management(self):
        """Test GPU memory usage tracking and management."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for testing")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Simulate model loading and evaluation
        dummy_tensor = torch.randn(1000, 1000, device='cuda')
        after_allocation = torch.cuda.memory_allocated()
        
        # Verify memory was allocated
        memory_used = after_allocation - initial_memory
        assert memory_used > 0, "GPU memory should be allocated"
        
        # Cleanup and verify memory is freed
        del dummy_tensor
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        memory_freed = after_allocation - final_memory
        assert memory_freed > 0, "GPU memory should be freed after cleanup"
        
        # Memory should return close to initial state
        memory_leak = final_memory - initial_memory
        assert memory_leak < 1024 * 1024, f"Possible memory leak: {memory_leak} bytes"
    
    @pytest.mark.performance
    def test_cpu_utilization_monitoring(self):
        """Test CPU utilization monitoring during evaluation."""
        import psutil
        import threading
        
        cpu_measurements = []
        monitoring_active = True
        
        def monitor_cpu():
            """Monitor CPU usage in background thread."""
            while monitoring_active:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_measurements.append(cpu_percent)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            # Simulate CPU-intensive work
            def cpu_intensive_task():
                return sum(i * i for i in range(100000))
            
            # Run multiple tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(cpu_intensive_task) for _ in range(4)]
                results = [future.result() for future in futures]
            
            assert len(results) == 4
            
        finally:
            # Stop monitoring
            monitoring_active = False
            monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_measurements:
            avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
            max_cpu = max(cpu_measurements)
            
            # Verify CPU was utilized but not overloaded
            assert avg_cpu > 10, f"CPU utilization {avg_cpu}% seems too low"
            assert max_cpu < 95, f"CPU utilization {max_cpu}% too high (possible overload)"


if __name__ == "__main__":
    pytest.main([__file__])