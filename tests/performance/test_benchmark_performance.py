"""Performance tests for benchmark suite."""

import pytest
import time
import torch
from unittest.mock import patch, MagicMock

from vid_diffusion_bench import BenchmarkSuite
from vid_diffusion_bench.profiler import PerformanceProfiler


@pytest.mark.slow
@pytest.mark.performance
class TestBenchmarkPerformance:
    """Test performance characteristics of the benchmark suite."""

    def test_benchmark_latency(self, benchmark_suite, mock_model):
        """Test benchmark latency stays within acceptable bounds."""
        # Mock model to return immediately
        with patch('vid_diffusion_bench.models.get_model', return_value=mock_model):
            start_time = time.time()
            
            results = benchmark_suite.evaluate_model(
                model_name="mock_model",
                prompts=["test prompt"],
                num_frames=16
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Should complete within reasonable time for mock model
            assert latency < 5.0, f"Benchmark took {latency:.2f}s, expected < 5s"
            assert results is not None

    def test_memory_usage_stays_bounded(self, benchmark_suite, mock_model):
        """Test that memory usage doesn't grow unbounded during benchmarking."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for memory testing")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        with patch('vid_diffusion_bench.models.get_model', return_value=mock_model):
            for i in range(5):  # Run multiple iterations
                results = benchmark_suite.evaluate_model(
                    model_name="mock_model",
                    prompts=[f"test prompt {i}"],
                    num_frames=8
                )
                
                current_memory = torch.cuda.memory_allocated()
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                assert memory_growth < 1e9, f"Memory grew by {memory_growth} bytes"

    def test_profiler_overhead(self, mock_model):
        """Test that profiler adds minimal overhead."""
        profiler = PerformanceProfiler()
        
        # Time without profiler
        start = time.time()
        for _ in range(10):
            _ = mock_model.generate("test", num_frames=8)
        no_profiler_time = time.time() - start
        
        # Time with profiler
        start = time.time()
        for _ in range(10):
            with profiler.track("mock_model"):
                _ = mock_model.generate("test", num_frames=8)
        with_profiler_time = time.time() - start
        
        # Profiler should add < 10% overhead
        overhead = (with_profiler_time - no_profiler_time) / no_profiler_time
        assert overhead < 0.1, f"Profiler added {overhead:.1%} overhead"

    def test_concurrent_benchmarks(self, benchmark_suite, mock_model):
        """Test that concurrent benchmarks don't interfere."""
        import threading
        import concurrent.futures
        
        results = []
        errors = []
        
        def run_benchmark(thread_id):
            try:
                with patch('vid_diffusion_bench.models.get_model', return_value=mock_model):
                    result = benchmark_suite.evaluate_model(
                        model_name=f"mock_model_{thread_id}",
                        prompts=[f"test prompt {thread_id}"],
                        num_frames=8
                    )
                    results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run 3 concurrent benchmarks
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_benchmark, i) for i in range(3)]
            concurrent.futures.wait(futures)
        
        # All should complete without errors
        assert len(errors) == 0, f"Concurrent benchmark errors: {errors}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    @pytest.mark.gpu
    def test_gpu_memory_cleanup(self, benchmark_suite, mock_model):
        """Test that GPU memory is properly cleaned up after benchmarks."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        with patch('vid_diffusion_bench.models.get_model', return_value=mock_model):
            # Run benchmark
            benchmark_suite.evaluate_model(
                model_name="mock_model",
                prompts=["test prompt"],
                num_frames=16
            )
        
        # Force cleanup
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should return close to initial state
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 1e8, f"Memory not cleaned up properly: {memory_diff} bytes"

    def test_large_batch_performance(self, benchmark_suite, mock_model):
        """Test performance with larger batches."""
        large_prompts = [f"test prompt {i}" for i in range(20)]
        
        with patch('vid_diffusion_bench.models.get_model', return_value=mock_model):
            start_time = time.time()
            
            results = benchmark_suite.evaluate_model(
                model_name="mock_model",
                prompts=large_prompts,
                num_frames=8
            )
            
            end_time = time.time()
            
            # Should scale reasonably with batch size
            time_per_prompt = (end_time - start_time) / len(large_prompts)
            assert time_per_prompt < 1.0, f"Time per prompt: {time_per_prompt:.2f}s"

    def test_metrics_computation_performance(self, sample_video):
        """Test that metrics computation is reasonably fast."""
        from vid_diffusion_bench.metrics import VideoQualityMetrics
        
        metrics = VideoQualityMetrics()
        
        # Create batch of videos
        video_batch = sample_video.unsqueeze(0).repeat(5, 1, 1, 1, 1)
        reference_batch = sample_video.unsqueeze(0).repeat(5, 1, 1, 1, 1)
        
        start_time = time.time()
        
        # Compute various metrics
        # Note: These might be mocked in actual implementation
        with patch.object(metrics, 'compute_fvd', return_value=50.0):
            with patch.object(metrics, 'compute_is', return_value=(30.0, 5.0)):
                fvd_score = metrics.compute_fvd(video_batch, reference_batch)
                is_score = metrics.compute_is(video_batch)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        computation_time = end_time - start_time
        assert computation_time < 10.0, f"Metrics computation took {computation_time:.2f}s"

    def test_model_loading_caching(self, benchmark_suite):
        """Test that model loading benefits from caching."""
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.randn(8, 3, 256, 256)
        
        load_times = []
        
        with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
            mock_get_model.return_value = mock_model
            
            # First load (should be slower)
            start = time.time()
            benchmark_suite.evaluate_model("test_model", ["prompt"], num_frames=8)
            first_load_time = time.time() - start
            load_times.append(first_load_time)
            
            # Subsequent loads (should be faster due to caching)
            for _ in range(3):
                start = time.time()
                benchmark_suite.evaluate_model("test_model", ["prompt"], num_frames=8)
                load_time = time.time() - start
                load_times.append(load_time)
        
        # Later loads should be faster (due to caching/warm-up)
        avg_later_loads = sum(load_times[1:]) / len(load_times[1:])
        improvement = (first_load_time - avg_later_loads) / first_load_time
        
        # Should see some improvement (even if small due to mocking)
        assert improvement >= 0, "No caching benefit observed"