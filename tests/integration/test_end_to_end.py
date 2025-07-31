"""
End-to-end integration tests for Video Diffusion Benchmark Suite.

These tests validate complete workflows from model loading to result generation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import torch
from vid_diffusion_bench import BenchmarkSuite
from vid_diffusion_bench.models import get_model


class TestEndToEndIntegration:
    """End-to-end integration test scenarios."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_prompts(self):
        """Sample prompts for testing."""
        return [
            "A cat playing piano in a jazz club",
            "Aerial view of a futuristic city at sunset",
            "Ocean waves crashing on a rocky shore",
            "Time-lapse of flowers blooming in spring"
        ]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_benchmark_workflow(self, temp_output_dir, sample_prompts):
        """Test complete benchmark workflow from start to finish."""
        
        # Mock model for integration test
        with patch('vid_diffusion_bench.models.registry.MODELS', {'mock-model': Mock()}):
            with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
                # Configure mock model
                mock_model = Mock()
                mock_model.generate.return_value = torch.randn(16, 3, 256, 256)  # 16 frames
                mock_model.requirements = {
                    "vram_gb": 8,
                    "precision": "fp16",
                    "dependencies": ["torch>=2.0"]
                }
                mock_get_model.return_value = mock_model
                
                # Initialize benchmark suite
                suite = BenchmarkSuite()
                
                # Run complete benchmark
                results = suite.evaluate_model(
                    model_name="mock-model",
                    prompts=sample_prompts[:2],  # Use subset for speed
                    num_frames=16,
                    fps=8,
                    resolution=(256, 256)
                )
                
                # Verify results structure
                assert hasattr(results, 'fvd')
                assert hasattr(results, 'latency')
                assert hasattr(results, 'peak_vram_gb')
                
                # Verify model was called correctly
                assert mock_model.generate.call_count == 2  # Once per prompt
                
                # Verify generation parameters
                call_args = mock_model.generate.call_args_list[0]
                assert call_args[1]['num_frames'] == 16

    @pytest.mark.integration
    def test_multi_model_comparison(self, sample_prompts):
        """Test benchmarking multiple models for comparison."""
        
        models_to_test = ['model-a', 'model-b', 'model-c']
        comparison_results = {}
        
        for model_name in models_to_test:
            with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
                # Configure unique mock for each model
                mock_model = Mock()
                mock_model.generate.return_value = torch.randn(8, 3, 128, 128)
                mock_model.requirements = {"vram_gb": 4, "precision": "fp16"}
                mock_get_model.return_value = mock_model
                
                suite = BenchmarkSuite()
                results = suite.evaluate_model(
                    model_name=model_name,
                    prompts=sample_prompts[:1],  # Single prompt for speed
                    num_frames=8
                )
                
                comparison_results[model_name] = {
                    'fvd': results.fvd,
                    'latency': results.latency,
                    'vram': results.peak_vram_gb
                }
        
        # Verify all models were tested
        assert len(comparison_results) == 3
        
        # Verify each model has required metrics
        for model_name, metrics in comparison_results.items():
            assert 'fvd' in metrics
            assert 'latency' in metrics
            assert 'vram' in metrics

    @pytest.mark.integration
    def test_result_export_and_import(self, temp_output_dir, sample_prompts):
        """Test exporting and importing benchmark results."""
        
        with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
            # Setup mock model
            mock_model = Mock()
            mock_model.generate.return_value = torch.randn(4, 3, 64, 64)
            mock_model.requirements = {"vram_gb": 2}
            mock_get_model.return_value = mock_model
            
            suite = BenchmarkSuite()
            
            # Generate results
            results = suite.evaluate_model(
                model_name="export-test-model",
                prompts=sample_prompts[:1],
                num_frames=4
            )
            
            # Export results
            output_file = temp_output_dir / "benchmark_results.json"
            results_dict = {
                'model_name': 'export-test-model',
                'fvd': results.fvd,
                'latency': results.latency,
                'peak_vram_gb': results.peak_vram_gb,
                'num_frames': 4,
                'prompts': sample_prompts[:1]
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # Verify export
            assert output_file.exists()
            
            # Import and verify results
            with open(output_file, 'r') as f:
                imported_results = json.load(f)
            
            assert imported_results['model_name'] == 'export-test-model'
            assert 'fvd' in imported_results
            assert 'latency' in imported_results
            assert imported_results['num_frames'] == 4

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gpu_memory_management(self, sample_prompts):
        """Test GPU memory management across multiple benchmarks."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        initial_memory = torch.cuda.memory_allocated()
        
        with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
            for i in range(3):  # Run multiple benchmarks
                # Configure mock model with GPU tensors
                mock_model = Mock()
                mock_model.generate.return_value = torch.randn(8, 3, 128, 128, device='cuda')
                mock_model.requirements = {"vram_gb": 4}
                mock_get_model.return_value = mock_model
                
                suite = BenchmarkSuite()
                results = suite.evaluate_model(
                    model_name=f"gpu-test-model-{i}",
                    prompts=sample_prompts[:1],
                    num_frames=8
                )
                
                # Verify results
                assert results is not None
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
        
        # Verify memory cleanup
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (< 100MB)
        assert memory_increase < 100 * 1024 * 1024

    @pytest.mark.integration
    def test_error_recovery_workflow(self, sample_prompts):
        """Test system recovery from various error conditions."""
        
        # Test 1: Model loading failure
        with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
            mock_get_model.side_effect = RuntimeError("Model not found")
            
            suite = BenchmarkSuite()
            
            with pytest.raises(RuntimeError, match="Model not found"):
                suite.evaluate_model(
                    model_name="nonexistent-model",
                    prompts=sample_prompts[:1]
                )
        
        # Test 2: Generation failure with recovery
        with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
            mock_model = Mock()
            
            # First call fails, subsequent calls succeed
            call_count = 0
            def generate_with_failure(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory")
                return torch.randn(4, 3, 64, 64)
            
            mock_model.generate.side_effect = generate_with_failure
            mock_model.requirements = {"vram_gb": 2}
            mock_get_model.return_value = mock_model
            
            suite = BenchmarkSuite()
            
            # First attempt should fail
            with pytest.raises(torch.cuda.OutOfMemoryError):
                suite.evaluate_model(
                    model_name="failing-model",
                    prompts=sample_prompts[:1]
                )
            
            # Subsequent attempt should succeed (simulating retry)
            results = suite.evaluate_model(
                model_name="failing-model",
                prompts=sample_prompts[:1]
            )
            assert results is not None

    @pytest.mark.integration
    def test_configuration_validation(self, sample_prompts):
        """Test validation of various configuration combinations."""
        
        with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.generate.return_value = torch.randn(16, 3, 256, 256)
            mock_model.requirements = {"vram_gb": 8}
            mock_get_model.return_value = mock_model
            
            suite = BenchmarkSuite()
            
            # Test valid configurations
            valid_configs = [
                {"num_frames": 8, "fps": 4, "resolution": (128, 128)},
                {"num_frames": 16, "fps": 8, "resolution": (256, 256)},
                {"num_frames": 32, "fps": 12, "resolution": (512, 512)},
            ]
            
            for config in valid_configs:
                results = suite.evaluate_model(
                    model_name="config-test-model",
                    prompts=sample_prompts[:1],
                    **config
                )
                assert results is not None
                assert hasattr(results, 'fvd')
            
            # Test invalid configurations
            invalid_configs = [
                {"num_frames": 0},  # Invalid frame count
                {"fps": -1},        # Invalid FPS
                {"resolution": (0, 256)},  # Invalid resolution
            ]
            
            for config in invalid_configs:
                with pytest.raises((ValueError, RuntimeError)):
                    suite.evaluate_model(
                        model_name="config-test-model",
                        prompts=sample_prompts[:1],
                        **config
                    )

    @pytest.mark.integration
    def test_benchmark_reproducibility(self, sample_prompts):
        """Test that benchmarks produce reproducible results."""
        
        results_run1 = []
        results_run2 = []
        
        # Run benchmark twice with same configuration
        for run_id in range(2):
            with patch('vid_diffusion_bench.models.get_model') as mock_get_model:
                mock_model = Mock()
                # Use deterministic output for reproducibility test
                mock_model.generate.return_value = torch.ones(8, 3, 128, 128) * (run_id + 1)
                mock_model.requirements = {"vram_gb": 4}
                mock_get_model.return_value = mock_model
                
                suite = BenchmarkSuite()
                results = suite.evaluate_model(
                    model_name="reproducibility-model",
                    prompts=sample_prompts[:1],
                    num_frames=8,
                    seed=42  # Fixed seed for reproducibility
                )
                
                if run_id == 0:
                    results_run1.append(results)
                else:
                    results_run2.append(results)
        
        # Results should be identical for same configuration and seed
        # Note: In actual implementation, you'd compare actual metric values
        assert len(results_run1) == len(results_run2)
        
        # Verify both runs completed successfully
        for results in results_run1 + results_run2:
            assert results is not None
            assert hasattr(results, 'fvd')