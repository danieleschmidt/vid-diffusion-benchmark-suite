"""Enhanced comprehensive tests for research framework components.

This test suite validates the research framework enhancements including
adaptive algorithms, novel metrics, validation framework, error handling,
scaling systems, and quantum acceleration techniques.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vid_diffusion_bench.research.adaptive_algorithms import (
    AdaptiveDiffusionOptimizer, ContentAnalyzer, PerformancePredictor,
    AdaptiveConfig, ContentFeatures
)
from vid_diffusion_bench.research.novel_metrics import (
    NovelMetricsEvaluator, VideoMetricResult, ContentAdaptiveQualityScorer,
    TemporalCoherenceAnalyzer
)
from vid_diffusion_bench.research.validation_framework import (
    ComprehensiveValidator, InputValidator, StatisticalValidator,
    ValidationResult
)
from vid_diffusion_bench.robustness.advanced_error_handling import (
    AdvancedErrorHandler, ErrorClassifier, ResourceTracker,
    MemoryRecoveryStrategy, ErrorSeverity, ErrorCategory
)
from vid_diffusion_bench.scaling.intelligent_scaling import (
    IntelligentScaler, ResourceMonitor, WorkloadCharacteristics,
    ScalingMode, ResourceMetrics
)
from vid_diffusion_bench.optimization.quantum_acceleration import (
    QuantumAcceleratedDiffusion, QuantumCircuit, TensorNetworkDecomposer,
    QuantumGate
)


class TestAdaptiveAlgorithms:
    """Test adaptive algorithm components."""
    
    @pytest.fixture
    def adaptive_config(self):
        return AdaptiveConfig(
            learning_rate=0.001,
            memory_threshold=0.8,
            quality_target=0.85
        )
    
    @pytest.fixture
    def content_analyzer(self):
        return ContentAnalyzer()
    
    @pytest.fixture
    def sample_video(self):
        return torch.randn(16, 3, 64, 64)  # 16 frames, 3 channels, 64x64
    
    def test_content_analyzer_initialization(self, content_analyzer):
        """Test ContentAnalyzer initialization."""
        assert content_analyzer.device is not None
        assert hasattr(content_analyzer, 'complexity_net')
        assert hasattr(content_analyzer, 'motion_net')
    
    def test_content_analysis(self, content_analyzer, sample_video):
        """Test content analysis functionality."""
        features = content_analyzer.analyze_content(sample_video)
        
        assert isinstance(features, ContentFeatures)
        assert 0 <= features.complexity_score <= 1
        assert 0 <= features.motion_intensity <= 1
        assert 0 <= features.texture_density <= 1
        assert 0 <= features.temporal_coherence <= 1
        assert 0 <= features.semantic_complexity <= 1
    
    def test_adaptive_optimizer_initialization(self, adaptive_config):
        """Test AdaptiveDiffusionOptimizer initialization."""
        optimizer = AdaptiveDiffusionOptimizer(adaptive_config)
        
        assert optimizer.config == adaptive_config
        assert hasattr(optimizer, 'content_analyzer')
        assert hasattr(optimizer, 'performance_predictor')
        assert hasattr(optimizer, 'optimal_configs')
    
    def test_content_optimization(self, adaptive_config, sample_video):
        """Test content-specific optimization."""
        optimizer = AdaptiveDiffusionOptimizer(adaptive_config)
        
        base_config = {
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'height': 256,
            'width': 256
        }
        
        optimized_config = optimizer.optimize_for_content(
            sample_video, "test_model", base_config
        )
        
        assert isinstance(optimized_config, dict)
        assert 'num_inference_steps' in optimized_config
        assert optimized_config['num_inference_steps'] > 0
    
    def test_performance_predictor(self):
        """Test PerformancePredictor neural network."""
        predictor = PerformancePredictor()
        
        # Test forward pass
        features = torch.randn(1, 5)
        config = torch.randn(1, 3)
        
        output = predictor(features, config)
        
        assert output.shape == (1, 3)  # [latency, memory, quality]
        assert not torch.isnan(output).any()
    
    def test_adaptation_statistics(self, adaptive_config):
        """Test adaptation statistics tracking."""
        optimizer = AdaptiveDiffusionOptimizer(adaptive_config)
        
        stats = optimizer.get_adaptation_stats()
        
        assert isinstance(stats, dict)
        assert 'status' in stats
        assert 'cached_configs' in stats


class TestNovelMetrics:
    """Test novel metrics evaluation components."""
    
    @pytest.fixture
    def sample_video(self):
        return torch.randn(8, 3, 128, 128)  # 8 frames, 3 channels, 128x128
    
    @pytest.fixture
    def metrics_evaluator(self):
        device = "cpu"  # Use CPU for testing
        return NovelMetricsEvaluator(device=device)
    
    def test_metrics_evaluator_initialization(self, metrics_evaluator):
        """Test NovelMetricsEvaluator initialization."""
        assert hasattr(metrics_evaluator, 'perceptual_net')
        assert hasattr(metrics_evaluator, 'temporal_analyzer')
        assert hasattr(metrics_evaluator, 'adaptive_scorer')
    
    def test_comprehensive_video_evaluation(self, metrics_evaluator, sample_video):
        """Test comprehensive video evaluation."""
        results = metrics_evaluator.evaluate_video_comprehensive(
            sample_video,
            text_prompt="A test video"
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for metric_name, result in results.items():
            assert isinstance(result, VideoMetricResult)
            assert 0 <= result.overall_score <= 1
            assert 0 <= result.confidence <= 1
            assert isinstance(result.frame_scores, list)
    
    def test_temporal_coherence_analysis(self, sample_video):
        """Test temporal coherence analyzer."""
        analyzer = TemporalCoherenceAnalyzer(device="cpu")
        
        result = analyzer.compute_temporal_coherence(sample_video)
        
        assert isinstance(result, VideoMetricResult)
        assert result.metric_name == "temporal_coherence"
        assert 0 <= result.overall_score <= 1
        assert len(result.frame_scores) == sample_video.shape[0] - 1  # N-1 coherence scores
    
    def test_content_adaptive_quality_scorer(self, sample_video):
        """Test content-adaptive quality scoring."""
        scorer = ContentAdaptiveQualityScorer()
        
        result = scorer.compute_adaptive_quality(sample_video)
        
        assert isinstance(result, VideoMetricResult)
        assert result.metric_name == "content_adaptive_quality"
        assert 0 <= result.overall_score <= 1
        assert 'content_features' in result.metadata
        assert 'quality_components' in result.metadata
    
    def test_overall_score_computation(self, metrics_evaluator, sample_video):
        """Test overall score computation."""
        metric_results = metrics_evaluator.evaluate_video_comprehensive(sample_video)
        
        overall_score = metrics_evaluator.compute_overall_score(metric_results)
        
        assert isinstance(overall_score, float)
        assert 0 <= overall_score <= 1


class TestValidationFramework:
    """Test validation framework components."""
    
    @pytest.fixture
    def comprehensive_validator(self):
        return ComprehensiveValidator()
    
    @pytest.fixture
    def input_validator(self):
        return InputValidator()
    
    @pytest.fixture
    def sample_video(self):
        return torch.randn(10, 3, 64, 64)
    
    @pytest.fixture
    def sample_config(self):
        return {
            'models': ['model1', 'model2'],
            'metrics': ['fvd', 'is'],
            'seeds': [42, 123, 456],
            'num_samples_per_seed': 20
        }
    
    def test_input_validator_initialization(self, input_validator):
        """Test InputValidator initialization."""
        assert hasattr(input_validator, 'validation_cache')
        assert hasattr(input_validator, 'cache_lock')
    
    def test_video_tensor_validation(self, input_validator, sample_video):
        """Test video tensor validation."""
        result = input_validator.validate_video_tensor(sample_video)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.confidence > 0
        assert 'shape' in result.metadata
        assert result.metadata['shape'] == list(sample_video.shape)
    
    def test_invalid_video_validation(self, input_validator):
        """Test validation of invalid video tensor."""
        # Test with NaN values
        invalid_video = torch.ones(5, 3, 32, 32)
        invalid_video[0, 0, 0, 0] = float('nan')
        
        result = input_validator.validate_video_tensor(invalid_video)
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert any("NaN" in issue for issue in result.issues)
    
    def test_prompt_validation(self, input_validator):
        """Test text prompt validation."""
        # Valid prompt
        valid_result = input_validator.validate_prompt_text("A cat playing with a ball")
        assert valid_result.is_valid
        
        # Invalid prompts
        empty_result = input_validator.validate_prompt_text("")
        assert not empty_result.is_valid
        
        too_short_result = input_validator.validate_prompt_text("Hi")
        assert not too_short_result.is_valid
    
    def test_config_validation(self, input_validator, sample_config):
        """Test configuration validation."""
        result = input_validator.validate_config_dict(
            sample_config,
            required_keys=['models', 'metrics'],
            value_ranges={'num_samples_per_seed': (1, 100)}
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert 'num_keys' in result.metadata
    
    def test_statistical_validator(self):
        """Test statistical validation."""
        validator = StatisticalValidator()
        
        # Test sample size validation
        size_result = validator.validate_sample_size(50, effect_size=0.5)
        assert isinstance(size_result, ValidationResult)
        
        # Test distribution validation
        normal_data = np.random.normal(0, 1, 100)
        dist_result = validator.validate_distribution(normal_data)
        assert isinstance(dist_result, ValidationResult)
        
        # Test statistical significance
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)
        sig_result = validator.validate_statistical_significance(group1, group2)
        assert isinstance(sig_result, ValidationResult)
        assert 'p_value' in sig_result.metadata
    
    def test_comprehensive_validation(self, comprehensive_validator, sample_video, sample_config):
        """Test comprehensive research pipeline validation."""
        prompts = ["A test prompt", "Another test prompt"]
        
        results = comprehensive_validator.validate_research_pipeline(
            sample_video, prompts, sample_config
        )
        
        assert isinstance(results, dict)
        assert 'overall' in results
        assert 'video_data' in results
        assert 'prompts' in results
        assert 'config' in results
        
        # Generate report
        report = comprehensive_validator.generate_validation_report(results)
        assert isinstance(report, str)
        assert "VALIDATION REPORT" in report


class TestAdvancedErrorHandling:
    """Test advanced error handling components."""
    
    @pytest.fixture
    def error_handler(self):
        return AdvancedErrorHandler()
    
    @pytest.fixture
    def error_classifier(self):
        return ErrorClassifier()
    
    def test_error_classifier_initialization(self, error_classifier):
        """Test ErrorClassifier initialization."""
        assert hasattr(error_classifier, 'error_patterns')
        assert len(error_classifier.error_patterns) > 0
    
    def test_error_classification(self, error_classifier):
        """Test error classification."""
        # Memory error
        memory_error = RuntimeError("CUDA out of memory")
        category, severity = error_classifier.classify_error(memory_error)
        
        assert category == ErrorCategory.MEMORY
        assert severity == ErrorSeverity.HIGH
        
        # File error
        file_error = FileNotFoundError("File not found")
        category, severity = error_classifier.classify_error(file_error)
        
        assert category == ErrorCategory.IO
        assert severity == ErrorSeverity.MEDIUM
    
    def test_error_handling_workflow(self, error_handler):
        """Test complete error handling workflow."""
        # Simulate error
        test_error = RuntimeError("Test error")
        context = {'operation': 'test_operation'}
        
        error_info = error_handler.handle_error(test_error, context, "test_op")
        
        assert error_info.error_type == "RuntimeError"
        assert error_info.message == "Test error"
        assert error_info.context == context
        assert len(error_handler.error_history) == 1
    
    def test_memory_recovery_strategy(self):
        """Test memory recovery strategy."""
        strategy = MemoryRecoveryStrategy()
        
        # Create mock error info
        from vid_diffusion_bench.robustness.advanced_error_handling import ErrorInfo
        error_info = ErrorInfo(
            error_id="test_memory_error",
            timestamp=time.time(),
            error_type="RuntimeError",
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="CUDA out of memory",
            traceback_str="test traceback"
        )
        
        assert strategy.can_recover(error_info)
        
        context = {'batch_size': 8}
        success = strategy.recover(error_info, context)
        
        # Recovery should succeed (mocked)
        assert success
        assert context['batch_size'] == 4  # Should be halved
    
    def test_resource_tracker(self):
        """Test resource tracking."""
        tracker = ResourceTracker()
        
        # Register a test resource
        test_resource = {"data": "test"}
        tracker.register_resource("test_resource", test_resource)
        
        stats = tracker.get_resource_stats()
        assert stats['total_resources'] == 1
        
        # Cleanup
        success = tracker.cleanup_resource("test_resource")
        assert success
        
        stats_after = tracker.get_resource_stats()
        assert stats_after['total_resources'] == 0
    
    def test_error_context_manager(self, error_handler):
        """Test error context manager."""
        context = {'test_key': 'test_value'}
        
        # Test successful operation
        with error_handler.error_context("test_operation", context):
            result = 42
        
        assert result == 42
        
        # Test error handling in context
        try:
            with error_handler.error_context("failing_operation", context):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        # Check error was recorded
        assert len(error_handler.error_history) >= 1


class TestIntelligentScaling:
    """Test intelligent scaling components."""
    
    @pytest.fixture
    def resource_monitor(self):
        monitor = ResourceMonitor(monitoring_interval=0.1)  # Fast for testing
        return monitor
    
    @pytest.fixture
    def intelligent_scaler(self):
        return IntelligentScaler(scaling_mode=ScalingMode.BALANCED)
    
    @pytest.fixture
    def sample_workload(self):
        return WorkloadCharacteristics(
            model_count=2,
            avg_model_size=4.0,
            avg_inference_time=2.0,
            batch_size=2,
            video_resolution=(256, 256),
            video_length=16,
            complexity_score=0.7,
            memory_intensity=0.6,
            compute_intensity=0.8
        )
    
    def test_resource_monitor_initialization(self, resource_monitor):
        """Test ResourceMonitor initialization."""
        assert resource_monitor.monitoring_interval == 0.1
        assert not resource_monitor.monitoring_active
        assert len(resource_monitor.metrics_history) == 0
    
    def test_metrics_collection(self, resource_monitor):
        """Test resource metrics collection."""
        metrics = resource_monitor._collect_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert 0 <= metrics.cpu_usage <= 1
        assert 0 <= metrics.memory_usage <= 1
        assert 0 <= metrics.gpu_usage <= 1
        assert metrics.timestamp > 0
    
    def test_workload_prediction(self, intelligent_scaler, sample_workload):
        """Test workload resource prediction."""
        requirements = intelligent_scaler.predict_resource_requirements(sample_workload)
        
        assert isinstance(requirements, dict)
        assert 'cpu_cores' in requirements
        assert 'memory_gb' in requirements
        assert 'gpu_count' in requirements
        assert 'confidence' in requirements
        
        assert requirements['cpu_cores'] >= 1
        assert requirements['memory_gb'] >= 4
        assert requirements['gpu_count'] >= 1
    
    def test_scaling_decisions(self, intelligent_scaler):
        """Test scaling decision making."""
        # Create metrics that trigger scaling
        high_usage_metrics = ResourceMetrics(
            cpu_usage=0.9,
            memory_usage=0.85,
            gpu_usage=0.9,
            gpu_memory_usage=0.8,
            disk_io=0.5,
            network_io=0.3,
            active_workers=4,
            pending_tasks=10
        )
        
        decisions = intelligent_scaler.make_scaling_decision(high_usage_metrics)
        
        assert isinstance(decisions, list)
        # Should have scaling decisions due to high usage
        assert len(decisions) > 0
        
        for decision in decisions:
            assert hasattr(decision, 'resource_type')
            assert hasattr(decision, 'action')
            assert hasattr(decision, 'confidence')
    
    def test_scaling_statistics(self, intelligent_scaler):
        """Test scaling statistics tracking."""
        stats = intelligent_scaler.get_scaling_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_scaling_actions' in stats
        assert 'scaling_mode' in stats
        assert stats['scaling_mode'] == ScalingMode.BALANCED.value


class TestQuantumAcceleration:
    """Test quantum acceleration components."""
    
    @pytest.fixture
    def quantum_circuit(self):
        return QuantumCircuit(num_qubits=3, device="cpu")
    
    @pytest.fixture
    def tensor_decomposer(self):
        return TensorNetworkDecomposer(max_bond_dimension=16)
    
    @pytest.fixture
    def quantum_accelerator(self):
        return QuantumAcceleratedDiffusion(
            num_qubits=4,
            max_bond_dimension=16,
            device="cpu"
        )
    
    def test_quantum_circuit_initialization(self, quantum_circuit):
        """Test QuantumCircuit initialization."""
        assert quantum_circuit.num_qubits == 3
        assert quantum_circuit.state_dim == 8  # 2^3
        assert quantum_circuit.state.amplitudes[0] == 1.0 + 0.0j  # |000⟩ state
    
    def test_quantum_gates(self, quantum_circuit):
        """Test quantum gate operations."""
        # Test Hadamard gate
        quantum_circuit.apply_gate(QuantumGate.HADAMARD, 0)
        
        # State should be in superposition
        probs = quantum_circuit.get_probabilities()
        assert len(probs) == 8
        assert abs(probs[0] - 0.5) < 1e-6  # |000⟩ probability
        assert abs(probs[4] - 0.5) < 1e-6  # |100⟩ probability
        
        # Test CNOT gate
        quantum_circuit.apply_cnot(0, 1)
        
        # Test measurement
        measurement = quantum_circuit.measure()
        assert 0 <= measurement < 8
    
    def test_tensor_decomposition(self, tensor_decomposer):
        """Test tensor network decomposition."""
        # Test SVD decomposition
        test_tensor = torch.randn(32, 32)
        decomposition = tensor_decomposer.decompose_tensor(test_tensor, mode="svd")
        
        assert 'U' in decomposition
        assert 'S' in decomposition
        assert 'Vh' in decomposition
        assert 'compression_ratio' in decomposition
        
        # Test reconstruction
        reconstructed = tensor_decomposer.reconstruct_tensor(decomposition, mode="svd")
        
        # Should be close to original
        error = torch.norm(test_tensor - reconstructed) / torch.norm(test_tensor)
        assert error < 0.1  # Less than 10% error
    
    def test_quantum_model_compression(self, quantum_accelerator):
        """Test quantum model compression."""
        # Create test model
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        compression_result = quantum_accelerator.compress_model_tensors(model)
        
        assert 'compressed_tensors' in compression_result
        assert 'compression_ratio' in compression_result
        assert 'memory_savings' in compression_result
        
        assert compression_result['compression_ratio'] > 1.0  # Should be compressed
        assert 0 <= compression_result['memory_savings'] <= 1.0
    
    def test_quantum_sampling(self, quantum_accelerator):
        """Test quantum-enhanced sampling."""
        # Create test distribution
        distribution = torch.rand(16)
        distribution = distribution / distribution.sum()  # Normalize
        
        samples = quantum_accelerator.generate_quantum_samples(distribution, 100)
        
        assert len(samples) == 100
        assert all(0 <= sample < 16 for sample in samples)
    
    def test_quantum_acceleration_statistics(self, quantum_accelerator):
        """Test quantum acceleration statistics."""
        stats = quantum_accelerator.get_acceleration_statistics()
        
        assert isinstance(stats, dict)
        assert 'optimizations' in stats
        assert 'decompositions' in stats
        assert 'samples_generated' in stats
    
    def test_quantum_benchmark(self, quantum_accelerator):
        """Test quantum acceleration benchmarking."""
        tensor_sizes = [(32, 32), (64, 64)]
        
        benchmark_results = quantum_accelerator.benchmark_quantum_acceleration(tensor_sizes)
        
        assert isinstance(benchmark_results, dict)
        assert len(benchmark_results) == len(tensor_sizes)
        
        for size_key, results in benchmark_results.items():
            assert 'decomposition_time' in results
            assert 'compression_ratio' in results
            assert 'reconstruction_error' in results
            assert results['compression_ratio'] > 0


class TestIntegration:
    """Integration tests for research framework components."""
    
    def test_end_to_end_research_pipeline(self):
        """Test complete research pipeline integration."""
        # Create test data
        video_data = torch.randn(8, 3, 64, 64)
        prompts = ["Test video 1", "Test video 2"]
        config = {
            'models': ['test_model'],
            'metrics': ['test_metric'],
            'seeds': [42, 123],
            'num_samples_per_seed': 10
        }
        
        # Initialize components
        validator = ComprehensiveValidator()
        error_handler = AdvancedErrorHandler()
        metrics_evaluator = NovelMetricsEvaluator(device="cpu")
        
        try:
            # Validation phase
            with error_handler.error_context("validation"):
                validation_results = validator.validate_research_pipeline(
                    video_data, prompts, config
                )
                assert validation_results['overall'].is_valid
            
            # Metrics evaluation phase
            with error_handler.error_context("metrics_evaluation"):
                metric_results = metrics_evaluator.evaluate_video_comprehensive(video_data)
                assert len(metric_results) > 0
                
                overall_score = metrics_evaluator.compute_overall_score(metric_results)
                assert 0 <= overall_score <= 1
            
            # Check error handling worked
            error_stats = error_handler.get_error_statistics()
            assert isinstance(error_stats, dict)
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    def test_adaptive_optimization_with_quantum_acceleration(self):
        """Test integration of adaptive optimization with quantum acceleration."""
        # Initialize components
        adaptive_config = AdaptiveConfig()
        adaptive_optimizer = AdaptiveDiffusionOptimizer(adaptive_config)
        quantum_accelerator = QuantumAcceleratedDiffusion(device="cpu")
        
        # Test data
        video_data = torch.randn(4, 3, 32, 32)
        base_config = {'num_inference_steps': 20, 'height': 32, 'width': 32}
        
        try:
            # Adaptive optimization
            optimized_config = adaptive_optimizer.optimize_for_content(
                video_data, "test_model", base_config
            )
            
            # Quantum enhancement
            test_model = nn.Linear(16, 8)
            compression_result = quantum_accelerator.compress_model_tensors(test_model)
            
            # Verify results
            assert isinstance(optimized_config, dict)
            assert compression_result['compression_ratio'] > 1.0
            
        except Exception as e:
            pytest.fail(f"Adaptive-quantum integration test failed: {e}")


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def test_device():
    """Determine test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        from vid_diffusion_bench.research.adaptive_algorithms import AdaptiveDiffusionOptimizer
        from vid_diffusion_bench.research.novel_metrics import NovelMetricsEvaluator
        from vid_diffusion_bench.research.validation_framework import ComprehensiveValidator
        from vid_diffusion_bench.robustness.advanced_error_handling import AdvancedErrorHandler
        from vid_diffusion_bench.scaling.intelligent_scaling import IntelligentScaler
        from vid_diffusion_bench.optimization.quantum_acceleration import QuantumAcceleratedDiffusion
        
        assert True  # All imports successful
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])