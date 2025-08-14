#!/usr/bin/env python3
"""Test script for Generation 1: Basic video diffusion benchmark functionality."""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_registry():
    """Test model registry functionality."""
    logger.info("Testing model registry...")
    
    try:
        from vid_diffusion_bench.models.registry import list_models, get_model
        
        # List available models
        models = list_models()
        logger.info(f"Available models: {models}")
        
        assert len(models) > 0, "No models registered"
        
        # Test loading a mock model
        if "mock-fast" in models:
            model = get_model("mock-fast")
            logger.info(f"Loaded model: {model.name}")
            logger.info(f"Requirements: {model.requirements}")
            
        logger.info("✓ Model registry test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model registry test failed: {e}")
        return False

def test_benchmark_suite():
    """Test basic benchmark suite functionality."""
    logger.info("Testing benchmark suite...")
    
    try:
        from vid_diffusion_bench.benchmark import BenchmarkSuite
        
        # Initialize benchmark suite
        suite = BenchmarkSuite(device="cpu", output_dir="./test_results")
        logger.info("✓ BenchmarkSuite initialized")
        
        # Test single model evaluation
        if "mock-fast" in suite.list_available_models():
            result = suite.evaluate_model(
                model_name="mock-fast",
                prompts=["A test video prompt"],
                num_frames=8,
                fps=4
            )
            
            logger.info(f"Evaluation result:")
            logger.info(f"  Success rate: {result.success_rate:.1%}")
            logger.info(f"  Model: {result.model_name}")
            logger.info(f"  Prompts: {len(result.prompts)}")
            
            assert result.success_rate > 0, "Evaluation failed completely"
            
        logger.info("✓ Benchmark suite test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Benchmark suite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_computation():
    """Test metrics computation."""
    logger.info("Testing metrics computation...")
    
    try:
        from vid_diffusion_bench.metrics import VideoQualityMetrics
        import torch
        
        # Initialize metrics engine
        metrics_engine = VideoQualityMetrics(device="cpu")
        logger.info("✓ Metrics engine initialized")
        
        # Create mock video data
        videos = [torch.rand(8, 3, 64, 64)]  # Small for testing
        prompts = ["Test prompt"]
        
        # Compute metrics
        metrics = metrics_engine.compute_all_metrics(videos, prompts)
        
        logger.info(f"Computed metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")
            
        assert "fvd" in metrics, "FVD metric missing"
        assert "clip_similarity" in metrics, "CLIP similarity missing"
        assert "overall_quality_score" in metrics, "Overall quality score missing"
        
        logger.info("✓ Metrics computation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Metrics computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_interface():
    """Test CLI interface functionality."""
    logger.info("Testing CLI interface...")
    
    try:
        from vid_diffusion_bench.cli import main
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0, "CLI help failed"
        
        # Test list-models command
        result = runner.invoke(main, ['list-models'])
        logger.info(f"List models output: {result.output[:200]}...")
        
        logger.info("✓ CLI interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ CLI interface test failed: {e}")
        return False

def test_real_model_adapters():
    """Test real model adapters (mock mode)."""
    logger.info("Testing real model adapters...")
    
    try:
        from vid_diffusion_bench.models.real_adapters import CogVideoAdapter
        
        # Initialize adapter (will fall back to mock)
        adapter = CogVideoAdapter(device="cpu")
        logger.info(f"Adapter name: {adapter.name}")
        logger.info(f"Requirements: {adapter.requirements}")
        
        # Test generation (will use mock)
        video = adapter.generate("A test prompt", num_frames=4, width=64, height=64)
        logger.info(f"Generated video shape: {video.shape}")
        
        assert video.shape[0] == 4, "Wrong number of frames"
        assert video.shape[1] == 3, "Wrong number of channels"
        
        logger.info("✓ Real model adapters test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Real model adapters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Generation 1 tests."""
    logger.info("=" * 60)
    logger.info("GENERATION 1 FUNCTIONALITY TESTS")
    logger.info("=" * 60)
    
    tests = [
        test_model_registry,
        test_benchmark_suite,
        test_metrics_computation,
        test_cli_interface,
        test_real_model_adapters
    ]
    
    results = []
    for test in tests:
        results.append(test())
        logger.info("")  # Blank line
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("=" * 60)
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 All Generation 1 tests PASSED! Core functionality is working.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests FAILED. Need to fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())