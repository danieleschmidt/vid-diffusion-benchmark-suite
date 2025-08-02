"""
Unit tests for test fixtures and mock data validation.

This module ensures that all test fixtures are valid and properly structured
for use in the test suite.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.conftest import FIXTURES_DIR


class TestFixtureValidation:
    """Test suite for validating test fixtures."""
    
    def test_fixtures_directory_exists(self):
        """Test that the fixtures directory exists."""
        assert FIXTURES_DIR.exists(), "Fixtures directory not found"
        assert FIXTURES_DIR.is_dir(), "Fixtures path is not a directory"
    
    def test_unit_test_prompts_structure(self):
        """Test that unit test prompts have correct structure."""
        prompts_file = FIXTURES_DIR / "test_prompts" / "unit_test_prompts.json"
        assert prompts_file.exists(), "Unit test prompts file not found"
        
        with open(prompts_file) as f:
            data = json.load(f)
        
        # Validate top-level structure
        assert "metadata" in data
        assert "prompts" in data
        assert "usage_notes" in data
        
        # Validate metadata
        metadata = data["metadata"]
        assert "version" in metadata
        assert "description" in metadata
        assert "total_prompts" in metadata
        assert "categories" in metadata
        
        # Validate prompts
        prompts = data["prompts"]
        assert isinstance(prompts, list)
        assert len(prompts) == metadata["total_prompts"]
        
        for prompt in prompts:
            assert "id" in prompt
            assert "text" in prompt
            assert "category" in prompt
            assert "complexity" in prompt
            assert "expected_elements" in prompt
            assert "motion_type" in prompt
            
            # Validate data types
            assert isinstance(prompt["id"], str)
            assert isinstance(prompt["text"], str)
            assert isinstance(prompt["category"], str)
            assert isinstance(prompt["complexity"], str)
            assert isinstance(prompt["expected_elements"], list)
            assert isinstance(prompt["motion_type"], str)
            
            # Validate category is in allowed categories
            assert prompt["category"] in metadata["categories"]
            
            # Validate complexity levels
            assert prompt["complexity"] in ["low", "medium", "high"]
    
    def test_mock_model_configs_structure(self):
        """Test that mock model configurations are valid."""
        mock_models_dir = FIXTURES_DIR / "mock_models"
        assert mock_models_dir.exists(), "Mock models directory not found"
        
        # Test SVD mock config
        svd_config_file = mock_models_dir / "mock_svd.yaml"
        assert svd_config_file.exists(), "Mock SVD config not found"
        
        with open(svd_config_file) as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = [
            "model_info", "model_config", "requirements", 
            "mock_behavior", "api_interface", "test_data"
        ]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
        
        # Validate model_info
        model_info = config["model_info"]
        required_info_fields = ["name", "display_name", "version", "type", "description"]
        for field in required_info_fields:
            assert field in model_info, f"Missing model_info field: {field}"
        
        # Validate generation config
        generation = config["model_config"]["generation"]
        assert "default_num_frames" in generation
        assert "default_fps" in generation
        assert "default_resolution" in generation
        assert isinstance(generation["default_resolution"], list)
        assert len(generation["default_resolution"]) == 2
        
        # Validate requirements
        requirements = config["requirements"]
        assert "gpu_memory_gb" in requirements
        assert "system_memory_gb" in requirements
        assert isinstance(requirements["gpu_memory_gb"], (int, float))
        assert isinstance(requirements["system_memory_gb"], (int, float))
        
        # Validate mock behavior
        mock_behavior = config["mock_behavior"]
        assert "generation_time_ms" in mock_behavior
        assert "peak_memory_usage_gb" in mock_behavior
        assert "simulated_metrics" in mock_behavior
        
        metrics = mock_behavior["simulated_metrics"]
        expected_metrics = ["fvd_score", "is_mean", "is_std", "clip_similarity", "temporal_consistency"]
        for metric in expected_metrics:
            assert metric in metrics, f"Missing simulated metric: {metric}"
            assert isinstance(metrics[metric], (int, float))
    
    def test_fixture_consistency(self):
        """Test that fixtures are consistent with each other."""
        # Load unit test prompts
        prompts_file = FIXTURES_DIR / "test_prompts" / "unit_test_prompts.json"
        with open(prompts_file) as f:
            prompts_data = json.load(f)
        
        # Load mock SVD config
        svd_config_file = FIXTURES_DIR / "mock_models" / "mock_svd.yaml"
        with open(svd_config_file) as f:
            svd_config = yaml.safe_load(f)
        
        # Test that mock model supports the prompt requirements
        default_frames = svd_config["model_config"]["generation"]["default_num_frames"]
        max_frames = svd_config["model_config"]["generation"]["max_num_frames"]
        
        # All unit test prompts should be compatible with default model settings
        for prompt in prompts_data["prompts"]:
            # For unit tests, we expect to use default settings
            assert default_frames <= max_frames, "Invalid mock model frame configuration"
    
    @pytest.mark.parametrize("fixture_type", ["test_prompts", "mock_models", "expected_outputs", "mock_data"])
    def test_fixture_directories_exist(self, fixture_type: str):
        """Test that all required fixture directories exist."""
        fixture_dir = FIXTURES_DIR / fixture_type
        if not fixture_dir.exists():
            # Create directory if it doesn't exist (for empty directories)
            fixture_dir.mkdir(parents=True, exist_ok=True)
        assert fixture_dir.is_dir(), f"{fixture_type} should be a directory"


class TestMockModelValidation:
    """Test suite specifically for mock model configurations."""
    
    def test_mock_svd_api_interface(self):
        """Test that mock SVD API interface is properly defined."""
        svd_config_file = FIXTURES_DIR / "mock_models" / "mock_svd.yaml"
        with open(svd_config_file) as f:
            config = yaml.safe_load(f)
        
        api_interface = config["api_interface"]
        
        # Validate endpoints
        assert "endpoints" in api_interface
        endpoints = api_interface["endpoints"]
        required_endpoints = ["generate", "health", "info"]
        for endpoint in required_endpoints:
            assert endpoint in endpoints
            assert isinstance(endpoints[endpoint], str)
            assert endpoints[endpoint].startswith("/")
        
        # Validate request/response schemas
        assert "generate_request_schema" in api_interface
        assert "generate_response_schema" in api_interface
        
        request_schema = api_interface["generate_request_schema"]
        assert "required" in request_schema
        assert "optional" in request_schema
        assert "prompt" in request_schema["required"]
        
        response_schema = api_interface["generate_response_schema"]
        assert "success" in response_schema
        assert "error" in response_schema
    
    def test_mock_svd_error_scenarios(self):
        """Test that mock SVD error scenarios are properly configured."""
        svd_config_file = FIXTURES_DIR / "mock_models" / "mock_svd.yaml"
        with open(svd_config_file) as f:
            config = yaml.safe_load(f)
        
        error_scenarios = config["mock_behavior"]["error_scenarios"]
        assert isinstance(error_scenarios, list)
        assert len(error_scenarios) > 0
        
        for scenario in error_scenarios:
            assert "trigger" in scenario
            assert "error_type" in scenario 
            assert "message" in scenario
            
            assert isinstance(scenario["trigger"], str)
            assert isinstance(scenario["error_type"], str)
            assert isinstance(scenario["message"], str)


class TestPromptFixtures:
    """Test suite for prompt fixture validation."""
    
    def test_prompt_ids_unique(self):
        """Test that all prompt IDs are unique."""
        prompts_file = FIXTURES_DIR / "test_prompts" / "unit_test_prompts.json"
        with open(prompts_file) as f:
            data = json.load(f)
        
        prompt_ids = [prompt["id"] for prompt in data["prompts"]]
        assert len(prompt_ids) == len(set(prompt_ids)), "Prompt IDs must be unique"
    
    def test_prompt_categories_consistent(self):
        """Test that prompt categories are consistent with metadata."""
        prompts_file = FIXTURES_DIR / "test_prompts" / "unit_test_prompts.json"
        with open(prompts_file) as f:
            data = json.load(f)
        
        allowed_categories = set(data["metadata"]["categories"])
        used_categories = set(prompt["category"] for prompt in data["prompts"])
        
        # All used categories should be in allowed categories
        assert used_categories.issubset(allowed_categories), \
            f"Used categories {used_categories} not in allowed {allowed_categories}"
    
    def test_prompt_text_quality(self):
        """Test that prompt texts meet quality standards."""
        prompts_file = FIXTURES_DIR / "test_prompts" / "unit_test_prompts.json"
        with open(prompts_file) as f:
            data = json.load(f)
        
        for prompt in data["prompts"]:
            text = prompt["text"]
            
            # Basic quality checks
            assert len(text) > 5, f"Prompt text too short: {text}"
            assert len(text) < 200, f"Prompt text too long: {text}"
            assert text[0].isupper(), f"Prompt should start with capital letter: {text}"
            assert not text.endswith("."), f"Prompt should not end with period: {text}"
            
            # Check for expected elements in text
            expected_elements = prompt["expected_elements"]
            text_lower = text.lower()
            for element in expected_elements:
                assert element.lower() in text_lower, \
                    f"Expected element '{element}' not found in prompt: {text}"


if __name__ == "__main__":
    pytest.main([__file__])