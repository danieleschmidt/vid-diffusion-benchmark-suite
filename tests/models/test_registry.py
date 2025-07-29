"""Tests for model registry functionality."""

import pytest
from vid_diffusion_bench.models import register_model, get_model, list_models, ModelAdapter
import torch


@register_model("test_model")
class TestModelAdapter(ModelAdapter):
    """Test model adapter for registry testing."""
    
    def generate(self, prompt: str, num_frames: int = 16, fps: int = 8, **kwargs):
        """Generate mock video tensor."""
        return torch.randn(num_frames, 3, 256, 256)
        
    @property
    def requirements(self):
        """Return test requirements."""
        return {
            "vram_gb": 4,
            "precision": "fp16",
            "dependencies": ["torch>=2.0"]
        }


class TestModelRegistry:
    """Test cases for model registry."""
    
    def test_register_model_decorator(self):
        """Test model registration via decorator."""
        models = list_models()
        assert "test_model" in models
        
    def test_get_registered_model(self):
        """Test retrieving registered model."""
        model = get_model("test_model")
        assert isinstance(model, TestModelAdapter)
        
    def test_get_model_with_kwargs(self):
        """Test retrieving model with configuration."""
        model = get_model("test_model", device="cpu")
        assert model.device == "cpu"
        
    def test_get_nonexistent_model(self):
        """Test error when getting nonexistent model."""
        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            get_model("nonexistent")
            
    def test_list_models_returns_list(self):
        """Test that list_models returns a list."""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) >= 1
        
    def test_duplicate_registration_error(self):
        """Test error when registering duplicate model name."""
        with pytest.raises(ValueError, match="already registered"):
            @register_model("test_model")
            class DuplicateModel(ModelAdapter):
                def generate(self, prompt, **kwargs):
                    pass
                    
                @property  
                def requirements(self):
                    return {}