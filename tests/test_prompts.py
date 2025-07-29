"""Tests for prompt generation and management."""

import pytest
from vid_diffusion_bench.prompts import StandardPrompts, PromptGenerator, PromptCategories


class TestStandardPrompts:
    """Test cases for StandardPrompts class."""
    
    def test_diverse_set_v2_exists(self):
        """Test that DIVERSE_SET_V2 prompt set exists."""
        assert hasattr(StandardPrompts, "DIVERSE_SET_V2")
        assert isinstance(StandardPrompts.DIVERSE_SET_V2, list)
        assert len(StandardPrompts.DIVERSE_SET_V2) > 0
        
    def test_motion_focused_exists(self):
        """Test that MOTION_FOCUSED prompt set exists."""
        assert hasattr(StandardPrompts, "MOTION_FOCUSED")
        assert isinstance(StandardPrompts.MOTION_FOCUSED, list)
        assert len(StandardPrompts.MOTION_FOCUSED) > 0
        
    def test_camera_movements_exists(self):
        """Test that CAMERA_MOVEMENTS prompt set exists."""
        assert hasattr(StandardPrompts, "CAMERA_MOVEMENTS")
        assert isinstance(StandardPrompts.CAMERA_MOVEMENTS, list)
        assert len(StandardPrompts.CAMERA_MOVEMENTS) > 0
        
    def test_all_prompts_are_strings(self):
        """Test that all prompts are strings."""
        for prompt in StandardPrompts.DIVERSE_SET_V2:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestPromptCategories:
    """Test cases for PromptCategories enum."""
    
    def test_categories_exist(self):
        """Test that all expected categories exist."""
        expected_categories = [
            "MOTION_DYNAMICS",
            "SCENE_TRANSITIONS", 
            "CAMERA_MOVEMENTS",
            "TEMPORAL_CONSISTENCY"
        ]
        
        for category in expected_categories:
            assert hasattr(PromptCategories, category)
            
    def test_category_values(self):
        """Test category enum values."""
        assert PromptCategories.MOTION_DYNAMICS.value == "motion_dynamics"
        assert PromptCategories.SCENE_TRANSITIONS.value == "scene_transitions"
        assert PromptCategories.CAMERA_MOVEMENTS.value == "camera_movements"
        assert PromptCategories.TEMPORAL_CONSISTENCY.value == "temporal_consistency"


class TestPromptGenerator:
    """Test cases for PromptGenerator class."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = PromptGenerator()
        assert generator is not None
        
    def test_create_test_suite_basic(self):
        """Test basic test suite creation."""
        generator = PromptGenerator()
        categories = [PromptCategories.MOTION_DYNAMICS]
        
        prompts = generator.create_test_suite(categories)
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
        for prompt in prompts:
            assert isinstance(prompt, str)
            
    def test_create_test_suite_custom_count(self):
        """Test test suite creation with custom count."""
        generator = PromptGenerator()
        categories = [PromptCategories.MOTION_DYNAMICS]
        count = 3
        
        prompts = generator.create_test_suite(categories, count_per_category=count)
        assert len(prompts) == count
        
    def test_create_test_suite_multiple_categories(self):
        """Test test suite creation with multiple categories."""
        generator = PromptGenerator()
        categories = [
            PromptCategories.MOTION_DYNAMICS,
            PromptCategories.CAMERA_MOVEMENTS
        ]
        
        prompts = generator.create_test_suite(categories)
        assert isinstance(prompts, list)
        assert len(prompts) > 0