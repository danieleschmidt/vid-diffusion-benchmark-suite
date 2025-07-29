"""Standard prompt sets for evaluation."""

from typing import List, Dict
from enum import Enum


class PromptCategories(Enum):
    """Categories for prompt classification."""
    MOTION_DYNAMICS = "motion_dynamics"
    SCENE_TRANSITIONS = "scene_transitions" 
    CAMERA_MOVEMENTS = "camera_movements"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


class StandardPrompts:
    """Curated sets of standardized evaluation prompts."""
    
    DIVERSE_SET_V2: List[str] = [
        "A cat playing piano in a sunlit room",
        "Ocean waves crashing against rocky cliffs at sunset",
        "A person dancing in falling snow",
        "Time-lapse of flowers blooming in a garden",
        "A train moving through a mountain tunnel",
    ]
    
    MOTION_FOCUSED: List[str] = [
        "A hummingbird hovering near a flower",
        "Water droplets falling in slow motion", 
        "A gymnast performing a backflip",
        "Leaves rustling in the wind",
        "A spinning windmill in a field",
    ]
    
    CAMERA_MOVEMENTS: List[str] = [
        "Camera panning across a city skyline",
        "Drone shot flying over a forest",
        "Close-up zoom on a butterfly's wings",
        "Tracking shot following a running dog",
        "Aerial view descending to street level",
    ]


class PromptGenerator:
    """Dynamic prompt generation for testing."""
    
    def create_test_suite(
        self,
        categories: List[PromptCategories],
        count_per_category: int = 10,
        difficulty: str = "medium"
    ) -> List[str]:
        """Generate diverse test prompt suite.
        
        Args:
            categories: Prompt categories to include
            count_per_category: Number of prompts per category
            difficulty: Difficulty level (easy/medium/challenging)
            
        Returns:
            List of generated prompts
        """
        # Implementation placeholder
        return StandardPrompts.DIVERSE_SET_V2[:count_per_category]