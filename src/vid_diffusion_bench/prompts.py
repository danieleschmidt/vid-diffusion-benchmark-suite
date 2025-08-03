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
        "Lightning illuminating storm clouds at night",
        "A chef flipping pancakes in a kitchen",
        "Birds flying in formation across the sky",
        "A waterfall cascading down moss-covered rocks",
        "Children playing soccer in a park",
        "Fire crackling in a stone fireplace",
        "A hot air balloon drifting over countryside",
        "Waves lapping against a sandy beach",
        "A painter creating artwork on canvas",
        "Smoke rising from a mountain campfire",
        "A dog chasing its tail in a yard",
        "Rain droplets hitting a window",
        "A street musician playing violin",
        "Butterflies landing on spring flowers",
        "A fountain spraying water in patterns"
    ]
    
    MOTION_FOCUSED: List[str] = [
        "A hummingbird hovering near a flower",
        "Water droplets falling in slow motion", 
        "A gymnast performing a backflip",
        "Leaves rustling in the wind",
        "A spinning windmill in a field",
        "A basketball player shooting a three-pointer",
        "A dancer performing a pirouette",
        "A motorcyclist taking a sharp turn",
        "A swimmer diving into a pool",
        "A skateboarder performing tricks",
        "Hair flowing in the wind",
        "A flag waving in the breeze",
        "A yo-yo spinning up and down",
        "A pendulum swinging back and forth",
        "A tennis player serving an ace"
    ]
    
    CAMERA_MOVEMENTS: List[str] = [
        "Camera panning across a city skyline",
        "Drone shot flying over a forest",
        "Close-up zoom on a butterfly's wings",
        "Tracking shot following a running dog",
        "Aerial view descending to street level",
        "Dolly shot approaching a castle gate",
        "Orbit shot around a blooming tree",
        "Tilt shot revealing a mountain peak",
        "Push-in shot on a person's eyes",
        "Pull-back shot from close-up to wide view",
        "Handheld camera walking through a crowd",
        "Steadicam shot following stairs upward",
        "Bird's eye view of traffic intersection",
        "Low angle shot looking up at skyscrapers",
        "Whip pan between two characters talking"
    ]
    
    SCENE_TRANSITIONS: List[str] = [
        "Day transitioning to night over a city",
        "Seasons changing in a park",
        "Desert transforming into oasis",
        "Empty room filling with party guests",
        "Calm lake becoming turbulent storm",
        "Sunset fading to starry night",
        "Building construction time-lapse",
        "Ice melting and flowing as water",
        "Seed growing into tall tree",
        "Tide moving from low to high"
    ]
    
    TEMPORAL_CONSISTENCY: List[str] = [
        "A clock ticking through several minutes",
        "A candle burning down over time",
        "A plant growing day by day",
        "Traffic patterns throughout the day",
        "Weather changing hour by hour",
        "A person aging over decades",
        "Paint drying on a canvas",
        "Fruit ripening on a tree",
        "Shadows moving with the sun",
        "A river flowing consistently downstream"
    ]
    
    CHALLENGING_SET: List[str] = [
        "Multiple people having a conversation with natural gestures",
        "A complex dance routine with synchronized movements",
        "Realistic fire with proper physics and lighting",
        "Detailed facial expressions during emotional speech",
        "Crowd scene with individual character movements",
        "Water simulation with accurate reflections and refractions",
        "Animal behavior with natural movement patterns",
        "Weather effects with atmospheric perspective",
        "Night scene with multiple light sources",
        "Action sequence with fast-paced movements"
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