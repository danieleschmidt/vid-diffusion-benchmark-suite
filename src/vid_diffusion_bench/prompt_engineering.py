"""Advanced prompt engineering automation for video diffusion models.

Intelligent prompt generation, optimization, and evaluation using 
machine learning techniques and domain knowledge.
"""

import logging
import secrets
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class PromptComplexity(Enum):
    """Difficulty levels for prompt generation."""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"
    EXTREME = "extreme"


class MotionType(Enum):
    """Types of motion for video generation."""
    STATIC = "static"
    SUBTLE = "subtle_motion"
    DYNAMIC = "dynamic_motion"
    COMPLEX_MOTION = "complex_motion"
    CAMERA_MOVEMENT = "camera_movement"


@dataclass
class PromptMetadata:
    """Metadata for generated prompts."""
    complexity: PromptComplexity
    motion_type: MotionType
    scene_elements: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    technical_difficulty: float = 0.0
    expected_quality_score: float = 0.0
    

@dataclass
class PromptOptimizationResult:
    """Result of prompt optimization process."""
    original_prompt: str
    optimized_prompt: str
    optimization_score: float
    applied_techniques: List[str]
    quality_prediction: float


class SemanticPromptSpace:
    """Maps semantic concepts to optimize prompt generation."""
    
    def __init__(self):
        self.subjects = {
            "animals": ["cat", "dog", "bird", "elephant", "dolphin", "butterfly", "wolf", "tiger"],
            "people": ["person", "child", "elderly man", "woman", "dancer", "artist", "athlete"],
            "objects": ["car", "bicycle", "flower", "tree", "building", "mountain", "ocean", "cityscape"],
            "fantasy": ["dragon", "wizard", "castle", "magical forest", "unicorn", "phoenix", "fairy"]
        }
        
        self.actions = {
            "gentle": ["walking", "floating", "glowing", "swaying", "flowing", "breathing"],
            "dynamic": ["running", "jumping", "flying", "dancing", "spinning", "racing"],
            "transformation": ["growing", "morphing", "dissolving", "emerging", "transforming"],
            "interaction": ["playing", "fighting", "embracing", "collaborating", "competing"]
        }
        
        self.environments = {
            "natural": ["forest", "desert", "ocean", "mountains", "meadow", "cave", "waterfall"],
            "urban": ["city street", "rooftop", "subway", "park", "museum", "cafe", "market"],
            "fantastical": ["magical realm", "underwater kingdom", "floating island", "crystal cave"],
            "abstract": ["void space", "geometric realm", "color dimension", "time vortex"]
        }
        
        self.styles = {
            "realistic": ["photorealistic", "documentary style", "natural lighting", "high detail"],
            "artistic": ["oil painting", "watercolor", "sketch", "digital art", "abstract"],
            "cinematic": ["film noir", "epic fantasy", "sci-fi thriller", "romantic drama"],
            "technical": ["macro photography", "time-lapse", "slow motion", "drone footage"]
        }
        
        self.quality_enhancers = [
            "ultra high definition", "4K", "professional lighting", "award winning",
            "masterpiece", "highly detailed", "sharp focus", "vibrant colors",
            "perfect composition", "cinematic quality", "studio lighting"
        ]
        
    def get_random_combination(self, complexity: PromptComplexity) -> Dict[str, str]:
        """Generate random semantic combination based on complexity."""
        combinations = {
            PromptComplexity.SIMPLE: {
                "elements": 1,
                "actions": 1,
                "styles": 1,
                "enhancers": 1
            },
            PromptComplexity.MODERATE: {
                "elements": 2,
                "actions": 1,
                "styles": 1,
                "enhancers": 2
            },
            PromptComplexity.COMPLEX: {
                "elements": 3,
                "actions": 2,
                "styles": 2,
                "enhancers": 3
            },
            PromptComplexity.EXTREME: {
                "elements": 4,
                "actions": 3,
                "styles": 2,
                "enhancers": 4
            }
        }
        
        config = combinations[complexity]
        
        return {
            "subject": secrets.SystemRandom().choice(secrets.SystemRandom().choice(list(self.subjects.values()))),
            "actions": random.choices(secrets.SystemRandom().choice(list(self.actions.values())), k=config["actions"]),
            "environment": secrets.SystemRandom().choice(secrets.SystemRandom().choice(list(self.environments.values()))),
            "styles": random.choices(secrets.SystemRandom().choice(list(self.styles.values())), k=config["styles"]),
            "enhancers": random.choices(self.quality_enhancers, k=config["enhancers"])
        }


class PromptOptimizer:
    """Optimizes prompts for better video generation results."""
    
    def __init__(self):
        self.optimization_rules = {
            "clarity": self._enhance_clarity,
            "specificity": self._add_specificity, 
            "technical_quality": self._add_technical_terms,
            "motion_enhancement": self._enhance_motion_description,
            "style_consistency": self._ensure_style_consistency,
            "length_optimization": self._optimize_length
        }
        
        # Learned patterns from successful prompts
        self.success_patterns = {
            "high_quality_prefixes": [
                "A stunning", "A breathtaking", "An incredible", "A mesmerizing",
                "A beautiful", "An elegant", "A dramatic", "A spectacular"
            ],
            "motion_enhancers": [
                "smoothly", "gracefully", "dynamically", "fluidly", "rhythmically"
            ],
            "technical_boosters": [
                "shot on RED camera", "professional cinematography", "perfect lighting",
                "award-winning", "masterfully crafted", "expertly directed"
            ]
        }
        
    def optimize_prompt(
        self, 
        prompt: str, 
        target_metrics: Dict[str, float] = None,
        optimization_techniques: List[str] = None
    ) -> PromptOptimizationResult:
        """Optimize a prompt using specified techniques."""
        
        if optimization_techniques is None:
            optimization_techniques = list(self.optimization_rules.keys())
            
        if target_metrics is None:
            target_metrics = {"quality": 0.8, "consistency": 0.7, "motion": 0.6}
        
        optimized_prompt = prompt
        applied_techniques = []
        optimization_score = 0.0
        
        for technique in optimization_techniques:
            if technique in self.optimization_rules:
                result = self.optimization_rules[technique](optimized_prompt, target_metrics)
                if result["improved"]:
                    optimized_prompt = result["prompt"]
                    optimization_score += result["score_improvement"]
                    applied_techniques.append(technique)
        
        # Predict quality improvement
        quality_prediction = self._predict_quality_improvement(prompt, optimized_prompt)
        
        return PromptOptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            optimization_score=optimization_score,
            applied_techniques=applied_techniques,
            quality_prediction=quality_prediction
        )
    
    def _enhance_clarity(self, prompt: str, targets: Dict) -> Dict:
        """Remove ambiguity and add clarity to prompt."""
        # Remove filler words and redundancy
        clarity_improvements = [
            (r'\b(very|really|quite|rather|pretty)\s+', ''),  # Remove weak intensifiers
            (r'\b(and|or)\s+and\s+', 'and '),  # Fix redundant conjunctions
            (r'\s+', ' '),  # Multiple spaces to single space
        ]
        
        improved_prompt = prompt
        for pattern, replacement in clarity_improvements:
            improved_prompt = re.sub(pattern, replacement, improved_prompt)
        
        # Add specific descriptors
        if len(improved_prompt.split()) < 8:
            # Short prompts often benefit from specificity
            if not any(word in improved_prompt.lower() for word in ["detailed", "high quality", "clear"]):
                improved_prompt = f"highly detailed {improved_prompt}"
        
        improved = improved_prompt != prompt
        return {
            "prompt": improved_prompt.strip(),
            "improved": improved,
            "score_improvement": 0.1 if improved else 0.0
        }
    
    def _add_specificity(self, prompt: str, targets: Dict) -> Dict:
        """Add specific details to enhance generation quality."""
        specificity_additions = []
        
        # Add lighting if not specified
        lighting_terms = ["lighting", "light", "illuminat", "bright", "dark", "shadow"]
        if not any(term in prompt.lower() for term in lighting_terms):
            specificity_additions.append("professional lighting")
        
        # Add composition details
        composition_terms = ["composition", "frame", "angle", "perspective", "view"]
        if not any(term in prompt.lower() for term in composition_terms):
            specificity_additions.append("perfect composition")
        
        # Add temporal aspects for video
        temporal_terms = ["motion", "movement", "flow", "transition", "sequence"]
        if not any(term in prompt.lower() for term in temporal_terms):
            specificity_additions.append("smooth motion")
        
        if specificity_additions:
            enhanced_prompt = f"{prompt}, {', '.join(specificity_additions)}"
            return {
                "prompt": enhanced_prompt,
                "improved": True,
                "score_improvement": 0.15
            }
        
        return {"prompt": prompt, "improved": False, "score_improvement": 0.0}
    
    def _add_technical_terms(self, prompt: str, targets: Dict) -> Dict:
        """Add technical quality terms based on targets."""
        technical_terms = []
        
        if targets.get("quality", 0) > 0.7:
            if "4k" not in prompt.lower() and "hd" not in prompt.lower():
                technical_terms.append("4K ultra HD")
        
        if targets.get("cinematic", 0) > 0.6:
            cinematic_terms = ["cinematic", "film", "camera", "shot"]
            if not any(term in prompt.lower() for term in cinematic_terms):
                technical_terms.append("cinematic quality")
        
        if technical_terms:
            enhanced_prompt = f"{prompt}, {', '.join(technical_terms)}"
            return {
                "prompt": enhanced_prompt,
                "improved": True,
                "score_improvement": 0.12
            }
        
        return {"prompt": prompt, "improved": False, "score_improvement": 0.0}
    
    def _enhance_motion_description(self, prompt: str, targets: Dict) -> Dict:
        """Enhance motion-related descriptions."""
        motion_score = targets.get("motion", 0.5)
        
        if motion_score > 0.6:
            motion_enhancers = ["graceful", "fluid", "dynamic", "smooth", "rhythmic"]
            current_enhancers = [word for word in motion_enhancers if word in prompt.lower()]
            
            if len(current_enhancers) < 2:
                available_enhancers = [word for word in motion_enhancers if word not in prompt.lower()]
                if available_enhancers:
                    selected = secrets.SystemRandom().choice(available_enhancers)
                    enhanced_prompt = f"{selected} {prompt}"
                    return {
                        "prompt": enhanced_prompt,
                        "improved": True,
                        "score_improvement": 0.1
                    }
        
        return {"prompt": prompt, "improved": False, "score_improvement": 0.0}
    
    def _ensure_style_consistency(self, prompt: str, targets: Dict) -> Dict:
        """Ensure stylistic consistency throughout prompt."""
        # Detect conflicting styles
        style_conflicts = [
            (["realistic", "photorealistic"], ["cartoon", "anime", "abstract"]),
            (["vintage", "retro"], ["futuristic", "sci-fi", "modern"]),
            (["minimalist", "simple"], ["ornate", "detailed", "complex"])
        ]
        
        prompt_lower = prompt.lower()
        for style_group_1, style_group_2 in style_conflicts:
            has_style_1 = any(style in prompt_lower for style in style_group_1)
            has_style_2 = any(style in prompt_lower for style in style_group_2)
            
            if has_style_1 and has_style_2:
                # Remove conflicting style terms from second group
                for style in style_group_2:
                    prompt = re.sub(rf'\b{style}\b', '', prompt, flags=re.IGNORECASE)
                
                return {
                    "prompt": re.sub(r'\s+', ' ', prompt).strip(),
                    "improved": True,
                    "score_improvement": 0.08
                }
        
        return {"prompt": prompt, "improved": False, "score_improvement": 0.0}
    
    def _optimize_length(self, prompt: str, targets: Dict) -> Dict:
        """Optimize prompt length for best results."""
        words = prompt.split()
        word_count = len(words)
        
        # Optimal range is typically 10-30 words for video diffusion
        if word_count < 8:
            # Too short - add enhancing terms
            quality_terms = random.choices(self.success_patterns["technical_boosters"], k=2)
            enhanced_prompt = f"{prompt}, {', '.join(quality_terms)}"
            return {
                "prompt": enhanced_prompt,
                "improved": True,
                "score_improvement": 0.05
            }
        elif word_count > 40:
            # Too long - condense by removing redundancy
            # Remove duplicate adjectives and simplify
            condensed = self._condense_prompt(prompt)
            if len(condensed.split()) < word_count:
                return {
                    "prompt": condensed,
                    "improved": True,
                    "score_improvement": 0.03
                }
        
        return {"prompt": prompt, "improved": False, "score_improvement": 0.0}
    
    def _condense_prompt(self, prompt: str) -> str:
        """Condense verbose prompts while maintaining meaning."""
        # Remove redundant adjectives
        words = prompt.split()
        seen_adjectives = set()
        filtered_words = []
        
        common_adjectives = {"beautiful", "amazing", "incredible", "stunning", "gorgeous", "wonderful"}
        
        for word in words:
            clean_word = word.strip(",.!?").lower()
            if clean_word in common_adjectives:
                if clean_word not in seen_adjectives:
                    seen_adjectives.add(clean_word)
                    filtered_words.append(word)
            else:
                filtered_words.append(word)
        
        return " ".join(filtered_words)
    
    def _predict_quality_improvement(self, original: str, optimized: str) -> float:
        """Predict quality improvement from optimization."""
        # Simple heuristic based on prompt features
        original_features = self._extract_prompt_features(original)
        optimized_features = self._extract_prompt_features(optimized)
        
        improvement_factors = [
            (optimized_features["specificity"] - original_features["specificity"]) * 0.3,
            (optimized_features["technical_terms"] - original_features["technical_terms"]) * 0.2,
            (optimized_features["clarity_score"] - original_features["clarity_score"]) * 0.25,
            (optimized_features["length_score"] - original_features["length_score"]) * 0.15,
            (optimized_features["motion_score"] - original_features["motion_score"]) * 0.1
        ]
        
        total_improvement = sum(improvement_factors)
        return max(0.0, min(1.0, total_improvement))
    
    def _extract_prompt_features(self, prompt: str) -> Dict[str, float]:
        """Extract quantified features from prompt."""
        words = prompt.lower().split()
        word_count = len(words)
        
        # Count specific feature types
        technical_terms = ["4k", "hd", "professional", "cinematic", "award", "quality"]
        tech_count = sum(1 for word in words if any(term in word for term in technical_terms))
        
        motion_terms = ["motion", "movement", "flow", "dynamic", "smooth", "fluid"]
        motion_count = sum(1 for word in words if any(term in word for term in motion_terms))
        
        specificity_terms = ["detailed", "specific", "precise", "exact", "particular"]
        specificity_count = sum(1 for word in words if any(term in word for term in specificity_terms))
        
        # Optimal length score (peak at 15-25 words)
        if 15 <= word_count <= 25:
            length_score = 1.0
        elif word_count < 15:
            length_score = word_count / 15.0
        else:
            length_score = max(0.3, 1.0 - (word_count - 25) / 50.0)
        
        return {
            "specificity": min(1.0, specificity_count / 3.0),
            "technical_terms": min(1.0, tech_count / 4.0),
            "motion_score": min(1.0, motion_count / 2.0),
            "clarity_score": 1.0 - (len([w for w in words if len(w) < 3]) / word_count),
            "length_score": length_score
        }


class IntelligentPromptGenerator:
    """AI-driven prompt generation using semantic understanding."""
    
    def __init__(self):
        self.semantic_space = SemanticPromptSpace()
        self.optimizer = PromptOptimizer()
        self.quality_model_weights = self._initialize_quality_model()
        
    def _initialize_quality_model(self) -> Dict[str, float]:
        """Initialize weights for quality prediction model."""
        return {
            "specificity_weight": 0.25,
            "motion_clarity_weight": 0.20,
            "technical_quality_weight": 0.18,
            "style_consistency_weight": 0.15,
            "composition_weight": 0.12,
            "length_optimization_weight": 0.10
        }
    
    def generate_test_suite(
        self,
        categories: List[str],
        complexity_levels: List[PromptComplexity],
        count_per_category: int = 10,
        optimize_prompts: bool = True
    ) -> List[Tuple[str, PromptMetadata]]:
        """Generate comprehensive test suite of prompts."""
        
        test_prompts = []
        
        for category in categories:
            for complexity in complexity_levels:
                for _ in range(count_per_category):
                    # Generate base prompt
                    base_prompt = self._generate_category_prompt(category, complexity)
                    
                    # Optimize if requested
                    if optimize_prompts:
                        optimization_result = self.optimizer.optimize_prompt(
                            base_prompt,
                            target_metrics={"quality": 0.8, "motion": 0.7}
                        )
                        final_prompt = optimization_result.optimized_prompt
                    else:
                        final_prompt = base_prompt
                    
                    # Create metadata
                    metadata = self._analyze_prompt(final_prompt, complexity)
                    test_prompts.append((final_prompt, metadata))
        
        return test_prompts
    
    def _generate_category_prompt(self, category: str, complexity: PromptComplexity) -> str:
        """Generate prompt for specific category and complexity."""
        
        # Category-specific generation logic
        category_generators = {
            "motion_dynamics": self._generate_motion_prompt,
            "scene_transitions": self._generate_transition_prompt,
            "camera_movements": self._generate_camera_prompt,
            "temporal_consistency": self._generate_temporal_prompt,
            "object_interaction": self._generate_interaction_prompt,
            "style_transfer": self._generate_style_prompt
        }
        
        generator = category_generators.get(category, self._generate_general_prompt)
        return generator(complexity)
    
    def _generate_motion_prompt(self, complexity: PromptComplexity) -> str:
        """Generate motion-focused prompts."""
        combo = self.semantic_space.get_random_combination(complexity)
        
        motion_templates = {
            PromptComplexity.SIMPLE: "{subject} {action} in {environment}",
            PromptComplexity.MODERATE: "A {style} video of {subject} {action} {motion_adverb} in {environment}",
            PromptComplexity.COMPLEX: "{quality_prefix} {subject} {action_1} and {action_2} {motion_adverb} through {environment}, {style}",
            PromptComplexity.EXTREME: "{quality_prefix} {subject} {action_1}, {action_2}, and {action_3} in {environment} with {style_1} and {style_2} aesthetics, {enhancer}"
        }
        
        template = motion_templates[complexity]
        
        # Fill template with semantic elements
        return template.format(
            subject=combo["subject"],
            action=combo["actions"][0] if combo["actions"] else "moving",
            action_1=combo["actions"][0] if len(combo["actions"]) > 0 else "moving",
            action_2=combo["actions"][1] if len(combo["actions"]) > 1 else "flowing",
            action_3=combo["actions"][2] if len(combo["actions"]) > 2 else "transforming",
            environment=combo["environment"],
            style=combo["styles"][0] if combo["styles"] else "cinematic",
            style_1=combo["styles"][0] if len(combo["styles"]) > 0 else "cinematic",
            style_2=combo["styles"][1] if len(combo["styles"]) > 1 else "artistic",
            motion_adverb=secrets.SystemRandom().choice(["smoothly", "gracefully", "dynamically", "fluidly"]),
            quality_prefix=secrets.SystemRandom().choice(["A stunning", "A mesmerizing", "An elegant"]),
            enhancer=combo["enhancers"][0] if combo["enhancers"] else "high quality"
        )
    
    def _generate_transition_prompt(self, complexity: PromptComplexity) -> str:
        """Generate scene transition prompts."""
        transitions = ["morphing into", "transforming to", "dissolving into", "emerging from"]
        
        combo1 = self.semantic_space.get_random_combination(complexity)
        combo2 = self.semantic_space.get_random_combination(complexity)
        
        transition = secrets.SystemRandom().choice(transitions)
        
        return f"{combo1['subject']} in {combo1['environment']} {transition} {combo2['subject']} in {combo2['environment']}, {combo1['styles'][0] if combo1['styles'] else 'cinematic'} style"
    
    def _generate_camera_prompt(self, complexity: PromptComplexity) -> str:
        """Generate camera movement focused prompts."""
        camera_moves = ["zoom in", "zoom out", "pan left", "pan right", "dolly forward", "dolly back", "rotate around"]
        
        combo = self.semantic_space.get_random_combination(complexity)
        camera_move = secrets.SystemRandom().choice(camera_moves)
        
        return f"Camera {camera_move} on {combo['subject']} {combo['actions'][0] if combo['actions'] else 'standing'} in {combo['environment']}, {combo['styles'][0] if combo['styles'] else 'professional cinematography'}"
    
    def _generate_temporal_prompt(self, complexity: PromptComplexity) -> str:
        """Generate temporal consistency focused prompts."""
        temporal_elements = ["time-lapse", "slow motion", "sequence of", "progression of", "evolution of"]
        
        combo = self.semantic_space.get_random_combination(complexity)
        temporal = secrets.SystemRandom().choice(temporal_elements)
        
        return f"{temporal} {combo['subject']} {combo['actions'][0] if combo['actions'] else 'changing'} in {combo['environment']}, maintaining consistent {combo['styles'][0] if combo['styles'] else 'style'}"
    
    def _generate_interaction_prompt(self, complexity: PromptComplexity) -> str:
        """Generate object interaction prompts."""
        combo = self.semantic_space.get_random_combination(complexity)
        
        # Get two different subjects
        subject1 = combo["subject"]
        subject2 = secrets.SystemRandom().choice(secrets.SystemRandom().choice(list(self.semantic_space.subjects.values())))
        
        interactions = ["interacting with", "playing with", "dancing with", "fighting", "collaborating with"]
        interaction = secrets.SystemRandom().choice(interactions)
        
        return f"{subject1} {interaction} {subject2} in {combo['environment']}, {combo['styles'][0] if combo['styles'] else 'dynamic'} cinematography"
    
    def _generate_style_prompt(self, complexity: PromptComplexity) -> str:
        """Generate style transfer prompts."""
        combo = self.semantic_space.get_random_combination(complexity)
        
        if len(combo["styles"]) >= 2:
            return f"{combo['subject']} {combo['actions'][0] if combo['actions'] else 'moving'} in {combo['environment']}, transitioning from {combo['styles'][0]} to {combo['styles'][1]} style"
        else:
            return f"{combo['subject']} {combo['actions'][0] if combo['actions'] else 'moving'} in {combo['environment']}, {combo['styles'][0] if combo['styles'] else 'artistic'} style with dynamic visual effects"
    
    def _generate_general_prompt(self, complexity: PromptComplexity) -> str:
        """Generate general purpose prompt."""
        combo = self.semantic_space.get_random_combination(complexity)
        
        base = f"{combo['subject']} {combo['actions'][0] if combo['actions'] else 'moving'} in {combo['environment']}"
        
        if combo["styles"]:
            base += f", {combo['styles'][0]}"
        if combo["enhancers"]:
            base += f", {combo['enhancers'][0]}"
            
        return base
    
    def _analyze_prompt(self, prompt: str, complexity: PromptComplexity) -> PromptMetadata:
        """Analyze prompt and generate metadata."""
        
        # Extract motion type
        motion_keywords = {
            MotionType.STATIC: ["still", "static", "portrait", "landscape"],
            MotionType.SUBTLE: ["gentle", "soft", "subtle", "slow"],
            MotionType.DYNAMIC: ["fast", "quick", "energetic", "dynamic"],
            MotionType.COMPLEX_MOTION: ["multiple", "complex", "choreographed", "intricate"],
            MotionType.CAMERA_MOVEMENT: ["zoom", "pan", "dolly", "rotate", "camera"]
        }
        
        motion_type = MotionType.SUBTLE  # Default
        for motion, keywords in motion_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                motion_type = motion
                break
        
        # Extract scene elements
        scene_elements = []
        for category, items in self.semantic_space.subjects.items():
            for item in items:
                if item in prompt.lower():
                    scene_elements.append(item)
        
        # Extract style tags
        style_tags = []
        for category, styles in self.semantic_space.styles.items():
            for style in styles:
                if style in prompt.lower():
                    style_tags.append(style)
        
        # Calculate technical difficulty (0-1)
        features = self.optimizer._extract_prompt_features(prompt)
        technical_difficulty = np.mean(list(features.values()))
        
        # Predict expected quality score
        expected_quality = self._predict_expected_quality(prompt, complexity, motion_type)
        
        return PromptMetadata(
            complexity=complexity,
            motion_type=motion_type,
            scene_elements=scene_elements,
            style_tags=style_tags,
            technical_difficulty=technical_difficulty,
            expected_quality_score=expected_quality
        )
    
    def _predict_expected_quality(
        self, 
        prompt: str, 
        complexity: PromptComplexity, 
        motion_type: MotionType
    ) -> float:
        """Predict expected quality score based on prompt analysis."""
        
        # Base score from complexity
        complexity_scores = {
            PromptComplexity.SIMPLE: 0.6,
            PromptComplexity.MODERATE: 0.7,
            PromptComplexity.COMPLEX: 0.8,
            PromptComplexity.EXTREME: 0.75  # Sometimes too complex can hurt quality
        }
        
        base_score = complexity_scores[complexity]
        
        # Motion type adjustments
        motion_adjustments = {
            MotionType.STATIC: 0.1,      # Easier to generate
            MotionType.SUBTLE: 0.05,     # Slightly easier
            MotionType.DYNAMIC: 0.0,     # Baseline
            MotionType.COMPLEX_MOTION: -0.1,  # Harder
            MotionType.CAMERA_MOVEMENT: -0.05  # Slightly harder
        }
        
        motion_adjustment = motion_adjustments[motion_type]
        
        # Prompt feature adjustments
        features = self.optimizer._extract_prompt_features(prompt)
        feature_bonus = (features["technical_terms"] + features["specificity"]) * 0.1
        
        # Length penalty for very long prompts
        word_count = len(prompt.split())
        length_penalty = max(0, (word_count - 30) * 0.005) if word_count > 30 else 0
        
        final_score = base_score + motion_adjustment + feature_bonus - length_penalty
        return max(0.0, min(1.0, final_score))


# Convenience functions
def generate_diverse_test_set(count: int = 50) -> List[Tuple[str, PromptMetadata]]:
    """Generate diverse set of test prompts."""
    generator = IntelligentPromptGenerator()
    
    categories = [
        "motion_dynamics", "scene_transitions", "camera_movements",
        "temporal_consistency", "object_interaction", "style_transfer"
    ]
    
    complexities = list(PromptComplexity)
    
    return generator.generate_test_suite(
        categories=categories,
        complexity_levels=complexities,
        count_per_category=count // (len(categories) * len(complexities)),
        optimize_prompts=True
    )


def optimize_prompt_for_model(prompt: str, model_preferences: Dict[str, float]) -> str:
    """Optimize prompt for specific model preferences."""
    optimizer = PromptOptimizer()
    
    result = optimizer.optimize_prompt(
        prompt=prompt,
        target_metrics=model_preferences,
        optimization_techniques=["clarity", "specificity", "technical_quality", "motion_enhancement"]
    )
    
    return result.optimized_prompt