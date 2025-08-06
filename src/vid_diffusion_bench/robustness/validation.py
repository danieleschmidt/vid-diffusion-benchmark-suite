"""Input validation and sanitization for robust benchmarking.

This module provides comprehensive validation for all inputs to the benchmarking
system, ensuring data integrity and preventing errors before they occur.
"""

import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .error_handling import ValidationError, BenchmarkException


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    severity: ValidationSeverity
    field: str
    message: str
    suggested_fix: Optional[str] = None
    corrected_value: Optional[Any] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    is_valid: bool
    results: List[ValidationResult]
    errors: List[ValidationResult]
    warnings: List[ValidationResult]
    corrections_applied: int = 0
    
    def __post_init__(self):
        self.errors = [r for r in self.results if r.severity == ValidationSeverity.ERROR]
        self.warnings = [r for r in self.results if r.severity == ValidationSeverity.WARNING]


class BaseValidator:
    """Base class for validators."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: Whether to treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, data: Any) -> ValidationReport:
        """Validate data and return comprehensive report.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation report
        """
        results = self._perform_validation(data)
        
        # Determine overall validity
        has_errors = any(r.severity == ValidationSeverity.ERROR for r in results)
        has_critical = any(r.severity == ValidationSeverity.CRITICAL for r in results)
        has_warnings = any(r.severity == ValidationSeverity.WARNING for r in results)
        
        is_valid = not has_errors and not has_critical
        if self.strict_mode:
            is_valid = is_valid and not has_warnings
        
        report = ValidationReport(is_valid=is_valid, results=results)
        
        self.logger.debug(f"Validation completed: {len(results)} checks, valid={is_valid}")
        return report
    
    def _perform_validation(self, data: Any) -> List[ValidationResult]:
        """Perform specific validation checks. To be implemented by subclasses."""
        raise NotImplementedError


class InputValidator(BaseValidator):
    """Validator for general input parameters."""
    
    def _perform_validation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate input parameters."""
        results = []
        
        # Check required fields
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in data:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field=field,
                    message=f"Required field '{field}' is missing",
                    suggested_fix=f"Add '{field}' to input data"
                ))
            elif data[field] is None:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field=field,
                    message=f"Required field '{field}' cannot be None",
                    suggested_fix=f"Provide a valid value for '{field}'"
                ))
        
        # Type validation
        results.extend(self._validate_types(data))
        
        # Range validation
        results.extend(self._validate_ranges(data))
        
        # Format validation
        results.extend(self._validate_formats(data))
        
        # Dependency validation
        results.extend(self._validate_dependencies(data))
        
        return results
    
    def get_required_fields(self) -> List[str]:
        """Get list of required fields. Override in subclasses."""
        return []
    
    def _validate_types(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data types."""
        results = []
        type_specs = self.get_type_specifications()
        
        for field, expected_types in type_specs.items():
            if field in data and data[field] is not None:
                value = data[field]
                if not isinstance(value, expected_types):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Field '{field}' has type {type(value).__name__}, expected {expected_types}",
                        suggested_fix=f"Convert '{field}' to {expected_types}"
                    ))
        
        return results
    
    def _validate_ranges(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate numeric ranges."""
        results = []
        range_specs = self.get_range_specifications()
        
        for field, (min_val, max_val) in range_specs.items():
            if field in data and data[field] is not None:
                value = data[field]
                
                if isinstance(value, (int, float)):
                    if min_val is not None and value < min_val:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field=field,
                            message=f"Field '{field}' value {value} is below minimum {min_val}",
                            suggested_fix=f"Set '{field}' to at least {min_val}",
                            corrected_value=max(value, min_val)
                        ))
                    
                    if max_val is not None and value > max_val:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field=field,
                            message=f"Field '{field}' value {value} exceeds maximum {max_val}",
                            suggested_fix=f"Set '{field}' to at most {max_val}",
                            corrected_value=min(value, max_val)
                        ))
        
        return results
    
    def _validate_formats(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate string formats."""
        results = []
        format_specs = self.get_format_specifications()
        
        for field, pattern in format_specs.items():
            if field in data and data[field] is not None:
                value = data[field]
                
                if isinstance(value, str):
                    if not re.match(pattern, value):
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field=field,
                            message=f"Field '{field}' does not match expected format",
                            suggested_fix=f"Ensure '{field}' matches pattern: {pattern}"
                        ))
        
        return results
    
    def _validate_dependencies(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate field dependencies."""
        results = []
        dependencies = self.get_dependency_specifications()
        
        for field, required_fields in dependencies.items():
            if field in data and data[field] is not None:
                for required_field in required_fields:
                    if required_field not in data or data[required_field] is None:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field=field,
                            message=f"Field '{field}' requires '{required_field}' to be set",
                            suggested_fix=f"Set '{required_field}' when using '{field}'"
                        ))
        
        return results
    
    def get_type_specifications(self) -> Dict[str, type]:
        """Get type specifications. Override in subclasses."""
        return {}
    
    def get_range_specifications(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Get range specifications. Override in subclasses."""
        return {}
    
    def get_format_specifications(self) -> Dict[str, str]:
        """Get format specifications. Override in subclasses."""
        return {}
    
    def get_dependency_specifications(self) -> Dict[str, List[str]]:
        """Get dependency specifications. Override in subclasses."""
        return {}


class ModelValidator(InputValidator):
    """Validator for model-related parameters."""
    
    def get_required_fields(self) -> List[str]:
        return ["model_name"]
    
    def get_type_specifications(self) -> Dict[str, type]:
        return {
            "model_name": str,
            "device": str,
            "precision": str,
            "batch_size": int,
            "max_length": int,
            "num_frames": int,
            "fps": int,
            "resolution": tuple
        }
    
    def get_range_specifications(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {
            "batch_size": (1, 32),
            "max_length": (1, 1000),
            "num_frames": (1, 1000),
            "fps": (1, 60),
            "precision": None
        }
    
    def get_format_specifications(self) -> Dict[str, str]:
        return {
            "model_name": r"^[a-zA-Z0-9_\-\.]+$",
            "device": r"^(cpu|cuda(:\d+)?|mps)$",
            "precision": r"^(fp16|fp32|bf16|int8)$"
        }
    
    def _perform_validation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Enhanced validation for model parameters."""
        results = super()._perform_validation(data)
        
        # Validate resolution tuple
        if "resolution" in data and data["resolution"] is not None:
            resolution = data["resolution"]
            if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
                width, height = resolution
                if not all(isinstance(dim, int) and dim > 0 for dim in [width, height]):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field="resolution",
                        message="Resolution dimensions must be positive integers",
                        suggested_fix="Use positive integers for width and height"
                    ))
                
                # Check reasonable resolution limits
                if width > 4096 or height > 4096:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field="resolution",
                        message="Resolution is very high and may cause memory issues",
                        suggested_fix="Consider using lower resolution for stability"
                    ))
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field="resolution",
                    message="Resolution must be a tuple/list of two integers (width, height)",
                    suggested_fix="Use format: (width, height)"
                ))
        
        # Validate model name exists (if we have a model registry)
        if "model_name" in data:
            model_name = data["model_name"]
            if not self._check_model_availability(model_name):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field="model_name",
                    message=f"Model '{model_name}' may not be available",
                    suggested_fix="Check model registry or download model"
                ))
        
        # Validate device compatibility
        if "device" in data and "precision" in data:
            device = data["device"]
            precision = data["precision"]
            
            if device == "cpu" and precision in ["fp16", "bf16"]:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field="precision",
                    message="Half precision may not be supported on CPU",
                    suggested_fix="Use fp32 precision for CPU inference",
                    corrected_value="fp32"
                ))
        
        return results
    
    def _check_model_availability(self, model_name: str) -> bool:
        """Check if model is available. Override with actual implementation."""
        # Placeholder - in real implementation, check model registry
        known_models = [
            "svd-xt", "svd-xt-1.1", "pika-lumiere", "cogvideo", 
            "modelscope-v2", "dreamvideo-v3"
        ]
        return model_name in known_models


class MetricValidator(InputValidator):
    """Validator for evaluation metrics."""
    
    def get_required_fields(self) -> List[str]:
        return ["metrics"]
    
    def get_type_specifications(self) -> Dict[str, type]:
        return {
            "metrics": list,
            "reference_dataset": str,
            "compute_fvd": bool,
            "compute_is": bool,
            "compute_clip": bool,
            "compute_temporal": bool
        }
    
    def _perform_validation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Enhanced validation for metric parameters."""
        results = super()._perform_validation(data)
        
        # Validate metrics list
        if "metrics" in data and data["metrics"] is not None:
            metrics = data["metrics"]
            if isinstance(metrics, list):
                valid_metrics = [
                    "fvd", "is", "clip_similarity", "temporal_consistency",
                    "perceptual_quality", "motion_coherence", "semantic_consistency"
                ]
                
                for metric in metrics:
                    if metric not in valid_metrics:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field="metrics",
                            message=f"Unknown metric '{metric}'",
                            suggested_fix=f"Use one of: {', '.join(valid_metrics)}"
                        ))
                
                # Check for empty metrics list
                if len(metrics) == 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field="metrics",
                        message="Metrics list cannot be empty",
                        suggested_fix="Include at least one evaluation metric"
                    ))
        
        # Validate reference dataset
        if "reference_dataset" in data and data["reference_dataset"] is not None:
            dataset = data["reference_dataset"]
            valid_datasets = ["ucf101", "kinetics600", "sky_timelapse", "custom"]
            
            if dataset not in valid_datasets:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field="reference_dataset",
                    message=f"Unknown reference dataset '{dataset}'",
                    suggested_fix=f"Use one of: {', '.join(valid_datasets)}"
                ))
        
        return results


class ConfigValidator(InputValidator):
    """Validator for configuration files and parameters."""
    
    def get_required_fields(self) -> List[str]:
        return ["experiment_name", "models", "prompts"]
    
    def get_type_specifications(self) -> Dict[str, type]:
        return {
            "experiment_name": str,
            "models": list,
            "prompts": list,
            "metrics": list,
            "seeds": list,
            "output_dir": str,
            "save_results": bool,
            "parallel": bool,
            "max_workers": int
        }
    
    def get_range_specifications(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {
            "max_workers": (1, 16)
        }
    
    def _perform_validation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Enhanced validation for configuration."""
        results = super()._perform_validation(data)
        
        # Validate experiment name
        if "experiment_name" in data and data["experiment_name"] is not None:
            name = data["experiment_name"]
            if len(name.strip()) == 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field="experiment_name",
                    message="Experiment name cannot be empty",
                    suggested_fix="Provide a descriptive experiment name"
                ))
        
        # Validate models list
        if "models" in data and data["models"] is not None:
            models = data["models"]
            if isinstance(models, list):
                if len(models) == 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field="models",
                        message="Models list cannot be empty",
                        suggested_fix="Include at least one model to evaluate"
                    ))
                
                # Check for duplicates
                if len(models) != len(set(models)):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field="models",
                        message="Duplicate models found in list",
                        suggested_fix="Remove duplicate model entries"
                    ))
        
        # Validate prompts list
        if "prompts" in data and data["prompts"] is not None:
            prompts = data["prompts"]
            if isinstance(prompts, list):
                if len(prompts) == 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field="prompts",
                        message="Prompts list cannot be empty",
                        suggested_fix="Include at least one prompt for evaluation"
                    ))
                
                # Check prompt quality
                for i, prompt in enumerate(prompts):
                    if not isinstance(prompt, str):
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field=f"prompts[{i}]",
                            message=f"Prompt {i} is not a string",
                            suggested_fix="Ensure all prompts are strings"
                        ))
                    elif len(prompt.strip()) == 0:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field=f"prompts[{i}]",
                            message=f"Prompt {i} is empty",
                            suggested_fix="Provide descriptive prompts"
                        ))
                    elif len(prompt) > 500:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field=f"prompts[{i}]",
                            message=f"Prompt {i} is very long ({len(prompt)} chars)",
                            suggested_fix="Consider shorter, more focused prompts"
                        ))
        
        # Validate seeds
        if "seeds" in data and data["seeds"] is not None:
            seeds = data["seeds"]
            if isinstance(seeds, list):
                for i, seed in enumerate(seeds):
                    if not isinstance(seed, int):
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field=f"seeds[{i}]",
                            message=f"Seed {i} is not an integer",
                            suggested_fix="Use integer values for random seeds"
                        ))
                    elif seed < 0 or seed > 2**32 - 1:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field=f"seeds[{i}]",
                            message=f"Seed {i} is outside recommended range",
                            suggested_fix="Use seeds between 0 and 2^32-1"
                        ))
        
        # Validate output directory
        if "output_dir" in data and data["output_dir"] is not None:
            output_dir = Path(data["output_dir"])
            try:
                # Check if parent directory is writable
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                if not output_dir.parent.is_dir():
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field="output_dir",
                        message="Output directory parent does not exist",
                        suggested_fix="Create parent directories or use existing path"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field="output_dir",
                    message=f"Cannot access output directory: {e}",
                    suggested_fix="Check directory permissions and path validity"
                ))
        
        return results


class DataValidator(BaseValidator):
    """Validator for input data (tensors, videos, etc.)."""
    
    def _perform_validation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate input data."""
        results = []
        
        # Validate video tensors
        if "videos" in data:
            results.extend(self._validate_video_tensors(data["videos"]))
        
        # Validate prompts
        if "prompts" in data:
            results.extend(self._validate_prompts(data["prompts"]))
        
        # Validate reference data
        if "reference_videos" in data:
            results.extend(self._validate_reference_data(data["reference_videos"]))
        
        return results
    
    def _validate_video_tensors(self, videos: Any) -> List[ValidationResult]:
        """Validate video tensor data."""
        results = []
        
        if not isinstance(videos, (list, torch.Tensor, np.ndarray)):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field="videos",
                message="Videos must be list, torch.Tensor, or numpy.ndarray",
                suggested_fix="Convert videos to supported format"
            ))
            return results
        
        # Convert to list if single tensor
        if isinstance(videos, (torch.Tensor, np.ndarray)):
            videos = [videos]
        
        for i, video in enumerate(videos):
            video_results = self._validate_single_video_tensor(video, f"videos[{i}]")
            results.extend(video_results)
        
        return results
    
    def _validate_single_video_tensor(self, video: Any, field: str) -> List[ValidationResult]:
        """Validate single video tensor."""
        results = []
        
        # Check if it's a tensor
        if not isinstance(video, (torch.Tensor, np.ndarray)):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field=field,
                message="Video must be torch.Tensor or numpy.ndarray",
                suggested_fix="Convert to tensor format"
            ))
            return results
        
        # Check dimensionality
        if video.ndim not in [4, 5]:  # (T,C,H,W) or (B,T,C,H,W)
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field=field,
                message=f"Video tensor has {video.ndim} dimensions, expected 4 or 5",
                suggested_fix="Use format (T,C,H,W) or (B,T,C,H,W)"
            ))
        else:
            shape = video.shape
            
            # Check minimum dimensions
            if video.ndim == 4:  # (T,C,H,W)
                T, C, H, W = shape
                if T < 1:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Video has {T} frames, need at least 1",
                        suggested_fix="Ensure video has at least 1 frame"
                    ))
                
                if C not in [1, 3]:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field=field,
                        message=f"Video has {C} channels, expected 1 or 3",
                        suggested_fix="Use grayscale (1) or RGB (3) channels"
                    ))
                
                if H < 32 or W < 32:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field=field,
                        message=f"Video resolution {H}x{W} is very small",
                        suggested_fix="Use resolution of at least 32x32"
                    ))
        
        # Check data type and range
        if isinstance(video, torch.Tensor):
            if video.dtype not in [torch.float16, torch.float32, torch.uint8]:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field=field,
                    message=f"Video tensor dtype {video.dtype} may cause issues",
                    suggested_fix="Use float32, float16, or uint8 dtype"
                ))
        
        # Check value ranges
        video_np = video.detach().cpu().numpy() if isinstance(video, torch.Tensor) else video
        min_val, max_val = video_np.min(), video_np.max()
        
        if min_val < 0 and max_val > 1:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                field=field,
                message=f"Video values in range [{min_val:.3f}, {max_val:.3f}]",
                suggested_fix="Normalize values to [0,1] or [-1,1] range"
            ))
        
        return results
    
    def _validate_prompts(self, prompts: Any) -> List[ValidationResult]:
        """Validate prompts."""
        results = []
        
        if not isinstance(prompts, list):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field="prompts",
                message="Prompts must be a list",
                suggested_fix="Convert prompts to list format"
            ))
            return results
        
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field=f"prompts[{i}]",
                    message=f"Prompt {i} is not a string",
                    suggested_fix="Convert prompt to string"
                ))
            elif len(prompt.strip()) == 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field=f"prompts[{i}]",
                    message=f"Prompt {i} is empty",
                    suggested_fix="Provide descriptive prompt text"
                ))
        
        return results
    
    def _validate_reference_data(self, reference_data: Any) -> List[ValidationResult]:
        """Validate reference data for metrics computation."""
        results = []
        
        # Similar validation as video tensors but for reference data
        if isinstance(reference_data, str):
            # Reference dataset name
            valid_datasets = ["ucf101", "kinetics600", "sky_timelapse"]
            if reference_data not in valid_datasets:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field="reference_videos",
                    message=f"Unknown reference dataset '{reference_data}'",
                    suggested_fix=f"Use one of: {', '.join(valid_datasets)}"
                ))
        else:
            # Actual reference video data
            results.extend(self._validate_video_tensors(reference_data))
        
        return results


# Composite validator that combines multiple validators
class CompositeValidator:
    """Composite validator that runs multiple validators."""
    
    def __init__(self, validators: List[BaseValidator]):
        """Initialize with list of validators."""
        self.validators = validators
        self.logger = logging.getLogger(__name__)
    
    def validate(self, data: Dict[str, Any]) -> ValidationReport:
        """Run all validators and combine results."""
        all_results = []
        
        for validator in self.validators:
            try:
                report = validator.validate(data)
                all_results.extend(report.results)
            except Exception as e:
                self.logger.error(f"Validator {validator.__class__.__name__} failed: {e}")
                all_results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field="validator",
                    message=f"Validator {validator.__class__.__name__} failed: {e}",
                    suggested_fix="Check validator implementation"
                ))
        
        # Determine overall validity
        has_errors = any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                        for r in all_results)
        
        return ValidationReport(is_valid=not has_errors, results=all_results)


# Auto-correction utilities
def auto_correct_data(data: Dict[str, Any], validation_report: ValidationReport) -> Dict[str, Any]:
    """Automatically correct data based on validation results where possible.
    
    Args:
        data: Original data
        validation_report: Validation report with corrections
        
    Returns:
        Corrected data dictionary
    """
    corrected_data = data.copy()
    corrections_applied = 0
    
    for result in validation_report.results:
        if result.corrected_value is not None:
            try:
                # Apply correction
                corrected_data[result.field] = result.corrected_value
                corrections_applied += 1
                logger.info(f"Auto-corrected {result.field}: {result.corrected_value}")
            except Exception as e:
                logger.warning(f"Failed to auto-correct {result.field}: {e}")
    
    validation_report.corrections_applied = corrections_applied
    return corrected_data


# Factory function for creating appropriate validators
def create_validator(validation_type: str, **kwargs) -> BaseValidator:
    """Create validator based on type.
    
    Args:
        validation_type: Type of validator to create
        **kwargs: Additional arguments for validator
        
    Returns:
        Appropriate validator instance
    """
    validators = {
        "input": InputValidator,
        "model": ModelValidator,
        "metric": MetricValidator,
        "config": ConfigValidator,
        "data": DataValidator
    }
    
    if validation_type not in validators:
        raise ValueError(f"Unknown validator type: {validation_type}")
    
    return validators[validation_type](**kwargs)


# Validation decorators
def validate_inputs(validator_type: str = "input", auto_correct: bool = False):
    """Decorator for automatic input validation.
    
    Args:
        validator_type: Type of validator to use
        auto_correct: Whether to attempt auto-correction
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Combine args and kwargs into data dict for validation
            data = kwargs.copy()
            
            # Create validator
            validator = create_validator(validator_type)
            
            # Validate
            report = validator.validate(data)
            
            if not report.is_valid:
                if auto_correct:
                    # Attempt auto-correction
                    corrected_data = auto_correct_data(data, report)
                    kwargs.update(corrected_data)
                    
                    # Re-validate corrected data
                    corrected_report = validator.validate(kwargs)
                    if not corrected_report.is_valid:
                        # Still invalid after correction
                        error_messages = [r.message for r in corrected_report.errors]
                        raise ValidationError(
                            field="multiple",
                            message=f"Validation failed: {'; '.join(error_messages)}"
                        )
                else:
                    # Raise validation error
                    error_messages = [r.message for r in report.errors]
                    raise ValidationError(
                        field="multiple", 
                        message=f"Validation failed: {'; '.join(error_messages)}"
                    )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator