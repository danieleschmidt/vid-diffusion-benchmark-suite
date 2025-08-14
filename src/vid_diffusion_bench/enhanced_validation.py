"""Enhanced validation and error handling for video diffusion benchmarks."""

import logging
import re
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    RESEARCH = "research"


class ValidationError(Exception):
    """Custom validation error with detailed context."""
    
    def __init__(self, message: str, code: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}
        self.timestamp = time.time()


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[str]
    metadata: Dict[str, Any]
    validation_time: float


class PromptValidator:
    """Comprehensive prompt validation."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup validation patterns."""
        # Potentially harmful patterns
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'file://',
            r'\\\\[a-zA-Z0-9]',  # UNC paths
            r'exec\s*\(',
            r'eval\s*\(',
            r'import\s+os',
            r'subprocess\.',
            r'__import__'
        ]
        
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(?:nude|naked|sex|porn|explicit)\b',
            r'\b(?:violence|blood|gore|death|kill)\b',
            r'\b(?:drug|cocaine|heroin|methamphetamine)\b'
        ]
        
        # Quality indicators
        self.quality_patterns = [
            r'\b(?:high.?quality|8k|4k|detailed|realistic)\b',
            r'\b(?:professional|cinematic|beautiful)\b'
        ]
        
    def validate_prompt(self, prompt: str) -> ValidationResult:
        """Validate a single prompt."""
        start_time = time.time()
        errors = []
        warnings = []
        metadata = {}
        
        # Basic validation
        if not prompt or not prompt.strip():
            errors.append(ValidationError(
                "Empty or whitespace-only prompt",
                "EMPTY_PROMPT",
                {"prompt": prompt}
            ))
            
        # Length validation
        if len(prompt) > 1000:
            errors.append(ValidationError(
                f"Prompt too long: {len(prompt)} characters (max 1000)",
                "PROMPT_TOO_LONG",
                {"length": len(prompt), "max_length": 1000}
            ))
        elif len(prompt) < 3:
            warnings.append(f"Very short prompt: {len(prompt)} characters")
            
        # Security validation
        for pattern in self.dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                errors.append(ValidationError(
                    f"Potentially dangerous pattern detected: {pattern}",
                    "DANGEROUS_PATTERN",
                    {"pattern": pattern, "prompt": prompt[:100]}
                ))
        
        # Content validation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.RESEARCH]:
            for pattern in self.inappropriate_patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    errors.append(ValidationError(
                        f"Inappropriate content detected",
                        "INAPPROPRIATE_CONTENT",
                        {"pattern": pattern}
                    ))
        
        # Quality analysis
        quality_score = 0
        for pattern in self.quality_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                quality_score += 1
                
        metadata.update({
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "quality_score": quality_score,
            "has_quality_indicators": quality_score > 0,
            "complexity": self._calculate_complexity(prompt)
        })
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
            validation_time=validation_time
        )
        
    def validate_prompt_batch(self, prompts: List[str]) -> Dict[int, ValidationResult]:
        """Validate multiple prompts."""
        results = {}
        
        for i, prompt in enumerate(prompts):
            try:
                results[i] = self.validate_prompt(prompt)
            except Exception as e:
                results[i] = ValidationResult(
                    is_valid=False,
                    errors=[ValidationError(
                        f"Validation failed: {str(e)}",
                        "VALIDATION_ERROR",
                        {"exception": str(e), "traceback": traceback.format_exc()}
                    )],
                    warnings=[],
                    metadata={},
                    validation_time=0.0
                )
        
        return results
        
    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score."""
        # Simple complexity based on vocabulary diversity and structure
        words = prompt.lower().split()
        unique_words = set(words)
        
        if not words:
            return 0.0
            
        diversity = len(unique_words) / len(words)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Complexity factors
        has_adjectives = any(word.endswith('ly') or word.endswith('ful') for word in words)
        has_complex_structure = ',' in prompt or ';' in prompt
        
        complexity = (
            diversity * 0.4 + 
            min(avg_word_length / 10, 1.0) * 0.3 +
            (0.2 if has_adjectives else 0) +
            (0.1 if has_complex_structure else 0)
        )
        
        return min(complexity, 1.0)


class ModelParameterValidator:
    """Validate model generation parameters."""
    
    def __init__(self):
        self.parameter_limits = {
            'num_frames': {'min': 1, 'max': 300, 'default': 16},
            'fps': {'min': 1, 'max': 60, 'default': 8},
            'width': {'min': 64, 'max': 2048, 'default': 512},
            'height': {'min': 64, 'max': 2048, 'default': 512},
            'num_inference_steps': {'min': 1, 'max': 200, 'default': 50},
            'guidance_scale': {'min': 0.1, 'max': 30.0, 'default': 7.5},
            'batch_size': {'min': 1, 'max': 16, 'default': 1}
        }
        
    def validate_parameters(self, **kwargs) -> ValidationResult:
        """Validate generation parameters."""
        start_time = time.time()
        errors = []
        warnings = []
        metadata = {}
        
        for param_name, value in kwargs.items():
            if param_name in self.parameter_limits:
                limits = self.parameter_limits[param_name]
                
                # Type validation
                if not isinstance(value, (int, float)):
                    errors.append(ValidationError(
                        f"Parameter {param_name} must be numeric, got {type(value)}",
                        "INVALID_TYPE",
                        {"parameter": param_name, "value": value, "expected_type": "numeric"}
                    ))
                    continue
                
                # Range validation
                if value < limits['min']:
                    errors.append(ValidationError(
                        f"Parameter {param_name}={value} below minimum {limits['min']}",
                        "BELOW_MINIMUM",
                        {"parameter": param_name, "value": value, "minimum": limits['min']}
                    ))
                elif value > limits['max']:
                    errors.append(ValidationError(
                        f"Parameter {param_name}={value} above maximum {limits['max']}",
                        "ABOVE_MAXIMUM", 
                        {"parameter": param_name, "value": value, "maximum": limits['max']}
                    ))
                
                # Reasonable value warnings
                if param_name == 'num_frames' and value > 120:
                    warnings.append(f"Very high frame count: {value} (may use excessive memory)")
                elif param_name == 'width' and value > 1024:
                    warnings.append(f"Very high resolution width: {value} (may be slow)")
                elif param_name == 'height' and value > 1024:
                    warnings.append(f"Very high resolution height: {value} (may be slow)")
        
        # Cross-parameter validation
        width = kwargs.get('width', 512)
        height = kwargs.get('height', 512)
        num_frames = kwargs.get('num_frames', 16)
        
        # Memory usage estimation
        estimated_memory_gb = (width * height * num_frames * 3 * 4) / (1024**3)  # Rough estimate
        metadata['estimated_memory_gb'] = estimated_memory_gb
        
        if estimated_memory_gb > 10:
            warnings.append(f"High estimated memory usage: {estimated_memory_gb:.1f} GB")
            
        # Aspect ratio validation
        aspect_ratio = width / height
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            warnings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        metadata.update({
            'total_pixels': width * height,
            'total_frames': num_frames,
            'aspect_ratio': aspect_ratio,
            'validated_parameters': list(kwargs.keys())
        })
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
            validation_time=validation_time
        )


class SystemResourceValidator:
    """Validate system resources before benchmark execution."""
    
    def __init__(self):
        self.min_requirements = {
            'memory_gb': 8.0,
            'disk_space_gb': 10.0,
            'gpu_memory_gb': 6.0
        }
        
    def validate_system(self) -> ValidationResult:
        """Validate system resources."""
        start_time = time.time()
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Memory validation
            try:
                import psutil
                memory = psutil.virtual_memory()
                available_memory_gb = memory.available / (1024**3)
                total_memory_gb = memory.total / (1024**3)
                
                metadata['memory'] = {
                    'total_gb': total_memory_gb,
                    'available_gb': available_memory_gb,
                    'used_percent': memory.percent
                }
                
                if available_memory_gb < self.min_requirements['memory_gb']:
                    errors.append(ValidationError(
                        f"Insufficient memory: {available_memory_gb:.1f}GB available, need {self.min_requirements['memory_gb']}GB",
                        "INSUFFICIENT_MEMORY",
                        {"available": available_memory_gb, "required": self.min_requirements['memory_gb']}
                    ))
                elif available_memory_gb < self.min_requirements['memory_gb'] * 1.5:
                    warnings.append(f"Low memory: {available_memory_gb:.1f}GB available")
                    
            except ImportError:
                warnings.append("Cannot validate memory - psutil not available")
            
            # Disk space validation
            try:
                disk_usage = Path.cwd().stat()
                # Simple check - in production would check actual disk space
                metadata['disk'] = {'checked': True}
                
            except Exception as e:
                warnings.append(f"Cannot validate disk space: {e}")
            
            # GPU validation
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    metadata['gpu'] = {
                        'available': True,
                        'count': gpu_count,
                        'memory_gb': gpu_memory_gb,
                        'name': torch.cuda.get_device_name(0)
                    }
                    
                    if gpu_memory_gb < self.min_requirements['gpu_memory_gb']:
                        warnings.append(f"Low GPU memory: {gpu_memory_gb:.1f}GB, recommended {self.min_requirements['gpu_memory_gb']}GB")
                else:
                    metadata['gpu'] = {'available': False}
                    warnings.append("No GPU available - will use CPU (slower)")
                    
            except ImportError:
                metadata['gpu'] = {'torch_available': False}
                warnings.append("Cannot validate GPU - torch not available")
                
        except Exception as e:
            errors.append(ValidationError(
                f"System validation failed: {str(e)}",
                "SYSTEM_VALIDATION_ERROR",
                {"exception": str(e)}
            ))
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
            validation_time=validation_time
        )


class BenchmarkInputValidator:
    """Comprehensive validation for benchmark inputs."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.prompt_validator = PromptValidator(validation_level)
        self.param_validator = ModelParameterValidator()
        self.system_validator = SystemResourceValidator()
        
    def validate_benchmark_request(
        self,
        model_name: str,
        prompts: List[str],
        **kwargs
    ) -> Dict[str, ValidationResult]:
        """Validate complete benchmark request."""
        results = {}
        
        # Model name validation
        if not model_name or not isinstance(model_name, str):
            results['model'] = ValidationResult(
                is_valid=False,
                errors=[ValidationError("Invalid model name", "INVALID_MODEL_NAME")],
                warnings=[],
                metadata={},
                validation_time=0.0
            )
        else:
            results['model'] = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                metadata={'model_name': model_name},
                validation_time=0.0
            )
        
        # Prompt validation
        prompt_results = self.prompt_validator.validate_prompt_batch(prompts)
        results['prompts'] = ValidationResult(
            is_valid=all(r.is_valid for r in prompt_results.values()),
            errors=[e for r in prompt_results.values() for e in r.errors],
            warnings=[w for r in prompt_results.values() for w in r.warnings],
            metadata={
                'total_prompts': len(prompts),
                'valid_prompts': sum(1 for r in prompt_results.values() if r.is_valid),
                'individual_results': {str(k): {
                    'is_valid': v.is_valid,
                    'error_count': len(v.errors),
                    'warning_count': len(v.warnings)
                } for k, v in prompt_results.items()}
            },
            validation_time=sum(r.validation_time for r in prompt_results.values())
        )
        
        # Parameter validation
        results['parameters'] = self.param_validator.validate_parameters(**kwargs)
        
        # System validation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.RESEARCH]:
            results['system'] = self.system_validator.validate_system()
        
        return results
        
    def create_validation_report(self, validation_results: Dict[str, ValidationResult]) -> str:
        """Create human-readable validation report."""
        report_lines = ["Benchmark Validation Report", "=" * 30, ""]
        
        total_errors = 0
        total_warnings = 0
        
        for category, result in validation_results.items():
            report_lines.append(f"{category.upper()}:")
            
            if result.is_valid:
                report_lines.append("  ✓ VALID")
            else:
                report_lines.append("  ✗ INVALID")
            
            if result.errors:
                total_errors += len(result.errors)
                report_lines.append("  Errors:")
                for error in result.errors:
                    report_lines.append(f"    - {error}")
            
            if result.warnings:
                total_warnings += len(result.warnings)
                report_lines.append("  Warnings:")
                for warning in result.warnings:
                    report_lines.append(f"    - {warning}")
            
            if result.metadata:
                key_metadata = {k: v for k, v in result.metadata.items() 
                              if k not in ['individual_results']}
                if key_metadata:
                    report_lines.append(f"  Metadata: {key_metadata}")
            
            report_lines.append("")
        
        # Summary
        report_lines.extend([
            "SUMMARY:",
            f"  Total Errors: {total_errors}",
            f"  Total Warnings: {total_warnings}",
            f"  Overall Status: {'VALID' if total_errors == 0 else 'INVALID'}",
            ""
        ])
        
        return "\n".join(report_lines)


# Utility functions for enhanced validation
def sanitize_prompt(prompt: str) -> str:
    """Sanitize prompt by removing potentially dangerous content."""
    # Remove HTML tags
    import re
    prompt = re.sub(r'<[^>]+>', '', prompt)
    
    # Remove script-like content
    prompt = re.sub(r'javascript:', '', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'data:text/html', '', prompt, flags=re.IGNORECASE)
    
    # Normalize whitespace
    prompt = ' '.join(prompt.split())
    
    return prompt.strip()


def calculate_benchmark_fingerprint(
    model_name: str, 
    prompts: List[str], 
    **kwargs
) -> str:
    """Calculate unique fingerprint for benchmark configuration."""
    config_data = {
        'model': model_name,
        'prompts': sorted(prompts),  # Sort for consistency
        'parameters': sorted(kwargs.items())
    }
    
    config_str = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def estimate_benchmark_duration(
    model_name: str,
    num_prompts: int,
    num_frames: int = 16,
    **kwargs
) -> float:
    """Estimate benchmark execution time in seconds."""
    # Base time estimates per model type (very rough)
    base_times = {
        'mock-fast': 0.1,
        'mock-efficient': 0.05,
        'mock-high-quality': 2.0,
        'mock-memory-intensive': 1.0,
        'cogvideo': 30.0,
        'modelscope': 15.0,
        'zeroscope': 25.0,
        'animatediff': 20.0
    }
    
    base_time = base_times.get(model_name, 10.0)  # Default 10s per prompt
    
    # Scale by complexity
    frame_multiplier = num_frames / 16
    prompt_multiplier = num_prompts
    
    estimated_time = base_time * frame_multiplier * prompt_multiplier
    
    # Add overhead (loading, metrics computation, etc.)
    overhead = 30.0  # 30 seconds overhead
    
    return estimated_time + overhead