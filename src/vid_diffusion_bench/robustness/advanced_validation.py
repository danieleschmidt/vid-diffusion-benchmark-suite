"""Advanced validation framework for robust benchmark execution.

This module provides comprehensive validation capabilities including:
- Input sanitization and validation
- Model output verification
- Data integrity checks
- Performance boundary validation
- Security validation for model inputs
"""

import torch
import numpy as np
import logging
import re
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    validation_score: float  # 0.0 to 1.0
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)
        
    def update_score(self, penalty: float):
        """Update validation score with penalty."""
        self.validation_score = max(0.0, self.validation_score - penalty)


class InputValidator:
    """Validates benchmark inputs for safety and correctness."""
    
    def __init__(self):
        self.prompt_patterns = {
            'suspicious_code': r'(?i)(exec|eval|import|__import__|open|file|subprocess)',
            'sql_injection': r'(?i)(union|select|drop|delete|insert|update|alter)',
            'path_traversal': r'(\.\.\/|\\\.\.\\|\/\.\.)',
            'script_tags': r'<script[^>]*>.*?</script>',
            'malicious_urls': r'(?i)(javascript:|data:|vbscript:|about:)'
        }
        
        self.max_prompt_length = 500
        self.max_batch_size = 32
        self.valid_resolutions = [(256, 256), (512, 512), (576, 1024), (1024, 576)]
        
    def validate_prompts(self, prompts: List[str]) -> ValidationResult:
        """Validate text prompts for safety and correctness."""
        result = ValidationResult(
            is_valid=True, 
            errors=[], 
            warnings=[], 
            metadata={},
            validation_score=1.0
        )
        
        if not prompts:
            result.add_error("No prompts provided")
            return result
            
        if len(prompts) > 100:
            result.add_warning(f"Large number of prompts ({len(prompts)}), consider batching")
            result.update_score(0.1)
            
        for i, prompt in enumerate(prompts):
            # Check prompt validity
            prompt_validation = self._validate_single_prompt(prompt, i)
            if not prompt_validation.is_valid:
                result.errors.extend(prompt_validation.errors)
                result.is_valid = False
            result.warnings.extend(prompt_validation.warnings)
            result.validation_score = min(result.validation_score, prompt_validation.validation_score)
            
        result.metadata = {
            'num_prompts': len(prompts),
            'avg_prompt_length': np.mean([len(p) for p in prompts]),
            'max_prompt_length': max([len(p) for p in prompts]),
            'unique_prompts': len(set(prompts))
        }
        
        return result
        
    def _validate_single_prompt(self, prompt: str, index: int) -> ValidationResult:
        """Validate a single prompt."""
        result = ValidationResult(
            is_valid=True,
            errors=[], 
            warnings=[],
            metadata={},
            validation_score=1.0
        )
        
        if not isinstance(prompt, str):
            result.add_error(f"Prompt {index}: Must be a string, got {type(prompt)}")
            return result
            
        if len(prompt.strip()) == 0:
            result.add_error(f"Prompt {index}: Empty prompt")
            return result
            
        if len(prompt) > self.max_prompt_length:
            result.add_warning(f"Prompt {index}: Very long prompt ({len(prompt)} chars)")
            result.update_score(0.2)
            
        # Check for suspicious patterns
        for pattern_name, pattern in self.prompt_patterns.items():
            if re.search(pattern, prompt):
                result.add_error(f"Prompt {index}: Suspicious content detected ({pattern_name})")
                result.update_score(0.5)
                
        # Check encoding
        try:
            prompt.encode('utf-8')
        except UnicodeEncodeError:
            result.add_error(f"Prompt {index}: Invalid UTF-8 encoding")
            
        # Check for balanced quotes/brackets
        quote_count = prompt.count('"') + prompt.count("'")
        if quote_count % 2 != 0:
            result.add_warning(f"Prompt {index}: Unbalanced quotes")
            result.update_score(0.1)
            
        return result
        
    def validate_generation_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate video generation parameters."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            metadata={},
            validation_score=1.0
        )
        
        # Required parameters
        required_params = ['num_frames', 'fps']
        for param in required_params:
            if param not in params:
                result.add_error(f"Missing required parameter: {param}")
                
        # Validate num_frames
        if 'num_frames' in params:
            frames = params['num_frames']
            if not isinstance(frames, int) or frames <= 0:
                result.add_error("num_frames must be a positive integer")
            elif frames > 240:  # 10 seconds at 24fps
                result.add_warning(f"Very long video ({frames} frames)")
                result.update_score(0.1)
                
        # Validate fps
        if 'fps' in params:
            fps = params['fps']
            if not isinstance(fps, (int, float)) or fps <= 0:
                result.add_error("fps must be a positive number")
            elif fps > 60:
                result.add_warning(f"Very high fps ({fps})")
                result.update_score(0.1)
                
        # Validate resolution
        if 'resolution' in params:
            resolution = params['resolution']
            if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
                result.add_error("resolution must be a tuple/list of 2 integers")
            elif resolution not in self.valid_resolutions:
                result.add_warning(f"Non-standard resolution: {resolution}")
                result.update_score(0.1)
                
        # Validate batch_size
        if 'batch_size' in params:
            batch_size = params['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                result.add_error("batch_size must be a positive integer")
            elif batch_size > self.max_batch_size:
                result.add_error(f"batch_size too large (max {self.max_batch_size})")
                
        result.metadata = dict(params)
        return result


class OutputValidator:
    """Validates model outputs for correctness and quality."""
    
    def __init__(self):
        self.min_pixel_value = -1.0
        self.max_pixel_value = 1.0
        self.max_memory_gb = 48.0  # Maximum reasonable GPU memory
        
    def validate_video_tensor(self, tensor: torch.Tensor, expected_shape: Optional[Tuple[int, ...]] = None) -> ValidationResult:
        """Validate generated video tensor."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[], 
            metadata={},
            validation_score=1.0
        )
        
        if not isinstance(tensor, torch.Tensor):
            result.add_error(f"Expected torch.Tensor, got {type(tensor)}")
            return result
            
        # Check tensor shape
        if len(tensor.shape) != 4:
            result.add_error(f"Expected 4D tensor (T, C, H, W), got {len(tensor.shape)}D")
            return result
            
        T, C, H, W = tensor.shape
        
        # Validate dimensions
        if T <= 0:
            result.add_error(f"Invalid time dimension: {T}")
        elif T > 300:  # Very long video
            result.add_warning(f"Very long video: {T} frames")
            result.update_score(0.1)
            
        if C not in [1, 3, 4]:  # Grayscale, RGB, or RGBA
            result.add_error(f"Invalid channel dimension: {C}")
            
        if H <= 0 or W <= 0:
            result.add_error(f"Invalid spatial dimensions: {H}x{W}")
        elif H > 1080 or W > 1920:  # Very high resolution
            result.add_warning(f"Very high resolution: {H}x{W}")
            result.update_score(0.1)
            
        # Check expected shape if provided
        if expected_shape and tuple(tensor.shape) != expected_shape:
            result.add_error(f"Shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
            
        # Validate data type
        if tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
            result.add_warning(f"Unusual data type: {tensor.dtype}")
            result.update_score(0.1)
            
        # Check for NaN or infinite values
        if torch.isnan(tensor).any():
            result.add_error("Tensor contains NaN values")
            
        if torch.isinf(tensor).any():
            result.add_error("Tensor contains infinite values")
            
        # Check value range
        min_val, max_val = tensor.min().item(), tensor.max().item()
        if min_val < self.min_pixel_value or max_val > self.max_pixel_value:
            result.add_warning(f"Pixel values outside expected range [{self.min_pixel_value}, {self.max_pixel_value}]: [{min_val:.3f}, {max_val:.3f}]")
            result.update_score(0.2)
            
        # Check for all-zero or all-same tensors
        if tensor.std().item() < 1e-6:
            result.add_error("Tensor has no variation (all pixels nearly identical)")
            
        # Memory usage check
        memory_gb = tensor.numel() * tensor.element_size() / (1024**3)
        if memory_gb > self.max_memory_gb:
            result.add_error(f"Tensor too large: {memory_gb:.2f}GB")
            
        result.metadata = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'memory_gb': memory_gb,
            'min_value': min_val,
            'max_value': max_val,
            'mean_value': tensor.mean().item(),
            'std_value': tensor.std().item()
        }
        
        return result
        
    def validate_metrics(self, metrics: Dict[str, float]) -> ValidationResult:
        """Validate computed metrics."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            metadata={}, 
            validation_score=1.0
        )
        
        expected_ranges = {
            'fvd': (0, 1000),  # FVD typically 0-500 for good models
            'inception_score': (0, 100),  # IS typically 0-50
            'clip_similarity': (0, 1),  # CLIP similarity 0-1
            'temporal_consistency': (0, 1),  # Consistency score 0-1
            'overall_score': (0, 100)  # Overall score 0-100
        }
        
        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                result.add_error(f"Metric {metric_name}: Must be numeric, got {type(value)}")
                continue
                
            if np.isnan(value):
                result.add_error(f"Metric {metric_name}: NaN value")
                continue
                
            if np.isinf(value):
                result.add_error(f"Metric {metric_name}: Infinite value")
                continue
                
            # Check expected ranges
            if metric_name in expected_ranges:
                min_val, max_val = expected_ranges[metric_name]
                if value < min_val or value > max_val:
                    result.add_warning(f"Metric {metric_name}: Value {value:.3f} outside expected range [{min_val}, {max_val}]")
                    result.update_score(0.1)
                    
        result.metadata = dict(metrics)
        return result


class DataIntegrityValidator:
    """Validates data integrity and consistency."""
    
    def __init__(self):
        self.checksums = {}
        
    def compute_tensor_checksum(self, tensor: torch.Tensor) -> str:
        """Compute checksum for tensor data."""
        # Convert to bytes for hashing
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
        
    def validate_reproducibility(self, 
                               tensor1: torch.Tensor, 
                               tensor2: torch.Tensor,
                               tolerance: float = 1e-5) -> ValidationResult:
        """Validate that two tensors are reproductively identical."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            metadata={},
            validation_score=1.0
        )
        
        if tensor1.shape != tensor2.shape:
            result.add_error(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            return result
            
        # Check if tensors are exactly equal
        if torch.equal(tensor1, tensor2):
            result.metadata['exact_match'] = True
            result.metadata['max_difference'] = 0.0
            return result
            
        # Check approximate equality
        if torch.allclose(tensor1, tensor2, atol=tolerance):
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            result.metadata['exact_match'] = False
            result.metadata['max_difference'] = max_diff
            
            if max_diff > tolerance * 10:  # Significant but within tolerance
                result.add_warning(f"Large but acceptable difference: {max_diff:.2e}")
                result.update_score(0.1)
        else:
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            result.add_error(f"Tensors not reproducible: max difference {max_diff:.2e} > tolerance {tolerance:.2e}")
            result.metadata['exact_match'] = False
            result.metadata['max_difference'] = max_diff
            
        return result
        
    def validate_batch_consistency(self, tensors: List[torch.Tensor]) -> ValidationResult:
        """Validate consistency across a batch of tensors."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            metadata={},
            validation_score=1.0
        )
        
        if not tensors:
            result.add_error("Empty tensor list")
            return result
            
        # Check shape consistency
        reference_shape = tensors[0].shape
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.shape != reference_shape:
                result.add_error(f"Shape inconsistency at index {i}: {tensor.shape} vs {reference_shape}")
                
        # Check dtype consistency
        reference_dtype = tensors[0].dtype
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.dtype != reference_dtype:
                result.add_warning(f"Dtype inconsistency at index {i}: {tensor.dtype} vs {reference_dtype}")
                result.update_score(0.1)
                
        # Check value distribution consistency
        means = [tensor.mean().item() for tensor in tensors]
        stds = [tensor.std().item() for tensor in tensors]
        
        if np.std(means) > 0.5:  # High variance in means
            result.add_warning(f"High variance in tensor means: {np.std(means):.3f}")
            result.update_score(0.1)
            
        if np.std(stds) > 0.3:  # High variance in standard deviations
            result.add_warning(f"High variance in tensor stds: {np.std(stds):.3f}")
            result.update_score(0.1)
            
        result.metadata = {
            'batch_size': len(tensors),
            'common_shape': reference_shape,
            'mean_statistics': {'mean': np.mean(means), 'std': np.std(means)},
            'std_statistics': {'mean': np.mean(stds), 'std': np.std(stds)}
        }
        
        return result


class PerformanceValidator:
    """Validates performance metrics and boundaries."""
    
    def __init__(self):
        self.performance_thresholds = {
            'max_latency_seconds': 120.0,  # 2 minutes max per generation
            'max_memory_gb': 32.0,  # 32GB max GPU memory
            'min_throughput_fps': 0.01,  # Minimum acceptable throughput
            'max_power_watts': 400.0  # Maximum power consumption
        }
        
    def validate_performance_metrics(self, metrics: Dict[str, float]) -> ValidationResult:
        """Validate performance metrics against thresholds."""
        result = ValidationResult(
            is_valid=True,
            errors=[], 
            warnings=[],
            metadata={},
            validation_score=1.0
        )
        
        # Check latency
        if 'latency_seconds' in metrics:
            latency = metrics['latency_seconds']
            if latency > self.performance_thresholds['max_latency_seconds']:
                result.add_error(f"Latency too high: {latency:.2f}s > {self.performance_thresholds['max_latency_seconds']}s")
            elif latency > self.performance_thresholds['max_latency_seconds'] * 0.8:
                result.add_warning(f"High latency: {latency:.2f}s")
                result.update_score(0.2)
                
        # Check memory usage
        if 'memory_gb' in metrics:
            memory = metrics['memory_gb']
            if memory > self.performance_thresholds['max_memory_gb']:
                result.add_error(f"Memory usage too high: {memory:.2f}GB > {self.performance_thresholds['max_memory_gb']}GB")
            elif memory > self.performance_thresholds['max_memory_gb'] * 0.8:
                result.add_warning(f"High memory usage: {memory:.2f}GB")
                result.update_score(0.1)
                
        # Check throughput
        if 'throughput_fps' in metrics:
            throughput = metrics['throughput_fps']
            if throughput < self.performance_thresholds['min_throughput_fps']:
                result.add_error(f"Throughput too low: {throughput:.4f} fps")
                
        result.metadata = dict(metrics)
        return result
        
    def validate_resource_availability(self) -> ValidationResult:
        """Validate system resource availability."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            metadata={},
            validation_score=1.0
        )
        
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                result.add_warning(f"High CPU usage: {cpu_percent:.1f}%")
                result.update_score(0.1)
                
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                result.add_error(f"System memory nearly full: {memory.percent:.1f}%")
            elif memory.percent > 80:
                result.add_warning(f"High system memory usage: {memory.percent:.1f}%")
                result.update_score(0.1)
                
            # Check GPU availability and memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_reserved() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_utilization = gpu_memory / gpu_memory_total * 100
                
                if gpu_utilization > 95:
                    result.add_error(f"GPU memory nearly full: {gpu_utilization:.1f}%")
                elif gpu_utilization > 85:
                    result.add_warning(f"High GPU memory usage: {gpu_utilization:.1f}%")
                    result.update_score(0.1)
                    
            result.metadata = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'gpu_available': torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                result.metadata.update({
                    'gpu_memory_used_gb': gpu_memory,
                    'gpu_memory_total_gb': gpu_memory_total,
                    'gpu_utilization_percent': gpu_utilization
                })
                
        except Exception as e:
            result.add_warning(f"Could not check system resources: {e}")
            result.update_score(0.1)
            
        return result


class ComprehensiveValidator:
    """Main validator that combines all validation components."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.output_validator = OutputValidator()
        self.integrity_validator = DataIntegrityValidator()
        self.performance_validator = PerformanceValidator()
        
        logger.info("ComprehensiveValidator initialized with all validation components")
        
    def validate_benchmark_input(self, 
                                prompts: List[str],
                                params: Dict[str, Any]) -> ValidationResult:
        """Comprehensive validation of benchmark inputs."""
        # Combine all validation results
        prompt_result = self.input_validator.validate_prompts(prompts)
        param_result = self.input_validator.validate_generation_parameters(params)
        resource_result = self.performance_validator.validate_resource_availability()
        
        # Create combined result
        combined_result = ValidationResult(
            is_valid=prompt_result.is_valid and param_result.is_valid and resource_result.is_valid,
            errors=prompt_result.errors + param_result.errors + resource_result.errors,
            warnings=prompt_result.warnings + param_result.warnings + resource_result.warnings,
            metadata={
                'prompt_validation': prompt_result.metadata,
                'parameter_validation': param_result.metadata,
                'resource_validation': resource_result.metadata
            },
            validation_score=min(prompt_result.validation_score, param_result.validation_score, resource_result.validation_score)
        )
        
        return combined_result
        
    def validate_benchmark_output(self, 
                                 tensor: torch.Tensor,
                                 metrics: Dict[str, float],
                                 performance: Dict[str, float]) -> ValidationResult:
        """Comprehensive validation of benchmark outputs."""
        # Validate all output components
        tensor_result = self.output_validator.validate_video_tensor(tensor)
        metrics_result = self.output_validator.validate_metrics(metrics) 
        performance_result = self.performance_validator.validate_performance_metrics(performance)
        
        # Create combined result
        combined_result = ValidationResult(
            is_valid=tensor_result.is_valid and metrics_result.is_valid and performance_result.is_valid,
            errors=tensor_result.errors + metrics_result.errors + performance_result.errors,
            warnings=tensor_result.warnings + metrics_result.warnings + performance_result.warnings,
            metadata={
                'tensor_validation': tensor_result.metadata,
                'metrics_validation': metrics_result.metadata,
                'performance_validation': performance_result.metadata
            },
            validation_score=min(tensor_result.validation_score, metrics_result.validation_score, performance_result.validation_score)
        )
        
        return combined_result
        
    def generate_validation_report(self, 
                                  input_result: ValidationResult,
                                  output_result: ValidationResult,
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_validation': {
                'is_valid': input_result.is_valid and output_result.is_valid,
                'input_score': input_result.validation_score,
                'output_score': output_result.validation_score,
                'combined_score': (input_result.validation_score + output_result.validation_score) / 2
            },
            'input_validation': {
                'is_valid': input_result.is_valid,
                'errors': input_result.errors,
                'warnings': input_result.warnings,
                'score': input_result.validation_score,
                'metadata': input_result.metadata
            },
            'output_validation': {
                'is_valid': output_result.is_valid,
                'errors': output_result.errors,
                'warnings': output_result.warnings,
                'score': output_result.validation_score,
                'metadata': output_result.metadata
            },
            'summary': {
                'total_errors': len(input_result.errors) + len(output_result.errors),
                'total_warnings': len(input_result.warnings) + len(output_result.warnings),
                'validation_passed': len(input_result.errors) == 0 and len(output_result.errors) == 0
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to {save_path}")
            
        return report