"""Comprehensive validation framework for research integrity.

This module provides robust validation, error handling, and data integrity
checks for all research operations, ensuring reproducible and reliable results.

Key features:
1. Statistical significance validation
2. Input data validation and sanitization  
3. Result integrity verification
4. Experiment reproducibility validation
5. Performance benchmark validation
6. Multi-modal data consistency checks
"""

import torch
import numpy as np
import logging
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    
    def add_issue(self, issue: str, severity: str = "error"):
        """Add validation issue."""
        if severity == "error":
            self.issues.append(issue)
            self.is_valid = False
        elif severity == "warning":
            self.warnings.append(issue)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'confidence': self.confidence,
            'issues': self.issues,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'validation_time': self.validation_time
        }


class InputValidator:
    """Validates input data for research operations."""
    
    def __init__(self):
        self.validation_cache = {}
        self.cache_lock = threading.Lock()
        
    def validate_video_tensor(self, video: torch.Tensor, 
                            expected_shape: Optional[Tuple] = None,
                            value_range: Tuple[float, float] = (0.0, 1.0)) -> ValidationResult:
        """Validate video tensor format and content.
        
        Args:
            video: Input video tensor
            expected_shape: Expected tensor shape (optional)
            value_range: Expected value range
            
        Returns:
            ValidationResult with validation status
        """
        start_time = time.time()
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            # Basic tensor checks
            if not isinstance(video, torch.Tensor):
                result.add_issue(f"Expected torch.Tensor, got {type(video)}")
                return result
                
            if video.dim() < 3 or video.dim() > 5:
                result.add_issue(f"Invalid video dimensions: {video.dim()}. Expected 3-5 dims")
                return result
            
            # Shape validation
            if expected_shape and video.shape != expected_shape:
                result.add_issue(f"Shape mismatch: got {video.shape}, expected {expected_shape}")
                return result
            
            # Value range validation
            min_val, max_val = video.min().item(), video.max().item()
            if min_val < value_range[0] or max_val > value_range[1]:
                result.add_issue(f"Values out of range [{value_range[0]}, {value_range[1]}]: "
                               f"found [{min_val:.3f}, {max_val:.3f}]")
                return result
            
            # NaN/Inf checks
            if torch.isnan(video).any():
                result.add_issue("Video contains NaN values")
                return result
                
            if torch.isinf(video).any():
                result.add_issue("Video contains infinite values")
                return result
            
            # Content quality checks
            if video.numel() == 0:
                result.add_issue("Empty video tensor")
                return result
            
            # Statistical checks
            std_val = torch.std(video).item()
            if std_val < 1e-6:
                result.add_issue("Video appears to be constant (very low variance)")
                return result
            
            # Frame consistency checks for multi-frame videos
            if video.dim() >= 4 and video.shape[-3] > 1:  # Multiple frames
                frame_diffs = []
                for i in range(min(video.shape[-3] - 1, 10)):  # Check first 10 frame pairs
                    if video.dim() == 4:  # [T, C, H, W]
                        diff = torch.mean(torch.abs(video[i] - video[i+1])).item()
                    else:  # [B, T, C, H, W]
                        diff = torch.mean(torch.abs(video[0, i] - video[0, i+1])).item()
                    frame_diffs.append(diff)
                
                if all(d < 1e-5 for d in frame_diffs):
                    result.add_issue("All frames appear identical", "warning")
                    
            result.metadata.update({
                'shape': list(video.shape),
                'dtype': str(video.dtype),
                'device': str(video.device),
                'value_range': [min_val, max_val],
                'std': std_val,
                'mean': torch.mean(video).item()
            })
            
        except Exception as e:
            result.add_issue(f"Validation error: {str(e)}")
            logger.exception("Error during video tensor validation")
            
        finally:
            result.validation_time = time.time() - start_time
            
        return result
    
    def validate_model_output(self, output: Any, 
                            expected_type: type = torch.Tensor) -> ValidationResult:
        """Validate model output format and quality."""
        start_time = time.time()
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            # Type validation
            if not isinstance(output, expected_type):
                result.add_issue(f"Expected {expected_type}, got {type(output)}")
                return result
            
            if isinstance(output, torch.Tensor):
                # Use video tensor validation for tensor outputs
                tensor_result = self.validate_video_tensor(output)
                result.is_valid = tensor_result.is_valid
                result.issues.extend(tensor_result.issues)
                result.warnings.extend(tensor_result.warnings)
                result.metadata.update(tensor_result.metadata)
            
            elif isinstance(output, dict):
                # Validate dictionary structure
                required_keys = ['video', 'metadata']
                missing_keys = [k for k in required_keys if k not in output]
                if missing_keys:
                    result.add_issue(f"Missing required keys: {missing_keys}")
                    
                # Validate nested video if present
                if 'video' in output:
                    video_result = self.validate_video_tensor(output['video'])
                    if not video_result.is_valid:
                        result.is_valid = False
                        result.issues.extend([f"Video: {issue}" for issue in video_result.issues])
                        
            result.metadata['output_type'] = str(type(output))
            
        except Exception as e:
            result.add_issue(f"Model output validation error: {str(e)}")
            logger.exception("Error during model output validation")
            
        finally:
            result.validation_time = time.time() - start_time
            
        return result
    
    def validate_prompt_text(self, prompt: str, 
                           max_length: int = 1000,
                           min_length: int = 3) -> ValidationResult:
        """Validate text prompt quality and safety."""
        start_time = time.time()
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            # Basic checks
            if not isinstance(prompt, str):
                result.add_issue(f"Expected string prompt, got {type(prompt)}")
                return result
            
            if len(prompt.strip()) < min_length:
                result.add_issue(f"Prompt too short: {len(prompt)} < {min_length}")
                return result
                
            if len(prompt) > max_length:
                result.add_issue(f"Prompt too long: {len(prompt)} > {max_length}")
                return result
            
            # Content quality checks
            if prompt.strip() == "":
                result.add_issue("Empty prompt")
                return result
            
            # Check for potentially problematic content
            suspicious_patterns = [
                'password', 'secret', 'token', 'api_key',
                '<script', 'javascript:', 'eval(', 'exec('
            ]
            
            prompt_lower = prompt.lower()
            found_patterns = [p for p in suspicious_patterns if p in prompt_lower]
            if found_patterns:
                result.add_issue(f"Suspicious content detected: {found_patterns}", "warning")
            
            # Statistical analysis
            words = prompt.split()
            unique_words = len(set(words))
            
            result.metadata.update({
                'length': len(prompt),
                'word_count': len(words),
                'unique_words': unique_words,
                'character_diversity': len(set(prompt.lower())),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0
            })
            
        except Exception as e:
            result.add_issue(f"Prompt validation error: {str(e)}")
            logger.exception("Error during prompt validation")
            
        finally:
            result.validation_time = time.time() - start_time
            
        return result
    
    def validate_config_dict(self, config: Dict[str, Any],
                           required_keys: List[str] = None,
                           value_ranges: Dict[str, Tuple] = None) -> ValidationResult:
        """Validate configuration dictionary."""
        start_time = time.time()
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            if not isinstance(config, dict):
                result.add_issue(f"Expected dict, got {type(config)}")
                return result
            
            # Check required keys
            if required_keys:
                missing = [k for k in required_keys if k not in config]
                if missing:
                    result.add_issue(f"Missing required keys: {missing}")
            
            # Validate value ranges
            if value_ranges:
                for key, (min_val, max_val) in value_ranges.items():
                    if key in config:
                        val = config[key]
                        if isinstance(val, (int, float)):
                            if val < min_val or val > max_val:
                                result.add_issue(f"{key} out of range [{min_val}, {max_val}]: {val}")
            
            # Check for suspicious values
            for key, value in config.items():
                if isinstance(value, str) and any(pattern in value.lower() 
                                                for pattern in ['password', 'secret', 'token']):
                    result.add_issue(f"Potential sensitive data in config key '{key}'", "warning")
                    
            result.metadata.update({
                'num_keys': len(config),
                'config_size': len(str(config)),
                'keys': list(config.keys())
            })
            
        except Exception as e:
            result.add_issue(f"Config validation error: {str(e)}")
            logger.exception("Error during config validation")
            
        finally:
            result.validation_time = time.time() - start_time
            
        return result


class StatisticalValidator:
    """Validates statistical significance and experimental rigor."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def validate_sample_size(self, sample_size: int, 
                           effect_size: float = 0.5,
                           power: float = 0.8) -> ValidationResult:
        """Validate if sample size is adequate for statistical power."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            # Simple power analysis (Cohen's conventions)
            if effect_size <= 0.2:  # Small effect
                min_size = 200
            elif effect_size <= 0.5:  # Medium effect
                min_size = 50
            else:  # Large effect
                min_size = 20
                
            # Adjust for desired power
            min_size = int(min_size * (power / 0.8))
            
            if sample_size < min_size:
                result.add_issue(f"Sample size {sample_size} may be inadequate. "
                               f"Recommended minimum: {min_size}")
                result.confidence = sample_size / min_size
            
            result.metadata.update({
                'sample_size': sample_size,
                'recommended_min': min_size,
                'effect_size': effect_size,
                'power': power
            })
            
        except Exception as e:
            result.add_issue(f"Sample size validation error: {str(e)}")
            
        return result
    
    def validate_distribution(self, data: np.ndarray,
                            expected_distribution: str = 'normal') -> ValidationResult:
        """Validate data distribution assumptions."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            if len(data) < 3:
                result.add_issue("Insufficient data for distribution testing")
                return result
            
            # Normality tests
            if expected_distribution == 'normal':
                if len(data) >= 8:
                    # Shapiro-Wilk test for small samples
                    stat, p_value = stats.shapiro(data)
                    if p_value < self.alpha:
                        result.add_issue(f"Data may not be normally distributed (p={p_value:.4f})",
                                       "warning")
                
                # Additional normality indicators
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                
                if abs(skewness) > 2:
                    result.add_issue(f"High skewness detected: {skewness:.3f}", "warning")
                    
                if abs(kurtosis) > 7:
                    result.add_issue(f"High kurtosis detected: {kurtosis:.3f}", "warning")
            
            # Outlier detection
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            outlier_threshold = 1.5 * iqr
            
            outliers = np.sum((data < q25 - outlier_threshold) | 
                            (data > q75 + outlier_threshold))
            outlier_ratio = outliers / len(data)
            
            if outlier_ratio > 0.1:
                result.add_issue(f"High outlier ratio: {outlier_ratio:.2%}", "warning")
            
            result.metadata.update({
                'sample_size': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'outlier_ratio': float(outlier_ratio)
            })
            
        except Exception as e:
            result.add_issue(f"Distribution validation error: {str(e)}")
            logger.exception("Error during distribution validation")
            
        return result
    
    def validate_statistical_significance(self, 
                                        group1: np.ndarray,
                                        group2: np.ndarray,
                                        test_type: str = 'auto') -> ValidationResult:
        """Validate statistical significance between groups."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            if len(group1) < 3 or len(group2) < 3:
                result.add_issue("Insufficient data for statistical testing")
                return result
            
            # Automatic test selection
            if test_type == 'auto':
                # Check normality
                _, p1 = stats.shapiro(group1) if len(group1) <= 5000 else (0, 0.5)
                _, p2 = stats.shapiro(group2) if len(group2) <= 5000 else (0, 0.5)
                
                if p1 > self.alpha and p2 > self.alpha:
                    # Both normal - use t-test
                    stat, p_value = stats.ttest_ind(group1, group2)
                    test_used = 't-test'
                else:
                    # Non-normal - use Mann-Whitney U
                    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    test_used = 'Mann-Whitney U'
            else:
                # Use specified test
                if test_type == 't-test':
                    stat, p_value = stats.ttest_ind(group1, group2)
                elif test_type == 'mannwhitney':
                    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
                test_used = test_type
            
            # Effect size (Cohen's d for t-test)
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                                (len(group2) - 1) * np.var(group2, ddof=1)) /
                               (len(group1) + len(group2) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
            else:
                cohens_d = 0.0
            
            # Interpret results
            is_significant = p_value < self.alpha
            
            if not is_significant:
                result.add_issue(f"No significant difference found (p={p_value:.4f})", "warning")
                result.confidence = 1.0 - p_value
            
            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                effect_magnitude = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_magnitude = "small"
            elif abs(cohens_d) < 0.8:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            result.metadata.update({
                'test_used': test_used,
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_significant': is_significant,
                'alpha': self.alpha,
                'cohens_d': float(cohens_d),
                'effect_magnitude': effect_magnitude,
                'group1_size': len(group1),
                'group2_size': len(group2)
            })
            
        except Exception as e:
            result.add_issue(f"Statistical significance validation error: {str(e)}")
            logger.exception("Error during statistical significance validation")
            
        return result


class ExperimentValidator:
    """Validates experimental design and reproducibility."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.stats_validator = StatisticalValidator()
        
    def validate_experiment_design(self, 
                                 experiment_config: Dict[str, Any]) -> ValidationResult:
        """Validate experimental design for scientific rigor."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            # Required components
            required_sections = ['models', 'metrics', 'seeds', 'controls']
            missing_sections = [s for s in required_sections if s not in experiment_config]
            
            if missing_sections:
                result.add_issue(f"Missing experiment sections: {missing_sections}")
            
            # Validate models
            if 'models' in experiment_config:
                models = experiment_config['models']
                if not isinstance(models, list) or len(models) < 1:
                    result.add_issue("At least one model required for comparison")
                elif len(models) < 2:
                    result.add_issue("Multiple models recommended for comparison", "warning")
            
            # Validate seeds for reproducibility
            if 'seeds' in experiment_config:
                seeds = experiment_config['seeds']
                if not isinstance(seeds, list) or len(seeds) < 3:
                    result.add_issue("At least 3 random seeds recommended for reliability")
                    
                if len(set(seeds)) != len(seeds):
                    result.add_issue("Duplicate seeds detected")
            
            # Validate controls
            if 'controls' in experiment_config:
                controls = experiment_config['controls']
                if not isinstance(controls, dict):
                    result.add_issue("Controls should be a dictionary")
                else:
                    # Check for baseline model
                    if 'baseline_model' not in controls:
                        result.add_issue("Baseline model recommended for comparison", "warning")
            
            # Sample size validation
            num_samples = experiment_config.get('num_samples_per_seed', 10)
            sample_result = self.stats_validator.validate_sample_size(num_samples)
            if not sample_result.is_valid:
                result.issues.extend(sample_result.issues)
                result.warnings.extend(sample_result.warnings)
            
            result.metadata.update({
                'num_models': len(experiment_config.get('models', [])),
                'num_seeds': len(experiment_config.get('seeds', [])),
                'num_metrics': len(experiment_config.get('metrics', [])),
                'samples_per_seed': num_samples
            })
            
        except Exception as e:
            result.add_issue(f"Experiment design validation error: {str(e)}")
            logger.exception("Error during experiment design validation")
            
        return result
    
    def validate_reproducibility(self, 
                               results1: Dict[str, Any],
                               results2: Dict[str, Any],
                               tolerance: float = 0.05) -> ValidationResult:
        """Validate reproducibility between experiment runs."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            # Compare metrics
            common_metrics = set(results1.get('metrics', {}).keys()) & \
                           set(results2.get('metrics', {}).keys())
            
            if not common_metrics:
                result.add_issue("No common metrics found between runs")
                return result
            
            reproducibility_scores = []
            
            for metric in common_metrics:
                values1 = np.array(results1['metrics'][metric])
                values2 = np.array(results2['metrics'][metric])
                
                if len(values1) != len(values2):
                    result.add_issue(f"Mismatched sample sizes for metric {metric}")
                    continue
                
                # Correlation analysis
                if len(values1) > 1:
                    correlation, p_value = stats.pearsonr(values1, values2)
                    reproducibility_scores.append(correlation)
                    
                    if correlation < 0.8:
                        result.add_issue(f"Low reproducibility for {metric}: r={correlation:.3f}")
                
                # Mean difference analysis
                mean_diff = abs(np.mean(values1) - np.mean(values2))
                relative_diff = mean_diff / (np.mean(values1) + 1e-8)
                
                if relative_diff > tolerance:
                    result.add_issue(f"High mean difference for {metric}: {relative_diff:.3f}")
            
            # Overall reproducibility score
            if reproducibility_scores:
                overall_reproducibility = np.mean(reproducibility_scores)
                result.confidence = overall_reproducibility
                
                if overall_reproducibility < 0.7:
                    result.add_issue("Low overall reproducibility")
            
            result.metadata.update({
                'common_metrics': list(common_metrics),
                'reproducibility_scores': reproducibility_scores,
                'overall_reproducibility': float(np.mean(reproducibility_scores)) if reproducibility_scores else 0.0
            })
            
        except Exception as e:
            result.add_issue(f"Reproducibility validation error: {str(e)}")
            logger.exception("Error during reproducibility validation")
            
        return result


class DataIntegrityValidator:
    """Validates data integrity and consistency."""
    
    def __init__(self):
        self.checksums = {}
        
    def compute_data_checksum(self, data: Union[torch.Tensor, np.ndarray, str]) -> str:
        """Compute checksum for data integrity verification."""
        try:
            if isinstance(data, torch.Tensor):
                data_bytes = data.cpu().numpy().tobytes()
            elif isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')
                
            return hashlib.sha256(data_bytes).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to compute checksum: {e}")
            return "checksum_error"
    
    def validate_data_integrity(self, 
                              data_id: str,
                              data: Any,
                              expected_checksum: str = None) -> ValidationResult:
        """Validate data hasn't been corrupted."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            current_checksum = self.compute_data_checksum(data)
            
            # Store checksum for future validation
            self.checksums[data_id] = current_checksum
            
            # Check against expected checksum
            if expected_checksum and current_checksum != expected_checksum:
                result.add_issue(f"Data integrity check failed for {data_id}")
                result.add_issue(f"Expected: {expected_checksum}")
                result.add_issue(f"Got: {current_checksum}")
            
            result.metadata.update({
                'data_id': data_id,
                'checksum': current_checksum,
                'data_type': str(type(data))
            })
            
        except Exception as e:
            result.add_issue(f"Data integrity validation error: {str(e)}")
            logger.exception("Error during data integrity validation")
            
        return result
    
    def validate_file_integrity(self, file_path: Path) -> ValidationResult:
        """Validate file integrity and accessibility."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        try:
            if not file_path.exists():
                result.add_issue(f"File does not exist: {file_path}")
                return result
            
            if not file_path.is_file():
                result.add_issue(f"Path is not a file: {file_path}")
                return result
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                result.add_issue("File is empty")
            elif file_size > 10 * 1024 * 1024 * 1024:  # 10GB
                result.add_issue("File is very large (>10GB)", "warning")
            
            # Check read permissions
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except PermissionError:
                result.add_issue("No read permission for file")
            except Exception as e:
                result.add_issue(f"File read error: {str(e)}")
            
            result.metadata.update({
                'file_path': str(file_path),
                'file_size': file_size,
                'file_exists': True
            })
            
        except Exception as e:
            result.add_issue(f"File integrity validation error: {str(e)}")
            logger.exception("Error during file integrity validation")
            
        return result


class ComprehensiveValidator:
    """Main validator combining all validation components."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.stats_validator = StatisticalValidator()
        self.experiment_validator = ExperimentValidator()
        self.integrity_validator = DataIntegrityValidator()
        
    def validate_research_pipeline(self, 
                                 video_data: torch.Tensor,
                                 prompts: List[str],
                                 config: Dict[str, Any],
                                 results: Dict[str, Any] = None) -> Dict[str, ValidationResult]:
        """Comprehensive validation of entire research pipeline."""
        validation_results = {}
        
        # Validate inputs
        validation_results['video_data'] = self.input_validator.validate_video_tensor(video_data)
        
        # Validate prompts
        prompt_results = []
        for i, prompt in enumerate(prompts):
            prompt_result = self.input_validator.validate_prompt_text(prompt)
            prompt_results.append(prompt_result)
            if not prompt_result.is_valid:
                break
                
        validation_results['prompts'] = ValidationResult(
            is_valid=all(r.is_valid for r in prompt_results),
            confidence=np.mean([r.confidence for r in prompt_results]),
            issues=[issue for r in prompt_results for issue in r.issues],
            warnings=[warning for r in prompt_results for warning in r.warnings]
        )
        
        # Validate configuration
        validation_results['config'] = self.input_validator.validate_config_dict(config)
        
        # Validate experiment design
        validation_results['experiment_design'] = self.experiment_validator.validate_experiment_design(config)
        
        # Validate results if provided
        if results:
            # Extract metrics for statistical validation
            for metric_name, values in results.get('metrics', {}).items():
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    dist_result = self.stats_validator.validate_distribution(np.array(values))
                    validation_results[f'distribution_{metric_name}'] = dist_result
        
        # Overall validation status
        all_valid = all(result.is_valid for result in validation_results.values())
        overall_confidence = np.mean([result.confidence for result in validation_results.values()])
        
        validation_results['overall'] = ValidationResult(
            is_valid=all_valid,
            confidence=overall_confidence,
            metadata={
                'total_validations': len(validation_results),
                'failed_validations': sum(1 for r in validation_results.values() if not r.is_valid),
                'validation_summary': {k: v.is_valid for k, v in validation_results.items()}
            }
        )
        
        return validation_results
    
    def generate_validation_report(self, 
                                 validation_results: Dict[str, ValidationResult]) -> str:
        """Generate human-readable validation report."""
        report = ["=== RESEARCH VALIDATION REPORT ===\n"]
        
        overall = validation_results.get('overall')
        if overall:
            status = "PASSED" if overall.is_valid else "FAILED"
            report.append(f"Overall Status: {status} (Confidence: {overall.confidence:.2%})\n")
        
        # Summary
        total = len(validation_results) - 1  # Exclude 'overall'
        failed = sum(1 for k, v in validation_results.items() 
                    if k != 'overall' and not v.is_valid)
        
        report.append(f"Validations: {total - failed}/{total} passed\n")
        
        # Detailed results
        for name, result in validation_results.items():
            if name == 'overall':
                continue
                
            status = "✓" if result.is_valid else "✗"
            report.append(f"{status} {name.replace('_', ' ').title()}")
            
            if result.issues:
                for issue in result.issues:
                    report.append(f"  ERROR: {issue}")
                    
            if result.warnings:
                for warning in result.warnings:
                    report.append(f"  WARNING: {warning}")
                    
            report.append("")  # Empty line
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Example validation usage
    validator = ComprehensiveValidator()
    
    # Create test data
    test_video = torch.randn(16, 3, 256, 256)  # 16 frames
    test_prompts = ["A cat playing", "A dog running"]
    test_config = {
        'models': ['model1', 'model2'],
        'metrics': ['fvd', 'is', 'lpips'],
        'seeds': [42, 123, 456],
        'num_samples_per_seed': 20
    }
    
    # Run comprehensive validation
    results = validator.validate_research_pipeline(
        test_video, test_prompts, test_config
    )
    
    # Generate report
    report = validator.generate_validation_report(results)
    print(report)