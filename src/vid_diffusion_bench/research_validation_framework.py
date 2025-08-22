"""Research Validation Framework for Video Diffusion Benchmark Studies.

This module implements rigorous statistical validation, reproducibility testing,
and research methodology validation for video diffusion model benchmarks to
ensure academic-grade research quality and scientific rigor.
"""

import time
import logging
import asyncio
import numpy as np
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from scipy.stats import normaltest, shapiro, mannwhitneyu, ttest_ind, chi2_contingency
import warnings
from sklearn.utils import resample
from sklearn.metrics import cohen_kappa_score

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Suppress warnings during statistical tests
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ValidationLevel(Enum):
    """Research validation levels."""
    BASIC = "basic"          # Basic statistical tests
    CONFERENCE = "conference"  # Conference-level rigor
    JOURNAL = "journal"      # Journal-level rigor
    REPRODUCIBLE = "reproducible"  # Full reproducibility
    META_ANALYSIS = "meta_analysis"  # Meta-analysis ready


class StatisticalMethod(Enum):
    """Statistical methods for validation."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"
    COHEN_D = "cohen_d"
    CLIFF_DELTA = "cliff_delta"
    BONFERRONI = "bonferroni"
    FDR_CORRECTION = "fdr_correction"


class ReproducibilityLevel(Enum):
    """Levels of reproducibility validation."""
    COMPUTATIONAL = "computational"  # Same code, same results
    STATISTICAL = "statistical"     # Different samples, similar statistics
    CONCEPTUAL = "conceptual"       # Different implementation, same conclusions
    CROSS_PLATFORM = "cross_platform"  # Different platforms, same results


@dataclass
class StatisticalTestResult:
    """Results of a statistical test."""
    test_name: str
    test_statistic: float
    p_value: float
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    power: Optional[float]
    interpretation: str
    significant: bool
    assumptions_met: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproducibilityReport:
    """Report on reproducibility validation."""
    experiment_id: str
    original_results: Dict[str, float]
    reproduction_attempts: List[Dict[str, float]]
    reproducibility_score: float
    statistical_agreement: float
    computational_agreement: float
    cross_platform_agreement: float
    failed_reproductions: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchValidationResult:
    """Complete research validation result."""
    validation_id: str
    experiment_metadata: Dict[str, Any]
    validation_level: ValidationLevel
    statistical_tests: List[StatisticalTestResult]
    reproducibility_report: ReproducibilityReport
    methodological_assessment: Dict[str, Any]
    data_quality_metrics: Dict[str, float]
    publication_readiness: Dict[str, Any]
    overall_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalValidator:
    """Performs rigorous statistical validation of experimental results."""
    
    def __init__(self, significance_level: float = 0.05, 
                 effect_size_threshold: float = 0.5):
        self.significance_level = significance_level
        self.effect_size_threshold = effect_size_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_comparison(self, group_a: List[float], group_b: List[float],
                          test_name: str = "Comparison") -> StatisticalTestResult:
        """
        Validate a comparison between two groups with comprehensive statistical testing.
        
        Args:
            group_a: First group of measurements
            group_b: Second group of measurements
            test_name: Name of the test for reporting
        
        Returns:
            Comprehensive statistical test result
        """
        
        # Convert to numpy arrays
        a = np.array(group_a)
        b = np.array(group_b)
        
        # Remove NaN values
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        
        if len(a) < 3 or len(b) < 3:
            return StatisticalTestResult(
                test_name=test_name,
                test_statistic=0.0,
                p_value=1.0,
                confidence_interval=None,
                effect_size=0.0,
                power=0.0,
                interpretation="Insufficient data for statistical testing",
                significant=False,
                assumptions_met={"sample_size": False}
            )
        
        # Test assumptions
        assumptions = self._test_assumptions(a, b)
        
        # Choose appropriate test based on assumptions
        if assumptions["normality"] and assumptions["equal_variance"]:
            # Use parametric t-test
            test_stat, p_value = ttest_ind(a, b, equal_var=True)
            test_type = "Independent t-test (equal variance)"
        elif assumptions["normality"] and not assumptions["equal_variance"]:
            # Use Welch's t-test
            test_stat, p_value = ttest_ind(a, b, equal_var=False)
            test_type = "Welch's t-test (unequal variance)"
        else:
            # Use non-parametric Mann-Whitney U test
            test_stat, p_value = mannwhitneyu(a, b, alternative='two-sided')
            test_type = "Mann-Whitney U test (non-parametric)"
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(a, b)
        
        # Calculate confidence interval for the difference
        confidence_interval = self._calculate_confidence_interval(a, b)
        
        # Calculate statistical power
        power = self._calculate_power(a, b, effect_size)
        
        # Interpret results
        interpretation = self._interpret_results(p_value, effect_size, power, test_type)
        
        return StatisticalTestResult(
            test_name=test_name,
            test_statistic=float(test_stat),
            p_value=float(p_value),
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            interpretation=interpretation,
            significant=p_value < self.significance_level,
            assumptions_met=assumptions
        )
    
    def _test_assumptions(self, a: np.ndarray, b: np.ndarray) -> Dict[str, bool]:
        """Test statistical assumptions for parametric tests."""
        assumptions = {}
        
        # Test normality
        try:
            # Shapiro-Wilk test for small samples, D'Agostino for larger
            if len(a) <= 50 and len(b) <= 50:
                _, p_a = shapiro(a)
                _, p_b = shapiro(b)
            else:
                _, p_a = normaltest(a)
                _, p_b = normaltest(b)
            
            # Both groups should be approximately normal
            assumptions["normality"] = (p_a > 0.05 and p_b > 0.05)
            
        except Exception:
            assumptions["normality"] = False
        
        # Test equal variances (Levene's test)
        try:
            from scipy.stats import levene
            _, p_levene = levene(a, b)
            assumptions["equal_variance"] = p_levene > 0.05
        except Exception:
            # Fallback: simple variance ratio test
            var_ratio = np.var(a, ddof=1) / np.var(b, ddof=1)
            assumptions["equal_variance"] = 0.5 <= var_ratio <= 2.0
        
        # Test sample size adequacy
        assumptions["sample_size"] = len(a) >= 3 and len(b) >= 3
        
        # Test for outliers (using IQR method)
        assumptions["no_extreme_outliers"] = (
            self._count_extreme_outliers(a) < len(a) * 0.05 and
            self._count_extreme_outliers(b) < len(b) * 0.05
        )
        
        return assumptions
    
    def _count_extreme_outliers(self, data: np.ndarray) -> int:
        """Count extreme outliers using IQR method."""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3.0 * IQR  # 3.0 IQR for extreme outliers
        upper_bound = Q3 + 3.0 * IQR
        
        return np.sum((data < lower_bound) | (data > upper_bound))
    
    def _calculate_effect_size(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + 
                             (len(b) - 1) * np.var(b, ddof=1)) / 
                            (len(a) + len(b) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return abs(mean_a - mean_b) / pooled_std
    
    def _calculate_confidence_interval(self, a: np.ndarray, b: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the difference in means."""
        
        mean_diff = np.mean(a) - np.mean(b)
        
        # Standard error of the difference
        se_diff = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
        
        # Degrees of freedom (Welch's formula)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        n_a, n_b = len(a), len(b)
        
        if var_a == 0 and var_b == 0:
            return (mean_diff, mean_diff)
        
        df = (var_a / n_a + var_b / n_b) ** 2 / \
             ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
        
        # Critical t-value
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        margin_error = t_critical * se_diff
        
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def _calculate_power(self, a: np.ndarray, b: np.ndarray, effect_size: float) -> float:
        """Calculate statistical power of the test."""
        
        n_a, n_b = len(a), len(b)
        
        # For equal sample sizes, use simplified power calculation
        if n_a == n_b:
            n = n_a
            # Approximate power calculation for t-test
            delta = effect_size * np.sqrt(n / 2)
            power = 1 - stats.t.cdf(stats.t.ppf(1 - self.significance_level / 2, 2 * n - 2), 
                                   2 * n - 2, delta) + \
                   stats.t.cdf(stats.t.ppf(self.significance_level / 2, 2 * n - 2), 
                              2 * n - 2, delta)
        else:
            # More complex calculation for unequal samples
            n_eff = 2 * n_a * n_b / (n_a + n_b)  # Effective sample size
            delta = effect_size * np.sqrt(n_eff / 2)
            df = n_a + n_b - 2
            
            power = 1 - stats.t.cdf(stats.t.ppf(1 - self.significance_level / 2, df), 
                                   df, delta) + \
                   stats.t.cdf(stats.t.ppf(self.significance_level / 2, df), 
                              df, delta)
        
        return max(0.0, min(1.0, power))
    
    def _interpret_results(self, p_value: float, effect_size: float, 
                          power: float, test_type: str) -> str:
        """Generate interpretation of statistical results."""
        
        interpretations = []
        
        # Significance interpretation
        if p_value < 0.001:
            interpretations.append("highly significant (p < 0.001)")
        elif p_value < 0.01:
            interpretations.append("very significant (p < 0.01)")
        elif p_value < self.significance_level:
            interpretations.append(f"significant (p < {self.significance_level})")
        else:
            interpretations.append("not statistically significant")
        
        # Effect size interpretation (Cohen's guidelines)
        if effect_size < 0.2:
            interpretations.append("negligible effect size")
        elif effect_size < 0.5:
            interpretations.append("small effect size")
        elif effect_size < 0.8:
            interpretations.append("medium effect size")
        else:
            interpretations.append("large effect size")
        
        # Power interpretation
        if power < 0.5:
            interpretations.append("very low power - high risk of Type II error")
        elif power < 0.8:
            interpretations.append("moderate power - some risk of Type II error")
        else:
            interpretations.append("adequate statistical power")
        
        # Combine interpretations
        base_interpretation = f"Using {test_type}: {', '.join(interpretations)}."
        
        # Add recommendations
        if p_value >= self.significance_level and power < 0.8:
            base_interpretation += " Consider increasing sample size."
        elif p_value < self.significance_level and effect_size < 0.2:
            base_interpretation += " Statistically significant but practically negligible difference."
        
        return base_interpretation
    
    def validate_multiple_comparisons(self, groups: Dict[str, List[float]],
                                    correction_method: str = "bonferroni") -> List[StatisticalTestResult]:
        """
        Validate multiple pairwise comparisons with correction for multiple testing.
        
        Args:
            groups: Dictionary mapping group names to measurement lists
            correction_method: Method for multiple comparison correction
        
        Returns:
            List of corrected statistical test results
        """
        
        if len(groups) < 2:
            return []
        
        # Perform all pairwise comparisons
        group_names = list(groups.keys())
        raw_results = []
        
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name_a, name_b = group_names[i], group_names[j]
                test_name = f"{name_a} vs {name_b}"
                
                result = self.validate_comparison(groups[name_a], groups[name_b], test_name)
                raw_results.append(result)
        
        # Apply multiple comparison correction
        corrected_results = self._apply_multiple_comparison_correction(
            raw_results, correction_method
        )
        
        return corrected_results
    
    def _apply_multiple_comparison_correction(self, results: List[StatisticalTestResult],
                                           method: str) -> List[StatisticalTestResult]:
        """Apply multiple comparison correction to p-values."""
        
        if not results:
            return results
        
        p_values = [r.p_value for r in results]
        n_comparisons = len(p_values)
        
        if method == "bonferroni":
            corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
        elif method == "fdr":
            # Benjamini-Hochberg FDR correction
            corrected_p_values = self._benjamini_hochberg_correction(p_values)
        else:
            # No correction
            corrected_p_values = p_values
        
        # Create corrected results
        corrected_results = []
        for result, corrected_p in zip(results, corrected_p_values):
            corrected_result = StatisticalTestResult(
                test_name=f"{result.test_name} (corrected)",
                test_statistic=result.test_statistic,
                p_value=corrected_p,
                confidence_interval=result.confidence_interval,
                effect_size=result.effect_size,
                power=result.power,
                interpretation=self._interpret_results(corrected_p, result.effect_size or 0, 
                                                     result.power or 0, "corrected test"),
                significant=corrected_p < self.significance_level,
                assumptions_met=result.assumptions_met
            )
            corrected_results.append(corrected_result)
        
        return corrected_results
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction."""
        
        n = len(p_values)
        if n == 0:
            return p_values
        
        # Sort p-values with original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # Apply BH correction
        corrected_sorted = np.minimum(1.0, sorted_p_values * n / (np.arange(n) + 1))
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        
        # Restore original order
        corrected_p_values = np.empty(n)
        corrected_p_values[sorted_indices] = corrected_sorted
        
        return corrected_p_values.tolist()


class ReproducibilityValidator:
    """Validates reproducibility of experimental results."""
    
    def __init__(self, tolerance: float = 0.05, min_replications: int = 3):
        self.tolerance = tolerance
        self.min_replications = min_replications
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def validate_reproducibility(self, 
                                     experiment_function: Callable,
                                     experiment_params: Dict[str, Any],
                                     num_replications: int = 5,
                                     parallel_execution: bool = True) -> ReproducibilityReport:
        """
        Validate reproducibility by running multiple replications of an experiment.
        
        Args:
            experiment_function: Function to run the experiment
            experiment_params: Parameters for the experiment
            num_replications: Number of replications to run
            parallel_execution: Whether to run replications in parallel
        
        Returns:
            Comprehensive reproducibility report
        """
        
        experiment_id = f"repro_{int(time.time())}_{hash(str(experiment_params)) % 10000}"
        
        self.logger.info(f"Starting reproducibility validation: {experiment_id}")
        self.logger.info(f"Running {num_replications} replications")
        
        # Run the original experiment
        original_results = await self._run_single_experiment(experiment_function, experiment_params)
        
        # Run replications
        if parallel_execution:
            reproduction_attempts = await self._run_parallel_replications(
                experiment_function, experiment_params, num_replications
            )
        else:
            reproduction_attempts = await self._run_sequential_replications(
                experiment_function, experiment_params, num_replications
            )
        
        # Analyze reproducibility
        reproducibility_score = self._calculate_reproducibility_score(
            original_results, reproduction_attempts
        )
        
        statistical_agreement = self._calculate_statistical_agreement(
            original_results, reproduction_attempts
        )
        
        computational_agreement = self._calculate_computational_agreement(
            original_results, reproduction_attempts
        )
        
        # Identify failed reproductions
        failed_reproductions = self._identify_failed_reproductions(
            original_results, reproduction_attempts
        )
        
        # Generate recommendations
        recommendations = self._generate_reproducibility_recommendations(
            reproducibility_score, statistical_agreement, computational_agreement,
            len(failed_reproductions)
        )
        
        report = ReproducibilityReport(
            experiment_id=experiment_id,
            original_results=original_results,
            reproduction_attempts=reproduction_attempts,
            reproducibility_score=reproducibility_score,
            statistical_agreement=statistical_agreement,
            computational_agreement=computational_agreement,
            cross_platform_agreement=0.0,  # Would require actual cross-platform testing
            failed_reproductions=failed_reproductions,
            recommendations=recommendations
        )
        
        self.logger.info(f"Reproducibility validation complete: score={reproducibility_score:.3f}")
        
        return report
    
    async def _run_single_experiment(self, experiment_function: Callable,
                                   params: Dict[str, Any]) -> Dict[str, float]:
        """Run a single experiment and return results."""
        
        try:
            if asyncio.iscoroutinefunction(experiment_function):
                results = await experiment_function(**params)
            else:
                # Run in executor for non-async functions
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, lambda: experiment_function(**params))
            
            # Ensure results are numeric
            numeric_results = {}
            for key, value in results.items():
                try:
                    numeric_results[key] = float(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Non-numeric result ignored: {key}={value}")
            
            return numeric_results
            
        except Exception as e:
            self.logger.error(f"Experiment execution failed: {e}")
            return {}
    
    async def _run_parallel_replications(self, experiment_function: Callable,
                                       params: Dict[str, Any],
                                       num_replications: int) -> List[Dict[str, float]]:
        """Run replications in parallel."""
        
        tasks = []
        for i in range(num_replications):
            # Add replication index to parameters
            replication_params = params.copy()
            replication_params["_replication_id"] = i
            
            task = self._run_single_experiment(experiment_function, replication_params)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Replication {i} failed: {result}")
            elif isinstance(result, dict):
                valid_results.append(result)
        
        return valid_results
    
    async def _run_sequential_replications(self, experiment_function: Callable,
                                         params: Dict[str, Any],
                                         num_replications: int) -> List[Dict[str, float]]:
        """Run replications sequentially."""
        
        results = []
        
        for i in range(num_replications):
            self.logger.debug(f"Running replication {i+1}/{num_replications}")
            
            replication_params = params.copy()
            replication_params["_replication_id"] = i
            
            result = await self._run_single_experiment(experiment_function, replication_params)
            if result:
                results.append(result)
            
            # Small delay between replications
            await asyncio.sleep(0.1)
        
        return results
    
    def _calculate_reproducibility_score(self, original: Dict[str, float],
                                       replications: List[Dict[str, float]]) -> float:
        """Calculate overall reproducibility score (0-1)."""
        
        if not replications:
            return 0.0
        
        # Find common metrics across all runs
        common_metrics = set(original.keys())
        for replication in replications:
            common_metrics &= set(replication.keys())
        
        if not common_metrics:
            return 0.0
        
        # Calculate reproducibility for each metric
        metric_scores = []
        
        for metric in common_metrics:
            original_value = original[metric]
            replication_values = [r[metric] for r in replications]
            
            # Calculate relative deviations
            if original_value != 0:
                relative_deviations = [abs(v - original_value) / abs(original_value) 
                                     for v in replication_values]
            else:
                relative_deviations = [abs(v - original_value) for v in replication_values]
            
            # Score based on how many replications are within tolerance
            within_tolerance = sum(1 for dev in relative_deviations if dev <= self.tolerance)
            metric_score = within_tolerance / len(replication_values)
            
            metric_scores.append(metric_score)
        
        return np.mean(metric_scores) if metric_scores else 0.0
    
    def _calculate_statistical_agreement(self, original: Dict[str, float],
                                       replications: List[Dict[str, float]]) -> float:
        """Calculate statistical agreement between original and replications."""
        
        if not replications:
            return 0.0
        
        common_metrics = set(original.keys())
        for replication in replications:
            common_metrics &= set(replication.keys())
        
        if not common_metrics:
            return 0.0
        
        agreement_scores = []
        
        for metric in common_metrics:
            original_value = original[metric]
            replication_values = [r[metric] for r in replications]
            
            # Use coefficient of variation as agreement measure
            all_values = [original_value] + replication_values
            mean_value = np.mean(all_values)
            std_value = np.std(all_values)
            
            if mean_value != 0:
                cv = std_value / abs(mean_value)
                # Convert CV to agreement score (lower CV = higher agreement)
                agreement = max(0.0, 1.0 - cv / 0.5)  # 50% CV gives 0 agreement
            else:
                agreement = 1.0 if std_value < 1e-10 else 0.0
            
            agreement_scores.append(agreement)
        
        return np.mean(agreement_scores) if agreement_scores else 0.0
    
    def _calculate_computational_agreement(self, original: Dict[str, float],
                                         replications: List[Dict[str, float]]) -> float:
        """Calculate computational reproducibility (exact agreement)."""
        
        if not replications:
            return 0.0
        
        common_metrics = set(original.keys())
        for replication in replications:
            common_metrics &= set(replication.keys())
        
        if not common_metrics:
            return 0.0
        
        exact_matches = []
        
        for metric in common_metrics:
            original_value = original[metric]
            replication_values = [r[metric] for r in replications]
            
            # Check for exact matches (within floating point precision)
            exact_match_count = sum(1 for v in replication_values 
                                  if abs(v - original_value) < 1e-12)
            
            match_ratio = exact_match_count / len(replication_values)
            exact_matches.append(match_ratio)
        
        return np.mean(exact_matches) if exact_matches else 0.0
    
    def _identify_failed_reproductions(self, original: Dict[str, float],
                                     replications: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify replications that failed to reproduce original results."""
        
        failed = []
        
        for i, replication in enumerate(replications):
            failures = []
            
            for metric, original_value in original.items():
                if metric not in replication:
                    failures.append({
                        "metric": metric,
                        "issue": "missing_metric",
                        "original_value": original_value,
                        "replication_value": None
                    })
                else:
                    replication_value = replication[metric]
                    
                    # Check if within tolerance
                    if original_value != 0:
                        relative_error = abs(replication_value - original_value) / abs(original_value)
                    else:
                        relative_error = abs(replication_value - original_value)
                    
                    if relative_error > self.tolerance:
                        failures.append({
                            "metric": metric,
                            "issue": "value_mismatch",
                            "original_value": original_value,
                            "replication_value": replication_value,
                            "relative_error": relative_error
                        })
            
            if failures:
                failed.append({
                    "replication_id": i,
                    "failures": failures,
                    "failure_count": len(failures)
                })
        
        return failed
    
    def _generate_reproducibility_recommendations(self, reproducibility_score: float,
                                                statistical_agreement: float,
                                                computational_agreement: float,
                                                num_failures: int) -> List[str]:
        """Generate recommendations for improving reproducibility."""
        
        recommendations = []
        
        if reproducibility_score < 0.8:
            recommendations.append("Low reproducibility score detected. Consider fixing random seeds and ensuring deterministic computation.")
        
        if statistical_agreement < 0.9:
            recommendations.append("Low statistical agreement. Check for sources of variability in the experimental setup.")
        
        if computational_agreement < 0.5:
            recommendations.append("Low computational reproducibility. Ensure identical software versions and computational environments.")
        
        if num_failures > 0:
            recommendations.append(f"{num_failures} replications failed. Investigate failure patterns and fix experimental procedures.")
        
        if reproducibility_score > 0.95 and statistical_agreement > 0.95:
            recommendations.append("Excellent reproducibility achieved. Results are suitable for publication.")
        
        if not recommendations:
            recommendations.append("Good reproducibility achieved. Minor improvements may be possible.")
        
        return recommendations


class MethodologicalValidator:
    """Validates research methodology and experimental design."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_experimental_design(self, experiment_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experimental design for methodological soundness."""
        
        validation_results = {
            "design_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "methodological_issues": []
        }
        
        design_score = 0.0
        max_score = 0.0
        
        # 1. Sample size adequacy
        sample_size_score, sample_size_feedback = self._validate_sample_size(experiment_metadata)
        design_score += sample_size_score
        max_score += 1.0
        validation_results["strengths"].extend(sample_size_feedback["strengths"])
        validation_results["weaknesses"].extend(sample_size_feedback["weaknesses"])
        
        # 2. Control variables
        control_score, control_feedback = self._validate_controls(experiment_metadata)
        design_score += control_score
        max_score += 1.0
        validation_results["strengths"].extend(control_feedback["strengths"])
        validation_results["weaknesses"].extend(control_feedback["weaknesses"])
        
        # 3. Randomization
        randomization_score, randomization_feedback = self._validate_randomization(experiment_metadata)
        design_score += randomization_score
        max_score += 1.0
        validation_results["strengths"].extend(randomization_feedback["strengths"])
        validation_results["weaknesses"].extend(randomization_feedback["weaknesses"])
        
        # 4. Blinding
        blinding_score, blinding_feedback = self._validate_blinding(experiment_metadata)
        design_score += blinding_score
        max_score += 1.0
        validation_results["strengths"].extend(blinding_feedback["strengths"])
        validation_results["weaknesses"].extend(blinding_feedback["weaknesses"])
        
        # 5. Measurement validity
        measurement_score, measurement_feedback = self._validate_measurements(experiment_metadata)
        design_score += measurement_score
        max_score += 1.0
        validation_results["strengths"].extend(measurement_feedback["strengths"])
        validation_results["weaknesses"].extend(measurement_feedback["weaknesses"])
        
        # Calculate overall design score
        validation_results["design_score"] = design_score / max_score if max_score > 0 else 0.0
        
        # Generate overall recommendations
        validation_results["recommendations"] = self._generate_design_recommendations(
            validation_results["design_score"], validation_results["weaknesses"]
        )
        
        return validation_results
    
    def _validate_sample_size(self, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Validate sample size adequacy."""
        
        feedback = {"strengths": [], "weaknesses": []}
        score = 0.0
        
        sample_size = metadata.get("sample_size", 0)
        num_conditions = metadata.get("num_conditions", 1)
        
        if sample_size >= 30 * num_conditions:
            score = 1.0
            feedback["strengths"].append(f"Adequate sample size (n={sample_size}) for {num_conditions} conditions")
        elif sample_size >= 15 * num_conditions:
            score = 0.7
            feedback["strengths"].append("Moderate sample size")
            feedback["weaknesses"].append("Sample size could be larger for more robust results")
        else:
            score = 0.3
            feedback["weaknesses"].append(f"Small sample size (n={sample_size}) may limit statistical power")
        
        # Check for power analysis
        if metadata.get("power_analysis_conducted", False):
            score += 0.1
            feedback["strengths"].append("Power analysis was conducted")
        else:
            feedback["weaknesses"].append("No power analysis reported")
        
        return min(1.0, score), feedback
    
    def _validate_controls(self, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Validate control variables and conditions."""
        
        feedback = {"strengths": [], "weaknesses": []}
        score = 0.0
        
        # Check for control groups
        has_control_group = metadata.get("has_control_group", False)
        if has_control_group:
            score += 0.5
            feedback["strengths"].append("Control group included in design")
        else:
            feedback["weaknesses"].append("No control group specified")
        
        # Check for baseline measurements
        has_baseline = metadata.get("has_baseline_measurements", False)
        if has_baseline:
            score += 0.3
            feedback["strengths"].append("Baseline measurements included")
        
        # Check for controlled variables
        controlled_variables = metadata.get("controlled_variables", [])
        if len(controlled_variables) >= 3:
            score += 0.2
            feedback["strengths"].append(f"Multiple variables controlled ({len(controlled_variables)})")
        elif len(controlled_variables) > 0:
            score += 0.1
            feedback["strengths"].append("Some variables controlled")
        else:
            feedback["weaknesses"].append("No controlled variables specified")
        
        return min(1.0, score), feedback
    
    def _validate_randomization(self, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Validate randomization procedures."""
        
        feedback = {"strengths": [], "weaknesses": []}
        score = 0.0
        
        randomization_method = metadata.get("randomization_method", "none")
        
        if randomization_method in ["stratified", "block"]:
            score = 1.0
            feedback["strengths"].append(f"Sophisticated randomization method: {randomization_method}")
        elif randomization_method == "simple":
            score = 0.7
            feedback["strengths"].append("Simple randomization used")
        elif randomization_method == "systematic":
            score = 0.5
            feedback["strengths"].append("Systematic assignment used")
            feedback["weaknesses"].append("Systematic assignment may introduce bias")
        else:
            score = 0.0
            feedback["weaknesses"].append("No randomization method specified")
        
        # Check for random seed reporting
        if metadata.get("random_seed_reported", False):
            score += 0.1
            feedback["strengths"].append("Random seed reported for reproducibility")
        
        return min(1.0, score), feedback
    
    def _validate_blinding(self, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Validate blinding procedures."""
        
        feedback = {"strengths": [], "weaknesses": []}
        score = 0.0
        
        blinding_level = metadata.get("blinding_level", "none")
        
        if blinding_level == "double":
            score = 1.0
            feedback["strengths"].append("Double-blind design implemented")
        elif blinding_level == "single":
            score = 0.7
            feedback["strengths"].append("Single-blind design implemented")
        elif blinding_level == "assessor":
            score = 0.5
            feedback["strengths"].append("Assessor blinding implemented")
        else:
            score = 0.2
            feedback["weaknesses"].append("No blinding reported - may introduce bias")
        
        # For computational studies, blinding may not always be applicable
        study_type = metadata.get("study_type", "computational")
        if study_type == "computational" and blinding_level == "none":
            score = max(score, 0.6)  # Don't penalize too heavily
            feedback["weaknesses"] = [w for w in feedback["weaknesses"] 
                                     if "blinding" not in w.lower()]
            feedback["strengths"].append("Computational study - blinding considerations addressed")
        
        return score, feedback
    
    def _validate_measurements(self, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Validate measurement procedures and metrics."""
        
        feedback = {"strengths": [], "weaknesses": []}
        score = 0.0
        
        # Check for multiple metrics
        num_metrics = metadata.get("num_primary_metrics", 0)
        if num_metrics >= 3:
            score += 0.4
            feedback["strengths"].append(f"Multiple primary metrics ({num_metrics})")
        elif num_metrics > 0:
            score += 0.2
            feedback["strengths"].append("Primary metrics specified")
        else:
            feedback["weaknesses"].append("No primary metrics specified")
        
        # Check for validation of metrics
        metrics_validated = metadata.get("metrics_validated", False)
        if metrics_validated:
            score += 0.3
            feedback["strengths"].append("Metrics validation reported")
        else:
            feedback["weaknesses"].append("No metric validation reported")
        
        # Check for inter-rater reliability (if applicable)
        if metadata.get("inter_rater_reliability_assessed", False):
            score += 0.2
            feedback["strengths"].append("Inter-rater reliability assessed")
        
        # Check for measurement protocol
        has_protocol = metadata.get("has_measurement_protocol", False)
        if has_protocol:
            score += 0.1
            feedback["strengths"].append("Measurement protocol specified")
        
        return min(1.0, score), feedback
    
    def _generate_design_recommendations(self, design_score: float, 
                                       weaknesses: List[str]) -> List[str]:
        """Generate recommendations for improving experimental design."""
        
        recommendations = []
        
        if design_score < 0.6:
            recommendations.append("Experimental design needs significant improvement before publication.")
        elif design_score < 0.8:
            recommendations.append("Good experimental design with room for improvement.")
        else:
            recommendations.append("Excellent experimental design suitable for high-impact publication.")
        
        # Specific recommendations based on weaknesses
        weakness_recommendations = {
            "sample size": "Consider power analysis to determine adequate sample size",
            "control group": "Include appropriate control conditions for comparison",
            "randomization": "Implement proper randomization procedures to reduce bias",
            "blinding": "Consider blinding procedures where applicable to reduce evaluation bias",
            "baseline": "Include baseline measurements for better change detection",
            "metrics": "Validate measurement instruments and report reliability statistics"
        }
        
        for weakness in weaknesses:
            for key, recommendation in weakness_recommendations.items():
                if key in weakness.lower():
                    recommendations.append(recommendation)
                    break
        
        return recommendations


class ResearchValidationFramework:
    """Complete research validation framework."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.JOURNAL):
        self.validation_level = validation_level
        self.statistical_validator = StatisticalValidator()
        self.reproducibility_validator = ReproducibilityValidator()
        self.methodological_validator = MethodologicalValidator()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def comprehensive_validation(self, 
                                     experiment_data: Dict[str, Any],
                                     experiment_function: Optional[Callable] = None,
                                     comparison_groups: Optional[Dict[str, List[float]]] = None) -> ResearchValidationResult:
        """
        Perform comprehensive research validation.
        
        Args:
            experiment_data: Complete experiment data and metadata
            experiment_function: Function for reproducibility testing
            comparison_groups: Groups for statistical comparison
        
        Returns:
            Complete validation result
        """
        
        validation_id = f"validation_{int(time.time())}_{hash(str(experiment_data)) % 10000}"
        
        self.logger.info(f"Starting comprehensive research validation: {validation_id}")
        self.logger.info(f"Validation level: {self.validation_level.value}")
        
        # 1. Statistical Validation
        statistical_tests = []
        
        if comparison_groups:
            self.logger.info("Performing statistical validation")
            
            # Multiple comparison testing with correction
            statistical_tests = self.statistical_validator.validate_multiple_comparisons(
                comparison_groups, correction_method="fdr"
            )
        
        # 2. Reproducibility Validation
        reproducibility_report = None
        
        if experiment_function:
            self.logger.info("Performing reproducibility validation")
            
            experiment_params = experiment_data.get("experiment_parameters", {})
            num_replications = self._get_replication_count_for_level()
            
            reproducibility_report = await self.reproducibility_validator.validate_reproducibility(
                experiment_function, experiment_params, num_replications
            )
        else:
            # Create empty reproducibility report
            reproducibility_report = ReproducibilityReport(
                experiment_id="none",
                original_results={},
                reproduction_attempts=[],
                reproducibility_score=0.0,
                statistical_agreement=0.0,
                computational_agreement=0.0,
                cross_platform_agreement=0.0,
                failed_reproductions=[],
                recommendations=["No experiment function provided for reproducibility testing"]
            )
        
        # 3. Methodological Validation
        self.logger.info("Performing methodological validation")
        
        experiment_metadata = experiment_data.get("experiment_metadata", {})
        methodological_assessment = self.methodological_validator.validate_experimental_design(
            experiment_metadata
        )
        
        # 4. Data Quality Assessment
        data_quality_metrics = self._assess_data_quality(experiment_data)
        
        # 5. Publication Readiness Assessment
        publication_readiness = self._assess_publication_readiness(
            statistical_tests, reproducibility_report, methodological_assessment,
            data_quality_metrics
        )
        
        # 6. Calculate Overall Score
        overall_score = self._calculate_overall_validation_score(
            statistical_tests, reproducibility_report, methodological_assessment,
            data_quality_metrics
        )
        
        # Create validation result
        validation_result = ResearchValidationResult(
            validation_id=validation_id,
            experiment_metadata=experiment_metadata,
            validation_level=self.validation_level,
            statistical_tests=statistical_tests,
            reproducibility_report=reproducibility_report,
            methodological_assessment=methodological_assessment,
            data_quality_metrics=data_quality_metrics,
            publication_readiness=publication_readiness,
            overall_score=overall_score
        )
        
        self.logger.info(f"Validation complete: overall score={overall_score:.3f}")
        
        return validation_result
    
    def _get_replication_count_for_level(self) -> int:
        """Get required number of replications based on validation level."""
        
        replication_counts = {
            ValidationLevel.BASIC: 3,
            ValidationLevel.CONFERENCE: 5,
            ValidationLevel.JOURNAL: 7,
            ValidationLevel.REPRODUCIBLE: 10,
            ValidationLevel.META_ANALYSIS: 15
        }
        
        return replication_counts.get(self.validation_level, 5)
    
    def _assess_data_quality(self, experiment_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of experimental data."""
        
        quality_metrics = {}
        
        # Data completeness
        raw_data = experiment_data.get("raw_data", {})
        if raw_data:
            total_expected_data_points = sum(len(values) if isinstance(values, list) else 1 
                                           for values in raw_data.values())
            missing_data_points = sum(1 for values in raw_data.values() 
                                    for value in (values if isinstance(values, list) else [values])
                                    if value is None or (isinstance(value, float) and np.isnan(value)))
            
            quality_metrics["data_completeness"] = 1.0 - (missing_data_points / max(total_expected_data_points, 1))
        else:
            quality_metrics["data_completeness"] = 0.0
        
        # Data consistency
        consistency_score = 0.8  # Default assumption
        if raw_data:
            # Check for extreme outliers across metrics
            outlier_ratios = []
            for key, values in raw_data.items():
                if isinstance(values, list) and len(values) > 3:
                    values_array = np.array([v for v in values if v is not None and not np.isnan(v)])
                    if len(values_array) > 0:
                        Q1 = np.percentile(values_array, 25)
                        Q3 = np.percentile(values_array, 75)
                        IQR = Q3 - Q1
                        outliers = np.sum((values_array < Q1 - 3*IQR) | (values_array > Q3 + 3*IQR))
                        outlier_ratio = outliers / len(values_array)
                        outlier_ratios.append(outlier_ratio)
            
            if outlier_ratios:
                avg_outlier_ratio = np.mean(outlier_ratios)
                consistency_score = max(0.0, 1.0 - avg_outlier_ratio * 5)  # Penalize outliers
        
        quality_metrics["data_consistency"] = consistency_score
        
        # Measurement precision
        precision_scores = []
        if raw_data:
            for key, values in raw_data.items():
                if isinstance(values, list) and len(values) > 1:
                    values_array = np.array([v for v in values if v is not None and not np.isnan(v)])
                    if len(values_array) > 1:
                        cv = np.std(values_array) / max(abs(np.mean(values_array)), 1e-10)
                        precision = max(0.0, 1.0 - cv)  # Lower CV = higher precision
                        precision_scores.append(precision)
        
        quality_metrics["measurement_precision"] = np.mean(precision_scores) if precision_scores else 0.5
        
        # Sample size adequacy
        sample_sizes = []
        if raw_data:
            for values in raw_data.values():
                if isinstance(values, list):
                    sample_sizes.append(len(values))
        
        min_sample_size = min(sample_sizes) if sample_sizes else 0
        quality_metrics["sample_size_adequacy"] = min(1.0, min_sample_size / 30.0)  # 30 as reference
        
        return quality_metrics
    
    def _assess_publication_readiness(self, statistical_tests: List[StatisticalTestResult],
                                    reproducibility_report: ReproducibilityReport,
                                    methodological_assessment: Dict[str, Any],
                                    data_quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess readiness for publication."""
        
        readiness = {
            "ready_for_submission": False,
            "readiness_score": 0.0,
            "publication_level": "none",
            "requirements_met": {},
            "blocking_issues": [],
            "recommendations": []
        }
        
        # Requirements for different publication levels
        requirements = {
            "workshop": {"statistical_significance": 0.3, "reproducibility": 0.5, "methodology": 0.4, "data_quality": 0.5},
            "conference": {"statistical_significance": 0.6, "reproducibility": 0.7, "methodology": 0.6, "data_quality": 0.7},
            "journal": {"statistical_significance": 0.8, "reproducibility": 0.8, "methodology": 0.8, "data_quality": 0.8},
            "top_journal": {"statistical_significance": 0.9, "reproducibility": 0.9, "methodology": 0.9, "data_quality": 0.9}
        }
        
        # Calculate component scores
        if statistical_tests:
            significant_tests = [t for t in statistical_tests if t.significant]
            statistical_score = len(significant_tests) / len(statistical_tests)
        else:
            statistical_score = 0.5  # Neutral if no tests
        
        reproducibility_score = reproducibility_report.reproducibility_score
        methodology_score = methodological_assessment["design_score"]
        data_quality_score = np.mean(list(data_quality_metrics.values()))
        
        readiness["requirements_met"] = {
            "statistical_significance": statistical_score,
            "reproducibility": reproducibility_score,
            "methodology": methodology_score,
            "data_quality": data_quality_score
        }
        
        # Determine publication level
        for level, thresholds in requirements.items():
            meets_all_requirements = all(
                readiness["requirements_met"][req] >= threshold
                for req, threshold in thresholds.items()
            )
            
            if meets_all_requirements:
                readiness["publication_level"] = level
                readiness["ready_for_submission"] = True
            else:
                # Identify blocking issues for this level
                blocking = []
                for req, threshold in thresholds.items():
                    if readiness["requirements_met"][req] < threshold:
                        blocking.append(f"{req}: {readiness['requirements_met'][req]:.2f} < {threshold:.2f}")
                
                if not readiness["blocking_issues"]:  # Store blocking issues for lowest unmet level
                    readiness["blocking_issues"] = blocking
        
        # Calculate overall readiness score
        readiness["readiness_score"] = np.mean(list(readiness["requirements_met"].values()))
        
        # Generate recommendations
        if not readiness["ready_for_submission"]:
            readiness["recommendations"].append("Address blocking issues before submission")
            
            if statistical_score < 0.7:
                readiness["recommendations"].append("Strengthen statistical analysis with larger sample sizes or additional tests")
            
            if reproducibility_score < 0.7:
                readiness["recommendations"].append("Improve reproducibility with better documentation and code sharing")
            
            if methodology_score < 0.7:
                readiness["recommendations"].append("Strengthen experimental design with proper controls and randomization")
            
            if data_quality_score < 0.7:
                readiness["recommendations"].append("Improve data quality with more careful measurement procedures")
        
        return readiness
    
    def _calculate_overall_validation_score(self, statistical_tests: List[StatisticalTestResult],
                                          reproducibility_report: ReproducibilityReport,
                                          methodological_assessment: Dict[str, Any],
                                          data_quality_metrics: Dict[str, float]) -> float:
        """Calculate overall validation score."""
        
        # Component weights based on validation level
        weights = {
            ValidationLevel.BASIC: {"stats": 0.4, "repro": 0.2, "method": 0.2, "quality": 0.2},
            ValidationLevel.CONFERENCE: {"stats": 0.3, "repro": 0.3, "method": 0.2, "quality": 0.2},
            ValidationLevel.JOURNAL: {"stats": 0.25, "repro": 0.25, "method": 0.25, "quality": 0.25},
            ValidationLevel.REPRODUCIBLE: {"stats": 0.2, "repro": 0.4, "method": 0.2, "quality": 0.2},
            ValidationLevel.META_ANALYSIS: {"stats": 0.3, "repro": 0.3, "method": 0.2, "quality": 0.2}
        }
        
        level_weights = weights.get(self.validation_level, weights[ValidationLevel.JOURNAL])
        
        # Calculate component scores
        if statistical_tests:
            stats_score = np.mean([1.0 if t.significant else 0.5 for t in statistical_tests])
        else:
            stats_score = 0.5
        
        repro_score = reproducibility_report.reproducibility_score
        method_score = methodological_assessment["design_score"]
        quality_score = np.mean(list(data_quality_metrics.values()))
        
        # Weighted combination
        overall_score = (level_weights["stats"] * stats_score +
                        level_weights["repro"] * repro_score +
                        level_weights["method"] * method_score +
                        level_weights["quality"] * quality_score)
        
        return overall_score
    
    def generate_validation_report(self, validation_result: ResearchValidationResult) -> str:
        """Generate a human-readable validation report."""
        
        report = []
        report.append(f"# Research Validation Report")
        report.append(f"**Validation ID:** {validation_result.validation_id}")
        report.append(f"**Validation Level:** {validation_result.validation_level.value.title()}")
        report.append(f"**Overall Score:** {validation_result.overall_score:.3f}/1.000")
        report.append("")
        
        # Publication Readiness
        pub_readiness = validation_result.publication_readiness
        report.append("## Publication Readiness")
        report.append(f"**Ready for Submission:** {'✅ Yes' if pub_readiness['ready_for_submission'] else '❌ No'}")
        report.append(f"**Publication Level:** {pub_readiness['publication_level'].title()}")
        report.append(f"**Readiness Score:** {pub_readiness['readiness_score']:.3f}")
        
        if pub_readiness["blocking_issues"]:
            report.append("\n**Blocking Issues:**")
            for issue in pub_readiness["blocking_issues"]:
                report.append(f"- {issue}")
        
        # Statistical Tests
        report.append("\n## Statistical Analysis")
        if validation_result.statistical_tests:
            significant_tests = [t for t in validation_result.statistical_tests if t.significant]
            report.append(f"**Tests Performed:** {len(validation_result.statistical_tests)}")
            report.append(f"**Significant Results:** {len(significant_tests)}")
            
            for test in validation_result.statistical_tests[:5]:  # Show top 5
                significance_marker = "✅" if test.significant else "❌"
                report.append(f"- {significance_marker} {test.test_name}: p={test.p_value:.4f}, "
                            f"effect size={test.effect_size:.3f if test.effect_size else 'N/A'}")
        else:
            report.append("No statistical tests performed.")
        
        # Reproducibility
        report.append("\n## Reproducibility")
        repro = validation_result.reproducibility_report
        report.append(f"**Reproducibility Score:** {repro.reproducibility_score:.3f}")
        report.append(f"**Statistical Agreement:** {repro.statistical_agreement:.3f}")
        report.append(f"**Replications Attempted:** {len(repro.reproduction_attempts)}")
        report.append(f"**Failed Reproductions:** {len(repro.failed_reproductions)}")
        
        # Methodology
        report.append("\n## Methodology")
        method = validation_result.methodological_assessment
        report.append(f"**Design Score:** {method['design_score']:.3f}")
        
        if method["strengths"]:
            report.append("\n**Methodological Strengths:**")
            for strength in method["strengths"][:5]:
                report.append(f"- {strength}")
        
        if method["weaknesses"]:
            report.append("\n**Methodological Weaknesses:**")
            for weakness in method["weaknesses"][:5]:
                report.append(f"- {weakness}")
        
        # Data Quality
        report.append("\n## Data Quality")
        quality = validation_result.data_quality_metrics
        for metric, score in quality.items():
            status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
            report.append(f"- {status} {metric.replace('_', '').title()}: {score:.3f}")
        
        # Recommendations
        if pub_readiness["recommendations"]:
            report.append("\n## Recommendations")
            for rec in pub_readiness["recommendations"]:
                report.append(f"- {rec}")
        
        return "\n".join(report)
    
    def export_validation_results(self, validation_result: ResearchValidationResult, 
                                filepath: str):
        """Export validation results to file."""
        
        export_data = validation_result.to_dict()
        export_data["export_timestamp"] = time.time()
        export_data["validation_report"] = self.generate_validation_report(validation_result)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Validation results exported to {filepath}")


# Example usage and testing
async def run_research_validation_example():
    """Example of comprehensive research validation."""
    
    print("=== Research Validation Framework Example ===")
    
    # Create validation framework
    framework = ResearchValidationFramework(ValidationLevel.JOURNAL)
    
    # Mock experiment function
    async def mock_experiment(model_name="test_model", num_samples=100, random_seed=42, _replication_id=0):
        """Mock experiment function for reproducibility testing."""
        
        # Simulate some variability between replications
        np.random.seed(random_seed + _replication_id * 1000)
        
        # Generate mock results with some controlled randomness
        base_fvd = 100.0
        base_is = 35.0
        base_clip = 0.3
        
        # Add replication variability
        fvd_score = base_fvd + np.random.normal(0, 2.0)
        inception_score = base_is + np.random.normal(0, 1.5)
        clip_similarity = base_clip + np.random.normal(0, 0.02)
        
        # Add systematic improvement for some replications (to test detection)
        if _replication_id > 2:
            fvd_score -= 5.0  # Systematic improvement
        
        return {
            "fvd_score": fvd_score,
            "inception_score": inception_score,
            "clip_similarity": clip_similarity,
            "latency_ms": 2000 + np.random.normal(0, 100),
            "memory_gb": 8.0 + np.random.normal(0, 0.5)
        }
    
    # Prepare experiment data
    experiment_data = {
        "experiment_metadata": {
            "sample_size": 150,
            "num_conditions": 3,
            "has_control_group": True,
            "has_baseline_measurements": True,
            "controlled_variables": ["resolution", "inference_steps", "guidance_scale"],
            "randomization_method": "stratified",
            "random_seed_reported": True,
            "blinding_level": "assessor",
            "study_type": "computational",
            "num_primary_metrics": 5,
            "metrics_validated": True,
            "power_analysis_conducted": True,
            "has_measurement_protocol": True
        },
        "experiment_parameters": {
            "model_name": "advanced_video_diffusion",
            "num_samples": 100,
            "random_seed": 42
        },
        "raw_data": {
            "group_a_fvd": [98.2, 97.1, 99.5, 96.8, 98.9, 97.3, 98.6, 99.1, 97.8, 98.4],
            "group_b_fvd": [102.3, 101.8, 103.1, 100.9, 102.7, 101.5, 103.4, 102.0, 101.2, 102.8],
            "group_c_fvd": [95.1, 94.8, 96.2, 95.7, 94.9, 95.4, 96.0, 95.2, 94.6, 95.8],
            "group_a_is": [36.2, 35.8, 36.5, 35.9, 36.1, 36.3, 36.0, 35.7, 36.4, 36.2],
            "group_b_is": [34.1, 33.8, 34.5, 34.2, 33.9, 34.0, 34.3, 34.1, 33.7, 34.4],
            "group_c_is": [37.8, 38.1, 37.5, 37.9, 38.0, 37.6, 38.2, 37.7, 38.3, 37.4]
        }
    }
    
    # Prepare comparison groups for statistical testing
    comparison_groups = {
        "Model_A": experiment_data["raw_data"]["group_a_fvd"],
        "Model_B": experiment_data["raw_data"]["group_b_fvd"],
        "Model_C": experiment_data["raw_data"]["group_c_fvd"]
    }
    
    # Run comprehensive validation
    print("Running comprehensive research validation...")
    
    validation_result = await framework.comprehensive_validation(
        experiment_data=experiment_data,
        experiment_function=mock_experiment,
        comparison_groups=comparison_groups
    )
    
    # Display results
    print(f"\n=== Validation Results ===")
    print(f"Overall Score: {validation_result.overall_score:.3f}")
    print(f"Publication Ready: {'✅ Yes' if validation_result.publication_readiness['ready_for_submission'] else '❌ No'}")
    print(f"Publication Level: {validation_result.publication_readiness['publication_level'].title()}")
    
    # Statistical tests summary
    print(f"\n--- Statistical Tests ---")
    significant_tests = [t for t in validation_result.statistical_tests if t.significant]
    print(f"Total tests: {len(validation_result.statistical_tests)}")
    print(f"Significant results: {len(significant_tests)}")
    
    for test in validation_result.statistical_tests:
        significance = "✅" if test.significant else "❌"
        print(f"  {significance} {test.test_name}: p={test.p_value:.4f}, "
              f"effect size={test.effect_size:.3f if test.effect_size else 'N/A'}")
    
    # Reproducibility summary
    print(f"\n--- Reproducibility ---")
    repro = validation_result.reproducibility_report
    print(f"Reproducibility score: {repro.reproducibility_score:.3f}")
    print(f"Statistical agreement: {repro.statistical_agreement:.3f}")
    print(f"Reproductions attempted: {len(repro.reproduction_attempts)}")
    print(f"Failed reproductions: {len(repro.failed_reproductions)}")
    
    # Methodology summary
    print(f"\n--- Methodology ---")
    method = validation_result.methodological_assessment
    print(f"Design score: {method['design_score']:.3f}")
    print(f"Strengths: {len(method['strengths'])}")
    print(f"Weaknesses: {len(method['weaknesses'])}")
    
    # Data quality summary
    print(f"\n--- Data Quality ---")
    for metric, score in validation_result.data_quality_metrics.items():
        status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
        print(f"  {status} {metric.replace('_', ' ').title()}: {score:.3f}")
    
    # Recommendations
    if validation_result.publication_readiness["recommendations"]:
        print(f"\n--- Recommendations ---")
        for rec in validation_result.publication_readiness["recommendations"]:
            print(f"  - {rec}")
    
    # Generate and display report
    print(f"\n=== Full Validation Report ===")
    report = framework.generate_validation_report(validation_result)
    print(report)
    
    # Export results
    export_path = "research_validation_results.json"
    framework.export_validation_results(validation_result, export_path)
    print(f"\nValidation results exported to {export_path}")
    
    # Additional statistical analysis example
    print(f"\n=== Additional Statistical Analysis ===")
    
    # Test individual comparison
    group_a = experiment_data["raw_data"]["group_a_fvd"]
    group_c = experiment_data["raw_data"]["group_c_fvd"]
    
    individual_test = framework.statistical_validator.validate_comparison(
        group_a, group_c, "Model A vs Model C (FVD)"
    )
    
    print(f"Individual test result:")
    print(f"  Test: {individual_test.test_name}")
    print(f"  Significant: {'✅ Yes' if individual_test.significant else '❌ No'}")
    print(f"  P-value: {individual_test.p_value:.6f}")
    print(f"  Effect size: {individual_test.effect_size:.3f}")
    print(f"  Interpretation: {individual_test.interpretation}")
    
    return {
        "framework": framework,
        "validation_result": validation_result,
        "individual_test": individual_test,
        "export_path": export_path
    }


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(run_research_validation_example())