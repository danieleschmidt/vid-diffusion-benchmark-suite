"""Research module for advanced video diffusion benchmarking.

This module provides comprehensive research tools for conducting reproducible,
publication-ready video diffusion model studies including:

- Novel video quality metrics with advanced analyzers
- Statistical significance analysis with multiple testing corrections
- Experimental framework with reproducibility controls  
- Context compression and efficiency optimization
- Bayesian analysis and effect size calculations

Designed for academic research and industrial R&D applications.
"""

from .context_compression import ContextCompressor, RetrievalFreeCompressor
from .novel_metrics import (
    NovelVideoMetrics, 
    AdvancedVideoMetrics,
    PerceptualQualityAnalyzer,
    MotionDynamicsAnalyzer,
    SemanticConsistencyAnalyzer,
    CrossModalAlignmentAnalyzer
)
from .statistical_analysis import (
    StatisticalSignificanceAnalyzer, 
    BenchmarkStatistics,
    BenchmarkStatisticalAnalysis,
    StatisticalTest,
    ComparisonResult,
    EffectSizeCalculator,
    PowerAnalyzer,
    BayesianAnalyzer
)
from .experimental_framework import (
    ExperimentalFramework, 
    ReproducibilityManager,
    ExperimentConfig,
    ExperimentResult,
    ReproducibilityReport
)

__all__ = [
    # Context compression
    "ContextCompressor",
    "RetrievalFreeCompressor", 
    
    # Novel metrics
    "NovelVideoMetrics",
    "AdvancedVideoMetrics",
    "PerceptualQualityAnalyzer",
    "MotionDynamicsAnalyzer", 
    "SemanticConsistencyAnalyzer",
    "CrossModalAlignmentAnalyzer",
    
    # Statistical analysis
    "StatisticalSignificanceAnalyzer",
    "BenchmarkStatistics",
    "BenchmarkStatisticalAnalysis",
    "StatisticalTest",
    "ComparisonResult",
    "EffectSizeCalculator",
    "PowerAnalyzer",
    "BayesianAnalyzer",
    
    # Experimental framework
    "ExperimentalFramework",
    "ReproducibilityManager",
    "ExperimentConfig", 
    "ExperimentResult",
    "ReproducibilityReport"
]