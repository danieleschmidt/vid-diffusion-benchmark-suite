"""Progressive Quality Gates v2.0 - Adaptive Learning Quality Assurance

Advanced quality gate system that learns from project patterns, adapts thresholds
dynamically, validates progressive enhancement implementations, and provides
autonomous improvement recommendations with temporal learning.

This system evolves beyond static quality checks to become an intelligent
quality assurance partner that understands your project's specific needs.
"""

import asyncio
import time
import json
import logging
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import numpy as np
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QualityEvolutionStage(Enum):
    """Progressive enhancement stages."""
    GENERATION_1_SIMPLE = "gen1_simple"
    GENERATION_2_ROBUST = "gen2_robust"
    GENERATION_3_OPTIMIZED = "gen3_optimized"
    ADAPTIVE_LEARNING = "adaptive_learning"
    AUTONOMOUS_IMPROVEMENT = "autonomous_improvement"


class LearningMode(Enum):
    """Quality gate learning modes."""
    STATIC = "static"              # Traditional fixed thresholds
    ADAPTIVE = "adaptive"          # Learns from project patterns
    PREDICTIVE = "predictive"      # Predicts quality issues
    AUTONOMOUS = "autonomous"      # Self-improving system


@dataclass
class QualityPattern:
    """Learned quality patterns from project history."""
    pattern_id: str
    category: str
    pattern_type: str  # "improvement", "regression", "stable"
    confidence: float  # 0.0 to 1.0
    success_rate: float
    occurrences: int
    last_seen: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressiveMetric:
    """Enhanced quality metric with learning capabilities."""
    name: str
    category: str
    score: float
    baseline_score: float
    trend: str  # "improving", "declining", "stable"
    confidence: float
    learning_data: Dict[str, Any]
    generation_compliance: Dict[QualityEvolutionStage, bool]
    recommendations: List[str]
    auto_fix_available: bool = False
    predicted_score: Optional[float] = None


@dataclass
class ProgressiveQualityReport:
    """Comprehensive progressive quality assessment."""
    timestamp: float
    generation_stage: QualityEvolutionStage
    learning_mode: LearningMode
    overall_score: float
    baseline_comparison: float  # vs historical baseline
    trend_analysis: Dict[str, float]
    progressive_metrics: List[ProgressiveMetric]
    learned_patterns: List[QualityPattern]
    autonomous_improvements: List[str]
    generation_readiness: Dict[QualityEvolutionStage, bool]
    quality_trajectory: List[Tuple[float, float]]  # (timestamp, score)
    predictive_insights: Dict[str, Any]
    execution_time: float


class HistoricalQualityDatabase:
    """Stores and manages historical quality data for learning."""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.quality_history: deque = deque(maxlen=1000)
        self.pattern_database: Dict[str, QualityPattern] = {}
        self.threshold_history: Dict[str, List[float]] = defaultdict(list)
        self.load_database()
    
    def load_database(self):
        """Load historical quality data."""
        try:
            history_file = self.db_path / "quality_history.pkl"
            if history_file.exists():
                with open(history_file, 'rb') as f:
                    data = pickle.load(f)
                    self.quality_history = data.get('history', deque(maxlen=1000))
                    self.pattern_database = data.get('patterns', {})
                    self.threshold_history = data.get('thresholds', defaultdict(list))
        except Exception as e:
            logger.warning(f"Could not load quality database: {e}")
    
    def save_database(self):
        """Save quality data to persistent storage."""
        try:
            history_file = self.db_path / "quality_history.pkl"
            data = {
                'history': self.quality_history,
                'patterns': self.pattern_database,
                'thresholds': dict(self.threshold_history)
            }
            with open(history_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save quality database: {e}")
    
    def add_quality_record(self, report: ProgressiveQualityReport):
        """Add quality report to history."""
        record = {
            'timestamp': report.timestamp,
            'overall_score': report.overall_score,
            'generation_stage': report.generation_stage.value,
            'metrics': {m.name: m.score for m in report.progressive_metrics},
            'categories': self._calculate_category_scores(report.progressive_metrics)
        }
        self.quality_history.append(record)
        
        # Update pattern database
        self._update_patterns(report)
        
        # Save to persistent storage
        self.save_database()
    
    def _update_patterns(self, report: ProgressiveQualityReport):
        """Update learned patterns from new report."""
        for pattern in report.learned_patterns:
            if pattern.pattern_id in self.pattern_database:
                existing = self.pattern_database[pattern.pattern_id]
                existing.occurrences += 1
                existing.last_seen = time.time()
                existing.confidence = min(1.0, existing.confidence + 0.05)
            else:
                self.pattern_database[pattern.pattern_id] = pattern
    
    def _calculate_category_scores(self, metrics: List[ProgressiveMetric]) -> Dict[str, float]:
        """Calculate category scores from metrics."""
        category_scores = defaultdict(list)
        for metric in metrics:
            category_scores[metric.category].append(metric.score)
        return {cat: statistics.mean(scores) for cat, scores in category_scores.items()}
    
    def get_baseline_scores(self) -> Dict[str, float]:
        """Get baseline scores for comparison."""
        if len(self.quality_history) < 5:
            return {}
        
        recent_records = list(self.quality_history)[-10:]  # Last 10 records
        baseline = {}
        
        # Calculate baseline from historical data
        all_metrics = set()
        for record in recent_records:
            all_metrics.update(record['metrics'].keys())
        
        for metric_name in all_metrics:
            scores = [r['metrics'].get(metric_name, 0.0) for r in recent_records if metric_name in r['metrics']]
            if scores:
                baseline[metric_name] = statistics.median(scores)
        
        return baseline
    
    def predict_quality_trend(self, metric_name: str) -> Tuple[float, str]:
        """Predict quality trend for a metric."""
        if len(self.quality_history) < 3:
            return 0.0, "insufficient_data"
        
        recent_scores = []
        for record in list(self.quality_history)[-10:]:
            if metric_name in record['metrics']:
                recent_scores.append(record['metrics'][metric_name])
        
        if len(recent_scores) < 3:
            return 0.0, "insufficient_data"
        
        # Simple linear regression for trend
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        slope = np.polyfit(x, y, 1)[0]
        
        predicted_next = recent_scores[-1] + slope
        
        if slope > 0.02:
            trend = "improving"
        elif slope < -0.02:
            trend = "declining"
        else:
            trend = "stable"
        
        return predicted_next, trend


class AdaptiveThresholdManager:
    """Manages dynamic quality thresholds based on project characteristics."""
    
    def __init__(self, db: HistoricalQualityDatabase):
        self.db = db
        self.base_thresholds = {
            # Code Quality Thresholds
            'syntax_errors': 0.0,
            'complexity_average': 8.0,
            'line_length_violations': 0.05,
            'documentation_coverage': 0.7,
            'code_duplication': 0.1,
            
            # Security Thresholds
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 2,
            'hardcoded_secrets': 0,
            
            # Performance Thresholds
            'startup_time_ms': 5000,
            'memory_usage_mb': 1024,
            'algorithmic_complexity': 2.0,
            
            # Progressive Enhancement Thresholds
            'generation1_completeness': 0.8,
            'generation2_robustness': 0.7,
            'generation3_optimization': 0.6
        }
        
    def get_adaptive_thresholds(self, project_characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adaptive thresholds based on project characteristics."""
        adaptive_thresholds = self.base_thresholds.copy()
        
        # Adjust based on project size
        project_size = project_characteristics.get('total_files', 50)
        if project_size > 100:
            adaptive_thresholds['complexity_average'] = 10.0
            adaptive_thresholds['documentation_coverage'] = 0.8
        elif project_size < 20:
            adaptive_thresholds['complexity_average'] = 6.0
            adaptive_thresholds['documentation_coverage'] = 0.6
        
        # Adjust based on project type
        project_type = project_characteristics.get('type', 'library')
        if project_type in ['web_app', 'api']:
            adaptive_thresholds['startup_time_ms'] = 3000
            adaptive_thresholds['high_vulnerabilities'] = 0
        elif project_type == 'research':
            adaptive_thresholds['documentation_coverage'] = 0.9
            adaptive_thresholds['complexity_average'] = 12.0
        
        # Learn from historical performance
        historical_baselines = self.db.get_baseline_scores()
        for metric_name, baseline in historical_baselines.items():
            if metric_name in adaptive_thresholds:
                # Adjust threshold based on project's historical performance
                current_threshold = adaptive_thresholds[metric_name]
                adjusted_threshold = (baseline * 0.7) + (current_threshold * 0.3)
                adaptive_thresholds[metric_name] = adjusted_threshold
        
        return adaptive_thresholds


class ProgressiveEnhancementValidator:
    """Validates implementation against progressive enhancement stages."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.stage_requirements = {
            QualityEvolutionStage.GENERATION_1_SIMPLE: {
                'required_modules': ['benchmark', 'metrics', 'models'],
                'min_test_coverage': 0.5,
                'basic_functionality': True,
                'error_handling': 'basic'
            },
            QualityEvolutionStage.GENERATION_2_ROBUST: {
                'required_modules': ['monitoring', 'validation', 'security'],
                'min_test_coverage': 0.7,
                'error_handling': 'comprehensive',
                'logging_system': True,
                'input_validation': True
            },
            QualityEvolutionStage.GENERATION_3_OPTIMIZED: {
                'required_modules': ['optimization', 'scaling', 'caching'],
                'min_test_coverage': 0.8,
                'performance_benchmarks': True,
                'scalability_features': True,
                'resource_optimization': True
            }
        }
    
    async def validate_generation_compliance(self) -> Dict[QualityEvolutionStage, bool]:
        """Validate compliance with each generation's requirements."""
        compliance = {}
        
        for stage, requirements in self.stage_requirements.items():
            compliance[stage] = await self._check_stage_compliance(stage, requirements)
        
        return compliance
    
    async def _check_stage_compliance(self, stage: QualityEvolutionStage, requirements: Dict[str, Any]) -> bool:
        """Check if project meets requirements for a specific stage."""
        checks_passed = 0
        total_checks = len(requirements)
        
        # Check required modules
        if 'required_modules' in requirements:
            modules_present = await self._check_required_modules(requirements['required_modules'])
            if modules_present:
                checks_passed += 1
        
        # Check test coverage
        if 'min_test_coverage' in requirements:
            coverage = await self._estimate_test_coverage()
            if coverage >= requirements['min_test_coverage']:
                checks_passed += 1
        
        # Check error handling implementation
        if 'error_handling' in requirements:
            error_handling_level = await self._assess_error_handling()
            expected_level = requirements['error_handling']
            if self._meets_error_handling_requirement(error_handling_level, expected_level):
                checks_passed += 1
        
        # Check logging system
        if requirements.get('logging_system', False):
            has_logging = await self._check_logging_system()
            if has_logging:
                checks_passed += 1
        
        # Check performance benchmarks
        if requirements.get('performance_benchmarks', False):
            has_benchmarks = await self._check_performance_benchmarks()
            if has_benchmarks:
                checks_passed += 1
        
        return checks_passed >= (total_checks * 0.8)  # 80% compliance required
    
    async def _check_required_modules(self, required_modules: List[str]) -> bool:
        """Check if required modules are present."""
        src_path = self.project_root / "src"
        if not src_path.exists():
            return False
        
        found_modules = 0
        for module_name in required_modules:
            # Look for module files or directories
            module_patterns = [
                f"**/*{module_name}*.py",
                f"**/{module_name}/**/*.py"
            ]
            
            for pattern in module_patterns:
                matches = list(src_path.glob(pattern))
                if matches:
                    found_modules += 1
                    break
        
        return found_modules >= len(required_modules) * 0.7  # 70% of modules required
    
    async def _estimate_test_coverage(self) -> float:
        """Estimate test coverage based on test files."""
        test_files = list(self.project_root.glob("test*.py")) + list(self.project_root.glob("**/test_*.py"))
        source_files = list(self.project_root.glob("**/*.py"))
        source_files = [f for f in source_files if "test" not in f.name.lower()]
        
        if not source_files:
            return 0.0
        
        # Simple heuristic: ratio of test files to source files
        coverage_estimate = min(1.0, len(test_files) / len(source_files))
        return coverage_estimate
    
    async def _assess_error_handling(self) -> str:
        """Assess the level of error handling implementation."""
        error_handling_patterns = {
            'basic': [r'try:', r'except:'],
            'comprehensive': [r'except\s+\w+\s+as', r'logger\.error', r'raise\s+\w+Error'],
            'advanced': [r'custom.*Error', r'error_handler', r'circuit.*breaker']
        }
        
        pattern_matches = defaultdict(int)
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for level, patterns in error_handling_patterns.items():
                    for pattern in patterns:
                        import re
                        matches = len(re.findall(pattern, content, re.IGNORECASE))
                        pattern_matches[level] += matches
            except Exception:
                continue
        
        if pattern_matches['advanced'] > 5:
            return 'advanced'
        elif pattern_matches['comprehensive'] > 10:
            return 'comprehensive'
        elif pattern_matches['basic'] > 5:
            return 'basic'
        else:
            return 'minimal'
    
    def _meets_error_handling_requirement(self, actual_level: str, required_level: str) -> bool:
        """Check if actual error handling meets requirement."""
        level_hierarchy = ['minimal', 'basic', 'comprehensive', 'advanced']
        actual_index = level_hierarchy.index(actual_level) if actual_level in level_hierarchy else 0
        required_index = level_hierarchy.index(required_level) if required_level in level_hierarchy else 0
        return actual_index >= required_index
    
    async def _check_logging_system(self) -> bool:
        """Check if comprehensive logging system is implemented."""
        logging_indicators = [
            'logger = logging.getLogger',
            'logging.basicConfig',
            'LOG_LEVEL',
            'log_config',
            'structured_logging'
        ]
        
        matches = 0
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for indicator in logging_indicators:
                    if indicator in content:
                        matches += 1
                        break  # Count each file only once
            except Exception:
                continue
        
        return matches >= 3  # At least 3 files should have logging
    
    async def _check_performance_benchmarks(self) -> bool:
        """Check if performance benchmarking is implemented."""
        benchmark_indicators = [
            'benchmark',
            'profiler',
            'performance',
            'timing',
            'memory_usage',
            'throughput'
        ]
        
        for indicator in benchmark_indicators:
            pattern_matches = list(self.project_root.glob(f"**/*{indicator}*.py"))
            if pattern_matches:
                return True
        
        return False


class AutonomousImprovementEngine:
    """Autonomous code improvement and optimization suggestions."""
    
    def __init__(self, project_root: Path, db: HistoricalQualityDatabase):
        self.project_root = Path(project_root)
        self.db = db
        
    async def generate_autonomous_improvements(
        self,
        metrics: List[ProgressiveMetric]
    ) -> List[str]:
        """Generate autonomous improvement suggestions."""
        improvements = []
        
        # Analyze patterns and generate improvements
        code_improvements = await self._generate_code_improvements(metrics)
        security_improvements = await self._generate_security_improvements(metrics)
        performance_improvements = await self._generate_performance_improvements(metrics)
        architectural_improvements = await self._generate_architectural_improvements(metrics)
        
        improvements.extend(code_improvements)
        improvements.extend(security_improvements)
        improvements.extend(performance_improvements)
        improvements.extend(architectural_improvements)
        
        return improvements[:20]  # Top 20 improvements
    
    async def _generate_code_improvements(self, metrics: List[ProgressiveMetric]) -> List[str]:
        """Generate code quality improvements."""
        improvements = []
        
        code_metrics = [m for m in metrics if m.category == 'code_quality']
        
        for metric in code_metrics:
            if metric.score < 0.7:
                if 'complexity' in metric.name:
                    improvements.append("🔧 Auto-refactor: Extract complex functions into smaller, focused methods")
                elif 'duplication' in metric.name:
                    improvements.append("🔧 Auto-refactor: Create utility functions for repeated code patterns")
                elif 'documentation' in metric.name:
                    improvements.append("🤖 AI-Generate: Auto-generate missing docstrings using code analysis")
        
        return improvements
    
    async def _generate_security_improvements(self, metrics: List[ProgressiveMetric]) -> List[str]:
        """Generate security improvements."""
        improvements = []
        
        security_metrics = [m for m in metrics if m.category == 'security']
        
        for metric in security_metrics:
            if metric.score < 0.9:
                if 'secrets' in metric.name:
                    improvements.append("🔐 Auto-secure: Replace hardcoded secrets with environment variable loading")
                elif 'injection' in metric.name:
                    improvements.append("🛡️ Auto-secure: Implement input validation and sanitization decorators")
                elif 'permissions' in metric.name:
                    improvements.append("⚡ Auto-fix: Correct file permissions to secure defaults")
        
        return improvements
    
    async def _generate_performance_improvements(self, metrics: List[ProgressiveMetric]) -> List[str]:
        """Generate performance improvements."""
        improvements = []
        
        performance_metrics = [m for m in metrics if m.category == 'performance']
        
        for metric in performance_metrics:
            if metric.score < 0.7:
                if 'complexity' in metric.name:
                    improvements.append("⚡ Auto-optimize: Replace O(n²) algorithms with more efficient alternatives")
                elif 'import' in metric.name:
                    improvements.append("📦 Auto-optimize: Implement lazy imports for heavyweight dependencies")
                elif 'antipattern' in metric.name:
                    improvements.append("🚀 Auto-optimize: Replace performance anti-patterns with optimized implementations")
        
        return improvements
    
    async def _generate_architectural_improvements(self, metrics: List[ProgressiveMetric]) -> List[str]:
        """Generate architectural improvements."""
        improvements = []
        
        # Analyze overall project structure
        low_scoring_metrics = [m for m in metrics if m.score < 0.6]
        
        if len(low_scoring_metrics) > len(metrics) * 0.3:
            improvements.append("🏗️ Architecture: Consider implementing comprehensive error handling framework")
            improvements.append("📊 Architecture: Add centralized logging and monitoring system")
            improvements.append("🔄 Architecture: Implement automated testing and CI/CD pipeline")
        
        return improvements


class ProgressiveQualityGateSystem:
    """Advanced progressive quality gate system with learning capabilities."""
    
    def __init__(self, project_root: str = "/root/repo", learning_mode: LearningMode = LearningMode.ADAPTIVE):
        self.project_root = Path(project_root)
        self.learning_mode = learning_mode
        
        # Initialize components
        self.db = HistoricalQualityDatabase(self.project_root / ".quality_db")
        self.threshold_manager = AdaptiveThresholdManager(self.db)
        self.enhancement_validator = ProgressiveEnhancementValidator(self.project_root)
        self.improvement_engine = AutonomousImprovementEngine(self.project_root, self.db)
        
        # Initialize analyzers (reuse existing ones)
        from .autonomous_quality_gates import CodeQualityAnalyzer, SecurityAnalyzer, PerformanceAnalyzer
        self.code_analyzer = CodeQualityAnalyzer(self.project_root)
        self.security_analyzer = SecurityAnalyzer(self.project_root)
        self.performance_analyzer = PerformanceAnalyzer(self.project_root)
    
    async def run_progressive_quality_gates(
        self,
        target_generation: QualityEvolutionStage = QualityEvolutionStage.GENERATION_3_OPTIMIZED
    ) -> ProgressiveQualityReport:
        """Run comprehensive progressive quality gate analysis."""
        start_time = time.time()
        logger.info(f"Starting progressive quality analysis (target: {target_generation.value})...")
        
        # Determine project characteristics for adaptive thresholds
        project_characteristics = await self._analyze_project_characteristics()
        adaptive_thresholds = self.threshold_manager.get_adaptive_thresholds(project_characteristics)
        
        # Run quality analyses
        code_metrics = await self.code_analyzer.analyze_code_quality()
        security_metrics = await self.security_analyzer.analyze_security()
        performance_metrics = await self.performance_analyzer.analyze_performance()
        
        # Convert to progressive metrics with learning
        progressive_metrics = await self._convert_to_progressive_metrics(
            code_metrics + security_metrics + performance_metrics
        )
        
        # Validate generation compliance
        generation_compliance = await self.enhancement_validator.validate_generation_compliance()
        
        # Learn patterns from current analysis
        learned_patterns = await self._extract_quality_patterns(progressive_metrics)
        
        # Generate autonomous improvements
        autonomous_improvements = await self.improvement_engine.generate_autonomous_improvements(progressive_metrics)
        
        # Calculate scores and trends
        overall_score = self._calculate_progressive_score(progressive_metrics)
        baseline_comparison = self._calculate_baseline_comparison(progressive_metrics)
        trend_analysis = self._analyze_quality_trends(progressive_metrics)
        
        # Generate predictive insights
        predictive_insights = await self._generate_predictive_insights(progressive_metrics)
        
        # Create report
        report = ProgressiveQualityReport(
            timestamp=time.time(),
            generation_stage=self._determine_current_generation(generation_compliance),
            learning_mode=self.learning_mode,
            overall_score=overall_score,
            baseline_comparison=baseline_comparison,
            trend_analysis=trend_analysis,
            progressive_metrics=progressive_metrics,
            learned_patterns=learned_patterns,
            autonomous_improvements=autonomous_improvements,
            generation_readiness=generation_compliance,
            quality_trajectory=self._build_quality_trajectory(),
            predictive_insights=predictive_insights,
            execution_time=time.time() - start_time
        )
        
        # Store in database for learning
        self.db.add_quality_record(report)
        
        logger.info(f"Progressive quality analysis completed. Score: {overall_score:.2f}")
        return report
    
    async def _analyze_project_characteristics(self) -> Dict[str, Any]:
        """Analyze project characteristics for adaptive thresholds."""
        characteristics = {}
        
        # Count files and determine project size
        py_files = list(self.project_root.rglob("*.py"))
        characteristics['total_files'] = len(py_files)
        characteristics['total_lines'] = sum(len(open(f, 'r', encoding='utf-8', errors='ignore').readlines()) for f in py_files)
        
        # Determine project type based on structure
        if (self.project_root / "requirements.txt").exists() or (self.project_root / "pyproject.toml").exists():
            characteristics['type'] = 'library'
        if (self.project_root / "app.py").exists() or (self.project_root / "main.py").exists():
            characteristics['type'] = 'application'
        if any("benchmark" in str(f) for f in py_files):
            characteristics['type'] = 'research'
        
        # Analyze complexity
        total_complexity = 0
        function_count = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple complexity estimation
                    total_complexity += content.count('if ') + content.count('for ') + content.count('while ')
                    function_count += content.count('def ')
            except Exception:
                continue
        
        characteristics['avg_complexity'] = total_complexity / max(1, function_count)
        characteristics['function_count'] = function_count
        
        return characteristics
    
    async def _convert_to_progressive_metrics(self, basic_metrics: List) -> List[ProgressiveMetric]:
        """Convert basic metrics to progressive metrics with learning."""
        progressive_metrics = []
        baseline_scores = self.db.get_baseline_scores()
        
        for metric in basic_metrics:
            metric_name = metric.name
            current_score = metric.score
            baseline_score = baseline_scores.get(metric_name, current_score)
            
            # Predict trend
            predicted_score, trend = self.db.predict_quality_trend(metric_name)
            
            # Determine generation compliance
            generation_compliance = {
                QualityEvolutionStage.GENERATION_1_SIMPLE: current_score >= 0.5,
                QualityEvolutionStage.GENERATION_2_ROBUST: current_score >= 0.7,
                QualityEvolutionStage.GENERATION_3_OPTIMIZED: current_score >= 0.8
            }
            
            # Generate recommendations
            recommendations = []
            if hasattr(metric, 'remediation') and metric.remediation:
                recommendations.append(metric.remediation)
            
            progressive_metric = ProgressiveMetric(
                name=metric_name,
                category=metric.category,
                score=current_score,
                baseline_score=baseline_score,
                trend=trend,
                confidence=min(1.0, len(list(self.db.quality_history)) / 10.0),
                learning_data={
                    'historical_scores': [r['metrics'].get(metric_name, 0.0) for r in list(self.db.quality_history)[-10:]],
                    'improvement_rate': (current_score - baseline_score) / max(0.01, abs(baseline_score)),
                    'stability': self._calculate_metric_stability(metric_name)
                },
                generation_compliance=generation_compliance,
                recommendations=recommendations,
                predicted_score=predicted_score if predicted_score > 0 else None
            )
            
            progressive_metrics.append(progressive_metric)
        
        return progressive_metrics
    
    def _calculate_metric_stability(self, metric_name: str) -> float:
        """Calculate stability score for a metric based on historical variance."""
        scores = [r['metrics'].get(metric_name, 0.0) for r in list(self.db.quality_history)[-10:] if metric_name in r['metrics']]
        if len(scores) < 3:
            return 0.5  # Neutral stability for insufficient data
        
        variance = statistics.variance(scores)
        stability = max(0.0, 1.0 - (variance * 10))  # Invert variance to stability
        return min(1.0, stability)
    
    async def _extract_quality_patterns(self, metrics: List[ProgressiveMetric]) -> List[QualityPattern]:
        """Extract learned quality patterns from current analysis."""
        patterns = []
        
        # Pattern 1: Consistent improvement patterns
        improving_metrics = [m for m in metrics if m.trend == "improving" and m.confidence > 0.7]
        if len(improving_metrics) >= 3:
            pattern = QualityPattern(
                pattern_id=f"improvement_pattern_{hash(str(sorted([m.name for m in improving_metrics])))%10000}",
                category="improvement",
                pattern_type="improvement",
                confidence=statistics.mean([m.confidence for m in improving_metrics]),
                success_rate=len(improving_metrics) / len(metrics),
                occurrences=1,
                last_seen=time.time(),
                metadata={"improving_metrics": [m.name for m in improving_metrics]}
            )
            patterns.append(pattern)
        
        # Pattern 2: Category correlation patterns
        category_scores = defaultdict(list)
        for metric in metrics:
            category_scores[metric.category].append(metric.score)
        
        high_performing_categories = [cat for cat, scores in category_scores.items() if statistics.mean(scores) > 0.8]
        if len(high_performing_categories) >= 2:
            pattern = QualityPattern(
                pattern_id=f"category_excellence_{hash(str(sorted(high_performing_categories)))%10000}",
                category="excellence",
                pattern_type="stable",
                confidence=0.8,
                success_rate=len(high_performing_categories) / len(category_scores),
                occurrences=1,
                last_seen=time.time(),
                metadata={"excellent_categories": high_performing_categories}
            )
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_progressive_score(self, metrics: List[ProgressiveMetric]) -> float:
        """Calculate overall progressive quality score."""
        if not metrics:
            return 0.0
        
        # Weighted scoring based on category importance and trend
        category_weights = {
            'security': 0.4,
            'code_quality': 0.3,
            'performance': 0.2,
            'documentation': 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            base_weight = category_weights.get(metric.category, 0.1)
            
            # Adjust weight based on trend and confidence
            trend_multiplier = {
                'improving': 1.2,
                'stable': 1.0,
                'declining': 0.8
            }.get(metric.trend, 1.0)
            
            confidence_multiplier = metric.confidence
            
            final_weight = base_weight * trend_multiplier * confidence_multiplier
            weighted_sum += metric.score * final_weight
            total_weight += final_weight
        
        return weighted_sum / max(total_weight, 1.0)
    
    def _calculate_baseline_comparison(self, metrics: List[ProgressiveMetric]) -> float:
        """Calculate improvement compared to baseline."""
        if not metrics:
            return 0.0
        
        improvements = []
        for metric in metrics:
            if metric.baseline_score > 0:
                improvement = (metric.score - metric.baseline_score) / metric.baseline_score
                improvements.append(improvement)
        
        return statistics.mean(improvements) if improvements else 0.0
    
    def _analyze_quality_trends(self, metrics: List[ProgressiveMetric]) -> Dict[str, float]:
        """Analyze quality trends across categories."""
        category_trends = defaultdict(list)
        
        for metric in metrics:
            trend_score = {
                'improving': 1.0,
                'stable': 0.0,
                'declining': -1.0
            }.get(metric.trend, 0.0)
            category_trends[metric.category].append(trend_score)
        
        return {cat: statistics.mean(trends) for cat, trends in category_trends.items()}
    
    def _determine_current_generation(self, compliance: Dict[QualityEvolutionStage, bool]) -> QualityEvolutionStage:
        """Determine current generation stage based on compliance."""
        if compliance.get(QualityEvolutionStage.GENERATION_3_OPTIMIZED, False):
            return QualityEvolutionStage.GENERATION_3_OPTIMIZED
        elif compliance.get(QualityEvolutionStage.GENERATION_2_ROBUST, False):
            return QualityEvolutionStage.GENERATION_2_ROBUST
        elif compliance.get(QualityEvolutionStage.GENERATION_1_SIMPLE, False):
            return QualityEvolutionStage.GENERATION_1_SIMPLE
        else:
            return QualityEvolutionStage.GENERATION_1_SIMPLE
    
    def _build_quality_trajectory(self) -> List[Tuple[float, float]]:
        """Build quality trajectory from historical data."""
        trajectory = []
        for record in list(self.db.quality_history)[-20:]:  # Last 20 records
            trajectory.append((record['timestamp'], record['overall_score']))
        return trajectory
    
    async def _generate_predictive_insights(self, metrics: List[ProgressiveMetric]) -> Dict[str, Any]:
        """Generate predictive insights about quality trends."""
        insights = {
            'quality_forecast': {},
            'risk_areas': [],
            'improvement_opportunities': [],
            'confidence_level': 0.0
        }
        
        # Forecast quality for each metric
        for metric in metrics:
            if metric.predicted_score is not None:
                insights['quality_forecast'][metric.name] = {
                    'current': metric.score,
                    'predicted': metric.predicted_score,
                    'trend': metric.trend,
                    'confidence': metric.confidence
                }
        
        # Identify risk areas (declining trends with high impact)
        risk_areas = [m for m in metrics if m.trend == 'declining' and m.category in ['security', 'code_quality']]
        insights['risk_areas'] = [f"{m.category}.{m.name}" for m in risk_areas]
        
        # Identify improvement opportunities
        improvement_metrics = [m for m in metrics if m.score < 0.7 and m.trend != 'declining']
        insights['improvement_opportunities'] = [f"{m.category}.{m.name}" for m in improvement_metrics]
        
        # Calculate overall confidence
        if metrics:
            insights['confidence_level'] = statistics.mean([m.confidence for m in metrics])
        
        return insights
    
    async def save_progressive_report(self, report: ProgressiveQualityReport, output_path: Optional[str] = None):
        """Save progressive quality report to file."""
        if output_path is None:
            output_path = self.project_root / "progressive_quality_report.json"
        
        # Convert report to serializable format
        report_data = {
            'timestamp': report.timestamp,
            'generation_stage': report.generation_stage.value,
            'learning_mode': report.learning_mode.value,
            'overall_score': report.overall_score,
            'baseline_comparison': report.baseline_comparison,
            'trend_analysis': report.trend_analysis,
            'generation_readiness': {stage.value: ready for stage, ready in report.generation_readiness.items()},
            'quality_trajectory': report.quality_trajectory,
            'predictive_insights': report.predictive_insights,
            'execution_time': report.execution_time,
            'metrics': [
                {
                    'name': m.name,
                    'category': m.category,
                    'score': m.score,
                    'baseline_score': m.baseline_score,
                    'trend': m.trend,
                    'confidence': m.confidence,
                    'predicted_score': m.predicted_score,
                    'generation_compliance': {stage.value: ready for stage, ready in m.generation_compliance.items()},
                    'recommendations': m.recommendations
                }
                for m in report.progressive_metrics
            ],
            'learned_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'category': p.category,
                    'pattern_type': p.pattern_type,
                    'confidence': p.confidence,
                    'success_rate': p.success_rate,
                    'metadata': p.metadata
                }
                for p in report.learned_patterns
            ],
            'autonomous_improvements': report.autonomous_improvements
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Progressive quality report saved to {output_path}")


# Global progressive quality gate system
progressive_quality_system = ProgressiveQualityGateSystem()


async def run_progressive_quality_gates(
    project_root: str = "/root/repo",
    target_generation: QualityEvolutionStage = QualityEvolutionStage.GENERATION_3_OPTIMIZED,
    learning_mode: LearningMode = LearningMode.ADAPTIVE
) -> ProgressiveQualityReport:
    """Run progressive quality gates with learning capabilities."""
    system = ProgressiveQualityGateSystem(project_root, learning_mode)
    return await system.run_progressive_quality_gates(target_generation)


if __name__ == "__main__":
    async def main():
        report = await run_progressive_quality_gates()
        print(f"Progressive Quality Gates completed with score: {report.overall_score:.2f}")
        print(f"Current Generation: {report.generation_stage.value}")
        print(f"Autonomous Improvements Available: {len(report.autonomous_improvements)}")
    
    asyncio.run(main())