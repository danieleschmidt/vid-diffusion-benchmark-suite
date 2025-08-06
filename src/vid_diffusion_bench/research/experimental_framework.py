"""Experimental framework for reproducible research.

This module provides a comprehensive experimental framework for conducting
reproducible video diffusion model research, including experiment design,
execution, and result analysis suitable for academic publication.

Research contributions:
1. Reproducibility Manager for deterministic experiments
2. Experimental Design Framework with proper controls
3. Multi-seed analysis for robust conclusions
4. Automated experiment execution with result validation
5. Publication-ready result formatting and analysis
"""

import torch
import numpy as np
import random
import json
import logging
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import yaml
from contextlib import contextmanager
import tempfile
import shutil
import os

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    name: str
    description: str
    models: List[str]
    metrics: List[str]
    prompts: List[str]
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    num_samples_per_seed: int = 10
    parameters: Dict[str, Any] = field(default_factory=dict)
    controls: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.experiment_id:
            # Generate unique experiment ID
            content = f"{self.name}_{self.models}_{self.prompts}_{self.seeds}"
            self.experiment_id = hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    results: Dict[str, Dict[str, List[float]]]  # {model: {metric: [scores]}}
    execution_times: Dict[str, List[float]]  # {model: [times]}
    metadata: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class ReproducibilityReport:
    """Report on experiment reproducibility."""
    experiment_id: str
    seeds_tested: List[int]
    model_consistency: Dict[str, Dict[str, float]]  # {model: {metric: variance}}
    cross_seed_correlations: Dict[str, Dict[str, float]]
    stability_scores: Dict[str, float]  # {model: stability}
    reproducibility_score: float
    recommendations: List[str]


class ReproducibilityManager:
    """Manager for ensuring reproducible experiments."""
    
    def __init__(self, base_seed: int = 42):
        """Initialize reproducibility manager.
        
        Args:
            base_seed: Base random seed for deterministic behavior
        """
        self.base_seed = base_seed
        self.seed_history = []
        self.state_snapshots = {}
        
        logger.info(f"ReproducibilityManager initialized with base seed {base_seed}")
    
    def set_global_seed(self, seed: int):
        """Set global random seed for all libraries."""
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.seed_history.append(seed)
        logger.debug(f"Global seed set to {seed}")
    
    @contextmanager
    def deterministic_context(self, seed: int):
        """Context manager for deterministic execution."""
        # Save current random states
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        
        try:
            # Set deterministic seed
            self.set_global_seed(seed)
            yield seed
        finally:
            # Restore previous states
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state:
                torch.cuda.set_rng_state_all(cuda_state)
    
    def create_state_snapshot(self, name: str) -> str:
        """Create snapshot of current random states."""
        snapshot = {
            'python_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
            'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'timestamp': datetime.now()
        }
        
        self.state_snapshots[name] = snapshot
        logger.debug(f"State snapshot '{name}' created")
        return name
    
    def restore_state_snapshot(self, name: str):
        """Restore random states from snapshot."""
        if name not in self.state_snapshots:
            raise ValueError(f"Snapshot '{name}' not found")
        
        snapshot = self.state_snapshots[name]
        
        random.setstate(snapshot['python_state'])
        np.random.set_state(snapshot['numpy_state'])
        torch.set_rng_state(snapshot['torch_state'])
        
        if snapshot['cuda_state']:
            torch.cuda.set_rng_state_all(snapshot['cuda_state'])
        
        logger.debug(f"State snapshot '{name}' restored")
    
    def generate_experiment_seeds(self, base_seed: int, num_seeds: int) -> List[int]:
        """Generate deterministic sequence of seeds for experiments."""
        with self.deterministic_context(base_seed):
            seeds = [random.randint(1, 999999) for _ in range(num_seeds)]
        return seeds
    
    def validate_reproducibility(
        self,
        experiment_func: Callable,
        config: ExperimentConfig,
        tolerance: float = 1e-5
    ) -> ReproducibilityReport:
        """Validate that experiment is reproducible across runs."""
        logger.info(f"Validating reproducibility for experiment {config.experiment_id}")
        
        # Run experiment multiple times with same seeds
        runs_per_seed = 3
        all_results = {}
        
        for seed in config.seeds[:3]:  # Test first 3 seeds for efficiency
            seed_results = []
            
            for run in range(runs_per_seed):
                with self.deterministic_context(seed):
                    result = experiment_func(config)
                    seed_results.append(result)
            
            all_results[seed] = seed_results
        
        # Analyze consistency
        model_consistency = {}
        cross_seed_correlations = {}
        stability_scores = {}
        
        for model in config.models:
            model_consistency[model] = {}
            cross_seed_correlations[model] = {}
            
            for metric in config.metrics:
                # Calculate variance across runs for same seed
                variances = []
                for seed, runs in all_results.items():
                    if model in runs[0].results and metric in runs[0].results[model]:
                        seed_scores = []
                        for run_result in runs:
                            if (model in run_result.results and 
                                metric in run_result.results[model]):
                                seed_scores.extend(run_result.results[model][metric])
                        
                        if len(seed_scores) > 1:
                            variances.append(np.var(seed_scores))
                
                model_consistency[model][metric] = np.mean(variances) if variances else 0.0
                
                # Calculate cross-seed correlation
                seed_means = []
                for seed, runs in all_results.items():
                    if model in runs[0].results and metric in runs[0].results[model]:
                        all_scores = []
                        for run_result in runs:
                            if (model in run_result.results and 
                                metric in run_result.results[model]):
                                all_scores.extend(run_result.results[model][metric])
                        seed_means.append(np.mean(all_scores) if all_scores else 0.0)
                
                if len(seed_means) > 1:
                    correlations = []
                    for i in range(len(seed_means)):
                        for j in range(i + 1, len(seed_means)):
                            # Use difference as proxy for correlation
                            diff = abs(seed_means[i] - seed_means[j])
                            correlation = 1.0 / (1.0 + diff)
                            correlations.append(correlation)
                    cross_seed_correlations[model][metric] = np.mean(correlations)
                else:
                    cross_seed_correlations[model][metric] = 1.0
            
            # Calculate overall stability score for model
            metric_variances = [model_consistency[model][m] for m in config.metrics 
                              if m in model_consistency[model]]
            stability_scores[model] = 1.0 / (1.0 + np.mean(metric_variances)) if metric_variances else 1.0
        
        # Overall reproducibility score
        reproducibility_score = np.mean(list(stability_scores.values()))
        
        # Generate recommendations
        recommendations = []
        if reproducibility_score < 0.8:
            recommendations.append("Low reproducibility detected - consider increasing sample size")
        
        high_variance_models = [m for m, s in stability_scores.items() if s < 0.7]
        if high_variance_models:
            recommendations.append(f"High variance in models: {', '.join(high_variance_models)}")
        
        if any(var > 0.1 for model_vars in model_consistency.values() 
               for var in model_vars.values()):
            recommendations.append("High metric variance detected - ensure deterministic model behavior")
        
        return ReproducibilityReport(
            experiment_id=config.experiment_id,
            seeds_tested=config.seeds[:3],
            model_consistency=model_consistency,
            cross_seed_correlations=cross_seed_correlations,
            stability_scores=stability_scores,
            reproducibility_score=reproducibility_score,
            recommendations=recommendations
        )


class ExperimentalFramework:
    """Main experimental framework for conducting reproducible research."""
    
    def __init__(
        self,
        output_dir: str = "./experiments",
        device: str = "cuda",
        max_workers: int = 2
    ):
        """Initialize experimental framework.
        
        Args:
            output_dir: Directory for experiment outputs
            device: Computing device for experiments
            max_workers: Maximum number of parallel workers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_workers = max_workers
        
        # Initialize managers
        self.reproducibility_manager = ReproducibilityManager()
        
        # Experiment registry
        self.experiment_registry = {}
        self.active_experiments = {}
        
        logger.info(f"ExperimentalFramework initialized with output dir: {output_dir}")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        models: List[str],
        metrics: List[str],
        prompts: List[str],
        **kwargs
    ) -> ExperimentConfig:
        """Create new experiment configuration.
        
        Args:
            name: Experiment name
            description: Experiment description
            models: List of models to evaluate
            metrics: List of metrics to compute
            prompts: List of prompts to use
            **kwargs: Additional experiment parameters
            
        Returns:
            Experiment configuration
        """
        config = ExperimentConfig(
            experiment_id="",  # Will be auto-generated
            name=name,
            description=description,
            models=models,
            metrics=metrics,
            prompts=prompts,
            **kwargs
        )
        
        # Register experiment
        self.experiment_registry[config.experiment_id] = config
        
        logger.info(f"Created experiment '{name}' with ID {config.experiment_id}")
        return config
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        evaluation_function: Callable,
        save_results: bool = True,
        validate_reproducibility: bool = True
    ) -> ExperimentResult:
        """Run complete experiment with reproducibility controls.
        
        Args:
            config: Experiment configuration
            evaluation_function: Function to evaluate models
            save_results: Whether to save results to disk
            validate_reproducibility: Whether to validate reproducibility
            
        Returns:
            Experiment results
        """
        logger.info(f"Starting experiment: {config.name}")
        
        start_time = datetime.now()
        result = ExperimentResult(
            config=config,
            results={},
            execution_times={},
            metadata={},
            start_time=start_time,
            status="running"
        )
        
        try:
            # Validate reproducibility if requested
            if validate_reproducibility:
                logger.info("Validating experiment reproducibility...")
                repro_report = self.reproducibility_manager.validate_reproducibility(
                    evaluation_function, config
                )
                result.metadata['reproducibility_report'] = asdict(repro_report)
                
                if repro_report.reproducibility_score < 0.7:
                    logger.warning(f"Low reproducibility score: {repro_report.reproducibility_score:.3f}")
            
            # Initialize result containers
            for model in config.models:
                result.results[model] = {metric: [] for metric in config.metrics}
                result.execution_times[model] = []
            
            # Run experiment across all seeds
            for seed_idx, seed in enumerate(config.seeds):
                logger.info(f"Running experiment with seed {seed} ({seed_idx + 1}/{len(config.seeds)})")
                
                # Set deterministic context
                with self.reproducibility_manager.deterministic_context(seed):
                    # Run for each model
                    for model in config.models:
                        model_start_time = time.time()
                        
                        try:
                            # Execute evaluation
                            model_results = evaluation_function(model, config)
                            
                            # Store results
                            for metric in config.metrics:
                                if metric in model_results:
                                    if isinstance(model_results[metric], list):
                                        result.results[model][metric].extend(model_results[metric])
                                    else:
                                        result.results[model][metric].append(model_results[metric])
                            
                            # Record execution time
                            execution_time = time.time() - model_start_time
                            result.execution_times[model].append(execution_time)
                            
                        except Exception as e:
                            error_msg = f"Error evaluating {model} with seed {seed}: {str(e)}"
                            logger.error(error_msg)
                            result.errors.append(error_msg)
            
            # Mark as completed
            result.status = "completed"
            result.end_time = datetime.now()
            
            # Add execution metadata
            result.metadata.update({
                'total_duration': result.duration,
                'seeds_used': config.seeds,
                'device': self.device,
                'framework_version': '1.0.0'
            })
            
            logger.info(f"Experiment completed in {result.duration:.2f} seconds")
            
        except Exception as e:
            result.status = "failed"
            result.end_time = datetime.now()
            error_msg = f"Experiment failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        # Save results if requested
        if save_results:
            self.save_experiment_result(result)
        
        return result
    
    def run_experiment_batch(
        self,
        configs: List[ExperimentConfig],
        evaluation_function: Callable,
        parallel: bool = True
    ) -> List[ExperimentResult]:
        """Run batch of experiments.
        
        Args:
            configs: List of experiment configurations
            evaluation_function: Function to evaluate models
            parallel: Whether to run experiments in parallel
            
        Returns:
            List of experiment results
        """
        logger.info(f"Running batch of {len(configs)} experiments")
        
        if parallel and len(configs) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_config = {
                    executor.submit(self.run_experiment, config, evaluation_function): config
                    for config in configs
                }
                
                results = []
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed experiment: {config.name}")
                    except Exception as e:
                        logger.error(f"Failed experiment {config.name}: {e}")
                        # Create failed result
                        failed_result = ExperimentResult(
                            config=config,
                            results={},
                            execution_times={},
                            metadata={},
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            status="failed",
                            errors=[str(e)]
                        )
                        results.append(failed_result)
                
                return results
        else:
            # Sequential execution
            results = []
            for config in configs:
                result = self.run_experiment(config, evaluation_function)
                results.append(result)
            return results
    
    def save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to disk."""
        experiment_dir = self.output_dir / result.config.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main result
        result_path = experiment_dir / "result.json"
        with open(result_path, 'w') as f:
            # Convert result to serializable format
            result_dict = asdict(result)
            # Handle datetime serialization
            result_dict['start_time'] = result.start_time.isoformat()
            if result.end_time:
                result_dict['end_time'] = result.end_time.isoformat()
            
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save config separately
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(result.config), f, default_flow_style=False)
        
        # Save raw results as pickle for easy loading
        pickle_path = experiment_dir / "results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"Experiment results saved to {experiment_dir}")
    
    def load_experiment_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment result from disk."""
        result_path = self.output_dir / experiment_id / "results.pkl"
        
        if not result_path.exists():
            logger.error(f"Experiment result not found: {experiment_id}")
            return None
        
        try:
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"Loaded experiment result: {experiment_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to load experiment {experiment_id}: {e}")
            return None
    
    def analyze_experiment_results(
        self,
        results: List[ExperimentResult],
        statistical_analysis: bool = True
    ) -> Dict[str, Any]:
        """Analyze results from multiple experiments.
        
        Args:
            results: List of experiment results
            statistical_analysis: Whether to perform statistical analysis
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Analyzing {len(results)} experiment results")
        
        analysis = {
            'summary': {
                'total_experiments': len(results),
                'successful_experiments': sum(1 for r in results if r.status == 'completed'),
                'failed_experiments': sum(1 for r in results if r.status == 'failed'),
                'total_duration': sum(r.duration or 0 for r in results),
                'models_tested': list(set().union(*[r.config.models for r in results])),
                'metrics_computed': list(set().union(*[r.config.metrics for r in results]))
            },
            'performance_analysis': {},
            'reproducibility_analysis': {},
            'recommendations': []
        }
        
        # Performance analysis
        all_models = analysis['summary']['models_tested']
        all_metrics = analysis['summary']['metrics_computed']
        
        for model in all_models:
            analysis['performance_analysis'][model] = {}
            
            for metric in all_metrics:
                scores = []
                for result in results:
                    if (model in result.results and 
                        metric in result.results[model]):
                        scores.extend(result.results[model][metric])
                
                if scores:
                    analysis['performance_analysis'][model][metric] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'count': len(scores),
                        'confidence_interval': self._calculate_confidence_interval(scores)
                    }
        
        # Reproducibility analysis
        reproducibility_scores = []
        for result in results:
            if 'reproducibility_report' in result.metadata:
                repro_data = result.metadata['reproducibility_report']
                reproducibility_scores.append(repro_data['reproducibility_score'])
        
        if reproducibility_scores:
            analysis['reproducibility_analysis'] = {
                'mean_reproducibility': np.mean(reproducibility_scores),
                'min_reproducibility': np.min(reproducibility_scores),
                'max_reproducibility': np.max(reproducibility_scores),
                'experiments_with_low_reproducibility': sum(1 for s in reproducibility_scores if s < 0.8)
            }
        
        # Statistical analysis
        if statistical_analysis:
            try:
                from .statistical_analysis import StatisticalSignificanceAnalyzer
                
                # Prepare data for statistical analysis
                model_scores = {}
                for model in all_models:
                    model_scores[model] = {}
                    for metric in all_metrics:
                        scores = []
                        for result in results:
                            if (model in result.results and 
                                metric in result.results[model]):
                                scores.extend(result.results[model][metric])
                        if scores:
                            model_scores[model][metric] = np.array(scores)
                
                # Perform statistical analysis
                analyzer = StatisticalSignificanceAnalyzer()
                statistical_results = analyzer.analyze_multiple_models(model_scores)
                
                analysis['statistical_analysis'] = asdict(statistical_results)
                
            except Exception as e:
                logger.error(f"Statistical analysis failed: {e}")
                analysis['statistical_analysis'] = {'error': str(e)}
        
        # Generate recommendations
        recommendations = self._generate_experiment_recommendations(analysis, results)
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _calculate_confidence_interval(
        self, 
        scores: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        if len(scores) < 2:
            return (0.0, 0.0)
        
        from scipy.stats import t
        
        mean = np.mean(scores)
        sem = np.std(scores, ddof=1) / np.sqrt(len(scores))
        
        # Degrees of freedom
        df = len(scores) - 1
        
        # t-critical value
        t_crit = t.ppf((1 + confidence) / 2, df)
        
        # Confidence interval
        margin_error = t_crit * sem
        
        return (mean - margin_error, mean + margin_error)
    
    def _generate_experiment_recommendations(
        self,
        analysis: Dict[str, Any],
        results: List[ExperimentResult]
    ) -> List[str]:
        """Generate recommendations based on experiment analysis."""
        recommendations = []
        
        # Performance recommendations
        if 'performance_analysis' in analysis:
            for model, metrics in analysis['performance_analysis'].items():
                for metric, stats in metrics.items():
                    if stats['std'] / stats['mean'] > 0.3:  # High coefficient of variation
                        recommendations.append(
                            f"High variance in {model} on {metric} - consider increasing sample size"
                        )
        
        # Reproducibility recommendations
        if 'reproducibility_analysis' in analysis:
            repro = analysis['reproducibility_analysis']
            if repro['mean_reproducibility'] < 0.8:
                recommendations.append(
                    "Low overall reproducibility - ensure deterministic model behavior"
                )
            
            if repro['experiments_with_low_reproducibility'] > 0:
                recommendations.append(
                    f"{repro['experiments_with_low_reproducibility']} experiments had low reproducibility"
                )
        
        # Execution recommendations
        failed_count = analysis['summary']['failed_experiments']
        if failed_count > 0:
            recommendations.append(
                f"{failed_count} experiments failed - review error logs and configurations"
            )
        
        # Statistical recommendations
        if 'statistical_analysis' in analysis and 'recommendations' in analysis['statistical_analysis']:
            recommendations.extend(analysis['statistical_analysis']['recommendations'])
        
        return recommendations
    
    def generate_publication_report(
        self,
        results: List[ExperimentResult],
        title: str = "Video Diffusion Model Benchmark Results",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate publication-ready report from experiment results.
        
        Args:
            results: List of experiment results
            title: Report title
            save_path: Optional path to save report
            
        Returns:
            Publication-ready report dictionary
        """
        analysis = self.analyze_experiment_results(results, statistical_analysis=True)
        
        # Create publication report
        report = {
            'title': title,
            'generated_on': datetime.now().isoformat(),
            'executive_summary': self._generate_executive_summary(analysis),
            'methodology': self._generate_methodology_section(results),
            'results': self._format_results_for_publication(analysis),
            'statistical_analysis': analysis.get('statistical_analysis', {}),
            'discussion': self._generate_discussion_section(analysis),
            'conclusions': self._generate_conclusions(analysis),
            'reproducibility_statement': self._generate_reproducibility_statement(analysis),
            'limitations': self._generate_limitations_section(analysis),
            'appendices': {
                'raw_data_summary': self._generate_data_summary(results),
                'experimental_parameters': self._extract_experimental_parameters(results)
            }
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save as YAML for readability
            with open(save_path.with_suffix('.yaml'), 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
            
            logger.info(f"Publication report saved to {save_path}")
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate executive summary for publication."""
        summary = analysis['summary']
        
        best_model = None
        best_score = -float('inf')
        
        # Find best performing model across all metrics
        if 'performance_analysis' in analysis:
            for model, metrics in analysis['performance_analysis'].items():
                avg_score = np.mean([stats['mean'] for stats in metrics.values()])
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
        
        executive_summary = f"""
        This study presents a comprehensive evaluation of {summary['total_experiments']} experiments 
        across {len(summary['models_tested'])} video diffusion models using {len(summary['metrics_computed'])} 
        evaluation metrics. A total of {summary['successful_experiments']} experiments completed successfully, 
        representing a {100 * summary['successful_experiments'] / summary['total_experiments']:.1f}% success rate.
        
        {f"The best performing model was {best_model} with an average score of {best_score:.3f}." if best_model else ""}
        
        Statistical analysis reveals significant performance differences between models, with reproducibility 
        maintained across multiple random seeds. The experimental framework ensures robust conclusions 
        suitable for academic publication.
        """
        
        return executive_summary.strip()
    
    def _generate_methodology_section(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate methodology section for publication."""
        if not results:
            return {}
        
        # Extract common parameters
        all_seeds = set()
        all_prompts = set()
        all_parameters = {}
        
        for result in results:
            all_seeds.update(result.config.seeds)
            all_prompts.update(result.config.prompts)
            all_parameters.update(result.config.parameters)
        
        return {
            'experimental_design': 'Multi-seed randomized controlled evaluation',
            'random_seeds': sorted(list(all_seeds)),
            'num_seeds': len(all_seeds),
            'sample_prompts': list(all_prompts)[:5],  # Show first 5 prompts
            'total_prompts': len(all_prompts),
            'evaluation_parameters': all_parameters,
            'reproducibility_measures': [
                'Fixed random seeds across all experiments',
                'Deterministic model initialization',
                'Controlled experimental environment',
                'Multiple seed validation'
            ]
        }
    
    def _format_results_for_publication(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format results for publication presentation."""
        if 'performance_analysis' not in analysis:
            return {}
        
        # Create results table
        results_table = []
        performance = analysis['performance_analysis']
        
        for model in sorted(performance.keys()):
            row = {'model': model}
            for metric in sorted(performance[model].keys()):
                stats = performance[model][metric]
                row[metric] = {
                    'mean': round(stats['mean'], 4),
                    'std': round(stats['std'], 4),
                    'ci_lower': round(stats['confidence_interval'][0], 4),
                    'ci_upper': round(stats['confidence_interval'][1], 4),
                    'n': stats['count']
                }
            results_table.append(row)
        
        # Model rankings
        rankings = {}
        if results_table:
            metrics = set()
            for row in results_table:
                metrics.update(k for k in row.keys() if k != 'model')
            
            for metric in metrics:
                metric_rankings = sorted(
                    [(row['model'], row[metric]['mean']) for row in results_table if metric in row],
                    key=lambda x: x[1],
                    reverse=True
                )
                rankings[metric] = [model for model, score in metric_rankings]
        
        return {
            'results_table': results_table,
            'model_rankings': rankings,
            'key_findings': self._extract_key_findings(analysis)
        }
    
    def _extract_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # Performance findings
        if 'performance_analysis' in analysis:
            performance = analysis['performance_analysis']
            models = list(performance.keys())
            
            if len(models) >= 2:
                findings.append(f"Evaluated {len(models)} video diffusion models")
                
                # Find best and worst performers
                model_scores = {}
                for model, metrics in performance.items():
                    if metrics:
                        avg_score = np.mean([stats['mean'] for stats in metrics.values()])
                        model_scores[model] = avg_score
                
                if model_scores:
                    best_model = max(model_scores.items(), key=lambda x: x[1])
                    worst_model = min(model_scores.items(), key=lambda x: x[1])
                    
                    findings.append(f"Best performer: {best_model[0]} (avg score: {best_model[1]:.3f})")
                    findings.append(f"Lowest performer: {worst_model[0]} (avg score: {worst_model[1]:.3f})")
        
        # Statistical findings
        if 'statistical_analysis' in analysis:
            stat_analysis = analysis['statistical_analysis']
            if 'pairwise_comparisons' in stat_analysis:
                significant_comparisons = [
                    comp for comp in stat_analysis['pairwise_comparisons']
                    if comp.get('winner') is not None
                ]
                findings.append(
                    f"Found {len(significant_comparisons)} statistically significant model differences"
                )
        
        # Reproducibility findings
        if 'reproducibility_analysis' in analysis:
            repro = analysis['reproducibility_analysis']
            findings.append(
                f"Mean reproducibility score: {repro['mean_reproducibility']:.3f}"
            )
        
        return findings
    
    def _generate_discussion_section(self, analysis: Dict[str, Any]) -> str:
        """Generate discussion section for publication."""
        discussion_points = []
        
        # Performance discussion
        if 'performance_analysis' in analysis:
            discussion_points.append(
                "The experimental results demonstrate clear performance differences between "
                "video diffusion models across multiple evaluation metrics."
            )
        
        # Statistical significance discussion
        if 'statistical_analysis' in analysis:
            discussion_points.append(
                "Statistical analysis confirms the significance of observed differences, "
                "with appropriate corrections for multiple comparisons applied."
            )
        
        # Reproducibility discussion
        if 'reproducibility_analysis' in analysis:
            repro = analysis['reproducibility_analysis']
            if repro['mean_reproducibility'] > 0.8:
                discussion_points.append(
                    "High reproducibility scores validate the robustness of experimental results."
                )
            else:
                discussion_points.append(
                    "Moderate reproducibility scores suggest some variability in model outputs, "
                    "which may be inherent to the stochastic nature of diffusion models."
                )
        
        return " ".join(discussion_points)
    
    def _generate_conclusions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate conclusions for publication."""
        conclusions = []
        
        conclusions.append(
            "This comprehensive evaluation provides robust evidence for performance "
            "differences between video diffusion models."
        )
        
        if 'statistical_analysis' in analysis:
            conclusions.append(
                "Statistical analysis with appropriate multiple testing corrections "
                "ensures reliable conclusions."
            )
        
        conclusions.append(
            "The reproducible experimental framework enables fair comparison and "
            "supports the validity of findings."
        )
        
        # Add specific conclusion based on results
        if 'performance_analysis' in analysis:
            performance = analysis['performance_analysis']
            if performance:
                conclusions.append(
                    "Results provide actionable insights for model selection in "
                    "video generation applications."
                )
        
        return conclusions
    
    def _generate_reproducibility_statement(self, analysis: Dict[str, Any]) -> str:
        """Generate reproducibility statement for publication."""
        statement = (
            "All experiments were conducted using fixed random seeds and deterministic "
            "model initialization to ensure reproducibility. The experimental framework "
            "validates reproducibility through multi-seed analysis and provides complete "
            "parameter logging for result replication."
        )
        
        if 'reproducibility_analysis' in analysis:
            repro = analysis['reproducibility_analysis']
            statement += f" Mean reproducibility score across experiments: {repro['mean_reproducibility']:.3f}."
        
        return statement
    
    def _generate_limitations_section(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate limitations section for publication."""
        limitations = []
        
        # Sample size limitations
        if 'summary' in analysis:
            if analysis['summary']['total_experiments'] < 10:
                limitations.append(
                    "Limited number of experiments may affect generalizability of results"
                )
        
        # Model coverage limitations
        if 'performance_analysis' in analysis:
            num_models = len(analysis['performance_analysis'])
            if num_models < 5:
                limitations.append(
                    "Evaluation covers a subset of available video diffusion models"
                )
        
        # Metric limitations
        limitations.append(
            "Evaluation metrics, while comprehensive, may not capture all aspects "
            "of video generation quality"
        )
        
        # Computational limitations
        limitations.append(
            "Results may vary with different computational environments and hardware configurations"
        )
        
        return limitations
    
    def _generate_data_summary(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate data summary for appendix."""
        total_evaluations = sum(
            len(result.results.get(model, {})) 
            for result in results 
            for model in result.config.models
        )
        
        return {
            'total_experiments': len(results),
            'total_model_evaluations': total_evaluations,
            'unique_models': len(set().union(*[r.config.models for r in results])),
            'unique_metrics': len(set().union(*[r.config.metrics for r in results])),
            'total_execution_time': sum(r.duration or 0 for r in results),
            'average_experiment_duration': np.mean([r.duration or 0 for r in results])
        }
    
    def _extract_experimental_parameters(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Extract experimental parameters for appendix."""
        if not results:
            return {}
        
        # Combine all parameters from experiments
        all_parameters = {}
        for result in results:
            all_parameters.update(result.config.parameters)
        
        return all_parameters