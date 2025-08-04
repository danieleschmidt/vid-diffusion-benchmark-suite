"""Database service layer for benchmark operations."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import desc, asc, func, and_, or_
from sqlalchemy.orm import Session, joinedload

from .connection import db_session_scope, get_db_session
from .models import Model, EvaluationRun, BenchmarkResult, QualityMetrics, PerformanceMetrics
from ..benchmark import BenchmarkResult as BenchmarkResultData

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model-related database operations."""
    
    @staticmethod
    def create_model(
        name: str,
        display_name: str,
        model_type: str,
        requirements: Dict[str, Any],
        description: Optional[str] = None,
        **kwargs
    ) -> Model:
        """Create a new model entry."""
        with db_session_scope() as session:
            model = Model(
                name=name,
                display_name=display_name,
                description=description,
                model_type=model_type,
                vram_gb=requirements.get("vram_gb"),
                precision=requirements.get("precision"),
                model_size_gb=requirements.get("model_size_gb"),
                max_frames=requirements.get("max_frames"),
                supported_resolutions=requirements.get("supported_resolutions"),
                dependencies=requirements.get("dependencies"),
                **kwargs
            )
            session.add(model)
            session.flush()  # Get the ID
            session.refresh(model)
            
            logger.info(f"Created model: {name} (ID: {model.id})")
            return model
            
    @staticmethod
    def get_model_by_name(name: str) -> Optional[Model]:
        """Get model by name."""
        with db_session_scope() as session:
            return session.query(Model).filter(Model.name == name).first()
            
    @staticmethod
    def get_model_by_id(model_id: int) -> Optional[Model]:
        """Get model by ID."""
        with db_session_scope() as session:
            return session.query(Model).filter(Model.id == model_id).first()
            
    @staticmethod
    def list_models(
        active_only: bool = True,
        model_type: Optional[str] = None
    ) -> List[Model]:
        """List all models with optional filters."""
        with db_session_scope() as session:
            query = session.query(Model)
            
            if active_only:
                query = query.filter(Model.is_active == True)
                
            if model_type:
                query = query.filter(Model.model_type == model_type)
                
            return query.order_by(Model.name).all()
            
    @staticmethod
    def update_model(model_id: int, **updates) -> Optional[Model]:
        """Update model attributes."""
        with db_session_scope() as session:
            model = session.query(Model).filter(Model.id == model_id).first()
            if model:
                for key, value in updates.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                model.updated_at = datetime.utcnow()
                session.flush()
                session.refresh(model)
                logger.info(f"Updated model {model.name} (ID: {model_id})")
            return model
            
    @staticmethod
    def delete_model(model_id: int) -> bool:
        """Soft delete a model (set inactive)."""
        with db_session_scope() as session:
            model = session.query(Model).filter(Model.id == model_id).first()
            if model:
                model.is_active = False
                model.updated_at = datetime.utcnow()
                logger.info(f"Deactivated model {model.name} (ID: {model_id})")
                return True
            return False


class EvaluationService:
    """Service for evaluation run operations."""
    
    @staticmethod
    def create_evaluation_run(
        model_id: int,
        num_prompts: int,
        num_frames: int,
        fps: int,
        resolution: str,
        config: Dict[str, Any],
        prompts_used: List[str],
        system_specs: Optional[Dict[str, Any]] = None
    ) -> EvaluationRun:
        """Create a new evaluation run."""
        with db_session_scope() as session:
            run = EvaluationRun(
                model_id=model_id,
                num_prompts=num_prompts,
                num_frames=num_frames,
                fps=fps,
                resolution=resolution,
                config=config,
                prompts_used=prompts_used,
                system_specs=system_specs or {},
                status='pending'
            )
            session.add(run)
            session.flush()
            session.refresh(run)
            
            logger.info(f"Created evaluation run {run.id} for model ID {model_id}")
            return run
            
    @staticmethod
    def update_run_status(
        run_id: int, 
        status: str, 
        success_rate: Optional[float] = None,
        errors: Optional[List[Dict]] = None
    ) -> Optional[EvaluationRun]:
        """Update evaluation run status."""
        with db_session_scope() as session:
            run = session.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
            if run:
                run.status = status
                if success_rate is not None:
                    run.success_rate = success_rate
                if errors:
                    run.errors = errors
                    
                if status == 'completed':
                    run.completed_at = datetime.utcnow()
                    if run.started_at:
                        delta = run.completed_at - run.started_at
                        run.duration_seconds = delta.total_seconds()
                        
                session.flush()
                session.refresh(run)
                logger.info(f"Updated run {run_id} status to {status}")
            return run
            
    @staticmethod
    def get_evaluation_run(run_id: int) -> Optional[EvaluationRun]:
        """Get evaluation run by ID with all relationships."""
        with db_session_scope() as session:
            return session.query(EvaluationRun).options(
                joinedload(EvaluationRun.model),
                joinedload(EvaluationRun.benchmark_results),
                joinedload(EvaluationRun.quality_metrics),
                joinedload(EvaluationRun.performance_metrics)
            ).filter(EvaluationRun.id == run_id).first()
            
    @staticmethod
    def list_evaluation_runs(
        model_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[EvaluationRun]:
        """List evaluation runs with optional filters."""
        with db_session_scope() as session:
            query = session.query(EvaluationRun).options(joinedload(EvaluationRun.model))
            
            if model_id:
                query = query.filter(EvaluationRun.model_id == model_id)
            if status:
                query = query.filter(EvaluationRun.status == status)
                
            return query.order_by(desc(EvaluationRun.started_at)).limit(limit).all()


class BenchmarkService:
    """Service for benchmark result operations."""
    
    @staticmethod
    def save_benchmark_result(
        benchmark_result: BenchmarkResultData
    ) -> EvaluationRun:
        """Save complete benchmark result to database."""
        with db_session_scope() as session:
            # Get or create model
            model = session.query(Model).filter(Model.name == benchmark_result.model_name).first()
            if not model:
                # Create model if it doesn't exist
                model = Model(
                    name=benchmark_result.model_name,
                    display_name=benchmark_result.model_name.replace('-', ' ').title(),
                    model_type='diffusion',
                    is_active=True
                )
                session.add(model)
                session.flush()
                
            # Create evaluation run
            resolution = f"{512}x{512}"  # Default, could be extracted from results
            evaluation_run = EvaluationRun(
                model_id=model.id,
                num_prompts=len(benchmark_result.prompts),
                num_frames=16,  # Default, could be extracted
                fps=8,  # Default
                resolution=resolution,
                config={},
                prompts_used=benchmark_result.prompts,
                status='completed',
                success_rate=benchmark_result.success_rate,
                started_at=datetime.fromisoformat(benchmark_result.timestamp),
                completed_at=datetime.utcnow(),
                errors=benchmark_result.errors
            )
            session.add(evaluation_run)
            session.flush()
            
            # Save individual benchmark results
            for idx, (prompt, result_data) in enumerate(benchmark_result.results.items()):
                bench_result = BenchmarkResult(
                    evaluation_run_id=evaluation_run.id,
                    prompt_index=idx,
                    prompt_text=result_data["prompt"],
                    video_shape=str(result_data.get("video_shape", "unknown")),
                    generation_time_ms=result_data.get("generation_time", 0) * 1000,
                    memory_usage_mb=result_data.get("memory_usage", {}).get("peak_gpu_mb", 0),
                    status=result_data["status"],
                    error_message=result_data.get("error"),
                    error_type=None
                )
                session.add(bench_result)
                
            # Save quality metrics
            if benchmark_result.metrics:
                quality_metrics = QualityMetrics(
                    evaluation_run_id=evaluation_run.id,
                    fvd_score=benchmark_result.metrics.get("fvd"),
                    inception_score_mean=benchmark_result.metrics.get("inception_score"),
                    clip_similarity=benchmark_result.metrics.get("clip_similarity"),
                    temporal_consistency=benchmark_result.metrics.get("temporal_consistency"),
                    overall_quality_score=benchmark_result.metrics.get("overall_score"),
                    reference_dataset="ucf101"
                )
                session.add(quality_metrics)
                
            # Save performance metrics
            if benchmark_result.performance:
                perf_metrics = PerformanceMetrics(
                    evaluation_run_id=evaluation_run.id,
                    avg_latency_ms=benchmark_result.performance.get("avg_latency_ms"),
                    throughput_fps=benchmark_result.performance.get("throughput_fps"),
                    peak_vram_gb=benchmark_result.performance.get("peak_vram_gb"),
                    avg_power_watts=benchmark_result.performance.get("avg_power_watts"),
                    efficiency_score=benchmark_result.performance.get("efficiency_score")
                )
                session.add(perf_metrics)
                
            session.flush()
            session.refresh(evaluation_run)
            
            logger.info(f"Saved benchmark result for {benchmark_result.model_name} (Run ID: {evaluation_run.id})")
            return evaluation_run


class MetricsService:
    """Service for metrics and analytics operations."""
    
    @staticmethod
    def get_leaderboard(
        metric: str = "overall_quality_score",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get model leaderboard sorted by specified metric."""
        with db_session_scope() as session:
            if metric in ["fvd_score", "inception_score_mean", "clip_similarity", "temporal_consistency", "overall_quality_score"]:
                # Quality metrics
                query = (
                    session.query(Model, QualityMetrics, EvaluationRun)
                    .join(EvaluationRun, Model.id == EvaluationRun.model_id)
                    .join(QualityMetrics, EvaluationRun.id == QualityMetrics.evaluation_run_id)
                    .filter(Model.is_active == True)
                    .filter(EvaluationRun.status == 'completed')
                )
                
                # Sort by metric (lower is better for FVD, higher for others)
                if metric == "fvd_score":
                    query = query.order_by(asc(getattr(QualityMetrics, metric)))
                else:
                    query = query.order_by(desc(getattr(QualityMetrics, metric)))
                    
            elif metric in ["avg_latency_ms", "peak_vram_gb", "efficiency_score"]:
                # Performance metrics
                query = (
                    session.query(Model, PerformanceMetrics, EvaluationRun)
                    .join(EvaluationRun, Model.id == EvaluationRun.model_id)
                    .join(PerformanceMetrics, EvaluationRun.id == PerformanceMetrics.evaluation_run_id)
                    .filter(Model.is_active == True)
                    .filter(EvaluationRun.status == 'completed')
                )
                
                # Sort by metric (lower is better for latency/memory, higher for efficiency)
                if metric in ["avg_latency_ms", "peak_vram_gb"]:
                    query = query.order_by(asc(getattr(PerformanceMetrics, metric)))
                else:
                    query = query.order_by(desc(getattr(PerformanceMetrics, metric)))
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            results = query.limit(limit).all()
            
            leaderboard = []
            for rank, (model, metrics_obj, run) in enumerate(results, 1):
                entry = {
                    "rank": rank,
                    "model_name": model.name,
                    "model_display_name": model.display_name,
                    "model_type": model.model_type,
                    "metric_value": getattr(metrics_obj, metric),
                    "evaluation_run_id": run.id,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None
                }
                
                # Add additional context
                if hasattr(metrics_obj, 'overall_quality_score'):
                    entry["quality_score"] = metrics_obj.overall_quality_score
                if hasattr(metrics_obj, 'efficiency_score'):  
                    entry["efficiency_score"] = metrics_obj.efficiency_score
                    
                leaderboard.append(entry)
                
            return leaderboard
            
    @staticmethod
    def get_model_statistics() -> Dict[str, Any]:
        """Get overall model statistics."""
        with db_session_scope() as session:
            stats = {
                "total_models": session.query(Model).filter(Model.is_active == True).count(),
                "total_evaluation_runs": session.query(EvaluationRun).count(),
                "completed_runs": session.query(EvaluationRun).filter(EvaluationRun.status == 'completed').count(),
                "failed_runs": session.query(EvaluationRun).filter(EvaluationRun.status == 'failed').count(),
                "models_by_type": {}
            }
            
            # Get model type distribution
            type_counts = (
                session.query(Model.model_type, func.count(Model.id))
                .filter(Model.is_active == True)
                .group_by(Model.model_type)
                .all()
            )
            
            for model_type, count in type_counts:
                stats["models_by_type"][model_type] = count
                
            # Get average metrics
            avg_quality = session.query(func.avg(QualityMetrics.overall_quality_score)).scalar()
            avg_efficiency = session.query(func.avg(PerformanceMetrics.efficiency_score)).scalar()
            
            stats["average_quality_score"] = float(avg_quality) if avg_quality else None
            stats["average_efficiency_score"] = float(avg_efficiency) if avg_efficiency else None
            
            return stats
            
    @staticmethod
    def get_model_comparison(model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models across all metrics."""
        with db_session_scope() as session:
            comparison = {"models": {}, "comparison_matrix": {}}
            
            for model_name in model_names:
                # Get latest evaluation for each model
                latest_run = (
                    session.query(EvaluationRun)
                    .join(Model)
                    .filter(Model.name == model_name)
                    .filter(EvaluationRun.status == 'completed')
                    .order_by(desc(EvaluationRun.completed_at))
                    .first()
                )
                
                if not latest_run:
                    continue
                    
                model_data = {
                    "model_name": model_name,
                    "evaluation_run_id": latest_run.id,
                    "completed_at": latest_run.completed_at.isoformat() if latest_run.completed_at else None,
                    "success_rate": latest_run.success_rate
                }
                
                # Add quality metrics
                quality = session.query(QualityMetrics).filter(
                    QualityMetrics.evaluation_run_id == latest_run.id
                ).first()
                
                if quality:
                    model_data["quality_metrics"] = quality.to_dict()
                    
                # Add performance metrics
                performance = session.query(PerformanceMetrics).filter(
                    PerformanceMetrics.evaluation_run_id == latest_run.id
                ).first()
                
                if performance:
                    model_data["performance_metrics"] = performance.to_dict()
                    
                comparison["models"][model_name] = model_data
                
            return comparison


# Convenience functions
def register_model_from_adapter(adapter_class, model_name: str) -> Model:
    """Register a model in database from its adapter class."""
    requirements = adapter_class(device="cpu").requirements
    
    return ModelService.create_model(
        name=model_name,
        display_name=model_name.replace('-', ' ').title(),
        model_type='diffusion',
        requirements=requirements,
        description=f"Video diffusion model: {model_name}"
    )


def initialize_models_from_registry():
    """Initialize database with models from the model registry."""
    from ..models.registry import list_models
    
    logger.info("Initializing database with registered models...")
    
    registered_models = list_models()
    for model_name in registered_models:
        existing = ModelService.get_model_by_name(model_name)
        if not existing:
            try:
                from ..models.registry import get_model
                adapter = get_model(model_name, device="cpu")
                requirements = adapter.requirements
                
                ModelService.create_model(
                    name=model_name,
                    display_name=model_name.replace('-', ' ').title(),
                    model_type='diffusion',
                    requirements=requirements,
                    description=f"Video diffusion model: {model_name}",
                    is_verified=True
                )
                logger.info(f"Registered model in database: {model_name}")
            except Exception as e:
                logger.error(f"Failed to register model {model_name}: {e}")
                
    logger.info("Model initialization complete")