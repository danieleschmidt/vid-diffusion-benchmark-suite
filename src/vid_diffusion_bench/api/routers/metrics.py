"""Metrics and analytics endpoints."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from ...database.services import MetricsService
from ..schemas import Leaderboard, LeaderboardEntry, ModelStatistics

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/leaderboard", response_model=Leaderboard)
async def get_leaderboard(
    metric: str = Query("overall_quality_score", description="Metric to rank by"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of entries")
):
    """Get model leaderboard ranked by specified metric."""
    try:
        valid_metrics = [
            "overall_quality_score", "fvd_score", "inception_score_mean", 
            "clip_similarity", "temporal_consistency", "avg_latency_ms", 
            "peak_vram_gb", "efficiency_score"
        ]
        
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid metric '{metric}'. Valid metrics: {valid_metrics}"
            )
        
        leaderboard_data = MetricsService.get_leaderboard(metric=metric, limit=limit)
        
        entries = [
            LeaderboardEntry(**entry) for entry in leaderboard_data
        ]
        
        return Leaderboard(
            metric=metric,
            entries=entries,
            total_entries=len(entries)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve leaderboard")


@router.get("/statistics", response_model=ModelStatistics)
async def get_model_statistics():
    """Get overall model and evaluation statistics."""
    try:
        stats = MetricsService.get_model_statistics()
        return ModelStatistics(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/compare")
async def compare_models(
    models: List[str] = Query(..., description="List of model names to compare")
):
    """Compare multiple models across all metrics."""
    try:
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="At least 2 models required for comparison")
        
        if len(models) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 models allowed for comparison")
        
        comparison = MetricsService.get_model_comparison(models)
        
        if not comparison["models"]:
            raise HTTPException(status_code=404, detail="No evaluation data found for specified models")
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare models")


@router.get("/quality/{run_id}")
async def get_quality_metrics(run_id: int):
    """Get quality metrics for a specific evaluation run."""
    try:
        from ...database.services import EvaluationService
        
        run = EvaluationService.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Evaluation run {run_id} not found")
        
        if not run.quality_metrics:
            raise HTTPException(status_code=404, detail="No quality metrics found for this run")
        
        return run.quality_metrics[0].to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quality metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality metrics")


@router.get("/performance/{run_id}")
async def get_performance_metrics(run_id: int):
    """Get performance metrics for a specific evaluation run."""
    try:
        from ...database.services import EvaluationService
        
        run = EvaluationService.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Evaluation run {run_id} not found")
        
        if not run.performance_metrics:
            raise HTTPException(status_code=404, detail="No performance metrics found for this run")
        
        return run.performance_metrics[0].to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@router.get("/trends/{model_name}")
async def get_model_trends(
    model_name: str,
    limit: int = Query(20, ge=1, le=100, description="Number of recent evaluations")
):
    """Get performance trends for a specific model over time."""
    try:
        from ...database.services import EvaluationService, ModelService
        
        # Verify model exists
        model = ModelService.get_model_by_name(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Get recent evaluation runs
        runs = EvaluationService.list_evaluation_runs(
            model_id=model.id,
            status="completed",
            limit=limit
        )
        
        trends = []
        for run in runs:
            trend_point = {
                "evaluation_run_id": run.id,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "success_rate": run.success_rate,
                "duration_seconds": run.duration_seconds
            }
            
            # Add quality metrics if available
            if run.quality_metrics:
                qm = run.quality_metrics[0]
                trend_point.update({
                    "fvd_score": qm.fvd_score,
                    "overall_quality_score": qm.overall_quality_score,
                    "clip_similarity": qm.clip_similarity
                })
            
            # Add performance metrics if available
            if run.performance_metrics:
                pm = run.performance_metrics[0]
                trend_point.update({
                    "avg_latency_ms": pm.avg_latency_ms,
                    "peak_vram_gb": pm.peak_vram_gb,
                    "efficiency_score": pm.efficiency_score
                })
            
            trends.append(trend_point)
        
        return {
            "model_name": model_name,
            "trends": trends,
            "total_points": len(trends)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model trends")


@router.get("/pareto-frontier")
async def get_pareto_frontier():
    """Get Pareto frontier analysis (quality vs efficiency trade-offs)."""
    try:
        # Get all completed evaluations with both quality and performance metrics
        from ...database.connection import db_session_scope
        from ...database.models import Model, EvaluationRun, QualityMetrics, PerformanceMetrics
        
        with db_session_scope() as session:
            results = (
                session.query(Model, QualityMetrics, PerformanceMetrics, EvaluationRun)
                .join(EvaluationRun, Model.id == EvaluationRun.model_id)
                .join(QualityMetrics, EvaluationRun.id == QualityMetrics.evaluation_run_id)
                .join(PerformanceMetrics, EvaluationRun.id == PerformanceMetrics.evaluation_run_id)
                .filter(Model.is_active == True)
                .filter(EvaluationRun.status == 'completed')
                .all()
            )
        
        if not results:
            return {"pareto_frontier": [], "all_points": []}
        
        points = []
        for model, quality, performance, run in results:
            if quality.overall_quality_score and performance.efficiency_score:
                points.append({
                    "model_name": model.name,
                    "model_display_name": model.display_name,
                    "quality_score": quality.overall_quality_score,
                    "efficiency_score": performance.efficiency_score,
                    "evaluation_run_id": run.id
                })
        
        # Compute Pareto frontier (simplified)
        points.sort(key=lambda x: x["quality_score"], reverse=True)
        
        pareto_frontier = []
        max_efficiency = -1
        
        for point in points:
            if point["efficiency_score"] > max_efficiency:
                pareto_frontier.append(point)
                max_efficiency = point["efficiency_score"]
        
        return {
            "pareto_frontier": pareto_frontier,
            "all_points": points,
            "total_models": len(points)
        }
        
    except Exception as e:
        logger.error(f"Failed to compute Pareto frontier: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute Pareto frontier")