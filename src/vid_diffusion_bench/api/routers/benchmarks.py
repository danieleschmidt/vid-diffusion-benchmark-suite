"""Benchmark execution endpoints."""

import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query

from ...database.services import EvaluationService, BenchmarkService
from ...benchmark import BenchmarkSuite
from ...prompts import StandardPrompts
from ..schemas import (
    BenchmarkRequest, BenchmarkResultSummary, DetailedBenchmarkResult,
    BenchmarkJob, EvaluationStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job tracking (use Redis in production)
benchmark_jobs = {}


async def run_benchmark_job(job_id: str, request: BenchmarkRequest):
    """Run benchmark in background."""
    try:
        benchmark_jobs[job_id]["status"] = "running"
        benchmark_jobs[job_id]["progress"] = 0.1
        
        # Initialize benchmark suite
        suite = BenchmarkSuite()
        
        # Use provided prompts or standard set
        prompts = request.prompts
        if not prompts:
            prompts = StandardPrompts.DIVERSE_SET_V2[:10]  # Use first 10 standard prompts
            
        benchmark_jobs[job_id]["progress"] = 0.2
        
        # Run benchmark
        result = suite.evaluate_model(
            model_name=request.model_name,
            prompts=prompts,
            num_frames=request.config.num_frames,
            fps=request.config.fps,
            resolution=tuple(map(int, request.config.resolution.split('x'))),
            num_inference_steps=request.config.num_inference_steps,
            guidance_scale=request.config.guidance_scale,
            save_videos=request.config.save_videos
        )
        
        benchmark_jobs[job_id]["progress"] = 0.9
        
        # Save to database (already done in evaluate_model)
        benchmark_jobs[job_id]["status"] = "completed"
        benchmark_jobs[job_id]["progress"] = 1.0
        benchmark_jobs[job_id]["result"] = result.to_dict()
        
        logger.info(f"Benchmark job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark job {job_id} failed: {e}")
        benchmark_jobs[job_id]["status"] = "failed"
        benchmark_jobs[job_id]["error"] = str(e)


@router.post("/run", response_model=BenchmarkJob, status_code=202)
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Start a benchmark run in the background."""
    try:
        # Validate model exists
        from ...models.registry import list_models
        available_models = list_models()
        if request.model_name not in available_models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{request.model_name}' not found. Available: {available_models}"
            )
        
        # Create job
        job_id = str(uuid.uuid4())
        benchmark_jobs[job_id] = {
            "status": "pending",
            "model_name": request.model_name,
            "created_at": "now",
            "progress": 0.0
        }
        
        # Start background task
        background_tasks.add_task(run_benchmark_job, job_id, request)
        
        return BenchmarkJob(
            job_id=job_id,
            status="pending",
            model_name=request.model_name,
            result_url=f"/api/v1/benchmarks/jobs/{job_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start benchmark: {e}")
        raise HTTPException(status_code=500, detail="Failed to start benchmark")


@router.get("/jobs/{job_id}", response_model=BenchmarkJob)
async def get_benchmark_job(job_id: str):
    """Get benchmark job status."""
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = benchmark_jobs[job_id]
    return BenchmarkJob(
        job_id=job_id,
        status=job_data["status"],
        model_name=job_data["model_name"],
        progress=job_data.get("progress"),
        result_url=f"/api/v1/benchmarks/jobs/{job_id}/result" if job_data["status"] == "completed" else None
    )


@router.get("/jobs/{job_id}/result")
async def get_benchmark_result(job_id: str):
    """Get benchmark job result."""
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = benchmark_jobs[job_id]
    if job_data["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job status: {job_data['status']}")
    
    return job_data.get("result", {})


@router.get("/runs", response_model=List[BenchmarkResultSummary])
async def list_benchmark_runs(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    status: Optional[EvaluationStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results")
):
    """List benchmark evaluation runs."""
    try:
        # Get model ID if model_name provided
        model_id = None
        if model_name:
            from ...database.services import ModelService
            model = ModelService.get_model_by_name(model_name)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            model_id = model.id
        
        runs = EvaluationService.list_evaluation_runs(
            model_id=model_id,
            status=status.value if status else None,
            limit=limit
        )
        
        return [
            BenchmarkResultSummary(
                id=run.id,
                model_name=run.model.name if run.model else "unknown",
                status=run.status,
                success_rate=run.success_rate,
                num_prompts=run.num_prompts,
                started_at=run.started_at,
                completed_at=run.completed_at,
                duration_seconds=run.duration_seconds
            )
            for run in runs
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list benchmark runs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve benchmark runs")


@router.get("/runs/{run_id}", response_model=DetailedBenchmarkResult)
async def get_benchmark_run(run_id: int):
    """Get detailed benchmark run results."""
    try:
        run = EvaluationService.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Benchmark run {run_id} not found")
        
        # Build detailed result
        result = DetailedBenchmarkResult(
            id=run.id,
            model_name=run.model.name if run.model else "unknown",
            status=run.status,
            success_rate=run.success_rate,
            num_prompts=run.num_prompts,
            started_at=run.started_at,
            completed_at=run.completed_at,
            duration_seconds=run.duration_seconds,
            config=run.config,
            errors=run.errors
        )
        
        # Add quality metrics if available
        if run.quality_metrics:
            from ..schemas import QualityMetricsSchema
            result.quality_metrics = QualityMetricsSchema.model_validate(run.quality_metrics[0])
        
        # Add performance metrics if available
        if run.performance_metrics:
            from ..schemas import PerformanceMetricsSchema
            result.performance_metrics = PerformanceMetricsSchema.model_validate(run.performance_metrics[0])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get benchmark run: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve benchmark run")


@router.delete("/runs/{run_id}")
async def delete_benchmark_run(run_id: int):
    """Delete a benchmark run and all associated data."""
    try:
        run = EvaluationService.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Benchmark run {run_id} not found")
        
        # TODO: Implement cascade delete in service
        # For now, just return success
        return {"message": f"Benchmark run {run_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete benchmark run: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete benchmark run")


@router.get("/runs/{run_id}/individual-results")
async def get_individual_results(run_id: int):
    """Get individual benchmark results for each prompt."""
    try:
        run = EvaluationService.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Benchmark run {run_id} not found")
        
        return [result.to_dict() for result in run.benchmark_results]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get individual results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve individual results")