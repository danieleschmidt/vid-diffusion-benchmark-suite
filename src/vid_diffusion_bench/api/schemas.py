"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class ModelType(str, Enum):
    """Model type enumeration."""
    DIFFUSION = "diffusion"
    GAN = "gan"
    AUTOREGRESSIVE = "autoregressive"
    OTHER = "other"


class EvaluationStatus(str, Enum):
    """Evaluation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelRequirements(BaseModel):
    """Model hardware and software requirements."""
    vram_gb: Optional[float] = None
    precision: Optional[str] = None
    model_size_gb: Optional[float] = None
    max_frames: Optional[int] = None
    supported_resolutions: Optional[List[List[int]]] = None
    dependencies: Optional[List[str]] = None


class ModelBase(BaseModel):
    """Base model schema."""
    name: str = Field(..., description="Unique model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = None
    model_type: ModelType = Field(..., description="Type of model")
    version: Optional[str] = None
    author: Optional[str] = None
    paper_url: Optional[str] = None
    code_url: Optional[str] = None
    huggingface_id: Optional[str] = None


class ModelCreate(ModelBase):
    """Schema for creating a new model."""
    requirements: ModelRequirements = Field(..., description="Model requirements")


class Model(ModelBase):
    """Full model schema."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    requirements: ModelRequirements
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime


class BenchmarkConfig(BaseModel):
    """Benchmark configuration schema."""
    num_frames: int = Field(16, ge=1, le=128, description="Number of frames to generate")
    fps: int = Field(8, ge=1, le=30, description="Frames per second")
    resolution: str = Field("512x512", description="Output resolution (WxH)")
    num_inference_steps: int = Field(25, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    save_videos: bool = Field(False, description="Whether to save generated videos")


class BenchmarkRequest(BaseModel):
    """Benchmark request schema."""
    model_name: str = Field(..., description="Name of model to benchmark")
    prompts: Optional[List[str]] = Field(None, description="Custom prompts (uses standard if None)")
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


class BenchmarkResultSummary(BaseModel):
    """Summary of benchmark results."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    model_name: str
    status: EvaluationStatus
    success_rate: Optional[float]
    num_prompts: int
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]


class QualityMetricsSchema(BaseModel):
    """Quality metrics schema."""
    model_config = ConfigDict(from_attributes=True)
    
    fvd_score: Optional[float] = Field(None, description="Fréchet Video Distance (lower is better)")
    inception_score_mean: Optional[float] = Field(None, description="Inception Score mean")
    inception_score_std: Optional[float] = Field(None, description="Inception Score standard deviation")
    clip_similarity: Optional[float] = Field(None, description="CLIP text-video similarity")
    temporal_consistency: Optional[float] = Field(None, description="Temporal consistency score")
    overall_quality_score: Optional[float] = Field(None, description="Overall quality score (0-100)")
    reference_dataset: Optional[str] = Field(None, description="Reference dataset used")


class PerformanceMetricsSchema(BaseModel):
    """Performance metrics schema."""
    model_config = ConfigDict(from_attributes=True)
    
    avg_latency_ms: Optional[float] = Field(None, description="Average generation latency (ms)")
    throughput_fps: Optional[float] = Field(None, description="Throughput (videos per second)")
    peak_vram_gb: Optional[float] = Field(None, description="Peak VRAM usage (GB)")
    avg_power_watts: Optional[float] = Field(None, description="Average power consumption (watts)")
    efficiency_score: Optional[float] = Field(None, description="Overall efficiency score (0-100)")


class DetailedBenchmarkResult(BenchmarkResultSummary):
    """Detailed benchmark result with metrics."""
    quality_metrics: Optional[QualityMetricsSchema] = None
    performance_metrics: Optional[PerformanceMetricsSchema] = None
    config: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None


class LeaderboardEntry(BaseModel):
    """Leaderboard entry schema."""
    rank: int
    model_name: str
    model_display_name: str
    model_type: str
    metric_value: float
    quality_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    evaluation_run_id: int
    completed_at: Optional[datetime] = None


class Leaderboard(BaseModel):
    """Leaderboard schema."""
    metric: str = Field(..., description="Metric used for ranking")
    entries: List[LeaderboardEntry] = Field(..., description="Leaderboard entries")
    total_entries: int = Field(..., description="Total number of entries")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelStatistics(BaseModel):
    """Model statistics schema."""
    total_models: int
    total_evaluation_runs: int
    completed_runs: int
    failed_runs: int
    models_by_type: Dict[str, int]
    average_quality_score: Optional[float] = None
    average_efficiency_score: Optional[float] = None


class HealthStatus(BaseModel):
    """Health status schema."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    database: Dict[str, Any] = Field(..., description="Database health info")
    models_available: int = Field(..., description="Number of available models")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkJob(BaseModel):
    """Background benchmark job schema."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    model_name: str = Field(..., description="Model being benchmarked")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    progress: Optional[float] = Field(None, description="Job progress (0-1)")
    result_url: Optional[str] = Field(None, description="URL to get results when complete")