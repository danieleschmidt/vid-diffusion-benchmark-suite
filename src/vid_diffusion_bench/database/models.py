"""Database models for Video Diffusion Benchmark Suite."""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Boolean, 
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from .connection import Base


class Model(Base):
    """Model registry table."""
    
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    model_type = Column(String(100), nullable=False)  # 'diffusion', 'gan', etc.
    version = Column(String(100))
    
    # Model specifications
    vram_gb = Column(Float)
    precision = Column(String(50))  # 'fp16', 'fp32', etc.
    model_size_gb = Column(Float)
    max_frames = Column(Integer)
    supported_resolutions = Column(JSON)  # List of [width, height] pairs
    dependencies = Column(JSON)  # List of required packages
    
    # Metadata
    author = Column(String(255))
    paper_url = Column(String(500))
    code_url = Column(String(500))
    huggingface_id = Column(String(500))
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    evaluation_runs = relationship("EvaluationRun", back_populates="model")
    
    def __repr__(self):
        return f"<Model(name='{self.name}', type='{self.model_type}')>"
        
    @hybrid_property
    def requirements_dict(self) -> Dict[str, Any]:
        """Get model requirements as dictionary."""
        return {
            "vram_gb": self.vram_gb,
            "precision": self.precision,
            "model_size_gb": self.model_size_gb,
            "max_frames": self.max_frames,
            "supported_resolutions": self.supported_resolutions or [],
            "dependencies": self.dependencies or []
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "model_type": self.model_type,
            "version": self.version,
            "requirements": self.requirements_dict,
            "author": self.author,
            "paper_url": self.paper_url,
            "code_url": self.code_url,
            "huggingface_id": self.huggingface_id,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class EvaluationRun(Base):
    """Evaluation run tracking table."""
    
    __tablename__ = 'evaluation_runs'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    
    # Run configuration
    num_prompts = Column(Integer, nullable=False)
    num_frames = Column(Integer, nullable=False)
    fps = Column(Integer, nullable=False)
    resolution = Column(String(50))  # "512x512"
    inference_steps = Column(Integer)
    
    # Run status
    status = Column(String(50), default='pending')  # pending, running, completed, failed
    success_rate = Column(Float)  # 0.0 to 1.0
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    # Configuration and metadata
    config = Column(JSON)  # Full configuration used for run
    prompts_used = Column(JSON)  # List of prompts
    errors = Column(JSON)  # List of error details
    system_specs = Column(JSON)  # Hardware specifications
    
    # Relationships
    model = relationship("Model", back_populates="evaluation_runs")
    benchmark_results = relationship("BenchmarkResult", back_populates="evaluation_run")
    quality_metrics = relationship("QualityMetrics", back_populates="evaluation_run")
    performance_metrics = relationship("PerformanceMetrics", back_populates="evaluation_run")
    
    __table_args__ = (
        Index('idx_evaluation_runs_model_status', 'model_id', 'status'),
        Index('idx_evaluation_runs_started_at', 'started_at'),
    )
    
    def __repr__(self):
        return f"<EvaluationRun(id={self.id}, model='{self.model.name if self.model else None}', status='{self.status}')>"
        
    @hybrid_property
    def is_completed(self) -> bool:
        """Check if evaluation run is completed."""
        return self.status == 'completed'
        
    @hybrid_property
    def is_successful(self) -> bool:
        """Check if evaluation run was successful."""
        return self.status == 'completed' and (self.success_rate or 0) > 0.5
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation run to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model.name if self.model else None,
            "num_prompts": self.num_prompts,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "resolution": self.resolution,
            "inference_steps": self.inference_steps,
            "status": self.status,
            "success_rate": self.success_rate,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "config": self.config,
            "errors": self.errors,
            "system_specs": self.system_specs
        }


class BenchmarkResult(Base):
    """Individual benchmark result for a specific prompt."""
    
    __tablename__ = 'benchmark_results'
    
    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(Integer, ForeignKey('evaluation_runs.id'), nullable=False)
    
    # Prompt details
    prompt_index = Column(Integer, nullable=False)
    prompt_text = Column(Text, nullable=False)
    prompt_category = Column(String(100))  # motion, camera, scene, etc.
    
    # Generation details
    video_shape = Column(String(100))  # "16x3x512x512"
    generation_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    
    # Status
    status = Column(String(50), default='success')  # success, failed
    error_message = Column(Text)
    error_type = Column(String(100))
    
    # File references (optional)
    video_file_path = Column(String(500))
    thumbnail_path = Column(String(500))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="benchmark_results")
    
    __table_args__ = (
        Index('idx_benchmark_results_run_prompt', 'evaluation_run_id', 'prompt_index'),
        Index('idx_benchmark_results_status', 'status'),
        UniqueConstraint('evaluation_run_id', 'prompt_index', name='uq_run_prompt'),
    )
    
    def __repr__(self):
        return f"<BenchmarkResult(id={self.id}, prompt_index={self.prompt_index}, status='{self.status}')>"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "id": self.id,
            "evaluation_run_id": self.evaluation_run_id,
            "prompt_index": self.prompt_index,
            "prompt_text": self.prompt_text,
            "prompt_category": self.prompt_category,
            "video_shape": self.video_shape,
            "generation_time_ms": self.generation_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "status": self.status,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "video_file_path": self.video_file_path,
            "thumbnail_path": self.thumbnail_path,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class QualityMetrics(Base):
    """Quality metrics for an evaluation run."""
    
    __tablename__ = 'quality_metrics'
    
    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(Integer, ForeignKey('evaluation_runs.id'), nullable=False, unique=True)
    
    # Quality metrics
    fvd_score = Column(Float)
    inception_score_mean = Column(Float)
    inception_score_std = Column(Float)
    clip_similarity = Column(Float)
    temporal_consistency = Column(Float)
    
    # Composite scores
    overall_quality_score = Column(Float)
    quality_rank = Column(Integer)  # Rank among all models
    
    # Reference dataset info
    reference_dataset = Column(String(100))
    reference_samples = Column(Integer)
    
    # Computation metadata
    computed_at = Column(DateTime, default=datetime.utcnow)
    computation_time_seconds = Column(Float)
    
    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="quality_metrics")
    
    def __repr__(self):
        return f"<QualityMetrics(id={self.id}, fvd={self.fvd_score}, overall={self.overall_quality_score})>"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert quality metrics to dictionary."""
        return {
            "id": self.id,
            "evaluation_run_id": self.evaluation_run_id,
            "fvd_score": self.fvd_score,
            "inception_score_mean": self.inception_score_mean,
            "inception_score_std": self.inception_score_std,
            "clip_similarity": self.clip_similarity,
            "temporal_consistency": self.temporal_consistency,
            "overall_quality_score": self.overall_quality_score,
            "quality_rank": self.quality_rank,
            "reference_dataset": self.reference_dataset,
            "reference_samples": self.reference_samples,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
            "computation_time_seconds": self.computation_time_seconds
        }


class PerformanceMetrics(Base):
    """Performance metrics for an evaluation run."""
    
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(Integer, ForeignKey('evaluation_runs.id'), nullable=False, unique=True)
    
    # Latency metrics
    avg_latency_ms = Column(Float)
    median_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    throughput_fps = Column(Float)
    
    # Memory metrics
    peak_vram_gb = Column(Float)
    avg_vram_gb = Column(Float)
    peak_cpu_memory_gb = Column(Float)
    
    # GPU metrics
    avg_gpu_utilization = Column(Float)
    peak_gpu_utilization = Column(Float)
    avg_temperature_celsius = Column(Float)
    peak_temperature_celsius = Column(Float)
    
    # Power metrics
    avg_power_watts = Column(Float)
    peak_power_watts = Column(Float)
    total_energy_wh = Column(Float)
    
    # Efficiency scores
    efficiency_score = Column(Float)
    efficiency_rank = Column(Integer)  # Rank among all models
    
    # System load
    avg_cpu_percent = Column(Float)
    peak_cpu_percent = Column(Float)
    
    # Timestamps
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="performance_metrics")
    
    def __repr__(self):
        return f"<PerformanceMetrics(id={self.id}, latency={self.avg_latency_ms}ms, vram={self.peak_vram_gb}GB)>"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary."""
        return {
            "id": self.id,
            "evaluation_run_id": self.evaluation_run_id,
            "avg_latency_ms": self.avg_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_fps": self.throughput_fps,
            "peak_vram_gb": self.peak_vram_gb,
            "avg_vram_gb": self.avg_vram_gb,
            "peak_cpu_memory_gb": self.peak_cpu_memory_gb,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "peak_gpu_utilization": self.peak_gpu_utilization,
            "avg_temperature_celsius": self.avg_temperature_celsius,
            "peak_temperature_celsius": self.peak_temperature_celsius,
            "avg_power_watts": self.avg_power_watts,
            "peak_power_watts": self.peak_power_watts,
            "total_energy_wh": self.total_energy_wh,
            "efficiency_score": self.efficiency_score,
            "efficiency_rank": self.efficiency_rank,
            "avg_cpu_percent": self.avg_cpu_percent,
            "peak_cpu_percent": self.peak_cpu_percent,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None
        }