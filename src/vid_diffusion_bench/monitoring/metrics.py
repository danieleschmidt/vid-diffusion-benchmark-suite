"""Metrics collection and monitoring."""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..monitoring.logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    model_name: str
    total_prompts: int
    successful_prompts: int
    failed_prompts: int
    avg_latency_ms: float
    p95_latency_ms: float
    total_duration_seconds: float
    peak_memory_mb: float
    avg_quality_score: float
    timestamp: datetime


class MetricsCollector:
    """Collects and tracks benchmark metrics."""
    
    def __init__(self):
        self._metrics_history: List[BenchmarkMetrics] = []
        self._current_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Real-time tracking
        self._active_benchmarks = {}
        self._latency_samples = defaultdict(deque)
        self._memory_samples = defaultdict(deque)
        
    def start_benchmark(self, model_name: str, run_id: str, num_prompts: int):
        """Start tracking a benchmark run."""
        with self._lock:
            self._active_benchmarks[run_id] = {
                "model_name": model_name,
                "start_time": time.time(),
                "num_prompts": num_prompts,
                "completed_prompts": 0,
                "failed_prompts": 0,
                "latencies": [],
                "memory_usage": []
            }
            
        logger.info(
            "Started tracking benchmark metrics",
            model_name=model_name,
            run_id=run_id,
            num_prompts=num_prompts
        )
        
    def record_prompt_completion(
        self, 
        run_id: str, 
        success: bool, 
        latency_ms: float, 
        memory_mb: float
    ):
        """Record completion of a single prompt."""
        with self._lock:
            if run_id not in self._active_benchmarks:
                logger.warning(f"Unknown benchmark run: {run_id}")
                return
                
            benchmark = self._active_benchmarks[run_id]
            
            if success:
                benchmark["completed_prompts"] += 1
            else:
                benchmark["failed_prompts"] += 1
                
            benchmark["latencies"].append(latency_ms)
            benchmark["memory_usage"].append(memory_mb)
            
            # Keep sliding window for real-time metrics
            model_name = benchmark["model_name"]
            self._latency_samples[model_name].append(latency_ms)
            self._memory_samples[model_name].append(memory_mb)
            
            # Keep only last 100 samples for real-time metrics
            if len(self._latency_samples[model_name]) > 100:
                self._latency_samples[model_name].popleft()
            if len(self._memory_samples[model_name]) > 100:
                self._memory_samples[model_name].popleft()
                
    def complete_benchmark(
        self, 
        run_id: str, 
        quality_metrics: Dict[str, float]
    ) -> BenchmarkMetrics:
        """Complete benchmark tracking and compute final metrics."""
        with self._lock:
            if run_id not in self._active_benchmarks:
                raise ValueError(f"Unknown benchmark run: {run_id}")
                
            benchmark = self._active_benchmarks[run_id]
            end_time = time.time()
            
            # Compute final metrics
            latencies = benchmark["latencies"]
            memory_usage = benchmark["memory_usage"]
            
            metrics = BenchmarkMetrics(
                model_name=benchmark["model_name"],
                total_prompts=benchmark["num_prompts"],
                successful_prompts=benchmark["completed_prompts"],
                failed_prompts=benchmark["failed_prompts"],
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
                p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
                total_duration_seconds=end_time - benchmark["start_time"],
                peak_memory_mb=max(memory_usage) if memory_usage else 0,
                avg_quality_score=quality_metrics.get("overall_quality_score", 0),
                timestamp=datetime.utcnow()
            )
            
            # Store in history
            self._metrics_history.append(metrics)
            
            # Clean up
            del self._active_benchmarks[run_id]
            
            logger.info(
                "Completed benchmark metrics collection",
                **asdict(metrics)
            )
            
            return metrics
            
    def get_real_time_metrics(self, model_name: str) -> Dict[str, float]:
        """Get real-time metrics for a model."""
        with self._lock:
            latencies = list(self._latency_samples.get(model_name, []))
            memory_usage = list(self._memory_samples.get(model_name, []))
            
            if not latencies:
                return {}
                
            return {
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": self._percentile(latencies, 95),
                "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "sample_count": len(latencies)
            }
            
    def get_model_summary(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary metrics for a model over specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        relevant_metrics = [
            m for m in self._metrics_history 
            if m.model_name == model_name and m.timestamp > cutoff_time
        ]
        
        if not relevant_metrics:
            return {"model_name": model_name, "no_data": True}
            
        total_runs = len(relevant_metrics)
        total_prompts = sum(m.total_prompts for m in relevant_metrics)
        successful_prompts = sum(m.successful_prompts for m in relevant_metrics)
        
        return {
            "model_name": model_name,
            "time_period_hours": hours,
            "total_runs": total_runs,
            "total_prompts": total_prompts,
            "success_rate": successful_prompts / total_prompts if total_prompts > 0 else 0,
            "avg_latency_ms": sum(m.avg_latency_ms for m in relevant_metrics) / total_runs,
            "avg_quality_score": sum(m.avg_quality_score for m in relevant_metrics) / total_runs,
            "peak_memory_mb": max(m.peak_memory_mb for m in relevant_metrics),
            "total_duration_hours": sum(m.total_duration_seconds for m in relevant_metrics) / 3600
        }
        
    def get_all_metrics_history(self, limit: int = 100) -> List[BenchmarkMetrics]:
        """Get recent metrics history."""
        with self._lock:
            return self._metrics_history[-limit:]
            
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c


class PrometheusMetrics:
    """Prometheus metrics integration."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics disabled")
            self.enabled = False
            return
            
        self.enabled = True
        self.registry = registry or CollectorRegistry()
        
        # Define metrics
        self.benchmark_runs_total = Counter(
            'vid_bench_runs_total',
            'Total number of benchmark runs',
            ['model_name', 'status'],
            registry=self.registry
        )
        
        self.prompt_processing_duration = Histogram(
            'vid_bench_prompt_duration_seconds',
            'Time spent processing individual prompts',
            ['model_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Histogram(
            'vid_bench_memory_usage_bytes',
            'Memory usage during benchmark',
            ['model_name', 'memory_type'],
            buckets=[1e6, 1e7, 1e8, 1e9, 2e9, 4e9, 8e9, 16e9, 32e9],
            registry=self.registry
        )
        
        self.quality_scores = Histogram(
            'vid_bench_quality_scores',
            'Quality scores from benchmarks',
            ['model_name', 'metric_type'],
            buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            registry=self.registry
        )
        
        self.active_benchmarks = Gauge(
            'vid_bench_active_benchmarks',
            'Number of currently running benchmarks',
            ['model_name'],
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
        
    def record_benchmark_start(self, model_name: str):
        """Record benchmark start."""
        if not self.enabled:
            return
        self.active_benchmarks.labels(model_name=model_name).inc()
        
    def record_benchmark_complete(self, model_name: str, success: bool):
        """Record benchmark completion."""
        if not self.enabled:
            return
        status = "success" if success else "failure"
        self.benchmark_runs_total.labels(model_name=model_name, status=status).inc()
        self.active_benchmarks.labels(model_name=model_name).dec()
        
    def record_prompt_duration(self, model_name: str, duration_seconds: float):
        """Record prompt processing duration."""
        if not self.enabled:
            return
        self.prompt_processing_duration.labels(model_name=model_name).observe(duration_seconds)
        
    def record_memory_usage(self, model_name: str, memory_type: str, bytes_used: float):
        """Record memory usage."""
        if not self.enabled:
            return
        self.memory_usage_bytes.labels(
            model_name=model_name, 
            memory_type=memory_type
        ).observe(bytes_used)
        
    def record_quality_score(self, model_name: str, metric_type: str, score: float):
        """Record quality score."""
        if not self.enabled:
            return
        self.quality_scores.labels(
            model_name=model_name,
            metric_type=metric_type
        ).observe(score)
        
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if not self.enabled:
            return "# Prometheus metrics disabled\n"
        return generate_latest(self.registry).decode('utf-8')


# Global metrics instances
_metrics_collector = MetricsCollector()
_prometheus_metrics = PrometheusMetrics()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _metrics_collector


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get global Prometheus metrics instance."""
    return _prometheus_metrics