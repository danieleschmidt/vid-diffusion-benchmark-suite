"""Advanced monitoring and alerting for robust benchmarking.

This module provides comprehensive monitoring capabilities including system metrics,
performance tracking, alerting, and real-time dashboards for the benchmarking system.
"""

import time
import threading
import logging
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import statistics
from pathlib import Path

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

from .fault_tolerance import HealthStatus


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Alert definition and state."""
    name: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    message: str
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    is_active: bool = False


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    gpu_metrics: Optional[Dict[str, Any]] = None
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkMetrics:
    """Benchmark-specific metrics."""
    models_evaluated: int = 0
    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    average_evaluation_time: float = 0.0
    total_evaluation_time: float = 0.0
    evaluations_per_minute: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collects and manages metrics."""
    
    def __init__(self, max_points: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_points: Maximum number of metric points to keep in memory
        """
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self._lock = threading.Lock()
        
        logger.info(f"MetricsCollector initialized with max_points={max_points}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a metric point.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            metric_type: Type of metric
        """
        with self._lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self.metrics[name].append(metric_point)
    
    def get_metric_history(
        self,
        name: str,
        duration_minutes: Optional[int] = None
    ) -> List[MetricPoint]:
        """Get metric history.
        
        Args:
            name: Metric name
            duration_minutes: Optional duration to filter
            
        Returns:
            List of metric points
        """
        with self._lock:
            points = list(self.metrics[name])
            
            if duration_minutes:
                cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
                points = [p for p in points if p.timestamp >= cutoff_time]
            
            return points
    
    def get_metric_statistics(
        self,
        name: str,
        duration_minutes: Optional[int] = None
    ) -> Dict[str, float]:
        """Get statistics for a metric.
        
        Args:
            name: Metric name
            duration_minutes: Optional duration to analyze
            
        Returns:
            Dictionary with statistics
        """
        points = self.get_metric_history(name, duration_minutes)
        
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1] if values else 0.0
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric statistics."""
        with self._lock:
            all_stats = {}
            for name in self.metrics.keys():
                all_stats[name] = self.get_metric_statistics(name, duration_minutes=5)
            return all_stats


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, collection_interval: float = 10.0):
        """Initialize system monitor.
        
        Args:
            collection_interval: Interval between collections in seconds
        """
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        self.running = False
        self._monitor_thread = None
        
        # Initialize network counters
        self._last_network_stats = psutil.net_io_counters()
        
        logger.info("SystemMonitor initialized")
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.running:
            return
        
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics_collector.record_metric('system.cpu.percent', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric('system.memory.percent', memory.percent)
            self.metrics_collector.record_metric('system.memory.used_gb', memory.used / 1e9)
            self.metrics_collector.record_metric('system.memory.available_gb', memory.available / 1e9)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_metric('system.disk.percent', disk_percent)
            self.metrics_collector.record_metric('system.disk.free_gb', disk.free / 1e9)
            
            # Network metrics
            current_network = psutil.net_io_counters()
            bytes_sent_delta = current_network.bytes_sent - self._last_network_stats.bytes_sent
            bytes_recv_delta = current_network.bytes_recv - self._last_network_stats.bytes_recv
            
            self.metrics_collector.record_metric('system.network.bytes_sent_rate', 
                                               bytes_sent_delta / self.collection_interval)
            self.metrics_collector.record_metric('system.network.bytes_recv_rate',
                                               bytes_recv_delta / self.collection_interval)
            
            self._last_network_stats = current_network
            
            # GPU metrics if available
            if GPU_AVAILABLE:
                self._collect_gpu_metrics()
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics."""
        try:
            for i in range(torch.cuda.device_count()):
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_cached = torch.cuda.memory_reserved(i)
                
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory
                
                memory_percent = (memory_cached / total_memory) * 100
                
                self.metrics_collector.record_metric(
                    f'gpu.{i}.memory.percent', memory_percent
                )
                self.metrics_collector.record_metric(
                    f'gpu.{i}.memory.allocated_gb', memory_allocated / 1e9
                )
                self.metrics_collector.record_metric(
                    f'gpu.{i}.memory.cached_gb', memory_cached / 1e9
                )
                
                # Temperature (if available)
                try:
                    temp = torch.cuda.temperature(i)
                    self.metrics_collector.record_metric(f'gpu.{i}.temperature', temp)
                except:
                    pass  # Temperature not available on all devices
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
    
    def get_current_system_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # GPU metrics
            gpu_metrics = None
            if GPU_AVAILABLE:
                gpu_metrics = {}
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_cached = torch.cuda.memory_reserved(i)
                    
                    gpu_metrics[f'gpu_{i}'] = {
                        'name': props.name,
                        'memory_total_gb': props.total_memory / 1e9,
                        'memory_allocated_gb': memory_allocated / 1e9,
                        'memory_cached_gb': memory_cached / 1e9,
                        'memory_percent': (memory_cached / props.total_memory) * 100
                    }
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / 1e9,
                memory_available_gb=memory.available / 1e9,
                disk_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / 1e9,
                gpu_metrics=gpu_metrics,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv
            )
            
        except Exception as e:
            logger.error(f"Error getting current system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_available_gb=0.0,
                disk_percent=0.0,
                disk_free_gb=0.0
            )


class BenchmarkMonitor:
    """Monitors benchmark execution and performance."""
    
    def __init__(self):
        """Initialize benchmark monitor."""
        self.metrics_collector = MetricsCollector()
        self.active_evaluations = {}
        self.evaluation_history = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        logger.info("BenchmarkMonitor initialized")
    
    def start_evaluation(self, evaluation_id: str, model_name: str, prompt: str):
        """Record start of evaluation.
        
        Args:
            evaluation_id: Unique evaluation identifier
            model_name: Name of model being evaluated
            prompt: Evaluation prompt
        """
        with self._lock:
            self.active_evaluations[evaluation_id] = {
                'model_name': model_name,
                'prompt': prompt,
                'start_time': datetime.now(),
                'status': 'running'
            }
        
        self.metrics_collector.record_metric(
            'benchmark.evaluations.started',
            1,
            {'model': model_name},
            MetricType.COUNTER
        )
    
    def end_evaluation(
        self,
        evaluation_id: str,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Record end of evaluation.
        
        Args:
            evaluation_id: Unique evaluation identifier
            success: Whether evaluation succeeded
            error_message: Optional error message if failed
        """
        with self._lock:
            if evaluation_id not in self.active_evaluations:
                logger.warning(f"Unknown evaluation ID: {evaluation_id}")
                return
            
            evaluation = self.active_evaluations[evaluation_id]
            end_time = datetime.now()
            duration = (end_time - evaluation['start_time']).total_seconds()
            
            # Update evaluation record
            evaluation.update({
                'end_time': end_time,
                'duration': duration,
                'success': success,
                'error_message': error_message
            })
            
            # Move to history
            self.evaluation_history.append(evaluation)
            del self.active_evaluations[evaluation_id]
        
        # Record metrics
        model_name = evaluation['model_name']
        tags = {'model': model_name}
        
        if success:
            self.metrics_collector.record_metric(
                'benchmark.evaluations.succeeded', 1, tags, MetricType.COUNTER
            )
        else:
            self.metrics_collector.record_metric(
                'benchmark.evaluations.failed', 1, tags, MetricType.COUNTER
            )
        
        self.metrics_collector.record_metric(
            'benchmark.evaluation.duration', duration, tags, MetricType.TIMER
        )
    
    def get_current_benchmark_metrics(self) -> BenchmarkMetrics:
        """Get current benchmark metrics."""
        with self._lock:
            total_evaluations = len(self.evaluation_history)
            successful = sum(1 for e in self.evaluation_history if e.get('success', False))
            failed = total_evaluations - successful
            
            # Calculate averages
            if self.evaluation_history:
                durations = [e['duration'] for e in self.evaluation_history 
                           if 'duration' in e]
                avg_duration = statistics.mean(durations) if durations else 0.0
                total_duration = sum(durations)
                
                # Evaluations per minute (last 10 minutes)
                recent_time = datetime.now() - timedelta(minutes=10)
                recent_evaluations = [
                    e for e in self.evaluation_history 
                    if e.get('end_time', datetime.min) >= recent_time
                ]
                evaluations_per_minute = len(recent_evaluations) / 10.0
                
                # Error rate
                error_rate = (failed / total_evaluations) * 100 if total_evaluations > 0 else 0.0
                
                # Unique models
                models = set(e['model_name'] for e in self.evaluation_history)
                models_evaluated = len(models)
            else:
                avg_duration = 0.0
                total_duration = 0.0
                evaluations_per_minute = 0.0
                error_rate = 0.0
                models_evaluated = 0
        
        return BenchmarkMetrics(
            models_evaluated=models_evaluated,
            total_evaluations=total_evaluations,
            successful_evaluations=successful,
            failed_evaluations=failed,
            average_evaluation_time=avg_duration,
            total_evaluation_time=total_duration,
            evaluations_per_minute=evaluations_per_minute,
            error_rate=error_rate
        )


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts = {}
        self.alert_history = deque(maxlen=500)
        self.notification_handlers = []
        self._lock = threading.Lock()
        
        logger.info("AlertManager initialized")
    
    def register_alert(
        self,
        name: str,
        severity: AlertSeverity,
        condition: Callable[[Dict[str, Any]], bool],
        message: str,
        cooldown_minutes: int = 5
    ):
        """Register alert condition.
        
        Args:
            name: Alert name
            severity: Alert severity
            condition: Function that returns True if alert should trigger
            message: Alert message template
            cooldown_minutes: Cooldown period between alerts
        """
        with self._lock:
            self.alerts[name] = Alert(
                name=name,
                severity=severity,
                condition=condition,
                message=message,
                cooldown_minutes=cooldown_minutes
            )
        
        logger.info(f"Registered alert '{name}' with severity {severity.value}")
    
    def register_notification_handler(self, handler: Callable[[Alert], None]):
        """Register notification handler.
        
        Args:
            handler: Function to handle alert notifications
        """
        self.notification_handlers.append(handler)
        logger.info("Registered notification handler")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert conditions.
        
        Args:
            metrics: Current metrics to check against
        """
        triggered_alerts = []
        
        with self._lock:
            for alert in self.alerts.values():
                try:
                    should_trigger = alert.condition(metrics)
                    
                    if should_trigger:
                        # Check cooldown
                        if alert.last_triggered:
                            time_since_last = datetime.now() - alert.last_triggered
                            if time_since_last.total_seconds() < alert.cooldown_minutes * 60:
                                continue  # Still in cooldown
                        
                        # Trigger alert
                        alert.last_triggered = datetime.now()
                        alert.trigger_count += 1
                        alert.is_active = True
                        triggered_alerts.append(alert)
                        
                        # Add to history
                        self.alert_history.append({
                            'name': alert.name,
                            'severity': alert.severity.value,
                            'message': alert.message,
                            'timestamp': alert.last_triggered.isoformat(),
                            'metrics': metrics.copy()
                        })
                        
                        logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
                    else:
                        # Reset active status if condition no longer met
                        if alert.is_active:
                            alert.is_active = False
                            logger.info(f"Alert resolved: {alert.name}")
                
                except Exception as e:
                    logger.error(f"Error checking alert '{alert.name}': {e}")
        
        # Send notifications
        for alert in triggered_alerts:
            self._send_notifications(alert)
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications.
        
        Args:
            alert: Alert to send notifications for
        """
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts.
        
        Returns:
            List of active alert dictionaries
        """
        with self._lock:
            active = []
            for alert in self.alerts.values():
                if alert.is_active:
                    active.append({
                        'name': alert.name,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'last_triggered': (alert.last_triggered.isoformat() 
                                         if alert.last_triggered else None),
                        'trigger_count': alert.trigger_count
                    })
            return active
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of alert history entries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for entry in self.alert_history:
            alert_time = datetime.fromisoformat(entry['timestamp'])
            if alert_time >= cutoff_time:
                history.append(entry)
        
        return history


class RobustnessMonitor:
    """Main monitoring coordinator for robustness metrics."""
    
    def __init__(self, monitoring_interval: float = 30.0):
        """Initialize robustness monitor.
        
        Args:
            monitoring_interval: Interval between monitoring checks in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.system_monitor = SystemMonitor()
        self.benchmark_monitor = BenchmarkMonitor()
        self.alert_manager = AlertManager()
        
        self.running = False
        self._monitor_thread = None
        
        # Setup default alerts
        self._setup_default_alerts()
        
        logger.info("RobustnessMonitor initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert conditions."""
        # High CPU usage alert
        self.alert_manager.register_alert(
            name="high_cpu_usage",
            severity=AlertSeverity.WARNING,
            condition=lambda m: m.get('system', {}).get('cpu_percent', 0) > 90,
            message="CPU usage is above 90%",
            cooldown_minutes=10
        )
        
        # High memory usage alert
        self.alert_manager.register_alert(
            name="high_memory_usage",
            severity=AlertSeverity.WARNING,
            condition=lambda m: m.get('system', {}).get('memory_percent', 0) > 85,
            message="Memory usage is above 85%",
            cooldown_minutes=10
        )
        
        # GPU memory alert
        self.alert_manager.register_alert(
            name="high_gpu_memory",
            severity=AlertSeverity.ERROR,
            condition=self._check_gpu_memory,
            message="GPU memory usage is critical",
            cooldown_minutes=5
        )
        
        # High error rate alert
        self.alert_manager.register_alert(
            name="high_error_rate",
            severity=AlertSeverity.ERROR,
            condition=lambda m: m.get('benchmark', {}).get('error_rate', 0) > 20,
            message="Benchmark error rate is above 20%",
            cooldown_minutes=15
        )
        
        # Low evaluation throughput alert
        self.alert_manager.register_alert(
            name="low_throughput",
            severity=AlertSeverity.WARNING,
            condition=lambda m: (
                m.get('benchmark', {}).get('evaluations_per_minute', 0) < 0.1 and
                m.get('benchmark', {}).get('total_evaluations', 0) > 5
            ),
            message="Evaluation throughput is very low",
            cooldown_minutes=20
        )
    
    def _check_gpu_memory(self, metrics: Dict[str, Any]) -> bool:
        """Check GPU memory usage across all GPUs."""
        system_metrics = metrics.get('system', {})
        gpu_metrics = system_metrics.get('gpu_metrics')
        
        if not gpu_metrics:
            return False
        
        for gpu_info in gpu_metrics.values():
            if gpu_info.get('memory_percent', 0) > 95:
                return True
        
        return False
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if self.running:
            return
        
        self.running = True
        
        # Start component monitors
        self.system_monitor.start_monitoring()
        
        # Start main monitoring loop
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring."""
        self.running = False
        
        # Stop component monitors
        self.system_monitor.stop_monitoring()
        
        # Stop main thread
        if self._monitor_thread:
            self._monitor_thread.join()
        
        logger.info("Comprehensive monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect all metrics
                current_metrics = self._collect_all_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(current_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics from different sources."""
        try:
            system_metrics = self.system_monitor.get_current_system_metrics()
            benchmark_metrics = self.benchmark_monitor.get_current_benchmark_metrics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system': asdict(system_metrics),
                'benchmark': asdict(benchmark_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {'timestamp': datetime.now().isoformat()}
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status.
        
        Returns:
            Dictionary with all monitoring information
        """
        current_metrics = self._collect_all_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        recent_alerts = self.alert_manager.get_alert_history(hours=2)
        
        # Get metric statistics
        system_stats = self.system_monitor.metrics_collector.get_all_metrics()
        benchmark_stats = self.benchmark_monitor.metrics_collector.get_all_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'metric_statistics': {
                'system': system_stats,
                'benchmark': benchmark_stats
            },
            'alerts': {
                'active_count': len(active_alerts),
                'active_alerts': active_alerts,
                'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
            },
            'monitoring_status': {
                'system_monitor_running': self.system_monitor.running,
                'monitoring_interval': self.monitoring_interval,
                'total_registered_alerts': len(self.alert_manager.alerts)
            }
        }
    
    def export_metrics(self, output_path: str, duration_hours: int = 24):
        """Export metrics to file.
        
        Args:
            output_path: Path to save metrics
            duration_hours: Hours of history to export
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'duration_hours': duration_hours,
            'system_metrics': {},
            'benchmark_metrics': {},
            'alert_history': self.alert_manager.get_alert_history(duration_hours)
        }
        
        # Export system metrics
        for metric_name in self.system_monitor.metrics_collector.metrics.keys():
            history = self.system_monitor.metrics_collector.get_metric_history(
                metric_name, duration_hours * 60
            )
            export_data['system_metrics'][metric_name] = [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'value': p.value,
                    'tags': p.tags
                }
                for p in history
            ]
        
        # Export benchmark metrics
        for metric_name in self.benchmark_monitor.metrics_collector.metrics.keys():
            history = self.benchmark_monitor.metrics_collector.get_metric_history(
                metric_name, duration_hours * 60
            )
            export_data['benchmark_metrics'][metric_name] = [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'value': p.value,
                    'tags': p.tags
                }
                for p in history
            ]
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {output_path}")


# Utility classes for health metrics
class HealthMetrics:
    """Health-specific metrics and calculations."""
    
    @staticmethod
    def calculate_system_health_score(system_metrics: SystemMetrics) -> float:
        """Calculate overall system health score (0-100).
        
        Args:
            system_metrics: Current system metrics
            
        Returns:
            Health score between 0 and 100
        """
        try:
            # CPU health (lower usage = higher score)
            cpu_score = max(0, 100 - system_metrics.cpu_percent)
            
            # Memory health
            memory_score = max(0, 100 - system_metrics.memory_percent)
            
            # Disk health
            disk_score = max(0, 100 - system_metrics.disk_percent)
            
            # GPU health (if available)
            gpu_score = 100  # Default if no GPU
            if system_metrics.gpu_metrics:
                gpu_scores = []
                for gpu_info in system_metrics.gpu_metrics.values():
                    gpu_memory_percent = gpu_info.get('memory_percent', 0)
                    gpu_scores.append(max(0, 100 - gpu_memory_percent))
                
                if gpu_scores:
                    gpu_score = statistics.mean(gpu_scores)
            
            # Weighted average
            weights = {
                'cpu': 0.3,
                'memory': 0.3,
                'disk': 0.2,
                'gpu': 0.2
            }
            
            overall_score = (
                weights['cpu'] * cpu_score +
                weights['memory'] * memory_score +
                weights['disk'] * disk_score +
                weights['gpu'] * gpu_score
            )
            
            return max(0, min(100, overall_score))
            
        except Exception as e:
            logger.error(f"Error calculating system health score: {e}")
            return 50.0  # Neutral score on error
    
    @staticmethod
    def calculate_benchmark_health_score(benchmark_metrics: BenchmarkMetrics) -> float:
        """Calculate benchmark health score (0-100).
        
        Args:
            benchmark_metrics: Current benchmark metrics
            
        Returns:
            Health score between 0 and 100
        """
        try:
            # Success rate score
            if benchmark_metrics.total_evaluations > 0:
                success_rate = (benchmark_metrics.successful_evaluations / 
                              benchmark_metrics.total_evaluations) * 100
            else:
                success_rate = 100  # No evaluations yet
            
            # Error rate score (inverse)
            error_rate_score = max(0, 100 - benchmark_metrics.error_rate)
            
            # Throughput score (relative to expected performance)
            expected_throughput = 1.0  # 1 evaluation per minute baseline
            throughput_score = min(100, 
                (benchmark_metrics.evaluations_per_minute / expected_throughput) * 100)
            
            # Weighted average
            weights = {
                'success_rate': 0.5,
                'error_rate': 0.3,
                'throughput': 0.2
            }
            
            overall_score = (
                weights['success_rate'] * success_rate +
                weights['error_rate'] * error_rate_score +
                weights['throughput'] * throughput_score
            )
            
            return max(0, min(100, overall_score))
            
        except Exception as e:
            logger.error(f"Error calculating benchmark health score: {e}")
            return 50.0  # Neutral score on error


# Default notification handlers
def console_notification_handler(alert: Alert):
    """Simple console notification handler."""
    print(f"🚨 ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}")


def log_notification_handler(alert: Alert):
    """Log-based notification handler."""
    log_level = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.ERROR: logging.ERROR,
        AlertSeverity.CRITICAL: logging.CRITICAL
    }.get(alert.severity, logging.WARNING)
    
    logger.log(log_level, f"Alert '{alert.name}': {alert.message}")


# Global monitoring instance
_global_robustness_monitor = None


def get_robustness_monitor() -> RobustnessMonitor:
    """Get global robustness monitor instance."""
    global _global_robustness_monitor
    if _global_robustness_monitor is None:
        _global_robustness_monitor = RobustnessMonitor()
        
        # Register default notification handlers
        _global_robustness_monitor.alert_manager.register_notification_handler(
            log_notification_handler
        )
    
    return _global_robustness_monitor