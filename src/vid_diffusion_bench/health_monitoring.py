"""Comprehensive health monitoring system for video diffusion benchmarks."""

import psutil
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

try:
    import torch
    import GPUtil
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SystemHealth:
    """System health status snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    process_count: int
    gpu_info: Optional[Dict[str, Any]] = None
    network_stats: Optional[Dict[str, Any]] = None
    alerts: List[str] = None


@dataclass
class HealthThresholds:
    """Health monitoring thresholds."""
    cpu_warning: float = 80.0
    cpu_critical: float = 95.0
    memory_warning: float = 85.0
    memory_critical: float = 95.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0
    gpu_memory_warning: float = 85.0
    gpu_memory_critical: float = 95.0
    gpu_temp_warning: float = 80.0
    gpu_temp_critical: float = 90.0


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, thresholds: HealthThresholds = None, history_size: int = 1000):
        self.thresholds = thresholds or HealthThresholds()
        self.history_size = history_size
        self.health_history: List[SystemHealth] = []
        self.lock = threading.Lock()
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 10.0  # seconds
        
        # Health check callbacks
        self.health_callbacks: List[Callable[[SystemHealth], None]] = []
        
        logger.info("Health monitor initialized")
    
    def start_monitoring(self, interval: float = 10.0):
        """Start continuous health monitoring."""
        if self.monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.monitor_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Health monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring:
            try:
                health = self.check_health()
                self._store_health(health)
                self._trigger_callbacks(health)
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.monitor_interval)
    
    def check_health(self) -> SystemHealth:
        """Perform comprehensive health check."""
        timestamp = time.time()
        alerts = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent >= self.thresholds.cpu_critical:
            alerts.append(f"CRITICAL: CPU usage {cpu_percent:.1f}%")
        elif cpu_percent >= self.thresholds.cpu_warning:
            alerts.append(f"WARNING: CPU usage {cpu_percent:.1f}%")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        if memory_percent >= self.thresholds.memory_critical:
            alerts.append(f"CRITICAL: Memory usage {memory_percent:.1f}%")
        elif memory_percent >= self.thresholds.memory_warning:
            alerts.append(f"WARNING: Memory usage {memory_percent:.1f}%")
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        if disk_usage_percent >= self.thresholds.disk_critical:
            alerts.append(f"CRITICAL: Disk usage {disk_usage_percent:.1f}%")
        elif disk_usage_percent >= self.thresholds.disk_warning:
            alerts.append(f"WARNING: Disk usage {disk_usage_percent:.1f}%")
        
        # Load average
        load_average = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        
        # Process count
        process_count = len(psutil.pids())
        
        # GPU info
        gpu_info = self._get_gpu_info()
        if gpu_info:
            for gpu_id, gpu_data in gpu_info.items():
                if 'memory_percent' in gpu_data:
                    mem_pct = gpu_data['memory_percent']
                    if mem_pct >= self.thresholds.gpu_memory_critical:
                        alerts.append(f"CRITICAL: GPU {gpu_id} memory {mem_pct:.1f}%")
                    elif mem_pct >= self.thresholds.gpu_memory_warning:
                        alerts.append(f"WARNING: GPU {gpu_id} memory {mem_pct:.1f}%")
                
                if 'temperature' in gpu_data:
                    temp = gpu_data['temperature']
                    if temp >= self.thresholds.gpu_temp_critical:
                        alerts.append(f"CRITICAL: GPU {gpu_id} temperature {temp}°C")
                    elif temp >= self.thresholds.gpu_temp_warning:
                        alerts.append(f"WARNING: GPU {gpu_id} temperature {temp}°C")
        
        # Network stats
        network_stats = self._get_network_stats()
        
        return SystemHealth(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            load_average=load_average,
            process_count=process_count,
            gpu_info=gpu_info,
            network_stats=network_stats,
            alerts=alerts
        )
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information if available."""
        if not GPU_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = {}
            
            for i, gpu in enumerate(gpus):
                gpu_info[f"gpu_{i}"] = {
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_utilization": gpu.load * 100,
                    "temperature": gpu.temperature,
                    "driver_version": gpu.driver
                }
            
            # Add PyTorch GPU info if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch_memory = torch.cuda.memory_stats(i)
                    if f"gpu_{i}" in gpu_info:
                        gpu_info[f"gpu_{i}"].update({
                            "torch_allocated_mb": torch_memory.get("allocated_bytes.all.current", 0) / (1024**2),
                            "torch_reserved_mb": torch_memory.get("reserved_bytes.all.current", 0) / (1024**2),
                            "torch_max_allocated_mb": torch_memory.get("allocated_bytes.all.peak", 0) / (1024**2)
                        })
            
            return gpu_info
            
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return None
    
    def _get_network_stats(self) -> Optional[Dict[str, Any]]:
        """Get network statistics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout
            }
        except Exception as e:
            logger.warning(f"Failed to get network stats: {e}")
            return None
    
    def _store_health(self, health: SystemHealth):
        """Store health data in history."""
        with self.lock:
            self.health_history.append(health)
            if len(self.health_history) > self.history_size:
                self.health_history.pop(0)
    
    def _trigger_callbacks(self, health: SystemHealth):
        """Trigger registered health callbacks."""
        for callback in self.health_callbacks:
            try:
                callback(health)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    def add_health_callback(self, callback: Callable[[SystemHealth], None]):
        """Add health status callback."""
        self.health_callbacks.append(callback)
    
    def get_health_history(self, minutes: int = 60) -> List[SystemHealth]:
        """Get health history for specified minutes."""
        cutoff_time = time.time() - (minutes * 60)
        with self.lock:
            return [h for h in self.health_history if h.timestamp >= cutoff_time]
    
    def get_current_health(self) -> SystemHealth:
        """Get current health status."""
        return self.check_health()
    
    def is_healthy(self) -> bool:
        """Check if system is currently healthy."""
        health = self.check_health()
        return len(health.alerts or []) == 0
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics."""
        history = self.get_health_history(60)  # Last hour
        if not history:
            return {"error": "No health data available"}
        
        cpu_values = [h.cpu_percent for h in history]
        memory_values = [h.memory_percent for h in history]
        
        return {
            "last_check": history[-1].timestamp,
            "is_healthy": len(history[-1].alerts or []) == 0,
            "active_alerts": history[-1].alerts or [],
            "cpu": {
                "current": history[-1].cpu_percent,
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "current": history[-1].memory_percent,
                "average": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "available_gb": history[-1].memory_available_gb
            },
            "disk": {
                "usage_percent": history[-1].disk_usage_percent,
                "free_gb": history[-1].disk_free_gb
            },
            "gpu": history[-1].gpu_info
        }
    
    def export_health_data(self, file_path: str, minutes: int = 60):
        """Export health data to JSON file."""
        history = self.get_health_history(minutes)
        data = {
            "export_time": datetime.now().isoformat(),
            "history_minutes": minutes,
            "health_data": [asdict(h) for h in history]
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Health data exported to {file_path}")


# Global health monitor instance
_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    return _health_monitor


@contextmanager
def health_monitoring(interval: float = 10.0):
    """Context manager for temporary health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring(interval)
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def health_check_decorator(alert_on_unhealthy: bool = True):
    """Decorator to check system health before function execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            health = _health_monitor.check_health()
            
            if alert_on_unhealthy and health.alerts:
                logger.warning(f"System health alerts before {func.__name__}: {health.alerts}")
            
            # Store pre-execution health
            pre_health = health
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Check health after execution
                post_health = _health_monitor.check_health()
                if post_health.alerts:
                    logger.warning(f"System health alerts after {func.__name__}: {post_health.alerts}")
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Create health monitor
    monitor = HealthMonitor()
    
    # Start monitoring
    monitor.start_monitoring(interval=5.0)
    
    # Add custom callback
    def alert_callback(health: SystemHealth):
        if health.alerts:
            print(f"HEALTH ALERT: {health.alerts}")
    
    monitor.add_health_callback(alert_callback)
    
    # Run for 30 seconds
    time.sleep(30)
    
    # Get health summary
    summary = monitor.get_health_summary()
    print(json.dumps(summary, indent=2))
    
    # Stop monitoring
    monitor.stop_monitoring()