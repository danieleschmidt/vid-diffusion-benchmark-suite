"""Distributed computing framework for scalable video diffusion benchmarking.

High-performance distributed execution using Ray, Dask, and custom orchestration
for massive-scale benchmarking across multiple nodes and GPUs.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue
import socket
import pickle
import zmq
import psutil

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import numpy as np
from collections import defaultdict, deque

from .benchmark import BenchmarkResult, BenchmarkSuite
from .models.registry import get_model, list_models
from .profiler import EfficiencyProfiler
from .robustness.fault_tolerance import CircuitBreaker, RetryHandler
from .monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Distributed execution modes."""
    LOCAL_PARALLEL = "local_parallel"
    DISTRIBUTED_NODES = "distributed_nodes"  
    GPU_CLUSTER = "gpu_cluster"
    CLOUD_BATCH = "cloud_batch"
    HYBRID = "hybrid"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class ComputeResource:
    """Represents a compute resource in the cluster."""
    node_id: str
    hostname: str
    gpu_count: int
    gpu_memory_gb: List[int]
    cpu_cores: int
    memory_gb: float
    available_gpus: List[int] = field(default_factory=list)
    current_load: float = 0.0
    status: str = "available"
    last_heartbeat: float = field(default_factory=time.time)


@dataclass 
class BenchmarkTask:
    """Individual benchmark task for distributed execution."""
    task_id: str
    model_name: str
    prompts: List[str]
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    result: Optional[BenchmarkResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ClusterConfiguration:
    """Configuration for distributed cluster."""
    master_address: str = "localhost"
    master_port: int = 6379
    node_discovery_port: int = 6380
    heartbeat_interval: int = 30
    task_timeout: int = 3600
    max_concurrent_tasks_per_node: int = 4
    load_balancing_strategy: str = "least_loaded"
    fault_tolerance: bool = True
    auto_scaling: bool = False


class ResourceManager:
    """Manages compute resources across the distributed cluster."""
    
    def __init__(self, config: ClusterConfiguration):
        self.config = config
        self.resources: Dict[str, ComputeResource] = {}
        self.resource_lock = threading.RLock()
        self.metrics_collector = MetricsCollector()
        self.circuit_breaker = CircuitBreaker()
        
    def register_node(self, resource: ComputeResource) -> bool:
        """Register a new compute node."""
        with self.resource_lock:
            if resource.node_id in self.resources:
                logger.warning(f"Node {resource.node_id} already registered, updating...")
            
            self.resources[resource.node_id] = resource
            logger.info(f"Registered node {resource.node_id} with {resource.gpu_count} GPUs")
            
            # Update metrics
            self.metrics_collector.update_gauge("cluster.nodes.total", len(self.resources))
            self.metrics_collector.update_gauge(
                "cluster.gpus.total", 
                sum(r.gpu_count for r in self.resources.values())
            )
            
            return True
    
    def allocate_resources(self, task: BenchmarkTask) -> Optional[ComputeResource]:
        """Allocate resources for a benchmark task."""
        with self.resource_lock:
            # Find suitable resource based on strategy
            if self.config.load_balancing_strategy == "least_loaded":
                candidate = self._find_least_loaded_node(task)
            elif self.config.load_balancing_strategy == "round_robin":
                candidate = self._find_round_robin_node(task)
            elif self.config.load_balancing_strategy == "gpu_memory":
                candidate = self._find_best_gpu_memory_node(task)
            else:
                candidate = self._find_random_available_node(task)
            
            if candidate:
                # Reserve resources
                candidate.current_load += self._estimate_task_load(task)
                candidate.available_gpus = candidate.available_gpus[1:]  # Reserve one GPU
                task.assigned_node = candidate.node_id
                
                logger.debug(f"Allocated node {candidate.node_id} for task {task.task_id}")
                return candidate
            
            return None
    
    def release_resources(self, task: BenchmarkTask):
        """Release resources when task completes."""
        if not task.assigned_node:
            return
            
        with self.resource_lock:
            if task.assigned_node in self.resources:
                resource = self.resources[task.assigned_node]
                resource.current_load = max(0, resource.current_load - self._estimate_task_load(task))
                
                # Return GPU to available pool
                if len(resource.available_gpus) < resource.gpu_count:
                    resource.available_gpus.append(0)  # Simplified GPU tracking
                
                logger.debug(f"Released resources for task {task.task_id} on node {task.assigned_node}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        with self.resource_lock:
            total_nodes = len(self.resources)
            available_nodes = sum(1 for r in self.resources.values() if r.status == "available")
            total_gpus = sum(r.gpu_count for r in self.resources.values())
            available_gpus = sum(len(r.available_gpus) for r in self.resources.values())
            avg_load = np.mean([r.current_load for r in self.resources.values()]) if self.resources else 0
            
            return {
                "total_nodes": total_nodes,
                "available_nodes": available_nodes,
                "total_gpus": total_gpus,
                "available_gpus": available_gpus,
                "average_load": avg_load,
                "node_details": {
                    node_id: {
                        "hostname": r.hostname,
                        "load": r.current_load,
                        "available_gpus": len(r.available_gpus),
                        "status": r.status
                    }
                    for node_id, r in self.resources.items()
                }
            }
    
    def _find_least_loaded_node(self, task: BenchmarkTask) -> Optional[ComputeResource]:
        """Find node with least current load."""
        available_nodes = [
            r for r in self.resources.values() 
            if r.status == "available" and len(r.available_gpus) > 0
        ]
        
        if not available_nodes:
            return None
            
        return min(available_nodes, key=lambda r: r.current_load)
    
    def _find_round_robin_node(self, task: BenchmarkTask) -> Optional[ComputeResource]:
        """Simple round-robin node selection."""
        available_nodes = [
            r for r in self.resources.values() 
            if r.status == "available" and len(r.available_gpus) > 0
        ]
        
        if not available_nodes:
            return None
            
        # Use task_id hash for deterministic round-robin
        index = hash(task.task_id) % len(available_nodes)
        return available_nodes[index]
    
    def _find_best_gpu_memory_node(self, task: BenchmarkTask) -> Optional[ComputeResource]:
        """Find node with best GPU memory for the task."""
        available_nodes = [
            r for r in self.resources.values() 
            if r.status == "available" and len(r.available_gpus) > 0
        ]
        
        if not available_nodes:
            return None
            
        # Estimate memory requirement (simplified)
        estimated_memory = self._estimate_memory_requirement(task)
        
        # Find nodes with sufficient memory
        suitable_nodes = [
            r for r in available_nodes
            if max(r.gpu_memory_gb) >= estimated_memory
        ]
        
        if suitable_nodes:
            # Choose node with most available memory
            return max(suitable_nodes, key=lambda r: max(r.gpu_memory_gb))
        else:
            # Fall back to least loaded
            return self._find_least_loaded_node(task)
    
    def _find_random_available_node(self, task: BenchmarkTask) -> Optional[ComputeResource]:
        """Find random available node."""
        available_nodes = [
            r for r in self.resources.values() 
            if r.status == "available" and len(r.available_gpus) > 0
        ]
        
        if available_nodes:
            return np.secrets.SystemRandom().choice(available_nodes)
        return None
    
    def _estimate_task_load(self, task: BenchmarkTask) -> float:
        """Estimate computational load of a task."""
        # Simplified load estimation based on task parameters
        base_load = 1.0
        
        # Adjust based on model complexity
        if "xl" in task.model_name.lower():
            base_load *= 2.0
        elif "large" in task.model_name.lower():
            base_load *= 1.5
            
        # Adjust based on number of prompts
        base_load *= min(len(task.prompts) / 10, 2.0)
        
        # Adjust based on parameters
        params = task.parameters
        if "num_frames" in params:
            base_load *= params["num_frames"] / 16  # Baseline 16 frames
        
        return base_load
    
    def _estimate_memory_requirement(self, task: BenchmarkTask) -> int:
        """Estimate GPU memory requirement in GB."""
        base_memory = 8  # Base memory requirement
        
        # Adjust for model size
        if "xl" in task.model_name.lower():
            base_memory = 24
        elif "large" in task.model_name.lower():
            base_memory = 16
            
        # Adjust for frame count
        params = task.parameters
        if "num_frames" in params:
            base_memory *= max(1, params["num_frames"] / 16)
            
        return int(base_memory)


class TaskScheduler:
    """Schedules and dispatches benchmark tasks across the cluster."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.task_queue = queue.PriorityQueue()
        self.running_tasks: Dict[str, BenchmarkTask] = {}
        self.completed_tasks: Dict[str, BenchmarkTask] = {}
        self.task_lock = threading.RLock()
        self.scheduler_thread = None
        self.is_running = False
        self.retry_handler = RetryHandler(max_retries=3)
        
    def submit_task(self, task: BenchmarkTask) -> str:
        """Submit a benchmark task for execution."""
        with self.task_lock:
            # Check dependencies
            if not self._check_dependencies(task):
                logger.warning(f"Task {task.task_id} has unmet dependencies")
                task.status = TaskStatus.PENDING
            
            # Add to queue (priority queue uses task priority)
            self.task_queue.put((task.priority, time.time(), task))
            logger.info(f"Submitted task {task.task_id} with priority {task.priority}")
            
            return task.task_id
    
    def start_scheduler(self):
        """Start the task scheduler."""
        if self.is_running:
            return
            
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Task scheduler started")
    
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        logger.info("Task scheduler stopped")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a specific task."""
        with self.task_lock:
            if task_id in self.running_tasks:
                return self.running_tasks[task_id].status
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id].status
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self.task_lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                logger.info(f"Cancelled task {task_id}")
                return True
            return False
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue status."""
        with self.task_lock:
            return {
                "pending": self.task_queue.qsize(),
                "running": len(self.running_tasks),
                "completed": len(self.completed_tasks)
            }
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Process pending tasks
                self._dispatch_pending_tasks()
                
                # Check running tasks
                self._check_running_tasks()
                
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                time.sleep(1)  # Scheduler tick rate
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)  # Back off on error
    
    def _dispatch_pending_tasks(self):
        """Dispatch pending tasks to available resources."""
        dispatched = 0
        max_dispatch_per_cycle = 10
        
        while not self.task_queue.empty() and dispatched < max_dispatch_per_cycle:
            try:
                # Get next task (blocks if queue is empty)
                _, _, task = self.task_queue.get(timeout=1)
                
                # Check if dependencies are met
                if not self._check_dependencies(task):
                    # Put back in queue
                    self.task_queue.put((task.priority, time.time(), task))
                    break
                
                # Try to allocate resources
                resource = self.resource_manager.allocate_resources(task)
                if resource:
                    # Dispatch task
                    self._dispatch_task(task, resource)
                    dispatched += 1
                else:
                    # No resources available, put back in queue
                    self.task_queue.put((task.priority, time.time(), task))
                    break
                    
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error dispatching task: {e}")
    
    def _dispatch_task(self, task: BenchmarkTask, resource: ComputeResource):
        """Dispatch a task to a specific resource."""
        with self.task_lock:
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            self.running_tasks[task.task_id] = task
            
            # Execute task asynchronously
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._execute_task_on_node, task, resource)
            
            # Store future for monitoring
            task._execution_future = future
            
            logger.info(f"Dispatched task {task.task_id} to node {resource.node_id}")
    
    def _execute_task_on_node(self, task: BenchmarkTask, resource: ComputeResource) -> BenchmarkResult:
        """Execute benchmark task on assigned node."""
        try:
            logger.info(f"Executing task {task.task_id} on node {resource.node_id}")
            
            # Create benchmark suite for this execution
            suite = BenchmarkSuite(device="cuda" if resource.gpu_count > 0 else "cpu")
            
            # Execute benchmark with task parameters
            result = suite.evaluate_model(
                model_name=task.model_name,
                prompts=task.prompts,
                **task.parameters
            )
            
            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completion_time = time.time()
            
            logger.info(f"Task {task.task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completion_time = time.time()
            
            # Consider retry if within limits
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
            
            raise e
        finally:
            # Always release resources
            self.resource_manager.release_resources(task)
    
    def _check_running_tasks(self):
        """Check status of running tasks."""
        with self.task_lock:
            completed_tasks = []
            
            for task_id, task in self.running_tasks.items():
                # Check for timeout
                if task.start_time and time.time() - task.start_time > 3600:  # 1 hour timeout
                    logger.warning(f"Task {task_id} timed out")
                    task.status = TaskStatus.FAILED
                    task.error = "Task timeout"
                    completed_tasks.append(task_id)
                
                # Check if execution finished
                elif hasattr(task, '_execution_future'):
                    future = task._execution_future
                    if future.done():
                        try:
                            result = future.result()
                            if task.status != TaskStatus.COMPLETED:
                                task.status = TaskStatus.COMPLETED
                        except Exception as e:
                            if task.status != TaskStatus.FAILED:
                                task.status = TaskStatus.FAILED
                                task.error = str(e)
                        
                        completed_tasks.append(task_id)
            
            # Move completed tasks
            for task_id in completed_tasks:
                task = self.running_tasks.pop(task_id)
                
                # Handle retries
                if task.status == TaskStatus.RETRYING:
                    # Reset for retry
                    task.status = TaskStatus.PENDING
                    task.start_time = None
                    task.assigned_node = None
                    self.task_queue.put((task.priority, time.time(), task))
                else:
                    self.completed_tasks[task_id] = task
    
    def _check_dependencies(self, task: BenchmarkTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            
            dep_task = self.completed_tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks to prevent memory leaks."""
        if len(self.completed_tasks) > 1000:  # Keep last 1000 completed tasks
            with self.task_lock:
                # Sort by completion time and keep most recent
                sorted_tasks = sorted(
                    self.completed_tasks.items(),
                    key=lambda x: x[1].completion_time or 0,
                    reverse=True
                )
                
                # Keep only the most recent 1000
                keep_tasks = dict(sorted_tasks[:1000])
                self.completed_tasks = keep_tasks
                
                logger.debug("Cleaned up old completed tasks")


class DistributedBenchmarkOrchestrator:
    """Main orchestrator for distributed video diffusion benchmarking."""
    
    def __init__(self, config: ClusterConfiguration = None):
        self.config = config or ClusterConfiguration()
        self.resource_manager = ResourceManager(self.config)
        self.task_scheduler = TaskScheduler(self.resource_manager)
        self.node_manager = None
        self.results_aggregator = ResultsAggregator()
        
        # ZeroMQ contexts for distributed communication
        self.zmq_context = zmq.Context()
        self.master_socket = None
        self.is_master = True
        
    def initialize_cluster(self, is_master: bool = True) -> bool:
        """Initialize the distributed cluster."""
        self.is_master = is_master
        
        try:
            if is_master:
                # Initialize as master node
                self._setup_master_node()
                logger.info("Initialized as master node")
            else:
                # Initialize as worker node
                self._setup_worker_node()
                logger.info("Initialized as worker node")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cluster: {e}")
            return False
    
    def submit_benchmark_campaign(
        self,
        models: List[str],
        prompts: List[str],
        parameter_grid: Dict[str, List[Any]],
        priority: int = 1
    ) -> List[str]:
        """Submit a comprehensive benchmark campaign."""
        
        logger.info(f"Submitting benchmark campaign: {len(models)} models, {len(prompts)} prompts")
        
        task_ids = []
        
        # Generate all parameter combinations
        import itertools
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        for model in models:
            for param_combo in itertools.product(*param_values):
                # Create parameter dict
                parameters = dict(zip(param_names, param_combo))
                
                # Create task
                task = BenchmarkTask(
                    task_id=f"benchmark_{uuid.uuid4().hex[:8]}",
                    model_name=model,
                    prompts=prompts.copy(),
                    parameters=parameters,
                    priority=priority
                )
                
                # Submit to scheduler
                task_id = self.task_scheduler.submit_task(task)
                task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} benchmark tasks")
        return task_ids
    
    def wait_for_campaign_completion(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None
    ) -> Dict[str, BenchmarkResult]:
        """Wait for benchmark campaign to complete and return results."""
        
        logger.info(f"Waiting for completion of {len(task_ids)} tasks")
        start_time = time.time()
        
        while True:
            # Check completion status
            completed_count = 0
            results = {}
            
            for task_id in task_ids:
                status = self.task_scheduler.get_task_status(task_id)
                
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    completed_count += 1
                    
                    if status == TaskStatus.COMPLETED:
                        task = self.task_scheduler.completed_tasks.get(task_id)
                        if task and task.result:
                            results[task_id] = task.result
            
            # Check if all completed
            if completed_count == len(task_ids):
                logger.info(f"Campaign completed: {len(results)} successful results")
                return results
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                logger.warning(f"Campaign timeout: {completed_count}/{len(task_ids)} completed")
                return results
            
            # Progress update
            if completed_count % 10 == 0:
                logger.info(f"Campaign progress: {completed_count}/{len(task_ids)} completed")
            
            time.sleep(10)  # Check every 10 seconds
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cluster performance metrics."""
        cluster_status = self.resource_manager.get_cluster_status()
        queue_status = self.task_scheduler.get_queue_status()
        
        return {
            "cluster": cluster_status,
            "queue": queue_status,
            "timestamp": time.time(),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    def scale_cluster(self, target_nodes: int) -> bool:
        """Scale cluster to target number of nodes (if auto-scaling enabled)."""
        if not self.config.auto_scaling:
            logger.warning("Auto-scaling not enabled")
            return False
        
        current_nodes = len(self.resource_manager.resources)
        
        if target_nodes > current_nodes:
            # Scale up
            logger.info(f"Scaling up cluster from {current_nodes} to {target_nodes} nodes")
            # Implementation would trigger cloud instance creation
            return self._scale_up_cluster(target_nodes - current_nodes)
        elif target_nodes < current_nodes:
            # Scale down
            logger.info(f"Scaling down cluster from {current_nodes} to {target_nodes} nodes")
            return self._scale_down_cluster(current_nodes - target_nodes)
        
        return True  # No scaling needed
    
    def shutdown_cluster(self):
        """Gracefully shutdown the distributed cluster."""
        logger.info("Shutting down distributed cluster")
        
        # Stop task scheduler
        self.task_scheduler.stop_scheduler()
        
        # Close ZeroMQ sockets
        if self.master_socket:
            self.master_socket.close()
        
        if self.zmq_context:
            self.zmq_context.term()
        
        logger.info("Cluster shutdown complete")
    
    def _setup_master_node(self):
        """Setup master node for cluster coordination."""
        # Setup ZeroMQ master socket for worker communication
        self.master_socket = self.zmq_context.socket(zmq.REP)
        self.master_socket.bind(f"tcp://*:{self.config.master_port}")
        
        # Start task scheduler
        self.task_scheduler.start_scheduler()
        
        # Register local resources
        local_resource = self._detect_local_resources()
        self.resource_manager.register_node(local_resource)
        
        # Start node discovery service
        self._start_node_discovery()
        
        self._start_time = time.time()
    
    def _setup_worker_node(self):
        """Setup worker node to connect to master."""
        # Connect to master
        worker_socket = self.zmq_context.socket(zmq.REQ)
        worker_socket.connect(f"tcp://{self.config.master_address}:{self.config.master_port}")
        
        # Register with master
        local_resource = self._detect_local_resources()
        registration_msg = {
            "action": "register_node",
            "resource": local_resource.__dict__
        }
        
        worker_socket.send_json(registration_msg)
        response = worker_socket.recv_json()
        
        if response.get("status") == "success":
            logger.info("Successfully registered with master node")
        else:
            logger.error("Failed to register with master node")
    
    def _detect_local_resources(self) -> ComputeResource:
        """Detect local compute resources."""
        # Get system information
        hostname = socket.gethostname()
        cpu_cores = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detect GPUs
        gpu_count = 0
        gpu_memory_gb = []
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_memory_gb.append(int(props.total_memory / (1024**3)))
        
        return ComputeResource(
            node_id=f"{hostname}_{uuid.uuid4().hex[:8]}",
            hostname=hostname,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            available_gpus=list(range(gpu_count)),
            status="available"
        )
    
    def _start_node_discovery(self):
        """Start node discovery service for automatic cluster formation."""
        def discovery_service():
            discovery_socket = self.zmq_context.socket(zmq.REP)
            discovery_socket.bind(f"tcp://*:{self.config.node_discovery_port}")
            
            while True:
                try:
                    message = discovery_socket.recv_json(zmq.NOBLOCK)
                    if message.get("action") == "discover_master":
                        response = {
                            "master_address": socket.gethostname(),
                            "master_port": self.config.master_port
                        }
                        discovery_socket.send_json(response)
                except zmq.Again:
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Discovery service error: {e}")
        
        discovery_thread = threading.Thread(target=discovery_service, daemon=True)
        discovery_thread.start()
    
    def _scale_up_cluster(self, additional_nodes: int) -> bool:
        """Scale up cluster by adding nodes."""
        # Implementation would depend on cloud provider
        # For now, simulate successful scaling
        logger.info(f"Simulating scale up of {additional_nodes} nodes")
        return True
    
    def _scale_down_cluster(self, nodes_to_remove: int) -> bool:
        """Scale down cluster by removing nodes."""
        logger.info(f"Simulating scale down of {nodes_to_remove} nodes")
        return True


class ResultsAggregator:
    """Aggregates and analyzes results from distributed benchmark execution."""
    
    def __init__(self):
        self.results_db = {}
        self.aggregation_lock = threading.RLock()
        
    def add_result(self, task_id: str, result: BenchmarkResult):
        """Add a benchmark result."""
        with self.aggregation_lock:
            self.results_db[task_id] = result
            
    def aggregate_campaign_results(
        self, 
        task_ids: List[str],
        grouping_keys: List[str] = None
    ) -> Dict[str, Any]:
        """Aggregate results from a benchmark campaign."""
        
        if grouping_keys is None:
            grouping_keys = ["model_name"]
        
        with self.aggregation_lock:
            # Get all results for the campaign
            campaign_results = {}
            for task_id in task_ids:
                if task_id in self.results_db:
                    campaign_results[task_id] = self.results_db[task_id]
            
            if not campaign_results:
                return {}
            
            # Group results by specified keys
            grouped_results = defaultdict(list)
            for task_id, result in campaign_results.items():
                group_key = tuple(getattr(result, key, "unknown") for key in grouping_keys)
                grouped_results[group_key].append(result)
            
            # Compute aggregated metrics for each group
            aggregated = {}
            for group_key, results in grouped_results.items():
                group_name = "_".join(str(k) for k in group_key)
                aggregated[group_name] = self._aggregate_group_results(results)
            
            return aggregated
    
    def _aggregate_group_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Aggregate results within a group."""
        if not results:
            return {}
        
        # Collect metrics
        all_metrics = defaultdict(list)
        all_performance = defaultdict(list)
        
        for result in results:
            if result.metrics:
                for metric, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        all_metrics[metric].append(value)
            
            if result.performance:
                for metric, value in result.performance.items():
                    if isinstance(value, (int, float)):
                        all_performance[metric].append(value)
        
        # Compute statistics
        aggregated_metrics = {}
        for metric, values in all_metrics.items():
            aggregated_metrics[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
        
        aggregated_performance = {}
        for metric, values in all_performance.items():
            aggregated_performance[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
        
        return {
            "metrics": aggregated_metrics,
            "performance": aggregated_performance,
            "sample_count": len(results),
            "success_rate": sum(1 for r in results if r.success_rate > 0.8) / len(results)
        }


# High-level convenience functions
def create_distributed_cluster(
    master_address: str = "localhost",
    auto_discover: bool = True
) -> DistributedBenchmarkOrchestrator:
    """Create and initialize a distributed benchmark cluster."""
    
    config = ClusterConfiguration(
        master_address=master_address,
        auto_scaling=True,
        fault_tolerance=True
    )
    
    orchestrator = DistributedBenchmarkOrchestrator(config)
    
    # Auto-discover if requested
    if auto_discover:
        orchestrator.initialize_cluster(is_master=True)
    
    return orchestrator


def run_distributed_benchmark(
    models: List[str],
    prompts: List[str],
    parameter_variations: Dict[str, List[Any]] = None,
    cluster_config: ClusterConfiguration = None
) -> Dict[str, Any]:
    """Run a comprehensive distributed benchmark campaign."""
    
    if parameter_variations is None:
        parameter_variations = {
            "num_frames": [16, 24],
            "guidance_scale": [5.0, 7.5],
            "num_inference_steps": [20, 50]
        }
    
    # Create orchestrator
    orchestrator = DistributedBenchmarkOrchestrator(cluster_config)
    orchestrator.initialize_cluster(is_master=True)
    
    try:
        # Submit benchmark campaign
        task_ids = orchestrator.submit_benchmark_campaign(
            models=models,
            prompts=prompts,
            parameter_grid=parameter_variations,
            priority=1
        )
        
        # Wait for completion
        results = orchestrator.wait_for_campaign_completion(task_ids, timeout=7200)  # 2 hours
        
        # Aggregate results
        aggregated = orchestrator.results_aggregator.aggregate_campaign_results(
            task_ids, 
            grouping_keys=["model_name"]
        )
        
        return {
            "individual_results": results,
            "aggregated_results": aggregated,
            "cluster_metrics": orchestrator.get_cluster_metrics()
        }
        
    finally:
        orchestrator.shutdown_cluster()


def join_distributed_cluster(
    master_address: str,
    master_port: int = 6379
) -> bool:
    """Join an existing distributed benchmark cluster as a worker node."""
    
    config = ClusterConfiguration(
        master_address=master_address,
        master_port=master_port
    )
    
    orchestrator = DistributedBenchmarkOrchestrator(config)
    return orchestrator.initialize_cluster(is_master=False)


# Example usage
if __name__ == "__main__":
    # Example: Run distributed benchmark
    models = ["svd-xt", "cogvideo", "pika-lumiere"]
    prompts = [
        "A cat playing piano in a cozy living room",
        "Aerial view of a futuristic city at sunset",
        "Ocean waves crashing on a rocky shore"
    ]
    
    parameter_variations = {
        "num_frames": [16, 24],
        "fps": [8, 12],
        "guidance_scale": [5.0, 7.5, 10.0]
    }
    
    results = run_distributed_benchmark(
        models=models,
        prompts=prompts,
        parameter_variations=parameter_variations
    )
    
    print(f"Distributed benchmark completed with {len(results['individual_results'])} results")