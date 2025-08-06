"""Distributed computing capabilities for scalable benchmarking.

This module provides distributed processing capabilities to scale video diffusion
benchmarking across multiple nodes, GPUs, and compute resources.
"""

import asyncio
import threading
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import socket
import pickle

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as torch_mp
    TORCH_DISTRIBUTED_AVAILABLE = True
except ImportError:
    TORCH_DISTRIBUTED_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status in distributed cluster."""
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Distributed task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    capabilities: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    load_average: float = 0.0


@dataclass
class DistributedTask:
    """Distributed benchmark task."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class DistributedBenchmarkConfig:
    """Configuration for distributed benchmarking."""
    max_workers_per_node: int = 4
    task_timeout: float = 3600.0  # 1 hour
    heartbeat_interval: float = 30.0
    node_discovery_port: int = 8765
    max_retries: int = 3
    load_balancing_strategy: str = "round_robin"  # or "least_loaded", "random"
    enable_fault_tolerance: bool = True
    result_storage_path: str = "./distributed_results"


class NodeManager:
    """Manages individual compute nodes in the distributed cluster."""
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        port: int = 8766,
        config: Optional[DistributedBenchmarkConfig] = None
    ):
        """Initialize node manager.
        
        Args:
            node_id: Unique node identifier
            port: Port for node communication
            config: Distributed benchmark configuration
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.config = config or DistributedBenchmarkConfig()
        
        # Node information
        self.node_info = NodeInfo(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=port,
            status=NodeStatus.OFFLINE
        )
        
        # Task management
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.task_history: List[DistributedTask] = []
        
        # Resource monitoring
        self.resource_monitor = None
        self.heartbeat_thread = None
        self.running = False
        
        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers_per_node)
        
        logger.info(f"NodeManager initialized: {self.node_id} on {self.node_info.ip_address}:{port}")
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def start_node(self):
        """Start the node and begin accepting tasks."""
        if self.running:
            return
        
        self.running = True
        self.node_info.status = NodeStatus.ONLINE
        
        # Update node capabilities
        self._update_node_capabilities()
        
        # Start heartbeat
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        
        logger.info(f"Node {self.node_id} started and online")
    
    def stop_node(self):
        """Stop the node gracefully."""
        self.running = False
        self.node_info.status = NodeStatus.OFFLINE
        
        # Cancel active tasks
        for task in self.active_tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info(f"Node {self.node_id} stopped")
    
    def _update_node_capabilities(self):
        """Update node capabilities and resources."""
        try:
            import psutil
            
            # System resources
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            self.node_info.resources = {
                "cpu_count": cpu_count,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent
            }
            
            # GPU capabilities
            if TORCH_DISTRIBUTED_AVAILABLE and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = {}
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info[f"gpu_{i}"] = {
                        "name": props.name,
                        "memory_total_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}"
                    }
                
                self.node_info.resources["gpu_count"] = gpu_count
                self.node_info.resources["gpu_info"] = gpu_info
                self.node_info.capabilities["has_gpu"] = True
            else:
                self.node_info.capabilities["has_gpu"] = False
            
            # Benchmark-specific capabilities
            self.node_info.capabilities.update({
                "supports_video_generation": True,
                "supports_metric_computation": True,
                "max_concurrent_tasks": self.config.max_workers_per_node,
                "frameworks": ["torch", "transformers", "diffusers"]
            })
            
        except Exception as e:
            logger.error(f"Error updating node capabilities: {e}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat updates."""
        while self.running:
            try:
                self._update_load_average()
                self.node_info.last_heartbeat = datetime.now()
                
                # Update resource usage
                self._update_node_capabilities()
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.config.heartbeat_interval)
    
    def _update_load_average(self):
        """Update node load average."""
        try:
            import psutil
            
            # Calculate load based on active tasks and system resources
            active_task_count = len([t for t in self.active_tasks.values() 
                                   if t.status == TaskStatus.RUNNING])
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            
            # Weighted load calculation
            task_load = active_task_count / self.config.max_workers_per_node
            resource_load = (cpu_percent + memory_percent) / 200  # Normalize to 0-1
            
            self.node_info.load_average = (task_load * 0.6 + resource_load * 0.4)
            
        except Exception as e:
            logger.error(f"Error updating load average: {e}")
            self.node_info.load_average = 0.5  # Default moderate load
    
    def execute_task(self, task: DistributedTask) -> DistributedTask:
        """Execute a distributed task.
        
        Args:
            task: Task to execute
            
        Returns:
            Completed task with results
        """
        if task.task_id in self.active_tasks:
            logger.warning(f"Task {task.task_id} already active")
            return task
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        task.status = TaskStatus.RUNNING
        task.assigned_node = self.node_id
        task.started_at = datetime.now()
        
        logger.info(f"Starting task {task.task_id} of type {task.task_type}")
        
        try:
            # Submit task to executor
            future = self.executor.submit(self._execute_task_worker, task)
            
            # Wait for completion with timeout
            result = future.result(timeout=self.config.task_timeout)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            self.node_info.tasks_completed += 1
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = str(e)
            
            self.node_info.tasks_failed += 1
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Move to history and remove from active
            self.task_history.append(task)
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
        
        return task
    
    def _execute_task_worker(self, task: DistributedTask) -> Any:
        """Worker function to execute task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        try:
            if task.task_type == "model_evaluation":
                return self._execute_model_evaluation_task(task)
            elif task.task_type == "metric_computation":
                return self._execute_metric_computation_task(task)
            elif task.task_type == "data_preprocessing":
                return self._execute_data_preprocessing_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    def _execute_model_evaluation_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute model evaluation task.
        
        Args:
            task: Model evaluation task
            
        Returns:
            Evaluation results
        """
        params = task.parameters
        model_name = params.get("model_name")
        prompts = params.get("prompts", [])
        
        logger.info(f"Executing model evaluation: {model_name} with {len(prompts)} prompts")
        
        # Simulate model evaluation (replace with actual implementation)
        import random
        time.sleep(random.uniform(10, 30))  # Simulate processing time
        
        # Mock results
        results = {
            "model_name": model_name,
            "evaluations": [],
            "summary": {
                "total_prompts": len(prompts),
                "success_rate": random.uniform(0.8, 1.0),
                "average_time": random.uniform(5, 15)
            }
        }
        
        for i, prompt in enumerate(prompts):
            results["evaluations"].append({
                "prompt_idx": i,
                "prompt": prompt,
                "success": random.random() > 0.1,
                "generation_time": random.uniform(3, 20),
                "metrics": {
                    "fvd": random.uniform(80, 120),
                    "is": random.uniform(25, 45),
                    "clip_sim": random.uniform(0.2, 0.4)
                }
            })
        
        return results
    
    def _execute_metric_computation_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute metric computation task.
        
        Args:
            task: Metric computation task
            
        Returns:
            Computed metrics
        """
        params = task.parameters
        video_paths = params.get("video_paths", [])
        metrics_to_compute = params.get("metrics", ["fvd", "is"])
        
        logger.info(f"Computing metrics for {len(video_paths)} videos")
        
        # Simulate metric computation
        import random
        time.sleep(random.uniform(5, 15))
        
        results = {
            "metrics": {},
            "video_count": len(video_paths)
        }
        
        for metric in metrics_to_compute:
            if metric == "fvd":
                results["metrics"]["fvd"] = random.uniform(80, 150)
            elif metric == "is":
                results["metrics"]["is"] = random.uniform(20, 50)
            elif metric == "clip_sim":
                results["metrics"]["clip_sim"] = random.uniform(0.1, 0.5)
        
        return results
    
    def _execute_data_preprocessing_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute data preprocessing task.
        
        Args:
            task: Data preprocessing task
            
        Returns:
            Preprocessing results
        """
        params = task.parameters
        data_paths = params.get("data_paths", [])
        
        logger.info(f"Preprocessing {len(data_paths)} data files")
        
        # Simulate preprocessing
        import random
        time.sleep(random.uniform(2, 8))
        
        return {
            "processed_files": len(data_paths),
            "output_paths": [f"processed_{i}.npy" for i in range(len(data_paths))],
            "preprocessing_time": random.uniform(1, 10)
        }
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get current node status.
        
        Returns:
            Node status dictionary
        """
        return {
            "node_info": asdict(self.node_info),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len([t for t in self.task_history 
                                  if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.task_history 
                               if t.status == TaskStatus.FAILED]),
            "running": self.running
        }


class ClusterManager:
    """Manages a cluster of distributed nodes."""
    
    def __init__(self, config: Optional[DistributedBenchmarkConfig] = None):
        """Initialize cluster manager.
        
        Args:
            config: Distributed benchmark configuration
        """
        self.config = config or DistributedBenchmarkConfig()
        self.nodes: Dict[str, NodeInfo] = {}
        self.pending_tasks: List[DistributedTask] = []
        self.completed_tasks: List[DistributedTask] = []
        
        # Cluster state
        self.running = False
        self.task_distributor = None
        self.management_thread = None
        
        logger.info("ClusterManager initialized")
    
    def register_node(self, node_info: NodeInfo):
        """Register a new node in the cluster.
        
        Args:
            node_info: Node information
        """
        self.nodes[node_info.node_id] = node_info
        logger.info(f"Registered node {node_info.node_id} ({node_info.hostname})")
    
    def unregister_node(self, node_id: str):
        """Unregister a node from the cluster.
        
        Args:
            node_id: Node ID to unregister
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Unregistered node {node_id}")
    
    def start_cluster(self):
        """Start cluster management."""
        if self.running:
            return
        
        self.running = True
        self.task_distributor = TaskDistributor(self, self.config)
        
        # Start management thread
        self.management_thread = threading.Thread(
            target=self._cluster_management_loop, daemon=True
        )
        self.management_thread.start()
        
        logger.info("Cluster management started")
    
    def stop_cluster(self):
        """Stop cluster management."""
        self.running = False
        
        if self.task_distributor:
            self.task_distributor.stop()
        
        logger.info("Cluster management stopped")
    
    def _cluster_management_loop(self):
        """Main cluster management loop."""
        while self.running:
            try:
                # Check node health
                self._check_node_health()
                
                # Distribute pending tasks
                if self.task_distributor:
                    self.task_distributor.distribute_tasks(self.pending_tasks)
                    self.pending_tasks.clear()
                
                time.sleep(10)  # Management loop interval
                
            except Exception as e:
                logger.error(f"Error in cluster management loop: {e}")
                time.sleep(10)
    
    def _check_node_health(self):
        """Check health of all registered nodes."""
        current_time = datetime.now()
        unhealthy_nodes = []
        
        for node_id, node_info in self.nodes.items():
            if node_info.last_heartbeat:
                time_since_heartbeat = current_time - node_info.last_heartbeat
                
                # Mark nodes as offline if no heartbeat for too long
                if time_since_heartbeat.total_seconds() > self.config.heartbeat_interval * 3:
                    if node_info.status != NodeStatus.OFFLINE:
                        node_info.status = NodeStatus.OFFLINE
                        logger.warning(f"Node {node_id} marked as offline (no heartbeat)")
                        unhealthy_nodes.append(node_id)
        
        # Handle unhealthy nodes
        for node_id in unhealthy_nodes:
            self._handle_unhealthy_node(node_id)
    
    def _handle_unhealthy_node(self, node_id: str):
        """Handle unhealthy node.
        
        Args:
            node_id: ID of unhealthy node
        """
        logger.info(f"Handling unhealthy node: {node_id}")
        
        if self.config.enable_fault_tolerance:
            # Reschedule tasks from unhealthy node
            # (This would require tracking which tasks were assigned to which nodes)
            pass
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        self.pending_tasks.append(task)
        logger.info(f"Submitted task {task.task_id} of type {task.task_type}")
        return task.task_id
    
    def submit_benchmark_job(
        self,
        models: List[str],
        prompts: List[str],
        metrics: List[str]
    ) -> List[str]:
        """Submit complete benchmark job as distributed tasks.
        
        Args:
            models: Models to evaluate
            prompts: Prompts to use
            metrics: Metrics to compute
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        # Create model evaluation tasks
        for model in models:
            task = DistributedTask(
                task_id=str(uuid.uuid4()),
                task_type="model_evaluation",
                parameters={
                    "model_name": model,
                    "prompts": prompts,
                    "metrics": metrics
                },
                priority=1
            )
            task_ids.append(self.submit_task(task))
        
        logger.info(f"Submitted benchmark job with {len(task_ids)} tasks")
        return task_ids
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status.
        
        Returns:
            Cluster status dictionary
        """
        online_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        busy_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.BUSY]
        offline_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.OFFLINE]
        
        return {
            "total_nodes": len(self.nodes),
            "online_nodes": len(online_nodes),
            "busy_nodes": len(busy_nodes),
            "offline_nodes": len(offline_nodes),
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "cluster_capacity": sum(n.capabilities.get("max_concurrent_tasks", 1) 
                                  for n in online_nodes),
            "average_load": sum(n.load_average for n in online_nodes) / len(online_nodes) 
                           if online_nodes else 0.0
        }


class TaskDistributor:
    """Distributes tasks across cluster nodes."""
    
    def __init__(self, cluster_manager: ClusterManager, config: DistributedBenchmarkConfig):
        """Initialize task distributor.
        
        Args:
            cluster_manager: Cluster manager instance
            config: Configuration
        """
        self.cluster_manager = cluster_manager
        self.config = config
        self.running = True
        
        logger.info("TaskDistributor initialized")
    
    def distribute_tasks(self, tasks: List[DistributedTask]):
        """Distribute tasks to available nodes.
        
        Args:
            tasks: Tasks to distribute
        """
        if not tasks:
            return
        
        available_nodes = self._get_available_nodes()
        if not available_nodes:
            logger.warning("No available nodes for task distribution")
            return
        
        # Sort tasks by priority (higher priority first)
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Distribute tasks based on load balancing strategy
        for task in tasks:
            node = self._select_node(available_nodes, task)
            if node:
                self._assign_task_to_node(task, node)
                logger.info(f"Assigned task {task.task_id} to node {node.node_id}")
            else:
                logger.warning(f"Could not assign task {task.task_id} - no suitable nodes")
    
    def _get_available_nodes(self) -> List[NodeInfo]:
        """Get list of available nodes.
        
        Returns:
            List of available nodes
        """
        return [
            node for node in self.cluster_manager.nodes.values()
            if node.status in [NodeStatus.ONLINE, NodeStatus.BUSY] and
            self._can_accept_task(node)
        ]
    
    def _can_accept_task(self, node: NodeInfo) -> bool:
        """Check if node can accept new tasks.
        
        Args:
            node: Node to check
            
        Returns:
            True if node can accept tasks
        """
        max_tasks = node.capabilities.get("max_concurrent_tasks", 1)
        current_load = node.load_average
        
        # Node can accept task if not at maximum load
        return current_load < 0.9  # 90% load threshold
    
    def _select_node(self, available_nodes: List[NodeInfo], task: DistributedTask) -> Optional[NodeInfo]:
        """Select best node for task based on load balancing strategy.
        
        Args:
            available_nodes: Available nodes
            task: Task to assign
            
        Returns:
            Selected node or None
        """
        if not available_nodes:
            return None
        
        strategy = self.config.load_balancing_strategy
        
        if strategy == "round_robin":
            # Simple round-robin selection
            return available_nodes[0]
        
        elif strategy == "least_loaded":
            # Select node with lowest load
            return min(available_nodes, key=lambda n: n.load_average)
        
        elif strategy == "random":
            # Random selection
            import random
            return random.choice(available_nodes)
        
        elif strategy == "capability_based":
            # Select based on task requirements and node capabilities
            return self._select_by_capability(available_nodes, task)
        
        else:
            # Default to least loaded
            return min(available_nodes, key=lambda n: n.load_average)
    
    def _select_by_capability(
        self, 
        available_nodes: List[NodeInfo], 
        task: DistributedTask
    ) -> Optional[NodeInfo]:
        """Select node based on capabilities.
        
        Args:
            available_nodes: Available nodes
            task: Task to assign
            
        Returns:
            Best node for task
        """
        # Filter nodes by required capabilities
        suitable_nodes = []
        
        for node in available_nodes:
            if task.task_type == "model_evaluation":
                # Prefer nodes with GPUs for model evaluation
                if node.capabilities.get("has_gpu", False):
                    suitable_nodes.append(node)
            elif task.task_type == "metric_computation":
                # Metric computation can run on any node
                suitable_nodes.append(node)
            else:
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            suitable_nodes = available_nodes
        
        # Select least loaded among suitable nodes
        return min(suitable_nodes, key=lambda n: n.load_average)
    
    def _assign_task_to_node(self, task: DistributedTask, node: NodeInfo):
        """Assign task to specific node.
        
        Args:
            task: Task to assign
            node: Target node
        """
        task.assigned_node = node.node_id
        task.status = TaskStatus.RUNNING
        
        # Update node status
        node.status = NodeStatus.BUSY
        
        # In a real implementation, this would send the task to the node
        # For now, we'll simulate this
        logger.info(f"Task {task.task_id} assigned to node {node.node_id}")
    
    def stop(self):
        """Stop task distributor."""
        self.running = False
        logger.info("TaskDistributor stopped")


class DistributedBenchmarkRunner:
    """Main distributed benchmark runner."""
    
    def __init__(self, config: Optional[DistributedBenchmarkConfig] = None):
        """Initialize distributed benchmark runner.
        
        Args:
            config: Configuration for distributed benchmarking
        """
        self.config = config or DistributedBenchmarkConfig()
        self.cluster_manager = ClusterManager(self.config)
        self.local_node = None
        
        # Results storage
        self.result_storage = Path(self.config.result_storage_path)
        self.result_storage.mkdir(parents=True, exist_ok=True)
        
        logger.info("DistributedBenchmarkRunner initialized")
    
    def start_local_node(self, port: int = 8766) -> NodeManager:
        """Start a local node for distributed computing.
        
        Args:
            port: Port for local node
            
        Returns:
            Local node manager
        """
        self.local_node = NodeManager(port=port, config=self.config)
        self.local_node.start_node()
        
        # Register with cluster
        self.cluster_manager.register_node(self.local_node.node_info)
        
        return self.local_node
    
    def start_cluster(self):
        """Start the distributed cluster."""
        self.cluster_manager.start_cluster()
        logger.info("Distributed cluster started")
    
    def stop_cluster(self):
        """Stop the distributed cluster."""
        if self.local_node:
            self.local_node.stop_node()
        
        self.cluster_manager.stop_cluster()
        logger.info("Distributed cluster stopped")
    
    def run_distributed_benchmark(
        self,
        models: List[str],
        prompts: List[str],
        metrics: List[str] = ["fvd", "is", "clip_similarity"]
    ) -> Dict[str, Any]:
        """Run distributed benchmark across cluster.
        
        Args:
            models: Models to evaluate
            prompts: Prompts for evaluation
            metrics: Metrics to compute
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting distributed benchmark: {len(models)} models, {len(prompts)} prompts")
        
        # Submit benchmark job
        task_ids = self.cluster_manager.submit_benchmark_job(models, prompts, metrics)
        
        # Wait for completion
        start_time = time.time()
        completed_results = {}
        
        while len(completed_results) < len(task_ids):
            # Check for completed tasks
            for task in self.cluster_manager.completed_tasks:
                if task.task_id not in completed_results:
                    completed_results[task.task_id] = task
            
            # Check timeout
            if time.time() - start_time > self.config.task_timeout * len(models):
                logger.error("Distributed benchmark timed out")
                break
            
            time.sleep(5)  # Check every 5 seconds
        
        # Compile results
        benchmark_results = {
            "total_tasks": len(task_ids),
            "completed_tasks": len(completed_results),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": time.time() - start_time,
            "results_by_model": {},
            "cluster_stats": self.cluster_manager.get_cluster_status()
        }
        
        # Organize results by model
        for task in completed_results.values():
            if task.status == TaskStatus.COMPLETED and task.result:
                model_name = task.result.get("model_name")
                if model_name:
                    benchmark_results["results_by_model"][model_name] = task.result
        
        # Save results
        self._save_benchmark_results(benchmark_results)
        
        logger.info(f"Distributed benchmark completed: {len(completed_results)}/{len(task_ids)} tasks")
        return benchmark_results
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to storage.
        
        Args:
            results: Benchmark results to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.result_storage / f"distributed_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cluster metrics.
        
        Returns:
            Cluster metrics dictionary
        """
        cluster_status = self.cluster_manager.get_cluster_status()
        
        # Add node details
        node_details = {}
        for node_id, node_info in self.cluster_manager.nodes.items():
            node_details[node_id] = {
                "hostname": node_info.hostname,
                "status": node_info.status.value,
                "load_average": node_info.load_average,
                "tasks_completed": node_info.tasks_completed,
                "tasks_failed": node_info.tasks_failed,
                "resources": node_info.resources,
                "capabilities": node_info.capabilities
            }
        
        return {
            "cluster_status": cluster_status,
            "node_details": node_details,
            "timestamp": datetime.now().isoformat()
        }


# Ray-based distributed implementation (if available)
if RAY_AVAILABLE:
    @ray.remote
    class RayBenchmarkWorker:
        """Ray-based benchmark worker."""
        
        def __init__(self):
            self.model_cache = {}
        
        def evaluate_model(self, model_name: str, prompts: List[str]) -> Dict[str, Any]:
            """Evaluate model using Ray.
            
            Args:
                model_name: Model to evaluate
                prompts: Prompts for evaluation
                
            Returns:
                Evaluation results
            """
            # Simulate model evaluation
            import random
            import time
            
            time.sleep(random.uniform(5, 20))
            
            results = {
                "model_name": model_name,
                "evaluations": [],
                "worker_id": ray.get_runtime_context().worker.worker_id.hex()
            }
            
            for i, prompt in enumerate(prompts):
                results["evaluations"].append({
                    "prompt_idx": i,
                    "prompt": prompt,
                    "metrics": {
                        "fvd": random.uniform(80, 120),
                        "is": random.uniform(25, 45)
                    }
                })
            
            return results
    
    class RayDistributedRunner:
        """Ray-based distributed benchmark runner."""
        
        def __init__(self, num_workers: int = 4):
            """Initialize Ray distributed runner.
            
            Args:
                num_workers: Number of Ray workers
            """
            self.num_workers = num_workers
            self.workers = None
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init()
            
            logger.info(f"RayDistributedRunner initialized with {num_workers} workers")
        
        def start_workers(self):
            """Start Ray workers."""
            self.workers = [RayBenchmarkWorker.remote() for _ in range(self.num_workers)]
            logger.info(f"Started {len(self.workers)} Ray workers")
        
        def run_distributed_benchmark(
            self,
            models: List[str],
            prompts: List[str]
        ) -> Dict[str, Any]:
            """Run distributed benchmark using Ray.
            
            Args:
                models: Models to evaluate
                prompts: Prompts for evaluation
                
            Returns:
                Benchmark results
            """
            if not self.workers:
                self.start_workers()
            
            logger.info(f"Running Ray distributed benchmark: {len(models)} models")
            
            # Distribute tasks across workers
            futures = []
            for i, model in enumerate(models):
                worker_idx = i % len(self.workers)
                future = self.workers[worker_idx].evaluate_model.remote(model, prompts)
                futures.append(future)
            
            # Collect results
            results = ray.get(futures)
            
            # Compile final results
            benchmark_results = {
                "total_models": len(models),
                "total_prompts": len(prompts),
                "results_by_model": {},
                "ray_cluster_resources": ray.cluster_resources()
            }
            
            for result in results:
                model_name = result["model_name"]
                benchmark_results["results_by_model"][model_name] = result
            
            logger.info("Ray distributed benchmark completed")
            return benchmark_results
        
        def shutdown(self):
            """Shutdown Ray workers."""
            if self.workers:
                # Shutdown workers gracefully
                ray.shutdown()
            logger.info("Ray workers shut down")


# Factory function for creating distributed runners
def create_distributed_runner(
    backend: str = "native",
    config: Optional[DistributedBenchmarkConfig] = None,
    **kwargs
) -> Union[DistributedBenchmarkRunner, 'RayDistributedRunner']:
    """Create distributed benchmark runner.
    
    Args:
        backend: Backend to use ('native' or 'ray')
        config: Configuration
        **kwargs: Additional arguments
        
    Returns:
        Distributed runner instance
    """
    if backend == "ray" and RAY_AVAILABLE:
        num_workers = kwargs.get("num_workers", 4)
        return RayDistributedRunner(num_workers)
    else:
        return DistributedBenchmarkRunner(config)


# Global distributed runner instance
_global_distributed_runner = None


def get_distributed_runner() -> DistributedBenchmarkRunner:
    """Get global distributed runner instance."""
    global _global_distributed_runner
    if _global_distributed_runner is None:
        _global_distributed_runner = DistributedBenchmarkRunner()
    return _global_distributed_runner