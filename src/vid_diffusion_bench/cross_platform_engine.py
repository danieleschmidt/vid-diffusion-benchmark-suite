"""Cross-platform execution engine for video diffusion benchmarking.

Universal compatibility system that ensures consistent performance across
different operating systems, hardware architectures, and cloud platforms.
"""

import asyncio
import os
import sys
import platform
import subprocess
import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Supported platform types."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD_AWS = "aws"
    CLOUD_GCP = "gcp"
    CLOUD_AZURE = "azure"


class ArchitectureType(Enum):
    """Supported CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    POWER = "power"
    RISC_V = "riscv"


class AcceleratorType(Enum):
    """Supported hardware accelerators."""
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    INTEL_GPU = "intel_gpu"
    APPLE_SILICON = "apple_silicon"
    TPU = "tpu"
    FPGA = "fpga"
    NEURAL_ENGINE = "neural_engine"


@dataclass
class PlatformCapabilities:
    """Platform capability assessment."""
    platform_type: PlatformType
    architecture: ArchitectureType
    accelerators: List[AcceleratorType]
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    network_bandwidth_mbps: float
    container_support: bool
    gpu_memory_gb: float = 0.0
    performance_score: float = 1.0
    limitations: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)


@dataclass
class CrossPlatformConfig:
    """Configuration for cross-platform execution."""
    target_platforms: List[PlatformType]
    compatibility_mode: str = "strict"  # strict, permissive, fallback
    performance_priority: str = "balanced"  # speed, memory, compatibility
    auto_optimization: bool = True
    fallback_strategies: List[str] = field(default_factory=list)
    platform_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class PlatformDetector:
    """Advanced platform detection and capability assessment."""
    
    def __init__(self):
        self.platform_cache = {}
        self._detection_cache_ttl = 300  # 5 minutes
        
    def detect_platform(self) -> PlatformCapabilities:
        """Detect current platform and its capabilities."""
        
        cache_key = "current_platform"
        now = time.time()
        
        if (cache_key in self.platform_cache and 
            now - self.platform_cache[cache_key]["timestamp"] < self._detection_cache_ttl):
            return self.platform_cache[cache_key]["capabilities"]
            
        # Detect platform type
        platform_type = self._detect_platform_type()
        
        # Detect architecture
        architecture = self._detect_architecture()
        
        # Detect accelerators
        accelerators = self._detect_accelerators()
        
        # Detect system resources
        cpu_cores = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        storage_gb = self._detect_storage_capacity()
        network_bandwidth = self._estimate_network_bandwidth()
        
        # Detect container support
        container_support = self._detect_container_support()
        
        # Detect GPU memory
        gpu_memory_gb = self._detect_gpu_memory()
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            cpu_cores, memory_gb, accelerators, gpu_memory_gb
        )
        
        # Identify limitations and optimizations
        limitations = self._identify_limitations(platform_type, architecture, accelerators)
        optimizations = self._suggest_optimizations(platform_type, architecture, accelerators)
        
        capabilities = PlatformCapabilities(
            platform_type=platform_type,
            architecture=architecture,
            accelerators=accelerators,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            storage_gb=storage_gb,
            network_bandwidth_mbps=network_bandwidth,
            container_support=container_support,
            gpu_memory_gb=gpu_memory_gb,
            performance_score=performance_score,
            limitations=limitations,
            optimizations=optimizations
        )
        
        # Cache result
        self.platform_cache[cache_key] = {
            "capabilities": capabilities,
            "timestamp": now
        }
        
        return capabilities
        
    def _detect_platform_type(self) -> PlatformType:
        """Detect the platform type."""
        
        # Check for container environments
        if os.path.exists("/.dockerenv") or os.getenv("container"):
            return PlatformType.DOCKER
            
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            return PlatformType.KUBERNETES
            
        # Check for cloud platforms
        try:
            # AWS metadata service
            import urllib.request
            urllib.request.urlopen("http://169.254.169.254/latest/meta-data/", timeout=1)
            return PlatformType.CLOUD_AWS
        except:
            pass
            
        try:
            # GCP metadata service
            urllib.request.urlopen("http://metadata.google.internal/", timeout=1)
            return PlatformType.CLOUD_GCP
        except:
            pass
            
        # OS detection
        system = platform.system().lower()
        if system == "linux":
            return PlatformType.LINUX
        elif system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        else:
            return PlatformType.LINUX  # Default fallback
            
    def _detect_architecture(self) -> ArchitectureType:
        """Detect CPU architecture."""
        
        machine = platform.machine().lower()
        
        if machine in ["x86_64", "amd64"]:
            return ArchitectureType.X86_64
        elif machine in ["arm64", "aarch64"]:
            return ArchitectureType.ARM64
        elif machine.startswith("arm"):
            return ArchitectureType.ARM32
        elif machine.startswith("power"):
            return ArchitectureType.POWER
        elif machine.startswith("riscv"):
            return ArchitectureType.RISC_V
        else:
            return ArchitectureType.X86_64  # Default fallback
            
    def _detect_accelerators(self) -> List[AcceleratorType]:
        """Detect available hardware accelerators."""
        
        accelerators = []
        
        # NVIDIA GPU detection
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                accelerators.append(AcceleratorType.NVIDIA_GPU)
        except:
            pass
            
        # AMD GPU detection
        try:
            result = subprocess.run(
                ["rocm-smi", "-l"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                accelerators.append(AcceleratorType.AMD_GPU)
        except:
            pass
            
        # Apple Silicon detection
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            accelerators.append(AcceleratorType.APPLE_SILICON)
            accelerators.append(AcceleratorType.NEURAL_ENGINE)
            
        # Intel GPU detection (Linux)
        if os.path.exists("/dev/dri"):
            try:
                result = subprocess.run(
                    ["lspci"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if "Intel" in result.stdout and ("VGA" in result.stdout or "3D" in result.stdout):
                    accelerators.append(AcceleratorType.INTEL_GPU)
            except:
                pass
                
        # TPU detection (Google Cloud)
        if os.path.exists("/dev/accel0"):
            accelerators.append(AcceleratorType.TPU)
            
        return accelerators
        
    def _detect_storage_capacity(self) -> float:
        """Detect available storage capacity in GB."""
        
        try:
            if platform.system() == "Windows":
                import shutil
                _, _, free_bytes = shutil.disk_usage("C:\\")
            else:
                statvfs = os.statvfs("/")
                free_bytes = statvfs.f_frsize * statvfs.f_bavail
                
            return free_bytes / (1024**3)
        except:
            return 100.0  # Default fallback
            
    def _estimate_network_bandwidth(self) -> float:
        """Estimate network bandwidth in Mbps."""
        
        # Simple bandwidth estimation based on platform
        platform_bandwidth = {
            PlatformType.CLOUD_AWS: 10000,  # 10 Gbps
            PlatformType.CLOUD_GCP: 10000,
            PlatformType.CLOUD_AZURE: 10000,
            PlatformType.KUBERNETES: 1000,  # 1 Gbps
            PlatformType.DOCKER: 1000,
            PlatformType.LINUX: 100,       # 100 Mbps
            PlatformType.WINDOWS: 100,
            PlatformType.MACOS: 100
        }
        
        platform_type = self._detect_platform_type()
        return platform_bandwidth.get(platform_type, 100)
        
    def _detect_container_support(self) -> bool:
        """Detect if container runtime is available."""
        
        try:
            # Check for Docker
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                timeout=5
            )
            if result.returncode == 0:
                return True
        except:
            pass
            
        try:
            # Check for Podman
            result = subprocess.run(
                ["podman", "--version"], 
                capture_output=True, 
                timeout=5
            )
            if result.returncode == 0:
                return True
        except:
            pass
            
        return False
        
    def _detect_gpu_memory(self) -> float:
        """Detect GPU memory in GB."""
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                memory_mb = int(result.stdout.strip().split('\n')[0])
                return memory_mb / 1024
        except:
            pass
            
        return 0.0
        
    def _calculate_performance_score(
        self,
        cpu_cores: int,
        memory_gb: float,
        accelerators: List[AcceleratorType],
        gpu_memory_gb: float
    ) -> float:
        """Calculate overall performance score."""
        
        # Base score from CPU and memory
        base_score = min(1.0, (cpu_cores / 16) * 0.5 + (memory_gb / 32) * 0.5)
        
        # Accelerator bonus
        accelerator_bonus = 0.0
        if AcceleratorType.NVIDIA_GPU in accelerators:
            accelerator_bonus += 1.0 + (gpu_memory_gb / 24) * 0.5
        elif AcceleratorType.APPLE_SILICON in accelerators:
            accelerator_bonus += 0.8
        elif AcceleratorType.AMD_GPU in accelerators:
            accelerator_bonus += 0.7
        elif AcceleratorType.TPU in accelerators:
            accelerator_bonus += 1.5
            
        return min(3.0, base_score + accelerator_bonus)
        
    def _identify_limitations(
        self,
        platform_type: PlatformType,
        architecture: ArchitectureType,
        accelerators: List[AcceleratorType]
    ) -> List[str]:
        """Identify platform limitations."""
        
        limitations = []
        
        # Architecture limitations
        if architecture == ArchitectureType.ARM32:
            limitations.append("Limited memory support on ARM32")
            limitations.append("Reduced performance for large models")
            
        # Platform limitations
        if platform_type == PlatformType.WINDOWS:
            limitations.append("Limited container optimization on Windows")
            
        if not accelerators:
            limitations.append("No hardware acceleration available")
            
        if platform_type == PlatformType.MACOS and AcceleratorType.NVIDIA_GPU in accelerators:
            limitations.append("Limited NVIDIA CUDA support on macOS")
            
        return limitations
        
    def _suggest_optimizations(
        self,
        platform_type: PlatformType,
        architecture: ArchitectureType,
        accelerators: List[AcceleratorType]
    ) -> List[str]:
        """Suggest platform-specific optimizations."""
        
        optimizations = []
        
        # GPU optimizations
        if AcceleratorType.NVIDIA_GPU in accelerators:
            optimizations.append("Enable CUDA mixed precision training")
            optimizations.append("Use TensorRT for inference optimization")
            
        if AcceleratorType.APPLE_SILICON in accelerators:
            optimizations.append("Use Metal Performance Shaders")
            optimizations.append("Enable unified memory optimization")
            
        # Platform optimizations
        if platform_type in [PlatformType.CLOUD_AWS, PlatformType.CLOUD_GCP]:
            optimizations.append("Use cloud-native storage optimization")
            optimizations.append("Enable auto-scaling for variable workloads")
            
        if platform_type == PlatformType.LINUX:
            optimizations.append("Use NUMA-aware memory allocation")
            optimizations.append("Enable transparent huge pages")
            
        return optimizations


class CrossPlatformExecutionEngine:
    """Cross-platform execution engine with adaptive optimization."""
    
    def __init__(self, config: CrossPlatformConfig):
        self.config = config
        self.detector = PlatformDetector()
        self.platform_capabilities = self.detector.detect_platform()
        
        # Execution adapters for different platforms
        self.execution_adapters = self._initialize_adapters()
        
        # Performance monitoring
        self.performance_history = []
        self._monitoring_lock = threading.Lock()
        
    def _initialize_adapters(self) -> Dict[PlatformType, Callable]:
        """Initialize platform-specific execution adapters."""
        
        return {
            PlatformType.LINUX: self._execute_linux,
            PlatformType.WINDOWS: self._execute_windows,
            PlatformType.MACOS: self._execute_macos,
            PlatformType.DOCKER: self._execute_docker,
            PlatformType.KUBERNETES: self._execute_kubernetes,
            PlatformType.CLOUD_AWS: self._execute_aws,
            PlatformType.CLOUD_GCP: self._execute_gcp,
            PlatformType.CLOUD_AZURE: self._execute_azure
        }
        
    async def execute_benchmark(
        self,
        benchmark_config: Dict[str, Any],
        optimization_level: str = "auto"
    ) -> Dict[str, Any]:
        """Execute benchmark with cross-platform optimization."""
        
        start_time = time.time()
        
        # Optimize configuration for current platform
        optimized_config = await self._optimize_config_for_platform(
            benchmark_config, 
            optimization_level
        )
        
        # Select appropriate execution adapter
        platform_type = self.platform_capabilities.platform_type
        if platform_type in self.execution_adapters:
            adapter = self.execution_adapters[platform_type]
        else:
            # Fallback to Linux adapter
            adapter = self.execution_adapters[PlatformType.LINUX]
            
        try:
            # Execute benchmark
            result = await adapter(optimized_config)
            
            # Record performance metrics
            execution_time = time.time() - start_time
            await self._record_performance_metrics(
                platform_type, 
                optimization_level, 
                execution_time, 
                result
            )
            
            return {
                "status": "success",
                "platform": platform_type.value,
                "execution_time": execution_time,
                "optimization_level": optimization_level,
                "result": result,
                "platform_capabilities": self.platform_capabilities
            }
            
        except Exception as e:
            logger.error(f"Benchmark execution failed on {platform_type.value}: {e}")
            
            # Attempt fallback execution
            if self.config.compatibility_mode in ["permissive", "fallback"]:
                return await self._execute_fallback(benchmark_config, e)
            else:
                raise
                
    async def _optimize_config_for_platform(
        self,
        config: Dict[str, Any],
        optimization_level: str
    ) -> Dict[str, Any]:
        """Optimize configuration for current platform."""
        
        optimized_config = config.copy()
        platform_type = self.platform_capabilities.platform_type
        
        # Platform-specific optimizations
        if platform_type == PlatformType.MACOS and AcceleratorType.APPLE_SILICON in self.platform_capabilities.accelerators:
            # Apple Silicon optimizations
            optimized_config["device"] = "mps"  # Metal Performance Shaders
            optimized_config["mixed_precision"] = True
            optimized_config["memory_format"] = "channels_last"
            
        elif AcceleratorType.NVIDIA_GPU in self.platform_capabilities.accelerators:
            # NVIDIA GPU optimizations
            optimized_config["device"] = "cuda"
            optimized_config["mixed_precision"] = True
            optimized_config["compile_mode"] = "default"
            
            # Adjust batch size based on GPU memory
            if self.platform_capabilities.gpu_memory_gb >= 24:
                optimized_config["batch_size"] = min(config.get("batch_size", 1) * 2, 8)
            elif self.platform_capabilities.gpu_memory_gb < 8:
                optimized_config["batch_size"] = 1
                
        else:
            # CPU-only optimizations
            optimized_config["device"] = "cpu"
            optimized_config["num_threads"] = min(self.platform_capabilities.cpu_cores, 8)
            optimized_config["batch_size"] = 1
            
        # Memory optimizations
        if self.platform_capabilities.memory_gb < 16:
            optimized_config["low_memory_mode"] = True
            optimized_config["offload_to_disk"] = True
            
        # Optimization level adjustments
        if optimization_level == "speed":
            optimized_config["precision"] = "fp16"
            optimized_config["fast_math"] = True
        elif optimization_level == "memory":
            optimized_config["gradient_checkpointing"] = True
            optimized_config["cpu_offload"] = True
        elif optimization_level == "compatibility":
            optimized_config["precision"] = "fp32"
            optimized_config["fast_math"] = False
            
        return optimized_config
        
    async def _execute_linux(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark on Linux platform."""
        
        # Linux-specific optimizations
        if "NUMA_NODES" in os.environ:
            config["numa_aware"] = True
            
        # Use high-performance timer
        config["timer_precision"] = "high"
        
        return await self._execute_generic_benchmark(config)
        
    async def _execute_windows(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark on Windows platform."""
        
        # Windows-specific optimizations
        config["timer_precision"] = "medium"
        config["process_priority"] = "high"
        
        return await self._execute_generic_benchmark(config)
        
    async def _execute_macos(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark on macOS platform."""
        
        # macOS-specific optimizations
        if AcceleratorType.APPLE_SILICON in self.platform_capabilities.accelerators:
            config["unified_memory"] = True
            config["metal_optimization"] = True
            
        return await self._execute_generic_benchmark(config)
        
    async def _execute_docker(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark in Docker container."""
        
        # Container-specific optimizations
        config["container_optimized"] = True
        config["shared_memory_size"] = "2g"
        
        return await self._execute_generic_benchmark(config)
        
    async def _execute_kubernetes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark on Kubernetes cluster."""
        
        # Kubernetes-specific optimizations
        config["distributed_execution"] = True
        config["pod_affinity"] = "gpu"
        
        return await self._execute_generic_benchmark(config)
        
    async def _execute_aws(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark on AWS platform."""
        
        # AWS-specific optimizations
        config["instance_storage_optimization"] = True
        config["network_optimization"] = "enhanced"
        
        return await self._execute_generic_benchmark(config)
        
    async def _execute_gcp(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark on Google Cloud Platform."""
        
        # GCP-specific optimizations
        config["preemptible_instances"] = False
        config["persistent_disk_optimization"] = True
        
        return await self._execute_generic_benchmark(config)
        
    async def _execute_azure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark on Azure platform."""
        
        # Azure-specific optimizations
        config["spot_instances"] = False
        config["premium_storage"] = True
        
        return await self._execute_generic_benchmark(config)
        
    async def _execute_generic_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic benchmark execution logic."""
        
        # Simulate benchmark execution
        await asyncio.sleep(0.1)  # Simulated execution time
        
        return {
            "model_performance": {
                "inference_time": 1.5,
                "memory_usage": 8.2,
                "gpu_utilization": 0.85
            },
            "system_metrics": {
                "cpu_usage": 0.75,
                "memory_usage": 0.60,
                "disk_io": 250
            },
            "optimization_applied": list(config.keys()),
            "platform_score": self.platform_capabilities.performance_score
        }
        
    async def _execute_fallback(
        self,
        original_config: Dict[str, Any],
        original_error: Exception
    ) -> Dict[str, Any]:
        """Execute fallback strategy when primary execution fails."""
        
        logger.warning(f"Executing fallback strategy due to: {original_error}")
        
        # Create conservative fallback configuration
        fallback_config = {
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 1,
            "low_memory_mode": True,
            "fast_math": False
        }
        
        try:
            result = await self._execute_generic_benchmark(fallback_config)
            return {
                "status": "fallback_success",
                "original_error": str(original_error),
                "fallback_config": fallback_config,
                "result": result
            }
        except Exception as fallback_error:
            return {
                "status": "complete_failure",
                "original_error": str(original_error),
                "fallback_error": str(fallback_error)
            }
            
    async def _record_performance_metrics(
        self,
        platform_type: PlatformType,
        optimization_level: str,
        execution_time: float,
        result: Dict[str, Any]
    ):
        """Record performance metrics for analysis."""
        
        with self._monitoring_lock:
            self.performance_history.append({
                "timestamp": time.time(),
                "platform": platform_type.value,
                "optimization_level": optimization_level,
                "execution_time": execution_time,
                "performance_score": result.get("platform_score", 0),
                "memory_usage": result.get("system_metrics", {}).get("memory_usage", 0)
            })
            
            # Keep history manageable
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
                
    def get_platform_compatibility_report(self) -> Dict[str, Any]:
        """Generate platform compatibility report."""
        
        report = {
            "current_platform": {
                "type": self.platform_capabilities.platform_type.value,
                "architecture": self.platform_capabilities.architecture.value,
                "accelerators": [acc.value for acc in self.platform_capabilities.accelerators],
                "performance_score": self.platform_capabilities.performance_score
            },
            "compatibility_matrix": {},
            "optimization_recommendations": self.platform_capabilities.optimizations,
            "known_limitations": self.platform_capabilities.limitations,
            "cross_platform_features": {
                "container_support": self.platform_capabilities.container_support,
                "gpu_acceleration": len(self.platform_capabilities.accelerators) > 0,
                "distributed_execution": self.platform_capabilities.platform_type in [
                    PlatformType.KUBERNETES, PlatformType.CLOUD_AWS, 
                    PlatformType.CLOUD_GCP, PlatformType.CLOUD_AZURE
                ]
            }
        }
        
        # Test compatibility with each supported platform
        for platform in PlatformType:
            compatibility_score = self._calculate_compatibility_score(platform)
            report["compatibility_matrix"][platform.value] = {
                "score": compatibility_score,
                "status": "full" if compatibility_score > 0.9 else "partial" if compatibility_score > 0.5 else "limited"
            }
            
        return report
        
    def _calculate_compatibility_score(self, target_platform: PlatformType) -> float:
        """Calculate compatibility score for target platform."""
        
        current_platform = self.platform_capabilities.platform_type
        
        if current_platform == target_platform:
            return 1.0
            
        # Platform family compatibility
        cloud_platforms = {PlatformType.CLOUD_AWS, PlatformType.CLOUD_GCP, PlatformType.CLOUD_AZURE}
        container_platforms = {PlatformType.DOCKER, PlatformType.KUBERNETES}
        desktop_platforms = {PlatformType.LINUX, PlatformType.WINDOWS, PlatformType.MACOS}
        
        if current_platform in cloud_platforms and target_platform in cloud_platforms:
            return 0.9
        elif current_platform in container_platforms and target_platform in container_platforms:
            return 0.9
        elif current_platform in desktop_platforms and target_platform in desktop_platforms:
            return 0.8
        else:
            return 0.6


# Global cross-platform engine
cross_platform_engine = CrossPlatformExecutionEngine(
    CrossPlatformConfig(
        target_platforms=[platform for platform in PlatformType],
        compatibility_mode="permissive",
        auto_optimization=True
    )
)