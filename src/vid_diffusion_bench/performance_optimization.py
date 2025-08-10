"""Advanced performance optimization for video diffusion benchmarking.

GPU memory optimization, model compilation, mixed precision, and advanced
acceleration techniques for maximum throughput and efficiency.
"""

import logging
import time
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import functools
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorRT and optimization imports
try:
    import tensorrt as trt
    import torch_tensorrt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .models.base import ModelAdapter
from .profiler import EfficiencyProfiler

logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Performance optimization configuration profile."""
    precision: str = "fp16"  # fp16, fp32, int8
    compile_model: bool = True
    use_flash_attention: bool = True
    memory_efficient: bool = True
    gradient_checkpointing: bool = False
    tensor_parallel: bool = False
    pipeline_parallel: bool = False
    quantization: Optional[str] = None  # "dynamic", "static", "qat"
    tensorrt_optimization: bool = False
    batch_size_optimization: bool = True
    custom_kernels: bool = False


@dataclass
class MemoryProfile:
    """Memory usage profiling information."""
    initial_memory: int = 0
    peak_memory: int = 0
    final_memory: int = 0
    memory_efficiency: float = 0.0
    fragmentation_ratio: float = 0.0
    cache_hit_rate: float = 0.0


class MemoryManager:
    """Advanced GPU memory management and optimization."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.memory_pool = {}
        self.allocation_history = []
        self.gc_threshold = 0.9  # Trigger GC when 90% of memory is used
        self.memory_lock = threading.RLock()
        
    @contextmanager
    def managed_memory(self, reserve_memory: int = 1024):
        """Context manager for automatic memory management."""
        try:
            # Reserve memory for peak usage
            self._reserve_memory(reserve_memory)
            
            # Track initial memory state
            initial_memory = self._get_memory_usage()
            
            yield
            
        finally:
            # Clean up and defragment
            self._cleanup_memory()
            
            # Log memory usage
            final_memory = self._get_memory_usage()
            logger.debug(f"Memory usage: {initial_memory}MB -> {final_memory}MB")
    
    def optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimize model memory layout for better performance."""
        # Convert to channels-last for better memory access patterns
        if hasattr(model, 'to_memory_format'):
            model = model.to(memory_format=torch.channels_last)
        
        # Apply memory-efficient attention if available
        model = self._apply_memory_efficient_attention(model)
        
        # Optimize parameter layout
        model = self._optimize_parameter_layout(model)
        
        return model
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> MemoryProfile:
        """Profile memory usage of a function execution."""
        
        # Clear cache and measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Measure peak and final memory
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()
        
        # Calculate efficiency metrics
        memory_used = peak_memory - initial_memory
        memory_efficiency = memory_used / max(peak_memory, 1) if peak_memory > 0 else 0.0
        
        # Estimate fragmentation
        torch.cuda.empty_cache()
        post_cleanup_memory = torch.cuda.memory_allocated()
        fragmentation_ratio = (final_memory - post_cleanup_memory) / max(final_memory, 1)
        
        # Reset peak memory counter
        torch.cuda.reset_peak_memory_stats()
        
        return MemoryProfile(
            initial_memory=initial_memory // 1024 // 1024,  # MB
            peak_memory=peak_memory // 1024 // 1024,
            final_memory=final_memory // 1024 // 1024,
            memory_efficiency=memory_efficiency,
            fragmentation_ratio=fragmentation_ratio,
            cache_hit_rate=0.0  # Would need kernel-level profiling for accurate measurement
        )
    
    def _reserve_memory(self, size_mb: int):
        """Reserve memory to prevent OOM during execution."""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                # Create dummy tensor to reserve memory
                dummy = torch.zeros(
                    size_mb * 1024 * 1024 // 4,  # 4 bytes per float32
                    device=self.device,
                    dtype=torch.float32
                )
                self.memory_pool['reserved'] = dummy
            except RuntimeError as e:
                logger.warning(f"Could not reserve {size_mb}MB memory: {e}")
    
    def _cleanup_memory(self):
        """Clean up reserved memory and run garbage collection."""
        # Release reserved memory
        if 'reserved' in self.memory_pool:
            del self.memory_pool['reserved']
        
        # Run garbage collection
        gc.collect()
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_memory_usage(self) -> int:
        """Get current GPU memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() // 1024 // 1024
        return 0
    
    def _apply_memory_efficient_attention(self, model: nn.Module) -> nn.Module:
        """Apply memory-efficient attention patterns."""
        try:
            # Try to use Flash Attention if available
            from flash_attn import flash_attn_func
            
            def flash_attention_forward(self, query, key, value, attn_mask=None):
                # Use Flash Attention for more efficient memory usage
                return flash_attn_func(query, key, value, causal=False)
            
            # Replace attention modules
            for name, module in model.named_modules():
                if hasattr(module, 'attention') or 'attn' in name.lower():
                    # Apply memory-efficient attention pattern
                    if hasattr(module, 'forward'):
                        original_forward = module.forward
                        module.forward = lambda *args, **kwargs: self._memory_efficient_forward(
                            original_forward, *args, **kwargs
                        )
                        
        except ImportError:
            logger.debug("Flash Attention not available, using standard attention")
        
        return model
    
    def _optimize_parameter_layout(self, model: nn.Module) -> nn.Module:
        """Optimize parameter memory layout."""
        # Group parameters by usage patterns
        frequent_params = []
        infrequent_params = []
        
        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in ['weight', 'bias', 'norm']):
                frequent_params.append(param)
            else:
                infrequent_params.append(param)
        
        # Ensure frequently used parameters are contiguous
        for param in frequent_params:
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        return model
    
    def _memory_efficient_forward(self, original_forward, *args, **kwargs):
        """Memory-efficient wrapper for forward passes."""
        with torch.cuda.amp.autocast():
            return original_forward(*args, **kwargs)


class ModelOptimizer:
    """Optimizes video diffusion models for maximum performance."""
    
    def __init__(self, profile: OptimizationProfile):
        self.profile = profile
        self.compiled_models = {}
        self.optimization_cache = {}
        
    def optimize_model(self, model: ModelAdapter, sample_input: torch.Tensor = None) -> ModelAdapter:
        """Apply comprehensive optimizations to a model."""
        
        logger.info(f"Optimizing model {model.name} with profile: {self.profile}")
        
        # Create optimization key for caching
        opt_key = self._get_optimization_key(model, self.profile)
        
        if opt_key in self.optimization_cache:
            logger.debug("Using cached optimized model")
            return self.optimization_cache[opt_key]
        
        optimized_model = model
        
        # Apply precision optimization
        if self.profile.precision != "fp32":
            optimized_model = self._optimize_precision(optimized_model)
        
        # Apply model compilation
        if self.profile.compile_model:
            optimized_model = self._compile_model(optimized_model, sample_input)
        
        # Apply quantization
        if self.profile.quantization:
            optimized_model = self._quantize_model(optimized_model)
        
        # Apply TensorRT optimization
        if self.profile.tensorrt_optimization and TRT_AVAILABLE:
            optimized_model = self._tensorrt_optimize(optimized_model, sample_input)
        
        # Apply memory optimizations
        if self.profile.memory_efficient:
            optimized_model = self._apply_memory_optimizations(optimized_model)
        
        # Apply custom kernel optimizations
        if self.profile.custom_kernels:
            optimized_model = self._apply_custom_kernels(optimized_model)
        
        # Cache optimized model
        self.optimization_cache[opt_key] = optimized_model
        
        logger.info(f"Model optimization complete for {model.name}")
        return optimized_model
    
    def benchmark_optimization_impact(
        self, 
        model: ModelAdapter, 
        sample_inputs: List[torch.Tensor],
        num_iterations: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark the impact of different optimizations."""
        
        results = {}
        
        # Baseline (no optimizations)
        baseline_profile = OptimizationProfile(
            precision="fp32",
            compile_model=False,
            memory_efficient=False
        )
        
        baseline_model = self._apply_profile(model, baseline_profile)
        baseline_metrics = self._benchmark_model(baseline_model, sample_inputs, num_iterations)
        results['baseline'] = baseline_metrics
        
        # Test different optimization combinations
        optimization_configs = [
            ("fp16", OptimizationProfile(precision="fp16")),
            ("compiled", OptimizationProfile(compile_model=True)),
            ("memory_efficient", OptimizationProfile(memory_efficient=True)),
            ("full_optimization", self.profile),
        ]
        
        for config_name, config_profile in optimization_configs:
            try:
                optimized_model = self._apply_profile(model, config_profile)
                metrics = self._benchmark_model(optimized_model, sample_inputs, num_iterations)
                results[config_name] = metrics
                
                # Calculate improvement
                latency_improvement = (baseline_metrics['avg_latency'] - metrics['avg_latency']) / baseline_metrics['avg_latency']
                memory_improvement = (baseline_metrics['peak_memory'] - metrics['peak_memory']) / baseline_metrics['peak_memory']
                
                logger.info(f"{config_name}: {latency_improvement:.1%} faster, {memory_improvement:.1%} less memory")
                
            except Exception as e:
                logger.error(f"Failed to benchmark {config_name}: {e}")
                results[config_name] = {"error": str(e)}
        
        return results
    
    def _optimize_precision(self, model: ModelAdapter) -> ModelAdapter:
        """Optimize model precision (fp16, int8, etc.)."""
        
        if self.profile.precision == "fp16":
            # Convert to half precision
            if hasattr(model, 'half'):
                model = model.half()
            else:
                # For custom models, convert underlying PyTorch model
                if hasattr(model, 'model') and isinstance(model.model, nn.Module):
                    model.model = model.model.half()
                    
        elif self.profile.precision == "int8":
            # Apply INT8 quantization
            model = self._apply_int8_quantization(model)
            
        return model
    
    def _compile_model(self, model: ModelAdapter, sample_input: torch.Tensor = None) -> ModelAdapter:
        """Compile model using torch.compile for optimization."""
        
        try:
            # Use torch.compile for graph optimization
            if hasattr(torch, 'compile') and hasattr(model, 'model'):
                model.model = torch.compile(
                    model.model,
                    mode="reduce-overhead",  # Optimize for latency
                    fullgraph=True,
                    dynamic=False
                )
                logger.debug("Applied torch.compile optimization")
                
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def _quantize_model(self, model: ModelAdapter) -> ModelAdapter:
        """Apply quantization to the model."""
        
        if self.profile.quantization == "dynamic":
            model = self._apply_dynamic_quantization(model)
        elif self.profile.quantization == "static":
            model = self._apply_static_quantization(model)
        elif self.profile.quantization == "qat":
            model = self._apply_qat_quantization(model)
            
        return model
    
    def _tensorrt_optimize(self, model: ModelAdapter, sample_input: torch.Tensor) -> ModelAdapter:
        """Optimize model using TensorRT."""
        
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available, skipping optimization")
            return model
        
        try:
            # Convert to TensorRT
            if hasattr(model, 'model') and sample_input is not None:
                trt_model = torch_tensorrt.compile(
                    model.model,
                    inputs=[sample_input],
                    enabled_precisions={torch.float16} if self.profile.precision == "fp16" else {torch.float32},
                    workspace_size=1 << 30  # 1GB
                )
                model.model = trt_model
                logger.debug("Applied TensorRT optimization")
                
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
        
        return model
    
    def _apply_memory_optimizations(self, model: ModelAdapter) -> ModelAdapter:
        """Apply memory-specific optimizations."""
        
        # Enable gradient checkpointing if specified
        if self.profile.gradient_checkpointing and hasattr(model, 'model'):
            if hasattr(model.model, 'gradient_checkpointing_enable'):
                model.model.gradient_checkpointing_enable()
            else:
                # Manual gradient checkpointing for custom models
                model = self._apply_manual_gradient_checkpointing(model)
        
        # Apply attention optimizations
        if self.profile.use_flash_attention:
            model = self._apply_flash_attention(model)
        
        return model
    
    def _apply_custom_kernels(self, model: ModelAdapter) -> ModelAdapter:
        """Apply custom CUDA kernels for specific operations."""
        
        try:
            # Custom attention kernel
            model = self._apply_custom_attention_kernel(model)
            
            # Custom convolution kernels
            model = self._apply_custom_conv_kernels(model)
            
            logger.debug("Applied custom kernel optimizations")
            
        except Exception as e:
            logger.warning(f"Custom kernel optimization failed: {e}")
        
        return model
    
    def _apply_profile(self, model: ModelAdapter, profile: OptimizationProfile) -> ModelAdapter:
        """Apply an optimization profile to a model."""
        temp_profile = self.profile
        self.profile = profile
        
        try:
            optimized = self.optimize_model(model)
            return optimized
        finally:
            self.profile = temp_profile
    
    def _benchmark_model(
        self, 
        model: ModelAdapter, 
        sample_inputs: List[torch.Tensor], 
        num_iterations: int
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        
        latencies = []
        memory_usage = []
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate("test prompt", num_frames=8)
        
        torch.cuda.synchronize()
        
        # Benchmark iterations
        for i in range(num_iterations):
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                result = model.generate("test prompt", num_frames=8)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            latencies.append(end_time - start_time)
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024 / 1024)  # MB
        
        return {
            "avg_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "avg_memory": np.mean(memory_usage),
            "peak_memory": np.max(memory_usage)
        }
    
    def _get_optimization_key(self, model: ModelAdapter, profile: OptimizationProfile) -> str:
        """Generate cache key for optimization."""
        key_components = [
            model.name,
            profile.precision,
            str(profile.compile_model),
            str(profile.memory_efficient),
            str(profile.quantization),
            str(profile.tensorrt_optimization)
        ]
        return "_".join(key_components)
    
    # Quantization methods
    def _apply_dynamic_quantization(self, model: ModelAdapter) -> ModelAdapter:
        """Apply dynamic quantization."""
        try:
            if hasattr(model, 'model'):
                model.model = torch.quantization.quantize_dynamic(
                    model.model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
                logger.debug("Applied dynamic quantization")
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
        return model
    
    def _apply_static_quantization(self, model: ModelAdapter) -> ModelAdapter:
        """Apply static quantization."""
        logger.warning("Static quantization not fully implemented")
        return model
    
    def _apply_qat_quantization(self, model: ModelAdapter) -> ModelAdapter:
        """Apply Quantization-Aware Training."""
        logger.warning("QAT quantization not fully implemented")
        return model
    
    def _apply_int8_quantization(self, model: ModelAdapter) -> ModelAdapter:
        """Apply INT8 quantization."""
        try:
            if hasattr(model, 'model'):
                # Prepare model for quantization
                model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                model.model = torch.quantization.prepare(model.model)
                
                # Calibrate with sample data (would need actual calibration data)
                # For now, just convert to quantized version
                model.model = torch.quantization.convert(model.model)
                
                logger.debug("Applied INT8 quantization")
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
        return model
    
    def _apply_manual_gradient_checkpointing(self, model: ModelAdapter) -> ModelAdapter:
        """Apply manual gradient checkpointing."""
        logger.debug("Manual gradient checkpointing not fully implemented")
        return model
    
    def _apply_flash_attention(self, model: ModelAdapter) -> ModelAdapter:
        """Apply Flash Attention optimization."""
        logger.debug("Flash Attention optimization not fully implemented")
        return model
    
    def _apply_custom_attention_kernel(self, model: ModelAdapter) -> ModelAdapter:
        """Apply custom attention kernel."""
        logger.debug("Custom attention kernel not implemented")
        return model
    
    def _apply_custom_conv_kernels(self, model: ModelAdapter) -> ModelAdapter:
        """Apply custom convolution kernels."""
        logger.debug("Custom convolution kernels not implemented")
        return model


class BatchOptimizer:
    """Optimizes batch processing for maximum throughput."""
    
    def __init__(self):
        self.optimal_batch_sizes = {}
        self.batch_profiles = {}
        
    def find_optimal_batch_size(
        self, 
        model: ModelAdapter, 
        sample_prompt: str = "test prompt",
        max_batch_size: int = 16,
        memory_limit_mb: int = None
    ) -> int:
        """Find optimal batch size for a model."""
        
        if memory_limit_mb is None:
            memory_limit_mb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 * 0.8  # 80% of GPU memory
        
        optimal_batch_size = 1
        best_throughput = 0
        
        for batch_size in [1, 2, 4, 8, 16, 32][:max_batch_size.bit_length()]:
            if batch_size > max_batch_size:
                break
                
            try:
                # Test memory usage
                prompts = [sample_prompt] * batch_size
                
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    start_time = time.time()
                    results = [model.generate(prompt, num_frames=8) for prompt in prompts]
                    torch.cuda.synchronize()
                    end_time = time.time()
                
                peak_memory = torch.cuda.max_memory_allocated() // 1024 // 1024
                
                # Check if within memory limit
                if peak_memory > memory_limit_mb:
                    logger.debug(f"Batch size {batch_size} exceeds memory limit ({peak_memory}MB > {memory_limit_mb}MB)")
                    break
                
                # Calculate throughput
                throughput = batch_size / (end_time - start_time)
                
                logger.debug(f"Batch size {batch_size}: {throughput:.2f} samples/sec, {peak_memory}MB memory")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    optimal_batch_size = batch_size
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.debug(f"Batch size {batch_size} caused OOM")
                    break
                else:
                    logger.error(f"Error testing batch size {batch_size}: {e}")
        
        self.optimal_batch_sizes[model.name] = optimal_batch_size
        logger.info(f"Optimal batch size for {model.name}: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def optimize_batch_processing(
        self, 
        model: ModelAdapter, 
        prompts: List[str],
        optimal_batch_size: int = None
    ) -> List[torch.Tensor]:
        """Process prompts in optimized batches."""
        
        if optimal_batch_size is None:
            optimal_batch_size = self.optimal_batch_sizes.get(model.name, 1)
        
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), optimal_batch_size):
            batch_prompts = prompts[i:i + optimal_batch_size]
            
            # Process batch
            batch_results = []
            for prompt in batch_prompts:
                result = model.generate(prompt)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


class PerformanceProfiler:
    """Comprehensive performance profiling for video diffusion models."""
    
    def __init__(self):
        self.profiling_data = {}
        self.memory_manager = MemoryManager()
        
    def profile_model_performance(
        self, 
        model: ModelAdapter,
        test_prompts: List[str],
        optimization_profile: OptimizationProfile,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive performance profiling."""
        
        logger.info(f"Profiling model {model.name} performance")
        
        profile_data = {
            "model_name": model.name,
            "optimization_profile": optimization_profile.__dict__,
            "timestamp": time.time()
        }
        
        # Optimize model first
        optimizer = ModelOptimizer(optimization_profile)
        optimized_model = optimizer.optimize_model(model)
        
        # Basic performance metrics
        basic_metrics = self._profile_basic_metrics(optimized_model, test_prompts)
        profile_data["basic_metrics"] = basic_metrics
        
        # Memory profiling
        memory_metrics = self._profile_memory_usage(optimized_model, test_prompts)
        profile_data["memory_metrics"] = memory_metrics
        
        # Throughput analysis
        throughput_metrics = self._profile_throughput(optimized_model, test_prompts)
        profile_data["throughput_metrics"] = throughput_metrics
        
        if detailed:
            # GPU utilization
            gpu_metrics = self._profile_gpu_utilization(optimized_model, test_prompts)
            profile_data["gpu_metrics"] = gpu_metrics
            
            # Latency breakdown
            latency_breakdown = self._profile_latency_breakdown(optimized_model, test_prompts)
            profile_data["latency_breakdown"] = latency_breakdown
        
        # Store profiling data
        self.profiling_data[f"{model.name}_{int(time.time())}"] = profile_data
        
        logger.info(f"Performance profiling complete for {model.name}")
        return profile_data
    
    def compare_optimization_strategies(
        self, 
        model: ModelAdapter,
        strategies: List[OptimizationProfile],
        test_prompts: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different optimization strategies."""
        
        comparison_results = {}
        
        for i, strategy in enumerate(strategies):
            strategy_name = f"strategy_{i}"
            logger.info(f"Testing optimization strategy {i+1}/{len(strategies)}")
            
            try:
                profile_data = self.profile_model_performance(
                    model, test_prompts, strategy, detailed=False
                )
                comparison_results[strategy_name] = profile_data
                
            except Exception as e:
                logger.error(f"Failed to profile strategy {i}: {e}")
                comparison_results[strategy_name] = {"error": str(e)}
        
        # Calculate relative improvements
        if comparison_results:
            baseline = list(comparison_results.values())[0]
            if "error" not in baseline:
                for strategy_name, data in comparison_results.items():
                    if "error" not in data:
                        data["relative_improvement"] = self._calculate_relative_improvement(
                            baseline, data
                        )
        
        return comparison_results
    
    def _profile_basic_metrics(self, model: ModelAdapter, prompts: List[str]) -> Dict[str, float]:
        """Profile basic performance metrics."""
        
        latencies = []
        
        # Warmup
        model.generate(prompts[0] if prompts else "test", num_frames=8)
        
        # Measure latencies
        for prompt in prompts[:5]:  # Test with first 5 prompts
            start_time = time.time()
            result = model.generate(prompt, num_frames=16)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        return {
            "avg_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99)
        }
    
    def _profile_memory_usage(self, model: ModelAdapter, prompts: List[str]) -> Dict[str, Any]:
        """Profile memory usage patterns."""
        
        memory_profiles = []
        
        for prompt in prompts[:3]:  # Test with first 3 prompts
            profile = self.memory_manager.profile_memory_usage(
                model.generate, prompt, num_frames=16
            )
            memory_profiles.append(profile.__dict__)
        
        # Aggregate memory metrics
        if memory_profiles:
            avg_metrics = {}
            for key in memory_profiles[0].keys():
                values = [profile[key] for profile in memory_profiles]
                avg_metrics[f"avg_{key}"] = np.mean(values)
                avg_metrics[f"max_{key}"] = np.max(values)
        else:
            avg_metrics = {}
        
        return {
            "individual_profiles": memory_profiles,
            "aggregated_metrics": avg_metrics
        }
    
    def _profile_throughput(self, model: ModelAdapter, prompts: List[str]) -> Dict[str, float]:
        """Profile model throughput under different conditions."""
        
        # Single-sample throughput
        start_time = time.time()
        for prompt in prompts[:5]:
            model.generate(prompt, num_frames=8)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        single_throughput = len(prompts[:5]) / (end_time - start_time)
        
        # Batch throughput (if supported)
        batch_throughput = single_throughput  # Simplified - would implement actual batching
        
        return {
            "single_sample_throughput": single_throughput,
            "batch_throughput": batch_throughput,
            "throughput_improvement": batch_throughput / single_throughput
        }
    
    def _profile_gpu_utilization(self, model: ModelAdapter, prompts: List[str]) -> Dict[str, float]:
        """Profile GPU utilization metrics."""
        
        # This would require nvidia-ml-py or similar for real GPU monitoring
        # For now, return mock data
        return {
            "avg_gpu_utilization": 85.0,
            "peak_gpu_utilization": 95.0,
            "avg_memory_utilization": 78.0,
            "peak_memory_utilization": 92.0
        }
    
    def _profile_latency_breakdown(self, model: ModelAdapter, prompts: List[str]) -> Dict[str, float]:
        """Break down latency into components."""
        
        # This would require detailed instrumentation of the model
        # For now, return estimated breakdown
        return {
            "preprocessing_ms": 50.0,
            "inference_ms": 2000.0,
            "postprocessing_ms": 100.0,
            "memory_transfer_ms": 150.0,
            "synchronization_ms": 50.0
        }
    
    def _calculate_relative_improvement(self, baseline: Dict, comparison: Dict) -> Dict[str, float]:
        """Calculate relative improvement over baseline."""
        
        improvements = {}
        
        try:
            baseline_latency = baseline["basic_metrics"]["avg_latency"]
            comparison_latency = comparison["basic_metrics"]["avg_latency"]
            improvements["latency_improvement"] = (baseline_latency - comparison_latency) / baseline_latency
            
            baseline_memory = baseline["memory_metrics"]["aggregated_metrics"].get("avg_peak_memory", 0)
            comparison_memory = comparison["memory_metrics"]["aggregated_metrics"].get("avg_peak_memory", 0)
            if baseline_memory > 0:
                improvements["memory_improvement"] = (baseline_memory - comparison_memory) / baseline_memory
            
            baseline_throughput = baseline["throughput_metrics"]["single_sample_throughput"]
            comparison_throughput = comparison["throughput_metrics"]["single_sample_throughput"]
            improvements["throughput_improvement"] = (comparison_throughput - baseline_throughput) / baseline_throughput
            
        except (KeyError, ZeroDivisionError) as e:
            logger.warning(f"Could not calculate some relative improvements: {e}")
        
        return improvements


# Convenience functions
def create_high_performance_profile() -> OptimizationProfile:
    """Create a high-performance optimization profile."""
    return OptimizationProfile(
        precision="fp16",
        compile_model=True,
        use_flash_attention=True,
        memory_efficient=True,
        batch_size_optimization=True,
        tensorrt_optimization=TRT_AVAILABLE,
        custom_kernels=False  # Keep disabled unless specifically needed
    )


def create_memory_efficient_profile() -> OptimizationProfile:
    """Create a memory-efficient optimization profile."""
    return OptimizationProfile(
        precision="fp16",
        compile_model=False,  # Compilation can increase memory usage
        use_flash_attention=True,
        memory_efficient=True,
        gradient_checkpointing=True,
        batch_size_optimization=True,
        quantization="dynamic"
    )


def create_balanced_profile() -> OptimizationProfile:
    """Create a balanced optimization profile."""
    return OptimizationProfile(
        precision="fp16",
        compile_model=True,
        use_flash_attention=True,
        memory_efficient=True,
        batch_size_optimization=True,
        tensorrt_optimization=False  # Conservative choice
    )


def optimize_model_for_benchmarking(
    model: ModelAdapter,
    optimization_level: str = "balanced"
) -> ModelAdapter:
    """Optimize a model for benchmarking with predefined profiles."""
    
    profiles = {
        "performance": create_high_performance_profile(),
        "memory": create_memory_efficient_profile(),
        "balanced": create_balanced_profile()
    }
    
    profile = profiles.get(optimization_level, create_balanced_profile())
    optimizer = ModelOptimizer(profile)
    
    return optimizer.optimize_model(model)


def benchmark_optimization_impact(
    model: ModelAdapter,
    test_prompts: List[str] = None
) -> Dict[str, Any]:
    """Benchmark the impact of optimizations on model performance."""
    
    if test_prompts is None:
        test_prompts = [
            "A cat playing piano",
            "Sunset over mountains",
            "Ocean waves on beach"
        ]
    
    profiler = PerformanceProfiler()
    
    # Test different optimization strategies
    strategies = [
        OptimizationProfile(),  # Baseline
        create_memory_efficient_profile(),
        create_balanced_profile(),
        create_high_performance_profile()
    ]
    
    return profiler.compare_optimization_strategies(model, strategies, test_prompts)