#!/usr/bin/env python3
"""
GPU memory profiling script for video diffusion models.
Profiles memory usage patterns during model inference.
"""

import json
import argparse
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class GPUMemorySnapshot:
    """GPU memory usage snapshot."""
    timestamp: float
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    gpu_utilization: int
    temperature: int


class GPUProfiler:
    """Profiles GPU memory usage during model operations."""
    
    def __init__(self):
        self.snapshots: List[GPUMemorySnapshot] = []
        self.profiling = False
    
    def get_gpu_stats(self) -> GPUMemorySnapshot:
        """Get current GPU statistics."""
        try:
            # Query GPU memory and utilization
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True)
            
            values = result.stdout.strip().split(', ')
            total_mb = int(values[0])
            used_mb = int(values[1])
            free_mb = int(values[2])
            utilization = int(values[3])
            temperature = int(values[4])
            
            return GPUMemorySnapshot(
                timestamp=time.time(),
                total_memory_mb=total_mb,
                used_memory_mb=used_mb,
                free_memory_mb=free_mb,
                gpu_utilization=utilization,
                temperature=temperature
            )
        except (subprocess.CalledProcessError, ValueError, IndexError) as e:
            print(f"Error getting GPU stats: {e}")
            return GPUMemorySnapshot(
                timestamp=time.time(),
                total_memory_mb=0,
                used_memory_mb=0,
                free_memory_mb=0,
                gpu_utilization=0,
                temperature=0
            )
    
    def start_profiling(self, interval: float = 0.5):
        """Start continuous GPU profiling."""
        self.profiling = True
        self.snapshots = []
        
        # Take initial snapshot
        self.snapshots.append(self.get_gpu_stats())
    
    def take_snapshot(self, label: str = ""):
        """Take a single GPU memory snapshot."""
        snapshot = self.get_gpu_stats()
        self.snapshots.append(snapshot)
        return snapshot
    
    def stop_profiling(self):
        """Stop profiling and return results."""
        self.profiling = False
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze profiling results."""
        if not self.snapshots:
            return {"error": "No profiling data available"}
        
        used_memory = [s.used_memory_mb for s in self.snapshots]
        utilization = [s.gpu_utilization for s in self.snapshots]
        temperatures = [s.temperature for s in self.snapshots]
        
        analysis = {
            "duration_seconds": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "total_snapshots": len(self.snapshots),
            "memory_usage": {
                "peak_mb": max(used_memory),
                "min_mb": min(used_memory),
                "avg_mb": sum(used_memory) / len(used_memory),
                "baseline_mb": used_memory[0] if used_memory else 0,
                "allocation_mb": max(used_memory) - min(used_memory)
            },
            "gpu_utilization": {
                "peak_percent": max(utilization),
                "min_percent": min(utilization),
                "avg_percent": sum(utilization) / len(utilization)
            },
            "temperature": {
                "peak_celsius": max(temperatures),
                "min_celsius": min(temperatures),
                "avg_celsius": sum(temperatures) / len(temperatures)
            },
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "used_memory_mb": s.used_memory_mb,
                    "gpu_utilization": s.gpu_utilization,
                    "temperature": s.temperature
                }
                for s in self.snapshots
            ]
        }
        
        return analysis


def profile_model_category(category: str, batch_size: int, resolution: str) -> Dict[str, Any]:
    """Profile a specific model category configuration."""
    profiler = GPUProfiler()
    
    # Start profiling
    profiler.start_profiling()
    
    # Simulate model operations (replace with actual model inference)
    print(f"Profiling {category} models with batch_size={batch_size}, resolution={resolution}")
    
    # Take baseline snapshot
    baseline = profiler.take_snapshot("baseline")
    print(f"Baseline GPU memory: {baseline.used_memory_mb}MB")
    
    # Simulate model loading
    time.sleep(2)
    load_snapshot = profiler.take_snapshot("model_loaded")
    print(f"After model loading: {load_snapshot.used_memory_mb}MB")
    
    # Simulate inference
    for i in range(5):
        time.sleep(1)
        inference_snapshot = profiler.take_snapshot(f"inference_{i}")
        print(f"Inference {i}: {inference_snapshot.used_memory_mb}MB")
    
    # Take final snapshot
    final_snapshot = profiler.take_snapshot("final")
    
    # Stop profiling and analyze
    analysis = profiler.stop_profiling()
    
    # Add configuration info
    analysis["configuration"] = {
        "model_category": category,
        "batch_size": batch_size,
        "resolution": resolution
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Profile GPU memory usage")
    parser.add_argument("--model-category", required=True, 
                      choices=["tier1", "tier2"],
                      help="Model category to profile")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Batch size for inference")
    parser.add_argument("--resolution", default="512x512",
                      help="Resolution for inference")
    parser.add_argument("--output", type=Path, required=True,
                      help="Output JSON file path")
    
    args = parser.parse_args()
    
    print(f"Starting GPU memory profiling...")
    print(f"Configuration: {args.model_category}, batch_size={args.batch_size}, resolution={args.resolution}")
    
    # Check if nvidia-smi is available
    try:
        subprocess.run(["nvidia-smi", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: nvidia-smi not found. Creating mock profile data.")
        
        # Create mock profile for environments without GPU
        mock_analysis = {
            "configuration": {
                "model_category": args.model_category,
                "batch_size": args.batch_size,
                "resolution": args.resolution
            },
            "duration_seconds": 10.0,
            "total_snapshots": 8,
            "memory_usage": {
                "peak_mb": 8192 * args.batch_size if args.model_category == "tier1" else 16384 * args.batch_size,
                "min_mb": 1024,
                "avg_mb": 6144 * args.batch_size,
                "baseline_mb": 1024,
                "allocation_mb": 7168 * args.batch_size
            },
            "gpu_utilization": {
                "peak_percent": 95,
                "min_percent": 10,
                "avg_percent": 78
            },
            "temperature": {
                "peak_celsius": 75,
                "min_celsius": 45,
                "avg_celsius": 62
            },
            "mock_data": True
        }
        
        with open(args.output, 'w') as f:
            json.dump(mock_analysis, f, indent=2)
        
        print(f"Mock GPU profile saved to {args.output}")
        return
    
    # Run actual profiling
    analysis = profile_model_category(
        args.model_category, 
        args.batch_size, 
        args.resolution
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nGPU Profiling Complete:")
    print(f"  Peak memory usage: {analysis['memory_usage']['peak_mb']}MB")
    print(f"  Memory allocation: {analysis['memory_usage']['allocation_mb']}MB")
    print(f"  Average GPU utilization: {analysis['gpu_utilization']['avg_percent']:.1f}%")
    print(f"  Profile saved to: {args.output}")


if __name__ == "__main__":
    main()