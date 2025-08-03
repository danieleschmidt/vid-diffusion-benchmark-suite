"""Mock model adapters for testing and demonstration."""

import logging
import time
from typing import Dict, Any
import torch
import torch.nn.functional as F
import numpy as np

from .base import ModelAdapter
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model("mock-fast")
class MockFastAdapter(ModelAdapter):
    """Mock fast model for testing benchmarks."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 16,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> torch.Tensor:
        """Generate mock video quickly."""
        logger.debug(f"Mock fast generation: {prompt[:50]}...")
        
        # Simulate fast generation (minimal compute)
        time.sleep(0.1)  # 100ms simulation
        
        # Generate simple animated pattern
        video = torch.zeros(num_frames, 3, height, width)
        
        for t in range(num_frames):
            # Simple moving gradient
            progress = t / max(1, num_frames - 1)
            
            # Create gradient that moves based on progress
            y_grad = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)
            x_grad = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)
            
            # Animate the gradient
            animated_pattern = (y_grad + x_grad + progress) % 1.0
            
            # Apply to RGB channels with different phases
            video[t, 0] = animated_pattern  # Red
            video[t, 1] = (animated_pattern + 0.33) % 1.0  # Green
            video[t, 2] = (animated_pattern + 0.66) % 1.0  # Blue
            
        return video
        
    @property
    def requirements(self) -> Dict[str, Any]:
        """Mock fast model requirements."""
        return {
            "vram_gb": 2.0,
            "precision": "fp32",
            "dependencies": ["torch>=1.12.0"],
            "model_size_gb": 0.5,
            "max_frames": 64,
            "supported_resolutions": [
                (512, 512),
                (256, 256),
                (128, 128)
            ]
        }
        
    @property
    def name(self) -> str:
        return "mock-fast"


@register_model("mock-high-quality")
class MockHighQualityAdapter(ModelAdapter):
    """Mock high-quality model that takes longer but produces better results."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 32,
        fps: int = 24,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> torch.Tensor:
        """Generate mock high-quality video."""
        logger.debug(f"Mock high-quality generation: {prompt[:50]}...")
        
        # Simulate longer generation time
        time.sleep(2.0)  # 2 second simulation
        
        # Generate more complex pattern
        video = torch.zeros(num_frames, 3, height, width)
        
        # Create coordinate grids
        y_coords = torch.arange(height).float().unsqueeze(1) / height
        x_coords = torch.arange(width).float().unsqueeze(0) / width
        
        for t in range(num_frames):
            # Complex animated pattern with multiple frequencies
            time_phase = 2 * np.pi * t / num_frames
            
            # Multiple sine waves for complexity
            pattern1 = torch.sin(y_coords * 8 * np.pi + time_phase)
            pattern2 = torch.cos(x_coords * 6 * np.pi + time_phase * 1.5)
            pattern3 = torch.sin((y_coords + x_coords) * 4 * np.pi + time_phase * 0.5)
            
            # Combine patterns
            combined = (pattern1 + pattern2 + pattern3) / 3
            combined = (combined + 1) / 2  # Normalize to [0, 1]
            
            # Add noise for texture
            noise = torch.randn_like(combined) * 0.1
            combined = torch.clamp(combined + noise, 0, 1)
            
            # Apply different transformations to RGB channels
            video[t, 0] = combined
            video[t, 1] = torch.roll(combined, shifts=int(height*0.1), dims=0)
            video[t, 2] = torch.roll(combined, shifts=int(width*0.1), dims=1)
            
        return video
        
    @property
    def requirements(self) -> Dict[str, Any]:
        """Mock high-quality model requirements."""
        return {
            "vram_gb": 12.0,
            "precision": "fp16", 
            "dependencies": [
                "torch>=2.0.0",
                "torchvision>=0.15.0"
            ],
            "model_size_gb": 8.5,
            "max_frames": 128,
            "supported_resolutions": [
                (1024, 1024),
                (768, 768),
                (512, 512)
            ]
        }
        
    @property
    def name(self) -> str:
        return "mock-high-quality"


@register_model("mock-memory-intensive")
class MockMemoryIntensiveAdapter(ModelAdapter):
    """Mock model that uses a lot of GPU memory."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        
        # Allocate some GPU memory to simulate large model
        if device == "cuda" and torch.cuda.is_available():
            self.dummy_weights = torch.randn(1000, 1000, 1000, device=device) * 0.01
        else:
            self.dummy_weights = None
            
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 16,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> torch.Tensor:
        """Generate video with high memory usage."""
        logger.debug(f"Mock memory-intensive generation: {prompt[:50]}...")
        
        # Simulate memory-intensive computation
        if self.device == "cuda" and torch.cuda.is_available():
            # Allocate temporary large tensors
            temp_memory = torch.randn(num_frames, 3, height * 2, width * 2, device=self.device)
            
            # Do some computation to simulate model work
            time.sleep(0.5)
            processed = F.interpolate(temp_memory, size=(height, width), mode='bilinear')
            
            # Apply some transformations
            result = torch.sigmoid(processed + self.dummy_weights[:num_frames, :3, :height, :width])
            
            # Clean up temporary memory
            del temp_memory
            torch.cuda.empty_cache()
            
            return result
        else:
            # CPU fallback
            time.sleep(1.0)
            return torch.rand(num_frames, 3, height, width)
            
    @property
    def requirements(self) -> Dict[str, Any]:
        """Mock memory-intensive model requirements."""
        return {
            "vram_gb": 24.0,  # Requires high-end GPU
            "precision": "fp32",  # Uses more memory with fp32
            "dependencies": [
                "torch>=2.0.0",
                "torchvision>=0.15.0"
            ],
            "model_size_gb": 15.0,
            "max_frames": 32,
            "supported_resolutions": [
                (512, 512),
                (256, 256)
            ]
        }
        
    @property
    def name(self) -> str:
        return "mock-memory-intensive"


@register_model("mock-efficient")
class MockEfficientAdapter(ModelAdapter):
    """Mock efficient model optimized for speed and low memory usage."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 16,
        fps: int = 30,
        width: int = 256,
        height: int = 256,
        **kwargs
    ) -> torch.Tensor:
        """Generate video efficiently."""
        logger.debug(f"Mock efficient generation: {prompt[:50]}...")
        
        # Very fast generation
        time.sleep(0.05)  # 50ms simulation
        
        # Simple but smooth animation
        video = torch.zeros(num_frames, 3, height, width)
        
        # Create simple moving circle
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        
        for t in range(num_frames):
            frame = torch.zeros(height, width)
            
            # Move circle in circular pattern
            angle = 2 * np.pi * t / num_frames
            offset_y = int(radius * 0.5 * np.sin(angle))
            offset_x = int(radius * 0.5 * np.cos(angle))
            
            # Draw circle
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing='ij'
            )
            
            distances = ((y_grid - (center_y + offset_y))**2 + 
                        (x_grid - (center_x + offset_x))**2)**0.5
            
            circle = (distances <= radius).float()
            
            # Apply to all channels with different intensities
            video[t, 0] = circle
            video[t, 1] = circle * 0.7
            video[t, 2] = circle * 0.4
            
        return video
        
    @property
    def requirements(self) -> Dict[str, Any]:
        """Mock efficient model requirements."""
        return {
            "vram_gb": 1.0,
            "precision": "fp16",
            "dependencies": ["torch>=1.12.0"],
            "model_size_gb": 0.2,
            "max_frames": 256,
            "supported_resolutions": [
                (256, 256),
                (128, 128),
                (64, 64)
            ]
        }
        
    @property
    def name(self) -> str:
        return "mock-efficient"


@register_model("mock-unstable")
class MockUnstableAdapter(ModelAdapter):
    """Mock model that sometimes fails for testing error handling."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.failure_rate = kwargs.get("failure_rate", 0.3)  # 30% failure rate
        
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 16,
        fps: int = 8,
        **kwargs
    ) -> torch.Tensor:
        """Generate video with potential failures."""
        logger.debug(f"Mock unstable generation: {prompt[:50]}...")
        
        # Randomly fail based on failure rate
        if np.random.random() < self.failure_rate:
            error_types = [
                "CUDA out of memory",
                "Model weights corrupted", 
                "Invalid prompt format",
                "Network timeout",
                "Inference engine crashed"
            ]
            error_msg = np.random.choice(error_types)
            raise RuntimeError(f"Mock failure: {error_msg}")
            
        # Normal generation when not failing
        time.sleep(0.3)
        return torch.rand(num_frames, 3, 256, 256)
        
    @property
    def requirements(self) -> Dict[str, Any]:
        """Mock unstable model requirements."""
        return {
            "vram_gb": 4.0,
            "precision": "fp16",
            "dependencies": ["torch>=1.12.0"],
            "model_size_gb": 2.0,
            "max_frames": 32,
            "supported_resolutions": [(256, 256)],
            "stability": "experimental"  # Mark as unstable
        }
        
    @property
    def name(self) -> str:
        return "mock-unstable"