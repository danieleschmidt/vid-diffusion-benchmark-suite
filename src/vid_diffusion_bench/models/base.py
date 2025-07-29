"""Base model adapter interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class ModelAdapter(ABC):
    """Abstract base class for video diffusion model adapters."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        """Initialize model adapter.
        
        Args:
            device: Device to load model on
            **kwargs: Model-specific configuration
        """
        self.device = device
        self.config = kwargs
        
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 16,
        fps: int = 8,
        **kwargs
    ) -> torch.Tensor:
        """Generate video from text prompt.
        
        Args:
            prompt: Text description of desired video
            num_frames: Number of frames to generate
            fps: Target frames per second
            **kwargs: Model-specific generation parameters
            
        Returns:
            Generated video tensor of shape (frames, channels, height, width)
        """
        pass
        
    @property
    @abstractmethod
    def requirements(self) -> Dict[str, Any]:
        """Model hardware and software requirements.
        
        Returns:
            Dictionary with keys: vram_gb, precision, dependencies
        """
        pass
        
    @property
    def name(self) -> str:
        """Model name for identification."""
        return self.__class__.__name__.lower()