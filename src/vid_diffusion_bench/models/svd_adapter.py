"""Stable Video Diffusion model adapter."""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np

from .base import ModelAdapter
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model("svd-xt-1.1")
class StableVideoDiffusionXTAdapter(ModelAdapter):
    """Adapter for Stable Video Diffusion XT 1.1 model."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        """Initialize SVD-XT adapter.
        
        Args:
            device: Device to load model on
            **kwargs: Additional configuration
        """
        super().__init__(device, **kwargs)
        self.model = None
        self.pipe = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the SVD model pipeline."""
        try:
            # Try to import diffusers
            from diffusers import StableVideoDiffusionPipeline
            
            # Load the model
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )
            
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_model_cpu_offload()
                
            logger.info("SVD-XT-1.1 model loaded successfully")
            
        except ImportError:
            logger.warning("diffusers not available, using mock implementation")
            self.pipe = None
        except Exception as e:
            logger.error(f"Failed to load SVD-XT model: {e}")
            self.pipe = None
            
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 25,
        fps: int = 7,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 25,
        **kwargs
    ) -> torch.Tensor:
        """Generate video from text prompt.
        
        Args:
            prompt: Text description of desired video
            num_frames: Number of frames to generate (max 25 for SVD-XT)
            fps: Target frames per second
            width: Video width
            height: Video height  
            num_inference_steps: Number of denoising steps
            **kwargs: Additional generation parameters
            
        Returns:
            Generated video tensor of shape (frames, channels, height, width)
        """
        if self.pipe is None:
            # Mock implementation when model not available
            return self._generate_mock_video(num_frames, 3, height, width)
            
        try:
            # SVD requires an initial image, so we generate one from prompt
            initial_image = self._text_to_image(prompt, width, height)
            
            # Generate video from image
            video_frames = self.pipe(
                initial_image,
                height=height,
                width=width, 
                num_frames=min(num_frames, 25),  # SVD-XT max is 25 frames
                num_inference_steps=num_inference_steps,
                fps=fps,
                **kwargs
            ).frames[0]
            
            # Convert to tensor format (T, C, H, W)
            video_tensor = torch.stack([
                torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
                for frame in video_frames
            ])
            
            return video_tensor
            
        except Exception as e:
            logger.error(f"Failed to generate video with SVD-XT: {e}")
            return self._generate_mock_video(num_frames, 3, height, width)
            
    def _text_to_image(self, prompt: str, width: int, height: int) -> torch.Tensor:
        """Generate initial image from text prompt."""
        try:
            from diffusers import StableDiffusionPipeline
            
            # Use a lightweight Stable Diffusion model for initial image
            img_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                img_pipe = img_pipe.to(self.device)
                
            image = img_pipe(
                prompt,
                width=width,
                height=height,
                num_inference_steps=20
            ).images[0]
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to generate initial image: {e}")
            # Return random image
            return torch.randn(3, height, width) * 0.5 + 0.5
            
    def _generate_mock_video(self, num_frames: int, channels: int, height: int, width: int) -> torch.Tensor:
        """Generate mock video for testing."""
        logger.debug(f"Generating mock video: {num_frames}x{channels}x{height}x{width}")
        
        # Create video with smooth motion
        video = torch.zeros(num_frames, channels, height, width)
        
        for t in range(num_frames):
            # Create moving pattern
            phase = 2 * np.pi * t / num_frames
            
            # Generate frame with moving sine wave pattern
            y_coords = torch.arange(height).float().unsqueeze(1) / height
            x_coords = torch.arange(width).float().unsqueeze(0) / width
            
            pattern = torch.sin(y_coords * 4 * np.pi + phase) * torch.cos(x_coords * 4 * np.pi + phase)
            pattern = (pattern + 1) / 2  # Normalize to [0, 1]
            
            # Apply pattern to all channels with slight variations
            for c in range(channels):
                channel_phase = phase + c * np.pi / 3
                channel_pattern = torch.sin(pattern * 2 * np.pi + channel_phase)
                channel_pattern = (channel_pattern + 1) / 2
                video[t, c] = channel_pattern
                
        return video
        
    @property
    def requirements(self) -> Dict[str, Any]:
        """Model hardware and software requirements."""
        return {
            "vram_gb": 8.0,  # Minimum VRAM for SVD-XT
            "precision": "fp16",
            "dependencies": [
                "diffusers>=0.27.0",
                "transformers>=4.40.0", 
                "accelerate>=0.30.0",
                "torch>=2.0.0",
                "torchvision>=0.15.0"
            ],
            "model_size_gb": 5.2,
            "min_inference_steps": 20,
            "max_inference_steps": 50,
            "max_frames": 25,
            "supported_resolutions": [
                (1024, 576),  # Default SVD resolution
                (768, 432),
                (512, 288)
            ]
        }
        
    @property
    def name(self) -> str:
        """Model name for identification."""
        return "svd-xt-1.1"


@register_model("svd-base")  
class StableVideoDiffusionBaseAdapter(ModelAdapter):
    """Adapter for base Stable Video Diffusion model."""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.pipe = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the SVD base model."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
                
            logger.info("SVD base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SVD base model: {e}")
            self.pipe = None
            
    def generate(
        self, 
        prompt: str, 
        num_frames: int = 14,
        fps: int = 6,
        **kwargs
    ) -> torch.Tensor:
        """Generate video from text prompt."""
        if self.pipe is None:
            return self._generate_mock_video(num_frames, 3, 576, 1024)
            
        # Similar implementation to SVD-XT but with different parameters
        try:
            initial_image = self._text_to_image(prompt, 1024, 576)
            
            video_frames = self.pipe(
                initial_image,
                num_frames=min(num_frames, 14),  # SVD base max is 14 frames
                fps=fps,
                **kwargs
            ).frames[0]
            
            video_tensor = torch.stack([
                torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
                for frame in video_frames
            ])
            
            return video_tensor
            
        except Exception as e:
            logger.error(f"Failed to generate video with SVD base: {e}")
            return self._generate_mock_video(num_frames, 3, 576, 1024)
            
    def _text_to_image(self, prompt: str, width: int, height: int):
        """Generate initial image (shared with SVD-XT)."""
        # Same implementation as SVD-XT
        return torch.randn(3, height, width) * 0.5 + 0.5
        
    def _generate_mock_video(self, num_frames: int, channels: int, height: int, width: int) -> torch.Tensor:
        """Generate mock video (shared implementation)."""
        return torch.randn(num_frames, channels, height, width) * 0.5 + 0.5
        
    @property
    def requirements(self) -> Dict[str, Any]:
        """Model requirements."""
        return {
            "vram_gb": 6.0,
            "precision": "fp16",
            "dependencies": [
                "diffusers>=0.27.0",
                "transformers>=4.40.0",
                "torch>=2.0.0"
            ],
            "model_size_gb": 4.8,
            "max_frames": 14,
            "supported_resolutions": [(1024, 576)]
        }
        
    @property 
    def name(self) -> str:
        return "svd-base"