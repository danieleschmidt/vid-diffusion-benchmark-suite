"""
Mock model adapters for testing the benchmark suite.
Each adapter simulates a different video diffusion model with distinct characteristics.
"""
from __future__ import annotations
import numpy as np
from vid_bench.benchmark import VideoModelBenchmark


class StaticNoiseModel(VideoModelBenchmark):
    """Baseline: generates static noise (worst case quality)."""
    model_name = "static-noise-baseline"

    def get_param_count(self) -> int:
        return 1_000_000

    def generate_frames(self, prompt, num_frames, height, width):
        return np.random.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


class SolidColorModel(VideoModelBenchmark):
    """Degenerate: generates solid color frames (high temporal consistency, low diversity)."""
    model_name = "solid-color-model"

    def get_param_count(self) -> int:
        return 500_000

    def generate_frames(self, prompt, num_frames, height, width):
        color = np.array([128, 64, 200], dtype=np.uint8)
        frames = np.ones((num_frames, height, width, 3), dtype=np.uint8)
        return frames * color


class GradientModel(VideoModelBenchmark):
    """Generates smooth gradient frames with slow animation."""
    model_name = "gradient-animator"

    def get_param_count(self) -> int:
        return 50_000_000

    def generate_frames(self, prompt, num_frames, height, width):
        frames = []
        for i in range(num_frames):
            phase = i / max(num_frames - 1, 1)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                r = int(255 * y / max(height - 1, 1))
                g = int(255 * phase)
                b = int(255 * (1 - y / max(height - 1, 1)))
                frame[y, :] = [r, g, b]
            frames.append(frame)
        return np.stack(frames)


class LargeModelMock(VideoModelBenchmark):
    """Simulates a large video diffusion model (high VRAM)."""
    model_name = "large-video-diffusion-7b"

    def get_param_count(self) -> int:
        return 7_000_000_000

    def generate_frames(self, prompt, num_frames, height, width):
        # Sinusoidal pattern
        t = np.linspace(0, 2 * np.pi, num_frames)
        frames = []
        for i, phase in enumerate(t):
            x = np.linspace(0, 2 * np.pi, width)
            y = np.linspace(0, 2 * np.pi, height)
            xx, yy = np.meshgrid(x, y)
            pattern = (np.sin(xx + phase) * 127 + 128).astype(np.uint8)
            frame = np.stack([pattern, np.roll(pattern, 20), np.roll(pattern, 40)], axis=-1)
            frames.append(frame)
        return np.stack(frames)


class SmallFastModel(VideoModelBenchmark):
    """Simulates a small, fast video model (low VRAM, decent quality)."""
    model_name = "tiny-video-model-100m"

    def get_param_count(self) -> int:
        return 100_000_000

    def generate_frames(self, prompt, num_frames, height, width):
        # Coherent random walk for temporal consistency
        frames = []
        base = np.random.randint(100, 200, (height, width, 3), dtype=np.int32)
        for i in range(num_frames):
            noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int32)
            base = np.clip(base + noise, 0, 255)
            frames.append(base.astype(np.uint8))
        return np.stack(frames)
