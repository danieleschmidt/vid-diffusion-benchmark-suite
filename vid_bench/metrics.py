"""
Video quality metrics for benchmarking video diffusion models.
"""
from __future__ import annotations
import numpy as np


def compute_fvd_proxy(frames: np.ndarray) -> float:
    """
    FVD proxy metric: frame-level FID approximation via pixel statistics.
    Computes KL divergence between per-frame pixel distributions and a
    reference Gaussian. Returns a scalar (lower = better quality/diversity).

    frames: (T, H, W, C) uint8 or float32
    """
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0

    T = frames.shape[0]
    if T == 0:
        return float("inf")

    # Compute per-frame mean and std (flattened pixels)
    frame_stats = []
    for i in range(T):
        f = frames[i].flatten()
        frame_stats.append((f.mean(), f.std() + 1e-8))

    means = np.array([s[0] for s in frame_stats])
    stds = np.array([s[1] for s in frame_stats])

    # Reference: unit Gaussian (mean=0.5, std=0.2 for normalized images)
    ref_mean, ref_std = 0.5, 0.2

    # Fréchet-like distance between per-frame stats and reference
    mean_diff = float(np.mean((means - ref_mean) ** 2))
    std_diff = float(np.mean((stds - ref_std) ** 2))
    fvd_proxy = mean_diff + std_diff

    return float(fvd_proxy)


def compute_temporal_consistency(frames: np.ndarray) -> float:
    """
    Temporal consistency: mean cosine similarity between consecutive frames.
    Higher = more temporally consistent [0, 1].

    frames: (T, H, W, C)
    """
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0

    T = frames.shape[0]
    if T < 2:
        return 1.0

    similarities = []
    for i in range(T - 1):
        a = frames[i].flatten()
        b = frames[i + 1].flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            similarities.append(0.0)
        else:
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            similarities.append(np.clip(sim, -1.0, 1.0))

    return float(np.mean(similarities))


def estimate_vram_mb(param_count: int, dtype_bytes: int = 2,
                     activation_multiplier: float = 2.0) -> float:
    """
    Estimate VRAM usage in MB.
    Formula: (params x dtype_bytes x activation_multiplier) / (1024^2)

    dtype_bytes: 2 for float16, 4 for float32
    activation_multiplier: accounts for activations + optimizer states
    """
    return (param_count * dtype_bytes * activation_multiplier) / (1024 ** 2)


def compute_frame_diversity(frames: np.ndarray) -> float:
    """
    Compute frame diversity as mean pairwise distance between frames.
    Higher = more diverse frames.
    """
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0
    T = frames.shape[0]
    if T < 2:
        return 0.0
    flat = frames.reshape(T, -1)
    dists = []
    for i in range(T):
        for j in range(i + 1, T):
            d = np.linalg.norm(flat[i] - flat[j])
            dists.append(d)
    return float(np.mean(dists)) if dists else 0.0
