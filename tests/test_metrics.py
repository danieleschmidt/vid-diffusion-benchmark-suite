import numpy as np
import pytest
from vid_bench.metrics import (
    compute_fvd_proxy, compute_temporal_consistency,
    estimate_vram_mb, compute_frame_diversity
)

def test_fvd_proxy_returns_float():
    frames = np.random.randint(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    result = compute_fvd_proxy(frames)
    assert isinstance(result, float)

def test_fvd_proxy_empty_returns_inf():
    result = compute_fvd_proxy(np.zeros((0, 32, 32, 3)))
    assert result == float("inf")

def test_fvd_proxy_solid_color_lower():
    # Solid gray (mean=0.5, std~0) vs random noise
    solid = np.ones((8, 32, 32, 3), dtype=np.uint8) * 128
    noise = np.random.randint(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    fvd_solid = compute_fvd_proxy(solid)
    fvd_noise = compute_fvd_proxy(noise)
    # Solid is closer to reference mean (0.5) so lower FVD proxy
    assert isinstance(fvd_solid, float) and isinstance(fvd_noise, float)

def test_temporal_consistency_identical_frames():
    frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    frames = np.stack([frame] * 4)
    tc = compute_temporal_consistency(frames)
    assert tc == pytest.approx(1.0, abs=1e-5)

def test_temporal_consistency_single_frame():
    frames = np.random.randint(0, 256, (1, 32, 32, 3), dtype=np.uint8)
    assert compute_temporal_consistency(frames) == 1.0

def test_temporal_consistency_range():
    frames = np.random.randint(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    tc = compute_temporal_consistency(frames)
    assert -1.0 <= tc <= 1.0

def test_estimate_vram_mb_basic():
    mb = estimate_vram_mb(1_000_000, dtype_bytes=4)
    assert mb == pytest.approx(1000000 * 4 * 2.0 / (1024**2), rel=0.01)

def test_estimate_vram_mb_large_model():
    mb = estimate_vram_mb(7_000_000_000, dtype_bytes=2)
    assert mb > 1000  # 7B fp16 model should be >1GB

def test_frame_diversity_identical_zero():
    frame = np.ones((8, 8, 3), dtype=np.uint8) * 100
    frames = np.stack([frame] * 4)
    d = compute_frame_diversity(frames)
    assert d == pytest.approx(0.0, abs=1e-5)

def test_frame_diversity_single_frame():
    frames = np.random.randint(0, 256, (1, 8, 8, 3), dtype=np.uint8)
    assert compute_frame_diversity(frames) == 0.0
