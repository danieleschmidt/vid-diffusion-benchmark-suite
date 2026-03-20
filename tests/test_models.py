import numpy as np
import pytest
from vid_bench.models import (
    StaticNoiseModel, SolidColorModel, GradientModel,
    LargeModelMock, SmallFastModel
)
from vid_bench.benchmark import BenchmarkReport

MODELS = [StaticNoiseModel, SolidColorModel, GradientModel, LargeModelMock, SmallFastModel]

@pytest.mark.parametrize("ModelClass", MODELS)
def test_model_generates_frames(ModelClass):
    m = ModelClass()
    frames = m.generate_frames("test prompt", num_frames=4, height=16, width=16)
    assert frames.shape == (4, 16, 16, 3)
    assert frames.dtype == np.uint8

@pytest.mark.parametrize("ModelClass", MODELS)
def test_model_run_returns_report(ModelClass):
    m = ModelClass()
    report = m.run(num_frames=4, height=16, width=16)
    assert isinstance(report, BenchmarkReport)
    assert report.model_name == m.model_name
    assert report.fvd_proxy >= 0
    assert 0 <= report.temporal_consistency <= 1
    assert report.vram_estimate_mb > 0
    assert report.latency_ms >= 0

def test_solid_color_high_temporal_consistency():
    m = SolidColorModel()
    report = m.run(num_frames=8, height=16, width=16)
    # Solid color = perfect temporal consistency
    assert report.temporal_consistency == pytest.approx(1.0, abs=1e-5)

def test_large_model_high_vram():
    m = LargeModelMock()
    report = m.run(num_frames=4, height=16, width=16)
    assert report.vram_estimate_mb > 10_000  # 7B model

def test_report_to_json():
    m = SmallFastModel()
    report = m.run(num_frames=4, height=16, width=16)
    j = report.to_json()
    import json
    data = json.loads(j)
    assert data["model_name"] == m.model_name

def test_benchmark_report_from_dict():
    m = GradientModel()
    report = m.run(num_frames=4, height=16, width=16)
    import json
    data = json.loads(report.to_json())
    r2 = BenchmarkReport.from_dict(data)
    assert r2.model_name == report.model_name
