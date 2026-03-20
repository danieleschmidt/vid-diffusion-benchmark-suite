import json
import pytest
from vid_bench.runner import run_benchmarks, ALL_MODELS

def test_run_all_models():
    reports = run_benchmarks(num_frames=4, height=16, width=16)
    assert len(reports) == len(ALL_MODELS)

def test_run_single_model():
    reports = run_benchmarks(["static-noise"], num_frames=4, height=16, width=16)
    assert len(reports) == 1
    assert reports[0].model_name == "static-noise-baseline"

def test_run_outputs_json(tmp_path):
    out = str(tmp_path / "report.json")
    run_benchmarks(["gradient"], num_frames=4, height=16, width=16, output_path=out)
    with open(out) as f:
        data = json.load(f)
    assert len(data) == 1
    assert "fvd_proxy" in data[0]

def test_unknown_model_skipped(capsys):
    reports = run_benchmarks(["nonexistent-model"], num_frames=4, height=16, width=16)
    assert len(reports) == 0
