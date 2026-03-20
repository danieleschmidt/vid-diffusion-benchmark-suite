"""CLI runner for video diffusion benchmarks."""
from __future__ import annotations
import json
import argparse
import sys
import time
from pathlib import Path
from typing import Optional
from vid_bench.benchmark import BenchmarkReport
from vid_bench.models import (
    StaticNoiseModel, SolidColorModel, GradientModel,
    LargeModelMock, SmallFastModel
)

ALL_MODELS = {
    "static-noise": StaticNoiseModel,
    "solid-color": SolidColorModel,
    "gradient": GradientModel,
    "large-model": LargeModelMock,
    "small-fast": SmallFastModel,
}


def run_benchmarks(model_names: Optional[list] = None,
                   num_frames: int = 8, height: int = 64, width: int = 64,
                   prompt: str = "a dog running in a park",
                   output_path: Optional[str] = None) -> list:
    names = model_names or list(ALL_MODELS.keys())
    reports = []
    for name in names:
        if name not in ALL_MODELS:
            print(f"Unknown model: {name}", file=sys.stderr)
            continue
        model = ALL_MODELS[name]()
        print(f"Benchmarking {name}...", flush=True)
        report = model.run(prompt=prompt, num_frames=num_frames,
                           height=height, width=width)
        reports.append(report)
        print(f"  FVD proxy: {report.fvd_proxy:.4f}")
        print(f"  Temporal consistency: {report.temporal_consistency:.4f}")
        print(f"  VRAM estimate: {report.vram_estimate_mb:.1f} MB")
        print(f"  Latency: {report.latency_ms:.1f} ms")

    if output_path:
        data = [json.loads(r.to_json()) for r in reports]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nReport saved to {output_path}")

    return reports


def main():
    parser = argparse.ArgumentParser(description="Video Diffusion Benchmark Suite")
    parser.add_argument("--models", nargs="+", choices=list(ALL_MODELS.keys()),
                        help="Models to benchmark (default: all)")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--prompt", default="a dog running in a park")
    parser.add_argument("--output", help="JSON output file path")
    args = parser.parse_args()
    run_benchmarks(args.models, args.frames, args.height, args.width,
                   args.prompt, args.output)


if __name__ == "__main__":
    main()
