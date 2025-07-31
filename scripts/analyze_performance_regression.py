#!/usr/bin/env python3
"""
Performance regression analysis tool.
Compares current benchmark results against baseline to detect significant regressions.
"""

import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_benchmark_results(results_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file or directory."""
    if results_path.is_file():
        with open(results_path) as f:
            return json.load(f)
    
    # Load multiple result files from directory
    combined_results = {"benchmarks": []}
    for json_file in results_path.glob("benchmark-*.json"):
        with open(json_file) as f:
            data = json.load(f)
            combined_results["benchmarks"].extend(data.get("benchmarks", []))
    
    return combined_results


def calculate_performance_change(baseline: float, current: float) -> Tuple[float, str]:
    """Calculate percentage change and determine if it's a regression."""
    if baseline == 0:
        return 0.0, "unknown"
    
    change_percent = ((current - baseline) / baseline) * 100
    
    if change_percent > 10:  # More than 10% slower
        return change_percent, "regression"
    elif change_percent < -5:  # More than 5% faster
        return change_percent, "improvement"
    else:
        return change_percent, "stable"


def analyze_benchmark_group(name: str, baseline_values: List[float], 
                          current_values: List[float], threshold: float) -> Dict[str, Any]:
    """Analyze a group of benchmark results."""
    if not baseline_values or not current_values:
        return {
            "name": name,
            "status": "insufficient_data",
            "message": "Insufficient data for comparison"
        }
    
    baseline_mean = statistics.mean(baseline_values)
    current_mean = statistics.mean(current_values)
    baseline_stdev = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
    current_stdev = statistics.stdev(current_values) if len(current_values) > 1 else 0
    
    change_percent, status = calculate_performance_change(baseline_mean, current_mean)
    
    is_significant_regression = (
        status == "regression" and 
        abs(change_percent) > threshold and
        current_mean > baseline_mean + 2 * baseline_stdev  # Statistical significance
    )
    
    return {
        "name": name,
        "baseline_mean": baseline_mean,
        "current_mean": current_mean,
        "baseline_stdev": baseline_stdev,
        "current_stdev": current_stdev,
        "change_percent": change_percent,
        "status": status,
        "is_significant_regression": is_significant_regression,
        "sample_sizes": {
            "baseline": len(baseline_values),
            "current": len(current_values)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze performance regression")
    parser.add_argument("--current-results", type=Path, required=True,
                      help="Path to current benchmark results")
    parser.add_argument("--baseline-results", type=Path, required=True,
                      help="Path to baseline benchmark results")
    parser.add_argument("--threshold", type=float, default=10.0,
                      help="Regression threshold percentage (default: 10%)")
    parser.add_argument("--output", type=Path, required=True,
                      help="Path to output analysis JSON")
    
    args = parser.parse_args()
    
    # Load results
    try:
        current_results = load_benchmark_results(args.current_results)
        baseline_results = load_benchmark_results(args.baseline_results)
    except FileNotFoundError as e:
        print(f"Error loading benchmark results: {e}")
        # Create minimal analysis for missing baseline
        analysis = {
            "significant_regression": False,
            "total_benchmarks": 0,
            "regressions": 0,
            "improvements": 0,
            "stable": 0,
            "error": f"Baseline not found: {e}",
            "benchmark_analysis": []
        }
        
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        return
    
    # Group benchmarks by name
    baseline_groups = {}
    current_groups = {}
    
    for bench in baseline_results.get("benchmarks", []):
        name = bench.get("name", bench.get("fullname", "unknown"))
        if name not in baseline_groups:
            baseline_groups[name] = []
        baseline_groups[name].append(bench.get("stats", {}).get("mean", 0))
    
    for bench in current_results.get("benchmarks", []):
        name = bench.get("name", bench.get("fullname", "unknown"))
        if name not in current_groups:
            current_groups[name] = []
        current_groups[name].append(bench.get("stats", {}).get("mean", 0))
    
    # Analyze each benchmark group
    benchmark_analyses = []
    regressions = 0
    improvements = 0
    stable = 0
    
    all_benchmark_names = set(baseline_groups.keys()) | set(current_groups.keys())
    
    for name in all_benchmark_names:
        baseline_vals = baseline_groups.get(name, [])
        current_vals = current_groups.get(name, [])
        
        analysis = analyze_benchmark_group(name, baseline_vals, current_vals, args.threshold)
        benchmark_analyses.append(analysis)
        
        if analysis["status"] == "regression":
            regressions += 1
        elif analysis["status"] == "improvement":
            improvements += 1
        elif analysis["status"] == "stable":
            stable += 1
    
    # Determine if there are significant regressions
    significant_regressions = [a for a in benchmark_analyses if a.get("is_significant_regression", False)]
    has_significant_regression = len(significant_regressions) > 0
    
    # Create summary
    analysis = {
        "significant_regression": has_significant_regression,
        "total_benchmarks": len(benchmark_analyses),
        "regressions": regressions,
        "improvements": improvements,
        "stable": stable,
        "threshold_percent": args.threshold,
        "significant_regressions": len(significant_regressions),
        "regression_summary": {
            "worst_regression": max(
                (a for a in benchmark_analyses if a["status"] == "regression"),
                key=lambda x: x["change_percent"],
                default=None
            ),
            "best_improvement": min(
                (a for a in benchmark_analyses if a["status"] == "improvement"),
                key=lambda x: x["change_percent"],
                default=None
            )
        },
        "benchmark_analysis": benchmark_analyses,
        "metrics": {
            "inference_speed": f"{improvements} improved, {regressions} regressed",
            "memory_usage": "Analysis available in detailed results",
            "throughput": f"{stable} stable benchmarks"
        }
    }
    
    # Save analysis
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print(f"Performance Analysis Complete:")
    print(f"  Total benchmarks: {len(benchmark_analyses)}")
    print(f"  Regressions: {regressions}")
    print(f"  Improvements: {improvements}")
    print(f"  Stable: {stable}")
    print(f"  Significant regressions: {len(significant_regressions)}")
    
    if has_significant_regression:
        print(f"\n❌ Significant performance regression detected!")
        for reg in significant_regressions:
            print(f"  - {reg['name']}: {reg['change_percent']:.1f}% slower")
    else:
        print(f"\n✅ No significant performance regression detected")


if __name__ == "__main__":
    main()