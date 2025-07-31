#!/usr/bin/env python3
"""
Performance report generator.
Creates detailed HTML reports from benchmark results.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Diffusion Benchmark Performance Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #6c757d; margin-top: 5px; }}
        .benchmark-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .benchmark-table th, .benchmark-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .benchmark-table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .status-regression {{ color: #dc3545; font-weight: bold; }}
        .status-improvement {{ color: #28a745; font-weight: bold; }}
        .status-stable {{ color: #6c757d; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .timestamp {{ color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Video Diffusion Benchmark Performance Report</h1>
        <p>Generated on {timestamp}</p>
        <p class="timestamp">Commit: {commit_sha} | Branch: {branch}</p>
    </div>

    <div class="summary">
        <div class="metric-card">
            <div class="metric-value">{total_benchmarks}</div>
            <div class="metric-label">Total Benchmarks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_duration:.2f}s</div>
            <div class="metric-label">Average Duration</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{fastest_benchmark:.3f}s</div>
            <div class="metric-label">Fastest Benchmark</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{slowest_benchmark:.3f}s</div>
            <div class="metric-label">Slowest Benchmark</div>
        </div>
    </div>

    <div class="chart-container">
        <h2>📊 Benchmark Results</h2>
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Benchmark Name</th>
                    <th>Mean Duration (s)</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Rounds</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {benchmark_rows}
            </tbody>
        </table>
    </div>

    <div class="chart-container">
        <h2>🔍 Performance Categories</h2>
        {category_analysis}
    </div>

    <div class="chart-container">
        <h2>🔧 System Information</h2>
        <pre>{system_info}</pre>
    </div>

    <script>
        // Add interactive features if needed
        console.log('Performance report loaded');
    </script>
</body>
</html>
"""


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


def categorize_benchmarks(benchmarks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize benchmarks by type."""
    categories = {
        "Model Inference": [],
        "Memory Usage": [],
        "Throughput": [],
        "Load Testing": [],
        "Other": []
    }
    
    for bench in benchmarks:
        name = bench.get("name", "").lower()
        if "inference" in name or "speed" in name:
            categories["Model Inference"].append(bench)
        elif "memory" in name or "vram" in name:
            categories["Memory Usage"].append(bench)
        elif "throughput" in name or "requests" in name:
            categories["Throughput"].append(bench)
        elif "load" in name or "concurrent" in name:
            categories["Load Testing"].append(bench)
        else:
            categories["Other"].append(bench)
    
    return categories


def format_benchmark_row(benchmark: Dict[str, Any]) -> str:
    """Format a benchmark result as an HTML table row."""
    stats = benchmark.get("stats", {})
    name = benchmark.get("name", benchmark.get("fullname", "Unknown"))
    
    mean = stats.get("mean", 0)
    stddev = stats.get("stddev", 0)
    min_val = stats.get("min", 0)
    max_val = stats.get("max", 0)
    rounds = stats.get("rounds", 0)
    
    # Determine status based on performance
    if mean < 1.0:
        status = '<span class="status-improvement">Fast</span>'
    elif mean > 10.0:
        status = '<span class="status-regression">Slow</span>'
    else:
        status = '<span class="status-stable">Normal</span>'
    
    return f"""
    <tr>
        <td>{name}</td>
        <td>{mean:.4f}</td>
        <td>{stddev:.4f}</td>
        <td>{min_val:.4f}</td>
        <td>{max_val:.4f}</td>
        <td>{rounds}</td>
        <td>{status}</td>
    </tr>
    """


def generate_category_analysis(categories: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate category analysis HTML."""
    html = ""
    
    for category, benchmarks in categories.items():
        if not benchmarks:
            continue
            
        avg_duration = sum(b.get("stats", {}).get("mean", 0) for b in benchmarks) / len(benchmarks)
        
        html += f"""
        <div style="margin-bottom: 20px;">
            <h3>{category} ({len(benchmarks)} benchmarks)</h3>
            <p>Average Duration: <strong>{avg_duration:.4f}s</strong></p>
            <ul>
        """
        
        for bench in sorted(benchmarks, key=lambda x: x.get("stats", {}).get("mean", 0))[:5]:
            name = bench.get("name", "Unknown")
            duration = bench.get("stats", {}).get("mean", 0)
            html += f"<li>{name}: {duration:.4f}s</li>"
        
        html += "</ul></div>"
    
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--benchmark-results", type=Path, required=True,
                      help="Path to benchmark results file or directory")
    parser.add_argument("--output-format", choices=["html", "json"], default="html",
                      help="Output format")
    parser.add_argument("--output", type=Path, required=True,
                      help="Output file path")
    
    args = parser.parse_args()
    
    # Load benchmark results
    try:
        results = load_benchmark_results(args.benchmark_results)
    except FileNotFoundError:
        print("No benchmark results found")
        return
    
    benchmarks = results.get("benchmarks", [])
    
    if not benchmarks:
        print("No benchmarks found in results")
        return
    
    # Calculate summary statistics
    durations = [b.get("stats", {}).get("mean", 0) for b in benchmarks]
    total_benchmarks = len(benchmarks)
    avg_duration = sum(durations) / len(durations) if durations else 0
    fastest_benchmark = min(durations) if durations else 0
    slowest_benchmark = max(durations) if durations else 0
    
    # Categorize benchmarks
    categories = categorize_benchmarks(benchmarks)
    
    if args.output_format == "html":
        # Generate HTML report
        benchmark_rows = "".join(format_benchmark_row(b) for b in benchmarks)
        category_analysis = generate_category_analysis(categories)
        
        # System info (mock data for now)
        system_info = {
            "python_version": "3.11.0",
            "pytorch_version": "2.3.0",
            "cuda_version": "12.1",
            "gpu_name": "Tesla V100",
            "timestamp": datetime.now().isoformat()
        }
        
        html_content = HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            commit_sha="<env:GITHUB_SHA>",
            branch="<env:GITHUB_REF_NAME>",
            total_benchmarks=total_benchmarks,
            avg_duration=avg_duration,
            fastest_benchmark=fastest_benchmark,
            slowest_benchmark=slowest_benchmark,
            benchmark_rows=benchmark_rows,
            category_analysis=category_analysis,
            system_info=json.dumps(system_info, indent=2)
        )
        
        with open(args.output, 'w') as f:
            f.write(html_content)
    
    else:  # JSON format
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_benchmarks": total_benchmarks,
                "avg_duration": avg_duration,
                "fastest_benchmark": fastest_benchmark,
                "slowest_benchmark": slowest_benchmark
            },
            "categories": {k: len(v) for k, v in categories.items()},
            "benchmarks": benchmarks
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
    
    print(f"Performance report generated: {args.output}")


if __name__ == "__main__":
    main()