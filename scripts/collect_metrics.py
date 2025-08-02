#!/usr/bin/env python3
"""
Automated metrics collection for Video Diffusion Benchmark Suite.

This script collects various project metrics and updates the project metrics file.
Designed to run in CI/CD pipelines and scheduled jobs.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import argparse


class MetricsCollector:
    """Collect and update project metrics."""
    
    def __init__(self, project_root: Path, github_token: Optional[str] = None):
        self.project_root = project_root
        self.github_token = github_token
        self.metrics_file = project_root / ".github" / "project-metrics.json"
        
    def load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics structure."""
        return {
            "project": {
                "name": "Video Diffusion Benchmark Suite",
                "version": "0.1.0",
                "license": "MIT"
            },
            "metrics": {},
            "tracking": {
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "update_frequency": "weekly"
            }
        }
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage
        try:
            result = subprocess.run(
                ["pytest", "--cov=vid_diffusion_bench", "--cov-report=json", "--quiet"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and Path("coverage.json").exists():
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                metrics["test_coverage"] = {
                    "current": f"{coverage_data['totals']['percent_covered']:.1f}%",
                    "target": "90%"
                }
        except Exception as e:
            print(f"Failed to collect coverage metrics: {e}")
        
        # Code complexity with radon
        try:
            result = subprocess.run(
                ["radon", "cc", "src/", "-j"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                function_count = 0
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item["type"] == "function":
                            total_complexity += item["complexity"]
                            function_count += 1
                
                if function_count > 0:
                    metrics["code_complexity"] = {
                        "cyclomatic_complexity": {
                            "average": round(total_complexity / function_count, 1),
                            "max_allowed": 10
                        }
                    }
        except Exception as e:
            print(f"Failed to collect complexity metrics: {e}")
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub repository metrics."""
        if not self.github_token:
            print("GitHub token not provided, skipping GitHub metrics")
            return {}
        
        metrics = {}
        repo_name = "danieleschmidt/vid-diffusion-benchmark-suite"
        
        try:
            headers = {"Authorization": f"token {self.github_token}"}
            
            # Repository stats
            repo_response = requests.get(
                f"https://api.github.com/repos/{repo_name}",
                headers=headers
            )
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                metrics["github_metrics"] = {
                    "stars": repo_data["stargazers_count"],
                    "forks": repo_data["forks_count"],
                    "watchers": repo_data["watchers_count"],
                    "open_issues": repo_data["open_issues_count"]
                }
            
            # Pull requests
            pr_response = requests.get(
                f"https://api.github.com/repos/{repo_name}/pulls?state=open",
                headers=headers
            )
            if pr_response.status_code == 200:
                pr_data = pr_response.json()
                metrics["github_metrics"]["open_prs"] = len(pr_data)
            
            # Contributors
            contributors_response = requests.get(
                f"https://api.github.com/repos/{repo_name}/contributors",
                headers=headers
            )
            if contributors_response.status_code == 200:
                contributors_data = contributors_response.json()
                metrics["github_metrics"]["contributors"] = len(contributors_data)
                
        except Exception as e:
            print(f"Failed to collect GitHub metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from recent benchmark runs."""
        metrics = {}
        
        # Check for recent benchmark results
        results_dir = self.project_root / "results"
        if results_dir.exists():
            recent_results = sorted(results_dir.glob("*.json"))[-10:]  # Last 10 results
            
            if recent_results:
                total_time = 0
                model_count = 0
                
                for result_file in recent_results:
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        if "benchmark_time" in result_data:
                            total_time += result_data["benchmark_time"]
                            model_count += 1
                    except Exception:
                        continue
                
                if model_count > 0:
                    metrics["benchmark_execution"] = {
                        "average_time_per_model": f"{total_time / model_count:.1f} minutes",
                        "target": "<10 minutes"
                    }
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        metrics = {}
        
        # Safety check for known vulnerabilities
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                vulnerabilities = safety_data.get("vulnerabilities", [])
                
                critical = sum(1 for v in vulnerabilities if v.get("severity") == "critical")
                high = sum(1 for v in vulnerabilities if v.get("severity") == "high")
                medium = sum(1 for v in vulnerabilities if v.get("severity") == "medium")
                low = len(vulnerabilities) - critical - high - medium
                
                metrics["vulnerability_scan"] = {
                    "last_scan": datetime.utcnow().isoformat() + "Z",
                    "critical_vulnerabilities": critical,
                    "high_vulnerabilities": high,
                    "medium_vulnerabilities": medium,
                    "low_vulnerabilities": low
                }
        except Exception as e:
            print(f"Failed to collect security metrics: {e}")
        
        return metrics
    
    def collect_model_metrics(self) -> Dict[str, Any]:
        """Collect model-related metrics."""
        metrics = {}
        
        # Count supported models from registry
        try:
            result = subprocess.run(
                ["python", "-c", "from vid_diffusion_bench.models.registry import get_all_models; print(len(get_all_models()))"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                model_count = int(result.stdout.strip())
                metrics["supported_models"] = {
                    "total": model_count
                }
        except Exception as e:
            print(f"Failed to collect model metrics: {e}")
        
        return metrics
    
    def update_metrics(self) -> None:
        """Update all metrics and save to file."""
        print("Collecting project metrics...")
        
        current_metrics = self.load_current_metrics()
        
        # Collect different types of metrics
        collectors = [
            ("code_quality", self.collect_code_quality_metrics),
            ("community", self.collect_github_metrics),
            ("performance", self.collect_performance_metrics),
            ("security", self.collect_security_metrics),
            ("models", self.collect_model_metrics)
        ]
        
        for category, collector in collectors:
            try:
                print(f"Collecting {category} metrics...")
                new_metrics = collector()
                if new_metrics:
                    current_metrics["metrics"][category] = {
                        **current_metrics["metrics"].get(category, {}),
                        **new_metrics
                    }
            except Exception as e:
                print(f"Failed to collect {category} metrics: {e}")
        
        # Update tracking information
        current_metrics["tracking"]["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Save updated metrics
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(current_metrics, f, indent=2)
        
        print(f"Metrics updated and saved to {self.metrics_file}")
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate a human-readable metrics report."""
        metrics = self.load_current_metrics()
        
        report = f"""# Video Diffusion Benchmark Suite - Metrics Report

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Project Overview
- **Name**: {metrics['project']['name']}
- **Version**: {metrics['project']['version']}
- **License**: {metrics['project']['license']}

"""
        
        # Code Quality
        if "code_quality" in metrics["metrics"]:
            report += "## Code Quality\n"
            code_metrics = metrics["metrics"]["code_quality"]
            
            if "test_coverage" in code_metrics:
                coverage = code_metrics["test_coverage"]
                report += f"- **Test Coverage**: {coverage['current']} (Target: {coverage['target']})\n"
            
            if "code_complexity" in code_metrics:
                complexity = code_metrics["code_complexity"]["cyclomatic_complexity"]
                report += f"- **Average Complexity**: {complexity['average']} (Max: {complexity['max_allowed']})\n"
            
            report += "\n"
        
        # Community Metrics
        if "community" in metrics["metrics"]:
            report += "## Community\n"
            community = metrics["metrics"]["community"]
            
            if "github_metrics" in community:
                github = community["github_metrics"]
                report += f"- **GitHub Stars**: {github.get('stars', 'N/A')}\n"
                report += f"- **Forks**: {github.get('forks', 'N/A')}\n"
                report += f"- **Contributors**: {github.get('contributors', 'N/A')}\n"
                report += f"- **Open Issues**: {github.get('open_issues', 'N/A')}\n"
                report += f"- **Open PRs**: {github.get('open_prs', 'N/A')}\n"
            
            report += "\n"
        
        # Performance
        if "performance" in metrics["metrics"]:
            report += "## Performance\n"
            performance = metrics["metrics"]["performance"]
            
            if "benchmark_execution" in performance:
                bench = performance["benchmark_execution"]
                report += f"- **Average Benchmark Time**: {bench['average_time_per_model']} (Target: {bench['target']})\n"
            
            report += "\n"
        
        # Security
        if "security" in metrics["metrics"]:
            report += "## Security\n"
            security = metrics["metrics"]["security"]
            
            if "vulnerability_scan" in security:
                vuln = security["vulnerability_scan"]
                report += f"- **Critical Vulnerabilities**: {vuln['critical_vulnerabilities']}\n"
                report += f"- **High Vulnerabilities**: {vuln['high_vulnerabilities']}\n"
                report += f"- **Medium Vulnerabilities**: {vuln['medium_vulnerabilities']}\n"
                report += f"- **Low Vulnerabilities**: {vuln['low_vulnerabilities']}\n"
                report += f"- **Last Scan**: {vuln['last_scan']}\n"
            
            report += "\n"
        
        # Models
        if "models" in metrics["metrics"]:
            report += "## Models\n"
            models = metrics["metrics"]["models"]
            
            if "supported_models" in models:
                model_count = models["supported_models"]["total"]
                report += f"- **Supported Models**: {model_count}\n"
            
            report += "\n"
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--github-token", help="GitHub API token")
    parser.add_argument("--report", type=Path, help="Generate report to file")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    
    args = parser.parse_args()
    
    # Get GitHub token from environment if not provided
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    
    collector = MetricsCollector(args.project_root, github_token)
    
    try:
        collector.update_metrics()
        
        if args.report:
            collector.generate_report(args.report)
        
        print("Metrics collection completed successfully!")
        
    except Exception as e:
        print(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()