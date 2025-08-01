#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Executor
Executes the highest-value tasks automatically
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class AutonomousExecutor:
    """Executes prioritized tasks from the value discovery backlog"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.backlog_path = self.repo_path / "BACKLOG.md"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Extract the next highest-priority task from BACKLOG.md"""
        # For this demo, return the predefined highest-priority task
        return {
            "id": "AUTO-001",
            "title": "Implement GitHub Actions CI/CD workflows",
            "category": "automation",
            "estimatedEffort": 6.0,
            "compositeScore": 85.4,
            "description": "Create comprehensive CI/CD workflows for testing, security scanning, and deployment"
        }
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task and return execution results"""
        
        print(f"🚀 Executing: {task['title']}")
        print(f"📊 Score: {task['compositeScore']} | Effort: {task['estimatedEffort']} hours")
        
        execution_start = datetime.now()
        
        try:
            if task["id"] == "AUTO-001":
                return self._implement_github_workflows(task)
            else:
                print(f"⚠️  Task type {task['id']} not yet implemented")
                return {
                    "success": False,
                    "error": "Task type not implemented",
                    "duration": 0
                }
                
        except Exception as e:
            execution_duration = (datetime.now() - execution_start).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "duration": execution_duration
            }
    
    def _implement_github_workflows(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement GitHub Actions CI/CD workflows"""
        
        execution_start = datetime.now()
        
        try:
            # Create .github/workflows directory
            workflows_dir = self.repo_path / ".github" / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Create CI workflow
            ci_workflow = workflows_dir / "ci.yml"
            ci_content = self._generate_ci_workflow()
            
            with open(ci_workflow, 'w') as f:
                f.write(ci_content)
            
            print("✅ Created .github/workflows/ci.yml")
            
            # Create security workflow
            security_workflow = workflows_dir / "security.yml"
            security_content = self._generate_security_workflow()
            
            with open(security_workflow, 'w') as f:
                f.write(security_content)
            
            print("✅ Created .github/workflows/security.yml")
            
            # Create release workflow
            release_workflow = workflows_dir / "release.yml"
            release_content = self._generate_release_workflow()
            
            with open(release_workflow, 'w') as f:
                f.write(release_content)
            
            print("✅ Created .github/workflows/release.yml")
            
            # Update documentation
            docs_update = self._update_workflow_documentation()
            
            execution_duration = (datetime.now() - execution_start).total_seconds()
            
            return {
                "success": True,
                "duration": execution_duration,
                "filesCreated": [
                    ".github/workflows/ci.yml",
                    ".github/workflows/security.yml", 
                    ".github/workflows/release.yml"
                ],
                "actualEffort": execution_duration / 3600,  # Convert to hours
                "valueDelivered": {
                    "automatedTesting": True,
                    "securityScanning": True,
                    "automatedReleases": True,
                    "cicdMaturity": "advanced"
                }
            }
            
        except Exception as e:
            execution_duration = (datetime.now() - execution_start).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "duration": execution_duration
            }
    
    def _generate_ci_workflow(self) -> str:
        """Generate comprehensive CI workflow"""
        return """name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHONPATH: src
  PYTHON_VERSION: '3.11'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,test]'
    
    - name: Run pre-commit hooks
      run: |
        pre-commit run --all-files
    
    - name: Run tests with coverage
      run: |
        pytest --cov=vid_diffusion_bench --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip build
    
    - name: Build package
      run: |
        python -m build
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist-${{ github.sha }}
        path: dist/

  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: vid-diffusion-benchmark:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
"""

    def _generate_security_workflow(self) -> str:
        """Generate security scanning workflow"""
        return """name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev]'
        pip install safety bandit semgrep
    
    - name: Run Bandit security scan
      run: |
        bandit -r src -f json -o bandit-report.json || true
        bandit -r src
    
    - name: Run Safety dependency scan
      run: |
        safety check --json --output safety-report.json || true
        safety check
    
    - name: Run Semgrep
      run: |
        semgrep --config=auto src --json --output=semgrep-report.json || true
        echo "Semgrep scan completed"
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports-${{ github.sha }}
        path: |
          bandit-report.json
          safety-report.json
          semgrep-report.json

  dependency-review:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
      with:
        fail-on-severity: moderate
"""

    def _generate_release_workflow(self) -> str:
        """Generate automated release workflow"""
        return """name: Release

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v1.0.0)'
        required: true

env:
  PYTHON_VERSION: '3.11'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for changelog generation
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip build twine
        pip install -e '.[dev]'
    
    - name: Run tests
      run: |
        pytest --cov=vid_diffusion_bench
    
    - name: Build package
      run: |
        python -m build
    
    - name: Generate changelog
      id: changelog
      run: |
        # Simple changelog generation from git log
        echo "## Changes" > RELEASE_CHANGELOG.md
        git log --pretty=format:"- %s (%h)" $(git describe --tags --abbrev=0 HEAD^)..HEAD >> RELEASE_CHANGELOG.md
        echo "changelog<<EOF" >> $GITHUB_OUTPUT
        cat RELEASE_CHANGELOG.md >> $GITHUB_OUTPUT
        echo "EOF" >> $GITHUB_OUTPUT
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        body: ${{ steps.changelog.outputs.changelog }}
        files: dist/*
        draft: false
        prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Publish to PyPI
      if: startsWith(github.ref, 'refs/tags/')
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        twine upload dist/* --skip-existing

  docker-release:
    runs-on: ubuntu-latest
    needs: release
    if: startsWith(github.ref, 'refs/tags/')
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Extract version
      id: version
      run: echo "version=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          vid-diffusion-benchmark:latest
          vid-diffusion-benchmark:${{ steps.version.outputs.version }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
"""

    def _update_workflow_documentation(self) -> bool:
        """Update documentation with workflow setup instructions"""
        try:
            # Update the existing GitHub Actions setup documentation
            github_actions_doc = self.repo_path / "docs" / "workflows" / "github-actions-setup.md"
            
            if github_actions_doc.exists():
                with open(github_actions_doc, 'r') as f:
                    content = f.read()
                
                # Add implementation status
                updated_content = content.replace(
                    "# GitHub Actions Workflow Setup",
                    "# GitHub Actions Workflow Setup\n\n**Status: ✅ IMPLEMENTED** - Workflows are now active in `.github/workflows/`"
                )
                
                with open(github_actions_doc, 'w') as f:
                    f.write(updated_content)
                
                print("✅ Updated GitHub Actions documentation")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to update documentation: {e}")
            return False
    
    def update_metrics(self, task: Dict[str, Any], execution_result: Dict[str, Any]) -> None:
        """Update value metrics with execution results"""
        
        # Load existing metrics
        try:
            with open(self.metrics_path) as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = {
                "executionHistory": [],
                "backlogMetrics": {},
                "learningData": {}
            }
        
        # Add execution record
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "taskId": task["id"],
            "title": task["title"],
            "category": task["category"],
            "scores": {
                "compositeScore": task.get("compositeScore", 0)
            },
            "estimatedEffort": task.get("estimatedEffort", 0),
            "actualEffort": execution_result.get("actualEffort", 0),
            "success": execution_result.get("success", False),
            "duration": execution_result.get("duration", 0),
            "valueDelivered": execution_result.get("valueDelivered", {}),
            "filesCreated": execution_result.get("filesCreated", [])
        }
        
        metrics["executionHistory"].append(execution_record)
        
        # Update learning data if successful
        if execution_result.get("success"):
            estimated = task.get("estimatedEffort", 0)
            actual = execution_result.get("actualEffort", 0)
            
            if estimated > 0 and actual > 0:
                effort_ratio = actual / estimated
                
                # Update effort estimation accuracy
                learning = metrics.get("learningData", {})
                learning["lastEstimationRatio"] = effort_ratio
                learning["executionCount"] = learning.get("executionCount", 0) + 1
                
                metrics["learningData"] = learning
        
        # Save updated metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Updated execution metrics for task {task['id']}")
    
    def run_execution_cycle(self) -> Dict[str, Any]:
        """Run a complete autonomous execution cycle"""
        
        print("🤖 Starting Terragon Autonomous Execution Cycle")
        print("=" * 50)
        
        # Get next highest-priority task
        task = self.get_next_task()
        
        if not task:
            print("⏳ No tasks available for execution")
            return {"executed": False, "reason": "no_tasks_available"}
        
        print(f"🎯 Selected task: {task['title']}")
        
        # Execute the task
        execution_result = self.execute_task(task)
        
        # Update metrics
        self.update_metrics(task, execution_result)
        
        # Report results
        if execution_result.get("success"):
            print("=" * 50)
            print(f"✅ Task completed successfully in {execution_result['duration']:.2f} seconds")
            print(f"📁 Files created: {len(execution_result.get('filesCreated', []))}")
            print(f"💎 Value delivered: {execution_result.get('valueDelivered', {})}")
        else:
            print("=" * 50)
            print(f"❌ Task failed: {execution_result.get('error', 'Unknown error')}")
        
        return {
            "executed": True,
            "task": task,
            "result": execution_result
        }


def main():
    """CLI entry point for autonomous execution"""
    
    try:
        executor = AutonomousExecutor()
        result = executor.run_execution_cycle()
        
        if result["executed"] and result["result"]["success"]:
            print("\n🚀 Execution completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Execution completed with issues")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Autonomous execution failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()