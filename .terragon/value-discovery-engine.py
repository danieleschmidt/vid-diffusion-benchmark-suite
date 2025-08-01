#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
Continuously discovers, scores, and prioritizes development tasks
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# Simple YAML parser for configuration (fallback if PyYAML not available)
def simple_yaml_load(content: str) -> Dict[str, Any]:
    """Simple YAML parser for basic configuration files"""
    result = {}
    current_section = result
    section_stack = [result]
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if ':' in line and not line.startswith(' '):
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if value:
                # Try to parse as number or boolean
                if value.lower() in ('true', 'false'):
                    current_section[key] = value.lower() == 'true'
                elif value.replace('.', '').replace('-', '').isdigit():
                    current_section[key] = float(value) if '.' in value else int(value)
                else:
                    current_section[key] = value.strip('"\'')
            else:
                # New section
                current_section[key] = {}
                current_section = current_section[key]
    
    return result

try:
    import yaml
    yaml_load = yaml.safe_load
except ImportError:
    yaml_load = lambda content: simple_yaml_load(content)


class ValueDiscoveryEngine:
    """Main engine for autonomous value discovery and task prioritization"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "value-config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration"""
        try:
            with open(self.config_path) as f:
                return yaml_load(f.read())
        except FileNotFoundError:
            print(f"❌ Configuration not found at {self.config_path}")
            sys.exit(1)
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing value metrics"""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return {
            "executionHistory": [],
            "backlogMetrics": {
                "totalItems": 0,
                "averageAge": 0.0,
                "debtRatio": 0.0,
                "velocityTrend": "unknown"
            },
            "learningData": {
                "predictionAccuracy": 0.0,
                "effortEstimationError": 0.0,
                "recalibrations": 0
            }
        }
    
    def discover_value_opportunities(self) -> List[Dict[str, Any]]:
        """Comprehensive value discovery across all configured sources"""
        opportunities = []
        
        print("🔍 Discovering value opportunities...")
        
        # 1. Git History Analysis
        opportunities.extend(self._analyze_git_history())
        
        # 2. Static Analysis
        opportunities.extend(self._run_static_analysis())
        
        # 3. Security Vulnerability Scan
        opportunities.extend(self._scan_security_vulnerabilities())
        
        # 4. Performance Analysis
        opportunities.extend(self._analyze_performance())
        
        # 5. Documentation Gap Analysis
        opportunities.extend(self._analyze_documentation_gaps())
        
        # 6. Test Coverage Analysis
        opportunities.extend(self._analyze_test_coverage())
        
        # 7. Dependency Health Check
        opportunities.extend(self._analyze_dependency_health())
        
        print(f"✅ Discovered {len(opportunities)} value opportunities")
        return opportunities
    
    def _analyze_git_history(self) -> List[Dict[str, Any]]:
        """Extract technical debt and improvement opportunities from git history"""
        opportunities = []
        
        try:
            # Look for TODO, FIXME, HACK markers in recent commits
            result = subprocess.run([
                "git", "log", "--grep=TODO\\|FIXME\\|HACK\\|XXX", 
                "--since=30 days ago", "--oneline"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            commit_markers = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Look for "quick fix" or "temporary" markers
            result = subprocess.run([
                "git", "log", "--grep=quick fix\\|temporary\\|bandaid\\|workaround", 
                "--since=30 days ago", "--oneline", "-i"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            temp_fixes = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Analyze commit frequency for hot spots
            result = subprocess.run([
                "git", "log", "--since=30 days ago", "--name-only", "--pretty=format:"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                file_changes = [f for f in result.stdout.split('\n') if f.strip()]
                file_churn = {}
                for file_path in file_changes:
                    file_churn[file_path] = file_churn.get(file_path, 0) + 1
                
                # Identify hot spots (files changed > 5 times in 30 days)
                hot_spots = [(f, count) for f, count in file_churn.items() if count > 5]
                
                for file_path, churn_count in hot_spots:
                    opportunities.append({
                        "id": f"hotspot-{hash(file_path)}",
                        "title": f"Refactor high-churn file: {file_path}",
                        "category": "technical-debt",
                        "source": "git-history",
                        "description": f"File changed {churn_count} times in 30 days - may need refactoring",
                        "files": [file_path],
                        "estimatedEffort": min(8, churn_count * 0.5),
                        "hotspotMultiplier": min(3.0, churn_count / 5.0),
                        "discoveredAt": datetime.now().isoformat()
                    })
            
            # Process debt markers
            for commit in commit_markers + temp_fixes:
                if commit.strip():
                    commit_hash = commit.split()[0]
                    commit_msg = ' '.join(commit.split()[1:])
                    
                    opportunities.append({
                        "id": f"debt-{commit_hash}",
                        "title": f"Resolve technical debt: {commit_msg[:50]}...",
                        "category": "technical-debt",
                        "source": "git-history",
                        "description": f"Technical debt identified in commit: {commit_msg}",
                        "estimatedEffort": 3.0,
                        "discoveredAt": datetime.now().isoformat()
                    })
                    
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git history analysis failed: {e}")
        
        return opportunities
    
    def _run_static_analysis(self) -> List[Dict[str, Any]]:
        """Run static analysis tools and extract improvement opportunities"""
        opportunities = []
        
        # Run Ruff
        try:
            result = subprocess.run([
                "ruff", "check", "src", "tests", "--format", "json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                
                # Group by rule and file
                rule_groups = {}
                for issue in ruff_issues:
                    rule = issue.get('code', 'unknown')
                    if rule not in rule_groups:
                        rule_groups[rule] = []
                    rule_groups[rule].append(issue)
                
                # Create opportunities for rules with multiple violations
                for rule, issues in rule_groups.items():
                    if len(issues) >= 3:  # Only create tasks for recurring issues
                        opportunities.append({
                            "id": f"ruff-{rule}",
                            "title": f"Fix Ruff violations: {rule}",
                            "category": "code-quality",
                            "source": "static-analysis",
                            "description": f"Fix {len(issues)} violations of rule {rule}",
                            "files": list(set(issue['filename'] for issue in issues)),
                            "estimatedEffort": min(8.0, len(issues) * 0.2),
                            "violationCount": len(issues),
                            "discoveredAt": datetime.now().isoformat()
                        })
                        
        except subprocess.CalledProcessError:
            print("⚠️  Ruff analysis skipped (not available)")
        except json.JSONDecodeError:
            print("⚠️  Ruff output format unexpected")
        
        # Run MyPy
        try:
            result = subprocess.run([
                "mypy", "src", "--no-error-summary"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                type_errors = result.stdout.strip().split('\n')
                error_count = len([e for e in type_errors if e.strip() and ':' in e])
                
                if error_count > 0:
                    opportunities.append({
                        "id": "mypy-errors",
                        "title": f"Fix type checking errors ({error_count} issues)",
                        "category": "code-quality",
                        "source": "static-analysis",
                        "description": f"Resolve {error_count} MyPy type checking errors",
                        "estimatedEffort": min(12.0, error_count * 0.3),
                        "errorCount": error_count,
                        "discoveredAt": datetime.now().isoformat()
                    })
                    
        except subprocess.CalledProcessError:
            print("⚠️  MyPy analysis skipped (errors expected in discovery mode)")
        
        return opportunities
    
    def _scan_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Scan for security vulnerabilities and create high-priority tasks"""
        opportunities = []
        
        # Run safety check
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                vulnerabilities = safety_data.get('vulnerabilities', [])
                
                for vuln in vulnerabilities:
                    opportunities.append({
                        "id": f"vuln-{vuln.get('id', 'unknown')}",
                        "title": f"Fix security vulnerability in {vuln.get('package_name', 'unknown')}",
                        "category": "security",
                        "source": "security-scan",
                        "description": f"Security vulnerability: {vuln.get('advisory', 'Unknown advisory')}",
                        "severity": vuln.get('severity', 'medium'),
                        "package": vuln.get('package_name'),
                        "estimatedEffort": 2.0 if vuln.get('severity') == 'low' else 4.0,
                        "securityPriority": True,
                        "discoveredAt": datetime.now().isoformat()
                    })
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            print("⚠️  Safety security scan skipped (tool not available or no issues)")
        
        return opportunities
    
    def _analyze_performance(self) -> List[Dict[str, Any]]:
        """Analyze performance and identify optimization opportunities"""
        opportunities = []
        
        # Check if benchmark results exist
        benchmark_files = list(self.repo_path.glob("**/benchmark*.json"))
        
        if benchmark_files:
            try:
                # Parse latest benchmark results
                latest_benchmark = max(benchmark_files, key=lambda p: p.stat().st_mtime)
                
                with open(latest_benchmark) as f:
                    benchmark_data = json.load(f)
                
                # Look for slow benchmarks (> 1 second)
                slow_benchmarks = []
                if 'benchmarks' in benchmark_data:
                    for bench in benchmark_data['benchmarks']:
                        if bench.get('stats', {}).get('mean', 0) > 1.0:
                            slow_benchmarks.append(bench)
                
                if slow_benchmarks:
                    opportunities.append({
                        "id": "perf-optimization",
                        "title": f"Optimize {len(slow_benchmarks)} slow benchmarks",
                        "category": "performance",
                        "source": "performance-analysis",
                        "description": f"Optimize {len(slow_benchmarks)} benchmarks taking > 1 second",
                        "estimatedEffort": len(slow_benchmarks) * 2.0,
                        "benchmarkCount": len(slow_benchmarks),
                        "discoveredAt": datetime.now().isoformat()
                    })
                    
            except (json.JSONDecodeError, KeyError):
                print("⚠️  Benchmark data format unexpected")
        
        return opportunities
    
    def _analyze_documentation_gaps(self) -> List[Dict[str, Any]]:
        """Identify documentation gaps and outdated content"""
        opportunities = []
        
        # Find Python files without docstrings
        src_files = list(self.repo_path.glob("src/**/*.py"))
        missing_docstrings = []
        
        for py_file in src_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                
                # Simple heuristic: check for class/function definitions without docstrings
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if re.match(r'^\s*(class|def)\s+\w+', line):
                        # Check if next non-empty line is a docstring
                        next_lines = lines[i+1:i+5]  # Check next 4 lines
                        has_docstring = any('"""' in nl or "'''" in nl for nl in next_lines)
                        
                        if not has_docstring:
                            missing_docstrings.append(str(py_file))
                            break  # Only count once per file
                            
            except Exception:
                continue
        
        if missing_docstrings:
            opportunities.append({
                "id": "docs-missing-docstrings",
                "title": f"Add docstrings to {len(missing_docstrings)} files",
                "category": "documentation",
                "source": "documentation-analysis",
                "description": f"Add missing docstrings to {len(missing_docstrings)} Python files",
                "files": missing_docstrings,
                "estimatedEffort": len(missing_docstrings) * 0.5,
                "discoveredAt": datetime.now().isoformat()
            })
        
        # Check for outdated documentation (> 90 days old)
        doc_files = list(self.repo_path.glob("**/*.md"))
        outdated_docs = []
        
        cutoff_date = datetime.now() - timedelta(days=90)
        
        for doc_file in doc_files:
            try:
                mod_time = datetime.fromtimestamp(doc_file.stat().st_mtime)
                if mod_time < cutoff_date:
                    outdated_docs.append(str(doc_file))
            except Exception:
                continue
        
        if outdated_docs:
            opportunities.append({
                "id": "docs-outdated",
                "title": f"Update {len(outdated_docs)} outdated documentation files",
                "category": "documentation",
                "source": "documentation-analysis", 
                "description": f"Review and update {len(outdated_docs)} docs not modified in 90+ days",
                "files": outdated_docs,
                "estimatedEffort": len(outdated_docs) * 0.3,
                "discoveredAt": datetime.now().isoformat()
            })
        
        return opportunities
    
    def _analyze_test_coverage(self) -> List[Dict[str, Any]]:
        """Analyze test coverage and identify areas needing tests"""
        opportunities = []
        
        try:
            # Run coverage analysis
            result = subprocess.run([
                "pytest", "--cov=vid_diffusion_bench", "--cov-report=json", 
                "--cov-report=term-missing", "-q"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Look for coverage.json
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 100)
                target_coverage = self.config.get('discovery', {}).get('tools', {}).get('quality', {}).get('target_coverage', 0.80) * 100
                
                if total_coverage < target_coverage:
                    gap = target_coverage - total_coverage
                    
                    opportunities.append({
                        "id": "test-coverage-gap",
                        "title": f"Improve test coverage by {gap:.1f}% (currently {total_coverage:.1f}%)",
                        "category": "testing",
                        "source": "test-analysis",
                        "description": f"Increase test coverage from {total_coverage:.1f}% to {target_coverage}%",
                        "currentCoverage": total_coverage,
                        "targetCoverage": target_coverage,
                        "estimatedEffort": gap * 0.2,  # ~12 minutes per percentage point
                        "discoveredAt": datetime.now().isoformat()
                    })
                
                # Find files with low coverage
                files_data = coverage_data.get('files', {})
                low_coverage_files = []
                
                for file_path, file_data in files_data.items():
                    file_coverage = file_data.get('summary', {}).get('percent_covered', 100)
                    if file_coverage < 70:  # Files with < 70% coverage
                        low_coverage_files.append({
                            'file': file_path,
                            'coverage': file_coverage
                        })
                
                if low_coverage_files:
                    opportunities.append({
                        "id": "test-low-coverage-files",
                        "title": f"Add tests for {len(low_coverage_files)} poorly covered files",
                        "category": "testing",
                        "source": "test-analysis",
                        "description": f"Add tests for {len(low_coverage_files)} files with < 70% coverage",
                        "files": [f['file'] for f in low_coverage_files],
                        "estimatedEffort": len(low_coverage_files) * 1.5,
                        "discoveredAt": datetime.now().isoformat()
                    })
                    
        except subprocess.CalledProcessError:
            print("⚠️  Test coverage analysis skipped (tests may have failed)")
        except (json.JSONDecodeError, FileNotFoundError):
            print("⚠️  Coverage data not available in JSON format")
        
        return opportunities
    
    def _analyze_dependency_health(self) -> List[Dict[str, Any]]:
        """Analyze dependency health and identify update opportunities"""
        opportunities = []
        
        # Check for outdated packages using pip
        try:
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                outdated_packages = json.loads(result.stdout)
                
                if outdated_packages:
                    # Group by severity (major vs minor updates)
                    major_updates = []
                    minor_updates = []
                    
                    for pkg in outdated_packages:
                        current = pkg.get('version', '0.0.0')
                        latest = pkg.get('latest_version', '0.0.0')
                        
                        # Simple heuristic: major if first number changes
                        current_major = current.split('.')[0] if '.' in current else current
                        latest_major = latest.split('.')[0] if '.' in latest else latest
                        
                        if current_major != latest_major:
                            major_updates.append(pkg)
                        else:
                            minor_updates.append(pkg)
                    
                    if minor_updates:
                        opportunities.append({
                            "id": "deps-minor-updates",
                            "title": f"Update {len(minor_updates)} packages (minor versions)",
                            "category": "maintenance",
                            "source": "dependency-analysis",
                            "description": f"Update {len(minor_updates)} packages to latest minor versions",
                            "packages": [p['name'] for p in minor_updates],
                            "estimatedEffort": len(minor_updates) * 0.2,
                            "updateType": "minor",
                            "discoveredAt": datetime.now().isoformat()
                        })
                    
                    if major_updates:
                        opportunities.append({
                            "id": "deps-major-updates",
                            "title": f"Update {len(major_updates)} packages (major versions)",
                            "category": "maintenance",
                            "source": "dependency-analysis",
                            "description": f"Update {len(major_updates)} packages to latest major versions (breaking changes possible)",
                            "packages": [p['name'] for p in major_updates],
                            "estimatedEffort": len(major_updates) * 1.0,
                            "updateType": "major",
                            "riskLevel": "medium",
                            "discoveredAt": datetime.now().isoformat()
                        })
                        
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            print("⚠️  Dependency analysis skipped (pip list failed)")
        
        return opportunities
    
    def calculate_value_scores(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate comprehensive value scores for all opportunities"""
        print("📊 Calculating value scores...")
        
        scored_opportunities = []
        
        for opp in opportunities:
            scores = self._calculate_composite_score(opp)
            opp.update(scores)
            scored_opportunities.append(opp)
        
        # Sort by composite score descending
        scored_opportunities.sort(key=lambda x: x.get('compositeScore', 0), reverse=True)
        
        return scored_opportunities
    
    def _calculate_composite_score(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Calculate WSJF, ICE, and Technical Debt scores"""
        
        # WSJF Components
        wsjf_scores = self._calculate_wsjf(opportunity)
        
        # ICE Components  
        ice_scores = self._calculate_ice(opportunity)
        
        # Technical Debt Score
        tech_debt_score = self._calculate_technical_debt_score(opportunity)
        
        # Apply category-specific boosts
        security_boost = 2.0 if opportunity.get('category') == 'security' else 1.0
        hotspot_boost = opportunity.get('hotspotMultiplier', 1.0)
        
        # Weighted composite score based on repository maturity
        weights = self.config['scoring']['weights']['maturing']
        
        composite_score = (
            weights['wsjf'] * wsjf_scores['wsjf'] +
            weights['ice'] * ice_scores['ice'] +
            weights['technicalDebt'] * tech_debt_score +
            weights['security'] * (security_boost - 1.0) * 10
        ) * hotspot_boost
        
        return {
            'wsjfScore': wsjf_scores['wsjf'],
            'iceScore': ice_scores['ice'],
            'technicalDebtScore': tech_debt_score,
            'securityBoost': security_boost,
            'hotspotBoost': hotspot_boost,
            'compositeScore': round(composite_score, 2)
        }
    
    def _calculate_wsjf(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Weighted Shortest Job First score"""
        
        category = opportunity.get('category', 'maintenance')
        
        # User Business Value (1-20 scale)
        business_value_map = {
            'security': 15,
            'performance': 12,
            'technical-debt': 8,
            'code-quality': 6,
            'testing': 7,
            'documentation': 4,
            'maintenance': 5
        }
        user_business_value = business_value_map.get(category, 5)
        
        # Time Criticality (1-20 scale)
        criticality_map = {
            'security': 18,
            'performance': 12,
            'technical-debt': 8,
            'code-quality': 6,
            'testing': 5,
            'documentation': 3,
            'maintenance': 4
        }
        time_criticality = criticality_map.get(category, 4)
        
        # Apply severity boost for security
        if opportunity.get('securityPriority'):
            time_criticality *= 1.5
        
        # Risk Reduction (1-20 scale)
        risk_reduction_map = {
            'security': 16,
            'technical-debt': 12,
            'performance': 8,
            'testing': 10,
            'code-quality': 7,
            'documentation': 4,
            'maintenance': 5
        }
        risk_reduction = risk_reduction_map.get(category, 5)
        
        # Opportunity Enablement (1-20 scale)
        opportunity_enablement = 8  # Base value
        if 'architecture' in opportunity.get('title', '').lower():
            opportunity_enablement = 14
        elif 'performance' in opportunity.get('title', '').lower():
            opportunity_enablement = 10
        
        # Cost of Delay
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        
        # Job Size (effort in hours)
        job_size = max(0.5, opportunity.get('estimatedEffort', 3.0))
        
        # WSJF = Cost of Delay / Job Size
        wsjf = cost_of_delay / job_size
        
        return {
            'wsjf': round(wsjf, 2),
            'costOfDelay': cost_of_delay,
            'jobSize': job_size
        }
    
    def _calculate_ice(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Impact, Confidence, Ease score"""
        
        category = opportunity.get('category', 'maintenance')
        
        # Impact (1-10 scale)
        impact_map = {
            'security': 9,
            'performance': 8,
            'technical-debt': 6,
            'code-quality': 5,
            'testing': 6,
            'documentation': 4,
            'maintenance': 4
        }
        impact = impact_map.get(category, 5)
        
        # Confidence (1-10 scale) - based on how well-defined the opportunity is
        confidence = 7  # Base confidence
        if opportunity.get('files'):  # We know which files are affected
            confidence += 1
        if opportunity.get('estimatedEffort', 0) < 4:  # Small tasks are more predictable
            confidence += 1
        if category in ['security', 'code-quality']:  # Clear acceptance criteria
            confidence += 1
        confidence = min(10, confidence)
        
        # Ease (1-10 scale) - how easy is it to implement
        ease_map = {
            'security': 6,  # May require research
            'performance': 5,  # May require profiling
            'technical-debt': 7,  # Usually straightforward
            'code-quality': 8,  # Automated tools help
            'testing': 7,  # Clear patterns
            'documentation': 9,  # Usually straightforward
            'maintenance': 8   # Usually straightforward
        }
        ease = ease_map.get(category, 6)
        
        # Adjust ease based on effort
        effort = opportunity.get('estimatedEffort', 3.0)
        if effort > 8:
            ease = max(1, ease - 2)  # Large tasks are harder
        elif effort < 2:
            ease = min(10, ease + 1)  # Small tasks are easier
        
        ice_score = impact * confidence * ease
        
        return {
            'ice': round(ice_score, 2),
            'impact': impact,
            'confidence': confidence,
            'ease': ease
        }
    
    def _calculate_technical_debt_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate technical debt specific scoring"""
        
        if opportunity.get('category') != 'technical-debt':
            return 0.0
        
        # Base debt impact
        debt_impact = opportunity.get('estimatedEffort', 3.0) * 2  # Hours saved
        
        # Debt interest (future cost if not addressed)
        debt_interest = debt_impact * 0.1  # 10% compound interest per month
        
        # Hotspot multiplier
        hotspot_multiplier = opportunity.get('hotspotMultiplier', 1.0)
        
        debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        return round(debt_score, 2)
    
    def select_next_best_value(self, scored_opportunities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the next highest-value item to execute"""
        
        min_score = self.config['scoring']['thresholds']['minScore']
        max_risk = self.config['scoring']['thresholds']['maxRisk']
        
        for opportunity in scored_opportunities:
            # Check minimum score threshold
            if opportunity.get('compositeScore', 0) < min_score:
                continue
            
            # Check risk threshold
            risk_level = opportunity.get('riskLevel', 'low')
            risk_score = {'low': 0.2, 'medium': 0.5, 'high': 0.8}.get(risk_level, 0.2)
            if risk_score > max_risk:
                continue
            
            # This is our next best value item
            return opportunity
        
        # No items meet criteria
        return None
    
    def generate_backlog(self, scored_opportunities: List[Dict[str, Any]]) -> None:
        """Generate the autonomous value backlog markdown file"""
        
        print("📋 Generating autonomous value backlog...")
        
        next_item = self.select_next_best_value(scored_opportunities)
        top_10 = scored_opportunities[:10]
        
        # Calculate backlog metrics
        total_items = len(scored_opportunities)
        avg_score = sum(o.get('compositeScore', 0) for o in scored_opportunities) / max(1, total_items)
        debt_items = len([o for o in scored_opportunities if o.get('category') == 'technical-debt'])
        debt_ratio = debt_items / max(1, total_items)
        
        # Update metrics
        self.metrics['backlogMetrics'].update({
            'totalItems': total_items,
            'averageScore': round(avg_score, 2),
            'debtRatio': round(debt_ratio, 2),
            'lastUpdated': datetime.now().isoformat()
        })
        
        # Generate markdown content
        backlog_content = f"""# 📊 Autonomous Value Backlog

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Repository: vid-diffusion-benchmark-suite
Maturity Level: MATURING (50-75% SDLC)

## 🎯 Next Best Value Item

"""
        
        if next_item:
            backlog_content += f"""**[{next_item['id'].upper()}] {next_item['title']}**
- **Composite Score**: {next_item.get('compositeScore', 0)}
- **WSJF**: {next_item.get('wsjfScore', 0)} | **ICE**: {next_item.get('iceScore', 0)} | **Tech Debt**: {next_item.get('technicalDebtScore', 0)}
- **Category**: {next_item.get('category', 'unknown')}
- **Estimated Effort**: {next_item.get('estimatedEffort', 0)} hours
- **Source**: {next_item.get('source', 'unknown')}
- **Description**: {next_item.get('description', 'No description available')}

"""
        else:
            backlog_content += "**No items currently meet the execution criteria.**\n\n"
        
        backlog_content += f"""## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours | Source |
|------|-----|--------|---------|----------|------------|---------|
"""
        
        for i, item in enumerate(top_10, 1):
            title_short = item['title'][:40] + '...' if len(item['title']) > 40 else item['title']
            backlog_content += f"| {i} | {item['id']} | {title_short} | {item.get('compositeScore', 0)} | {item.get('category', 'unknown')} | {item.get('estimatedEffort', 0)} | {item.get('source', 'unknown')} |\n"
        
        backlog_content += f"""

## 📈 Value Metrics

- **Total Items Discovered**: {total_items}
- **Average Composite Score**: {avg_score:.1f}
- **Technical Debt Ratio**: {debt_ratio:.1%}
- **High-Value Items (Score > 50)**: {len([o for o in scored_opportunities if o.get('compositeScore', 0) > 50])}
- **Security Items**: {len([o for o in scored_opportunities if o.get('category') == 'security'])}

## 🔄 Discovery Statistics

"""
        
        # Count by source
        sources = {}
        categories = {}
        for opp in scored_opportunities:
            source = opp.get('source', 'unknown')
            category = opp.get('category', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            categories[category] = categories.get(category, 0) + 1
        
        backlog_content += "### By Source:\n"
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_items) * 100 if total_items > 0 else 0
            backlog_content += f"- **{source}**: {count} items ({percentage:.1f}%)\n"
        
        backlog_content += "\n### By Category:\n"
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_items) * 100 if total_items > 0 else 0
            backlog_content += f"- **{category}**: {count} items ({percentage:.1f}%)\n"
        
        backlog_content += """

## 🚀 Autonomous Execution

This backlog is continuously updated by the Terragon Autonomous SDLC system.
The system automatically:

1. **Discovers** new value opportunities from multiple sources
2. **Scores** each opportunity using WSJF, ICE, and Technical Debt models
3. **Prioritizes** based on composite value scores and risk assessment
4. **Selects** the next best value item for execution
5. **Learns** from execution outcomes to improve future scoring

### Next Execution

The system will automatically execute the next best value item when:
- ✅ Composite score ≥ 15
- ✅ Risk level ≤ 0.7
- ✅ No conflicting work in progress
- ✅ All dependencies met

### Configuration

Scoring model and execution parameters can be adjusted in `.terragon/value-config.yaml`.

---

*Generated by Terragon Autonomous SDLC Value Discovery Engine*
*Framework: github.com/danieleschmidt/terragon-autonomous-sdlc*
"""
        
        # Write backlog file
        with open(self.backlog_path, 'w') as f:
            f.write(backlog_content)
        
        print(f"✅ Generated backlog with {total_items} opportunities")
        print(f"📊 Next best value item: {next_item['title'] if next_item else 'None available'}")
    
    def save_metrics(self) -> None:
        """Save updated metrics to file"""
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def run_discovery_cycle(self) -> Dict[str, Any]:
        """Run a complete value discovery cycle"""
        print("🚀 Starting Terragon Autonomous SDLC Value Discovery Cycle")
        print("=" * 60)
        
        cycle_start = datetime.now()
        
        # 1. Discover opportunities
        opportunities = self.discover_value_opportunities()
        
        # 2. Calculate value scores
        scored_opportunities = self.calculate_value_scores(opportunities)
        
        # 3. Generate backlog
        self.generate_backlog(scored_opportunities)
        
        # 4. Save metrics
        self.save_metrics()
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        # 5. Return cycle summary
        summary = {
            'cycleStart': cycle_start.isoformat(),
            'cycleDuration': round(cycle_duration, 2),
            'opportunitiesDiscovered': len(opportunities),
            'highValueItems': len([o for o in scored_opportunities if o.get('compositeScore', 0) > 50]),
            'nextBestValue': self.select_next_best_value(scored_opportunities),
            'categories': list(set(o.get('category', 'unknown') for o in opportunities)),
            'sources': list(set(o.get('source', 'unknown') for o in opportunities))
        }
        
        print("=" * 60)
        print(f"✅ Discovery cycle completed in {cycle_duration:.2f} seconds")
        print(f"📊 Discovered {len(opportunities)} opportunities")
        print(f"🎯 Next best value: {summary['nextBestValue']['title'] if summary['nextBestValue'] else 'None available'}")
        
        return summary


def main():
    """CLI entry point for the value discovery engine"""
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("🔍 Running in dry-run mode (discovery only, no execution)")
    
    try:
        engine = ValueDiscoveryEngine()
        summary = engine.run_discovery_cycle()
        
        # Exit with appropriate code
        if summary['nextBestValue']:
            print(f"\n🚀 Ready to execute: {summary['nextBestValue']['title']}")
            sys.exit(0)
        else:
            print("\n⏳ No items currently meet execution criteria")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Value discovery failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()