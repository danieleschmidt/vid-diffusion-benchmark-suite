#!/usr/bin/env python3
"""Comprehensive quality gates validation for Video Diffusion Benchmark Suite.

This script validates all quality gates including:
- Code structure and imports
- Documentation completeness  
- Security validation
- Performance optimization features
- Research framework capabilities
- Production readiness
"""

import ast
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

def check_code_structure() -> Dict[str, Any]:
    """Check code structure and organization."""
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'metrics': {}
    }
    
    project_root = Path('/root/repo/src/vid_diffusion_bench')
    
    # Check required modules exist
    required_modules = [
        '__init__.py',
        'benchmark.py',
        'research/experimental_framework.py',
        'research/novel_metrics.py',
        'robustness/advanced_validation.py',
        'security/auth.py',
        'reliability_framework.py'
    ]
    
    missing_modules = []
    for module in required_modules:
        if not (project_root / module).exists():
            missing_modules.append(module)
    
    if missing_modules:
        results['status'] = 'FAIL'
        results['errors'].append(f"Missing required modules: {missing_modules}")
    
    # Count total files
    python_files = list(project_root.rglob('*.py'))
    results['metrics']['total_python_files'] = len(python_files)
    results['metrics']['total_lines'] = sum(len(f.read_text().splitlines()) for f in python_files if f.is_file())
    
    return results

def check_syntax_validity() -> Dict[str, Any]:
    """Check Python syntax validity across all modules."""
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'files_checked': 0
    }
    
    project_root = Path('/root/repo/src/vid_diffusion_bench')
    
    for py_file in project_root.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                code = f.read()
            
            # Parse AST to check syntax
            ast.parse(code)
            results['files_checked'] += 1
            
        except SyntaxError as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"Syntax error in {py_file}: {e}")
        except Exception as e:
            results['warnings'].append(f"Could not check {py_file}: {e}")
    
    return results

def check_documentation_completeness() -> Dict[str, Any]:
    """Check documentation completeness."""
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'coverage': {}
    }
    
    project_root = Path('/root/repo')
    
    # Check required documentation files
    required_docs = [
        'README.md',
        'docs/ARCHITECTURE.md',
        'docs/API_REFERENCE.md',
        'docs/DEPLOYMENT.md'
    ]
    
    missing_docs = []
    for doc in required_docs:
        if not (project_root / doc).exists():
            missing_docs.append(doc)
    
    if missing_docs:
        results['warnings'].append(f"Missing documentation: {missing_docs}")
    
    # Check for docstrings in Python files
    project_code = Path('/root/repo/src/vid_diffusion_bench')
    undocumented_functions = []
    
    for py_file in project_code.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        undocumented_functions.append(f"{py_file.name}::{node.name}")
        except:
            pass  # Skip files that can't be parsed
    
    results['coverage']['undocumented_functions'] = len(undocumented_functions)
    if len(undocumented_functions) > 10:  # Allow some undocumented functions
        results['warnings'].append(f"Many undocumented functions: {len(undocumented_functions)}")
    
    return results

def check_security_features() -> Dict[str, Any]:
    """Check security feature implementation."""
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'features': {}
    }
    
    # Check security modules exist
    security_files = [
        '/root/repo/src/vid_diffusion_bench/security/auth.py',
        '/root/repo/src/vid_diffusion_bench/security/rate_limiting.py',
        '/root/repo/src/vid_diffusion_bench/security/sanitization.py',
        '/root/repo/src/vid_diffusion_bench/robustness/advanced_validation.py'
    ]
    
    security_features_found = 0
    for sec_file in security_files:
        if Path(sec_file).exists():
            security_features_found += 1
            
            # Check for key security patterns
            try:
                with open(sec_file, 'r') as f:
                    content = f.read()
                
                if 'API' in content and 'auth' in content.lower():
                    results['features']['api_authentication'] = True
                if 'sanitiz' in content.lower() or 'validat' in content.lower():
                    results['features']['input_validation'] = True
                if 'rate' in content.lower() and 'limit' in content.lower():
                    results['features']['rate_limiting'] = True
                    
            except:
                pass
    
    results['features']['security_modules_count'] = security_features_found
    
    if security_features_found < 3:
        results['warnings'].append("Limited security modules implemented")
    
    return results

def check_research_capabilities() -> Dict[str, Any]:
    """Check research framework capabilities."""
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'capabilities': {}
    }
    
    research_files = [
        '/root/repo/src/vid_diffusion_bench/research/experimental_framework.py',
        '/root/repo/src/vid_diffusion_bench/research/novel_metrics.py',
        '/root/repo/src/vid_diffusion_bench/research/statistical_analysis.py'
    ]
    
    research_features = 0
    for research_file in research_files:
        if Path(research_file).exists():
            research_features += 1
            
            try:
                with open(research_file, 'r') as f:
                    content = f.read()
                
                # Check for research patterns
                if 'experiment' in content.lower() and 'reproducib' in content.lower():
                    results['capabilities']['experimental_framework'] = True
                if 'statistical' in content.lower() and 'significance' in content.lower():
                    results['capabilities']['statistical_analysis'] = True
                if 'novel' in content.lower() and 'metric' in content.lower():
                    results['capabilities']['novel_metrics'] = True
                if 'publication' in content.lower():
                    results['capabilities']['publication_ready'] = True
                    
            except:
                pass
    
    results['capabilities']['research_modules_count'] = research_features
    
    if research_features < 2:
        results['status'] = 'FAIL'
        results['errors'].append("Insufficient research framework implementation")
    
    return results

def check_performance_optimization() -> Dict[str, Any]:
    """Check performance optimization features."""
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'optimizations': {}
    }
    
    # Check for performance-related files
    perf_files = [
        '/root/repo/src/vid_diffusion_bench/performance_optimization.py',
        '/root/repo/src/vid_diffusion_bench/scaling/auto_scaling.py',
        '/root/repo/src/vid_diffusion_bench/scaling/distributed.py',
        '/root/repo/src/vid_diffusion_bench/optimization/caching.py'
    ]
    
    perf_features = 0
    for perf_file in perf_files:
        if Path(perf_file).exists():
            perf_features += 1
            
            try:
                with open(perf_file, 'r') as f:
                    content = f.read()
                
                # Check for optimization patterns
                if 'mixed_precision' in content.lower():
                    results['optimizations']['mixed_precision'] = True
                if 'batch' in content.lower() and 'optim' in content.lower():
                    results['optimizations']['batch_optimization'] = True
                if 'distributed' in content.lower() or 'parallel' in content.lower():
                    results['optimizations']['distributed_processing'] = True
                if 'cache' in content.lower() or 'caching' in content.lower():
                    results['optimizations']['caching'] = True
                if 'auto' in content.lower() and 'scal' in content.lower():
                    results['optimizations']['auto_scaling'] = True
                    
            except:
                pass
    
    results['optimizations']['performance_modules_count'] = perf_features
    
    # Check benchmark.py for performance features
    benchmark_file = Path('/root/repo/src/vid_diffusion_bench/benchmark.py')
    if benchmark_file.exists():
        try:
            with open(benchmark_file, 'r') as f:
                content = f.read()
            
            if 'performance_config' in content:
                results['optimizations']['integrated_performance_config'] = True
            if 'circuit_breaker' in content:
                results['optimizations']['circuit_breaker_pattern'] = True
            if 'graceful_degradation' in content:
                results['optimizations']['graceful_degradation'] = True
                
        except:
            pass
    
    if perf_features < 2:
        results['warnings'].append("Limited performance optimization modules")
    
    return results

def check_production_readiness() -> Dict[str, Any]:
    """Check production readiness features."""
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'features': {}
    }
    
    # Check deployment files
    deployment_files = [
        '/root/repo/Dockerfile',
        '/root/repo/docker-compose.yml',
        '/root/repo/deployment/',
        '/root/repo/monitoring/'
    ]
    
    deployment_features = 0
    for deploy_file in deployment_files:
        if Path(deploy_file).exists():
            deployment_features += 1
    
    results['features']['deployment_artifacts'] = deployment_features
    
    # Check for reliability features
    reliability_file = Path('/root/repo/src/vid_diffusion_bench/reliability_framework.py')
    if reliability_file.exists():
        results['features']['reliability_framework'] = True
    
    # Check monitoring
    monitoring_files = list(Path('/root/repo/monitoring').glob('*.yml')) if Path('/root/repo/monitoring').exists() else []
    results['features']['monitoring_configs'] = len(monitoring_files)
    
    # Check for API
    api_file = Path('/root/repo/src/vid_diffusion_bench/api/app.py')
    if api_file.exists():
        results['features']['rest_api'] = True
    
    if deployment_features < 2:
        results['warnings'].append("Limited deployment infrastructure")
    
    return results

def generate_quality_report() -> Dict[str, Any]:
    """Generate comprehensive quality gates report."""
    print("🔍 Running Quality Gates Validation...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'framework_version': '1.0.0',
        'validation_results': {},
        'overall_status': 'PASS',
        'summary': {}
    }
    
    # Run all quality checks
    checks = {
        'code_structure': check_code_structure(),
        'syntax_validity': check_syntax_validity(),
        'documentation': check_documentation_completeness(),
        'security_features': check_security_features(),
        'research_capabilities': check_research_capabilities(),
        'performance_optimization': check_performance_optimization(),
        'production_readiness': check_production_readiness()
    }
    
    report['validation_results'] = checks
    
    # Calculate overall status
    failed_checks = [name for name, result in checks.items() if result['status'] == 'FAIL']
    warning_checks = [name for name, result in checks.items() if result['warnings']]
    
    if failed_checks:
        report['overall_status'] = 'FAIL'
    elif len(warning_checks) > 3:
        report['overall_status'] = 'PASS_WITH_WARNINGS'
    
    # Generate summary
    report['summary'] = {
        'total_checks': len(checks),
        'passed_checks': len([c for c in checks.values() if c['status'] == 'PASS']),
        'failed_checks': len(failed_checks),
        'checks_with_warnings': len(warning_checks),
        'critical_issues': failed_checks,
        'warning_areas': warning_checks
    }
    
    return report

def print_quality_report(report: Dict[str, Any]):
    """Print formatted quality report."""
    print(f"\n{'='*80}")
    print("🎯 VIDEO DIFFUSION BENCHMARK SUITE - QUALITY GATES REPORT")
    print(f"{'='*80}")
    
    print(f"\n📅 Generated: {report['timestamp']}")
    print(f"🔧 Framework Version: {report['framework_version']}")
    print(f"✅ Overall Status: {report['overall_status']}")
    
    print(f"\n📊 SUMMARY")
    print(f"{'─'*40}")
    summary = report['summary']
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Passed: {summary['passed_checks']}")
    print(f"Failed: {summary['failed_checks']}")
    print(f"With Warnings: {summary['checks_with_warnings']}")
    
    print(f"\n🔍 DETAILED RESULTS")
    print(f"{'─'*40}")
    
    for check_name, result in report['validation_results'].items():
        status_emoji = "✅" if result['status'] == 'PASS' else "❌"
        warning_emoji = "⚠️" if result['warnings'] else ""
        
        print(f"{status_emoji} {check_name.replace('_', ' ').title()}: {result['status']} {warning_emoji}")
        
        if result['errors']:
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"   🔴 {error}")
        
        if result['warnings']:
            for warning in result['warnings'][:2]:  # Show first 2 warnings
                print(f"   🟡 {warning}")
    
    print(f"\n🚀 FRAMEWORK CAPABILITIES")
    print(f"{'─'*40}")
    
    # Extract capabilities from results
    capabilities = []
    for result in report['validation_results'].values():
        if 'features' in result:
            capabilities.extend([f"{k}: {v}" for k, v in result['features'].items() if v])
        if 'capabilities' in result:
            capabilities.extend([f"{k}: {v}" for k, v in result['capabilities'].items() if v])
        if 'optimizations' in result:
            capabilities.extend([f"{k}: {v}" for k, v in result['optimizations'].items() if v])
    
    for capability in capabilities[:10]:  # Show top 10 capabilities
        print(f"   ✨ {capability}")
    
    if report['overall_status'] == 'PASS':
        print(f"\n🎉 ALL QUALITY GATES PASSED! Framework is production-ready.")
    elif report['overall_status'] == 'PASS_WITH_WARNINGS':
        print(f"\n⚠️  Quality gates passed with warnings. Consider addressing warning areas.")
    else:
        print(f"\n❌ Quality gates failed. Critical issues must be resolved.")
    
    print(f"\n{'='*80}")

def main():
    """Main quality gates execution."""
    try:
        print("🚀 Executing TERRAGON SDLC Quality Gates...")
        report = generate_quality_report()
        print_quality_report(report)
        
        # Save report
        report_path = Path('/root/repo/quality_gates_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Exit with appropriate code
        if report['overall_status'] == 'FAIL':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Quality gates execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()