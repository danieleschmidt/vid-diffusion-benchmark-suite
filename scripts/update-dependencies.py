#!/usr/bin/env python3
"""
Automated dependency update script for Video Diffusion Benchmark Suite.

This script provides manual dependency management capabilities and can be run
in CI/CD pipelines for automated dependency analysis and updates.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import tomllib


def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def get_current_dependencies() -> Dict[str, str]:
    """Extract current dependencies from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("ERROR: pyproject.toml not found")
        sys.exit(1)
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    deps = {}
    
    # Main dependencies
    if "project" in data and "dependencies" in data["project"]:
        for dep in data["project"]["dependencies"]:
            name = dep.split(">=")[0].split("==")[0].split(">")[0].split("<")[0]
            version = dep.replace(name, "").lstrip(">=").lstrip("==").lstrip(">").lstrip("<")
            deps[name] = version or "unknown"
    
    # Optional dependencies
    if "project" in data and "optional-dependencies" in data["project"]:
        for group, group_deps in data["project"]["optional-dependencies"].items():
            for dep in group_deps:
                name = dep.split(">=")[0].split("==")[0].split(">")[0].split("<")[0]
                version = dep.replace(name, "").lstrip(">=").lstrip("==").lstrip(">").lstrip("<")
                deps[f"{name} ({group})"] = version or "unknown"
    
    return deps


def check_outdated_packages() -> List[Dict[str, str]]:
    """Check for outdated packages using pip."""
    print("🔍 Checking for outdated packages...")
    
    code, stdout, stderr = run_command(["pip", "list", "--outdated", "--format=json"])
    
    if code != 0:
        print(f"ERROR: Failed to check outdated packages: {stderr}")
        return []
    
    try:
        outdated = json.loads(stdout)
        return outdated
    except json.JSONDecodeError:
        print("ERROR: Failed to parse pip output")
        return []


def check_security_vulnerabilities() -> List[Dict[str, str]]:
    """Check for security vulnerabilities using safety."""
    print("🔒 Checking for security vulnerabilities...")
    
    code, stdout, stderr = run_command(["safety", "check", "--json"])
    
    if code == 0:
        print("✅ No security vulnerabilities found")
        return []
    
    try:
        # Safety returns non-zero exit code when vulnerabilities are found
        if "No known security vulnerabilities found" in stderr:
            return []
        
        vulnerabilities = json.loads(stdout) if stdout else []
        return vulnerabilities
    except json.JSONDecodeError:
        print("⚠️  Could not parse safety output, manual review recommended")
        return []


def analyze_license_compatibility() -> Dict[str, List[str]]:
    """Analyze license compatibility of dependencies."""
    print("📄 Analyzing license compatibility...")
    
    code, stdout, stderr = run_command(["pip-licenses", "--format=json"])
    
    if code != 0:
        print("⚠️  pip-licenses not available, skipping license analysis")
        return {}
    
    try:
        licenses = json.loads(stdout)
        license_groups = {}
        
        for pkg in licenses:
            license_name = pkg.get("License", "Unknown")
            if license_name not in license_groups:
                license_groups[license_name] = []
            license_groups[license_name].append(pkg["Name"])
        
        return license_groups
    except json.JSONDecodeError:
        print("⚠️  Could not parse license information")
        return {}


def generate_dependency_report() -> None:
    """Generate comprehensive dependency report."""
    print("📊 Generating dependency report...\n")
    
    # Current dependencies
    current_deps = get_current_dependencies()
    print(f"📦 Current Dependencies ({len(current_deps)}):")
    for name, version in sorted(current_deps.items()):
        print(f"  • {name}: {version}")
    print()
    
    # Outdated packages
    outdated = check_outdated_packages()
    if outdated:
        print(f"⚠️  Outdated Packages ({len(outdated)}):")
        for pkg in outdated:
            print(f"  • {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
        print()
    else:
        print("✅ All packages are up to date\n")
    
    # Security vulnerabilities
    vulnerabilities = check_security_vulnerabilities()
    if vulnerabilities:
        print(f"🚨 Security Vulnerabilities ({len(vulnerabilities)}):")
        for vuln in vulnerabilities:
            print(f"  • {vuln.get('package', 'Unknown')}: {vuln.get('vulnerability', 'Unknown issue')}")
        print()
    else:
        print("✅ No known security vulnerabilities\n")
    
    # License analysis
    licenses = analyze_license_compatibility()
    if licenses:
        print("📄 License Distribution:")
        for license_name, packages in licenses.items():
            print(f"  • {license_name}: {len(packages)} packages")
            if license_name in ["GPL", "AGPL", "LGPL"]:
                print("    ⚠️  Copyleft license - review compatibility")
        print()


def update_pre_commit_hooks() -> None:
    """Update pre-commit hook versions."""
    print("🔄 Updating pre-commit hooks...")
    
    code, stdout, stderr = run_command(["pre-commit", "autoupdate"])
    
    if code == 0:
        print("✅ Pre-commit hooks updated successfully")
        print(stdout)
    else:
        print(f"❌ Failed to update pre-commit hooks: {stderr}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dependency management for vid-diffusion-bench")
    parser.add_argument(
        "--action",
        choices=["report", "update-hooks", "check-security", "all"],
        default="report",
        help="Action to perform"
    )
    parser.add_argument(
        "--output",
        help="Output file for dependency report (JSON format)"
    )
    
    args = parser.parse_args()
    
    if args.action in ["report", "all"]:
        generate_dependency_report()
    
    if args.action in ["update-hooks", "all"]:
        update_pre_commit_hooks()
    
    if args.action in ["check-security", "all"]:
        vulnerabilities = check_security_vulnerabilities()
        if vulnerabilities:
            print("🚨 Security vulnerabilities found!")
            sys.exit(1)
    
    if args.output:
        # Generate JSON report
        report_data = {
            "current_dependencies": get_current_dependencies(),
            "outdated_packages": check_outdated_packages(),
            "security_vulnerabilities": check_security_vulnerabilities(),
            "license_analysis": analyze_license_compatibility()
        }
        
        with open(args.output, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Report saved to {args.output}")


if __name__ == "__main__":
    main()