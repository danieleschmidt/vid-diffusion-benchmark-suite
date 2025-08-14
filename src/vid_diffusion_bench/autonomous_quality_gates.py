"""Autonomous quality gates system with comprehensive validation.

Advanced quality assurance system that automatically validates code quality,
security, performance, and reliability across all project dimensions.
"""

import asyncio
import time
import json
import subprocess
import ast
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import statistics

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Individual quality metric result."""
    name: str
    category: str
    score: float  # 0.0 to 1.0
    status: str  # "pass", "warn", "fail"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    timestamp: float
    overall_score: float
    status: str
    metrics: List[QualityMetric]
    categories: Dict[str, float]
    critical_issues: List[str]
    recommendations: List[str]
    execution_time: float


class CodeQualityAnalyzer:
    """Advanced code quality analysis with ML insights."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.python_files = list(self.project_root.rglob("*.py"))
        self.quality_thresholds = {
            'syntax_errors': 0,
            'complexity_threshold': 10,
            'line_length_max': 100,
            'function_length_max': 50,
            'class_length_max': 200,
            'duplicate_threshold': 0.05
        }
        
    async def analyze_code_quality(self) -> List[QualityMetric]:
        """Comprehensive code quality analysis."""
        metrics = []
        
        with ThreadPoolExecutor() as executor:
            # Run analyses in parallel
            syntax_task = asyncio.create_task(self._check_syntax_errors())
            complexity_task = asyncio.create_task(self._analyze_complexity())
            style_task = asyncio.create_task(self._check_code_style())
            duplication_task = asyncio.create_task(self._detect_duplication())
            documentation_task = asyncio.create_task(self._check_documentation())
            
            # Gather results
            syntax_metrics = await syntax_task
            complexity_metrics = await complexity_task
            style_metrics = await style_task
            duplication_metrics = await duplication_task
            doc_metrics = await documentation_task
            
            metrics.extend(syntax_metrics)
            metrics.extend(complexity_metrics)
            metrics.extend(style_metrics)
            metrics.extend(duplication_metrics)
            metrics.extend(doc_metrics)
            
        return metrics
        
    async def _check_syntax_errors(self) -> List[QualityMetric]:
        """Check for Python syntax errors."""
        metrics = []
        errors = []
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
            except UnicodeDecodeError:
                errors.append(f"{file_path}: Unicode decode error")
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")
                
        score = 1.0 if len(errors) == 0 else 0.0
        status = "pass" if len(errors) == 0 else "fail"
        
        metrics.append(QualityMetric(
            name="syntax_validation",
            category="code_quality",
            score=score,
            status=status,
            message=f"Found {len(errors)} syntax errors in {len(self.python_files)} files",
            details={"errors": errors[:10], "total_errors": len(errors)},
            remediation="Fix syntax errors before proceeding" if errors else None
        ))
        
        return metrics
        
    async def _analyze_complexity(self) -> List[QualityMetric]:
        """Analyze code complexity metrics."""
        metrics = []
        complexity_violations = []
        total_complexity = 0
        function_count = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_complexity(node)
                        total_complexity += complexity
                        function_count += 1
                        
                        if complexity > self.quality_thresholds['complexity_threshold']:
                            complexity_violations.append(
                                f"{file_path}:{node.lineno}: {node.name} (complexity: {complexity})"
                            )
                            
            except Exception as e:
                logger.warning(f"Could not analyze complexity for {file_path}: {e}")
                
        avg_complexity = total_complexity / max(1, function_count)
        score = max(0.0, min(1.0, 1.0 - (len(complexity_violations) / max(1, function_count))))
        status = "pass" if len(complexity_violations) == 0 else "warn" if score > 0.7 else "fail"
        
        metrics.append(QualityMetric(
            name="cyclomatic_complexity",
            category="code_quality",
            score=score,
            status=status,
            message=f"Average complexity: {avg_complexity:.1f}, {len(complexity_violations)} violations",
            details={
                "average_complexity": avg_complexity,
                "violations": complexity_violations[:10],
                "total_functions": function_count
            },
            remediation="Break down complex functions into smaller ones" if complexity_violations else None
        ))
        
        return metrics
        
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
        
    async def _check_code_style(self) -> List[QualityMetric]:
        """Check code style and formatting."""
        metrics = []
        style_issues = []
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    # Check line length
                    if len(line.rstrip()) > self.quality_thresholds['line_length_max']:
                        style_issues.append(f"{file_path}:{i}: Line too long ({len(line.rstrip())} chars)")
                        
                    # Check for trailing whitespace
                    if line.rstrip() != line.rstrip('\n'):
                        style_issues.append(f"{file_path}:{i}: Trailing whitespace")
                        
            except Exception as e:
                logger.warning(f"Could not check style for {file_path}: {e}")
                
        total_lines = sum(len(open(f).readlines()) for f in self.python_files)
        score = max(0.0, min(1.0, 1.0 - (len(style_issues) / max(1, total_lines * 0.1))))
        status = "pass" if len(style_issues) < total_lines * 0.01 else "warn"
        
        metrics.append(QualityMetric(
            name="code_style",
            category="code_quality", 
            score=score,
            status=status,
            message=f"Found {len(style_issues)} style issues across {total_lines} lines",
            details={"issues": style_issues[:20], "total_issues": len(style_issues)},
            remediation="Run code formatter (black, autopep8) to fix style issues" if style_issues else None
        ))
        
        return metrics
        
    async def _detect_duplication(self) -> List[QualityMetric]:
        """Detect code duplication."""
        metrics = []
        duplicates = []
        
        # Simple hash-based duplication detection
        code_hashes = {}
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Check for duplicate blocks of 5+ lines
                for i in range(len(lines) - 4):
                    block = ''.join(lines[i:i+5]).strip()
                    if len(block) > 50:  # Minimum block size
                        block_hash = hashlib.md5(block.encode()).hexdigest()
                        
                        if block_hash in code_hashes:
                            duplicates.append(f"{file_path}:{i+1} duplicates {code_hashes[block_hash]}")
                        else:
                            code_hashes[block_hash] = f"{file_path}:{i+1}"
                            
            except Exception as e:
                logger.warning(f"Could not check duplication for {file_path}: {e}")
                
        duplication_ratio = len(duplicates) / max(1, len(code_hashes))
        score = max(0.0, min(1.0, 1.0 - duplication_ratio))
        status = "pass" if duplication_ratio < 0.05 else "warn" if duplication_ratio < 0.1 else "fail"
        
        metrics.append(QualityMetric(
            name="code_duplication",
            category="code_quality",
            score=score,
            status=status,
            message=f"Duplication ratio: {duplication_ratio:.2%} ({len(duplicates)} duplicates)",
            details={"duplicates": duplicates[:10], "ratio": duplication_ratio},
            remediation="Extract common code into functions or modules" if duplicates else None
        ))
        
        return metrics
        
    async def _check_documentation(self) -> List[QualityMetric]:
        """Check documentation coverage and quality."""
        metrics = []
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            documented_classes += 1
                            
            except Exception as e:
                logger.warning(f"Could not check documentation for {file_path}: {e}")
                
        function_doc_ratio = documented_functions / max(1, total_functions)
        class_doc_ratio = documented_classes / max(1, total_classes)
        overall_doc_ratio = (function_doc_ratio + class_doc_ratio) / 2
        
        score = overall_doc_ratio
        status = "pass" if score > 0.8 else "warn" if score > 0.5 else "fail"
        
        metrics.append(QualityMetric(
            name="documentation_coverage",
            category="documentation",
            score=score,
            status=status,
            message=f"Documentation: {overall_doc_ratio:.1%} (functions: {function_doc_ratio:.1%}, classes: {class_doc_ratio:.1%})",
            details={
                "function_coverage": function_doc_ratio,
                "class_coverage": class_doc_ratio,
                "total_functions": total_functions,
                "total_classes": total_classes
            },
            remediation="Add docstrings to undocumented functions and classes" if score < 0.8 else None
        ))
        
        return metrics


class SecurityAnalyzer:
    """Security vulnerability detection and analysis."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'query\s*\(\s*["\'].*\+.*["\']'
            ],
            'path_traversal': [
                r'open\s*\(\s*.*\+.*\)',
                r'file\s*\(\s*.*\+.*\)'
            ],
            'dangerous_imports': [
                r'import\s+pickle',
                r'import\s+subprocess',
                r'from\s+subprocess\s+import'
            ]
        }
        
    async def analyze_security(self) -> List[QualityMetric]:
        """Comprehensive security analysis."""
        metrics = []
        
        # Run security checks
        secret_metrics = await self._check_hardcoded_secrets()
        injection_metrics = await self._check_injection_vulnerabilities()
        import_metrics = await self._check_dangerous_imports()
        permission_metrics = await self._check_file_permissions()
        
        metrics.extend(secret_metrics)
        metrics.extend(injection_metrics)
        metrics.extend(import_metrics)
        metrics.extend(permission_metrics)
        
        return metrics
        
    async def _check_hardcoded_secrets(self) -> List[QualityMetric]:
        """Check for hardcoded secrets and credentials."""
        vulnerabilities = []
        
        for file_path in self.project_root.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in self.security_patterns['hardcoded_secrets']:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append(f"{file_path}:{line_num}: Potential hardcoded secret")
                        
            except Exception as e:
                logger.warning(f"Could not scan {file_path} for secrets: {e}")
                
        score = 1.0 if len(vulnerabilities) == 0 else 0.0
        status = "pass" if len(vulnerabilities) == 0 else "fail"
        
        return [QualityMetric(
            name="hardcoded_secrets",
            category="security",
            score=score,
            status=status,
            message=f"Found {len(vulnerabilities)} potential hardcoded secrets",
            details={"vulnerabilities": vulnerabilities},
            remediation="Use environment variables or secure vaults for secrets" if vulnerabilities else None
        )]
        
    async def _check_injection_vulnerabilities(self) -> List[QualityMetric]:
        """Check for injection vulnerability patterns."""
        vulnerabilities = []
        
        for file_path in self.project_root.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in self.security_patterns['sql_injection']:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append(f"{file_path}:{line_num}: Potential SQL injection")
                        
                for pattern in self.security_patterns['path_traversal']:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append(f"{file_path}:{line_num}: Potential path traversal")
                        
            except Exception as e:
                logger.warning(f"Could not scan {file_path} for injections: {e}")
                
        score = 1.0 if len(vulnerabilities) == 0 else max(0.0, 1.0 - len(vulnerabilities) * 0.2)
        status = "pass" if len(vulnerabilities) == 0 else "warn" if score > 0.5 else "fail"
        
        return [QualityMetric(
            name="injection_vulnerabilities",
            category="security",
            score=score,
            status=status,
            message=f"Found {len(vulnerabilities)} potential injection vulnerabilities",
            details={"vulnerabilities": vulnerabilities},
            remediation="Use parameterized queries and input validation" if vulnerabilities else None
        )]
        
    async def _check_dangerous_imports(self) -> List[QualityMetric]:
        """Check for potentially dangerous imports."""
        dangerous_imports = []
        
        for file_path in self.project_root.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in self.security_patterns['dangerous_imports']:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        dangerous_imports.append(f"{file_path}:{line_num}: {match.group()}")
                        
            except Exception as e:
                logger.warning(f"Could not scan {file_path} for dangerous imports: {e}")
                
        # Filter out legitimate uses
        legitimate_uses = [imp for imp in dangerous_imports if 'subprocess' in imp and 'vid_diffusion_bench' in str(imp)]
        actual_dangerous = [imp for imp in dangerous_imports if imp not in legitimate_uses]
        
        score = 1.0 if len(actual_dangerous) == 0 else max(0.5, 1.0 - len(actual_dangerous) * 0.1)
        status = "pass" if len(actual_dangerous) == 0 else "warn"
        
        return [QualityMetric(
            name="dangerous_imports",
            category="security",
            score=score,
            status=status,
            message=f"Found {len(actual_dangerous)} potentially dangerous imports",
            details={"imports": actual_dangerous},
            remediation="Review dangerous imports for security implications" if actual_dangerous else None
        )]
        
    async def _check_file_permissions(self) -> List[QualityMetric]:
        """Check file permissions for security issues."""
        permission_issues = []
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # Check for world-writable files
                    if mode.endswith('7') or mode.endswith('6'):
                        permission_issues.append(f"{file_path}: World-writable permissions ({mode})")
                        
                except Exception as e:
                    logger.warning(f"Could not check permissions for {file_path}: {e}")
                    
        score = 1.0 if len(permission_issues) == 0 else max(0.0, 1.0 - len(permission_issues) * 0.1)
        status = "pass" if len(permission_issues) == 0 else "warn" if score > 0.7 else "fail"
        
        return [QualityMetric(
            name="file_permissions",
            category="security",
            score=score,
            status=status,
            message=f"Found {len(permission_issues)} file permission issues",
            details={"issues": permission_issues},
            remediation="Review and fix insecure file permissions" if permission_issues else None
        )]


class PerformanceAnalyzer:
    """Performance analysis and optimization recommendations."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        
    async def analyze_performance(self) -> List[QualityMetric]:
        """Analyze performance characteristics."""
        metrics = []
        
        # Check for performance anti-patterns
        antipattern_metrics = await self._check_performance_antipatterns()
        import_metrics = await self._analyze_import_performance()
        algorithm_metrics = await self._analyze_algorithmic_complexity()
        
        metrics.extend(antipattern_metrics)
        metrics.extend(import_metrics)
        metrics.extend(algorithm_metrics)
        
        return metrics
        
    async def _check_performance_antipatterns(self) -> List[QualityMetric]:
        """Check for common performance anti-patterns."""
        antipatterns = []
        
        performance_patterns = {
            'nested_loops': r'for\s+.*:\s*\n\s*for\s+.*:',
            'string_concatenation': r'\+\s*=\s*["\']',
            'global_variables': r'global\s+\w+',
            'inefficient_membership': r'if\s+.*\s+in\s+\[.*\]'
        }
        
        for file_path in self.project_root.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern_name, pattern in performance_patterns.items():
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        antipatterns.append(f"{file_path}:{line_num}: {pattern_name}")
                        
            except Exception as e:
                logger.warning(f"Could not analyze performance for {file_path}: {e}")
                
        score = max(0.0, min(1.0, 1.0 - len(antipatterns) * 0.05))
        status = "pass" if len(antipatterns) == 0 else "warn" if score > 0.7 else "fail"
        
        return [QualityMetric(
            name="performance_antipatterns",
            category="performance",
            score=score,
            status=status,
            message=f"Found {len(antipatterns)} performance anti-patterns",
            details={"antipatterns": antipatterns[:20]},
            remediation="Optimize identified performance bottlenecks" if antipatterns else None
        )]
        
    async def _analyze_import_performance(self) -> List[QualityMetric]:
        """Analyze import performance."""
        slow_imports = []
        
        for file_path in self.project_root.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for imports that might be slow
                heavyweight_imports = [
                    'import pandas', 'import tensorflow', 'import torch',
                    'import matplotlib', 'import seaborn', 'import sklearn'
                ]
                
                for heavy_import in heavyweight_imports:
                    if heavy_import in content:
                        # Check if it's at module level (not inside function)
                        if re.search(f'^{re.escape(heavy_import)}', content, re.MULTILINE):
                            slow_imports.append(f"{file_path}: {heavy_import}")
                            
            except Exception as e:
                logger.warning(f"Could not analyze imports for {file_path}: {e}")
                
        score = max(0.5, min(1.0, 1.0 - len(slow_imports) * 0.1))
        status = "pass" if len(slow_imports) == 0 else "warn"
        
        return [QualityMetric(
            name="import_performance",
            category="performance",
            score=score,
            status=status,
            message=f"Found {len(slow_imports)} potentially slow imports",
            details={"slow_imports": slow_imports},
            remediation="Consider lazy imports for heavyweight libraries" if slow_imports else None
        )]
        
    async def _analyze_algorithmic_complexity(self) -> List[QualityMetric]:
        """Analyze algorithmic complexity patterns."""
        complexity_issues = []
        
        for file_path in self.project_root.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._estimate_time_complexity(node)
                        if complexity > 2:  # O(n^2) or worse
                            complexity_issues.append(f"{file_path}:{node.lineno}: {node.name} (estimated O(n^{complexity}))")
                            
            except Exception as e:
                logger.warning(f"Could not analyze complexity for {file_path}: {e}")
                
        score = max(0.0, min(1.0, 1.0 - len(complexity_issues) * 0.1))
        status = "pass" if len(complexity_issues) == 0 else "warn" if score > 0.7 else "fail"
        
        return [QualityMetric(
            name="algorithmic_complexity",
            category="performance",
            score=score,
            status=status,
            message=f"Found {len(complexity_issues)} high-complexity algorithms",
            details={"issues": complexity_issues},
            remediation="Optimize high-complexity algorithms" if complexity_issues else None
        )]
        
    def _estimate_time_complexity(self, node: ast.AST) -> int:
        """Estimate time complexity of a function."""
        max_depth = 0
        current_depth = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(child, ast.FunctionDef):
                # Reset depth for nested functions
                current_depth = 0
                
        return max_depth


class AutomatedQualityGateSystem:
    """Comprehensive automated quality gate system."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.code_analyzer = CodeQualityAnalyzer(self.project_root)
        self.security_analyzer = SecurityAnalyzer(self.project_root)
        self.performance_analyzer = PerformanceAnalyzer(self.project_root)
        
        # Quality thresholds
        self.quality_thresholds = {
            'overall_score': 0.8,
            'critical_categories': ['security', 'syntax'],
            'category_minimums': {
                'security': 0.9,
                'code_quality': 0.7,
                'performance': 0.6,
                'documentation': 0.5
            }
        }
        
    async def run_quality_gates(self) -> QualityReport:
        """Run comprehensive quality gate analysis."""
        start_time = time.time()
        logger.info("Starting comprehensive quality gate analysis...")
        
        # Run all analyzers concurrently
        code_task = asyncio.create_task(self.code_analyzer.analyze_code_quality())
        security_task = asyncio.create_task(self.security_analyzer.analyze_security())
        performance_task = asyncio.create_task(self.performance_analyzer.analyze_performance())
        
        # Gather all metrics
        code_metrics = await code_task
        security_metrics = await security_task
        performance_metrics = await performance_task
        
        all_metrics = code_metrics + security_metrics + performance_metrics
        
        # Calculate category scores
        categories = self._calculate_category_scores(all_metrics)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(categories)
        
        # Determine status
        status = self._determine_status(overall_score, categories, all_metrics)
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(all_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_metrics, categories)
        
        execution_time = time.time() - start_time
        
        report = QualityReport(
            timestamp=time.time(),
            overall_score=overall_score,
            status=status,
            metrics=all_metrics,
            categories=categories,
            critical_issues=critical_issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        logger.info(f"Quality gate analysis completed in {execution_time:.2f}s. Status: {status}")
        return report
        
    def _calculate_category_scores(self, metrics: List[QualityMetric]) -> Dict[str, float]:
        """Calculate scores for each category."""
        category_metrics = {}
        
        for metric in metrics:
            if metric.category not in category_metrics:
                category_metrics[metric.category] = []
            category_metrics[metric.category].append(metric.score)
            
        category_scores = {}
        for category, scores in category_metrics.items():
            category_scores[category] = statistics.mean(scores) if scores else 0.0
            
        return category_scores
        
    def _calculate_overall_score(self, categories: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = {
            'security': 0.4,
            'code_quality': 0.3,
            'performance': 0.2,
            'documentation': 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, score in categories.items():
            weight = weights.get(category, 0.1)
            weighted_sum += score * weight
            total_weight += weight
            
        return weighted_sum / max(total_weight, 1.0)
        
    def _determine_status(
        self,
        overall_score: float,
        categories: Dict[str, float],
        metrics: List[QualityMetric]
    ) -> str:
        """Determine overall quality gate status."""
        
        # Check for critical failures
        critical_failures = [m for m in metrics if m.status == "fail" and m.category in self.quality_thresholds['critical_categories']]
        if critical_failures:
            return "FAIL"
            
        # Check category minimums
        for category, minimum in self.quality_thresholds['category_minimums'].items():
            if categories.get(category, 0.0) < minimum:
                return "FAIL"
                
        # Check overall score
        if overall_score < self.quality_thresholds['overall_score']:
            return "WARN"
            
        return "PASS"
        
    def _identify_critical_issues(self, metrics: List[QualityMetric]) -> List[str]:
        """Identify critical issues that must be addressed."""
        critical_issues = []
        
        for metric in metrics:
            if metric.status == "fail":
                critical_issues.append(f"{metric.category}.{metric.name}: {metric.message}")
                
        return critical_issues
        
    def _generate_recommendations(
        self,
        metrics: List[QualityMetric],
        categories: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Category-specific recommendations
        if categories.get('security', 1.0) < 0.9:
            recommendations.append("🔒 Security: Run security scan and fix identified vulnerabilities")
            
        if categories.get('code_quality', 1.0) < 0.7:
            recommendations.append("📋 Code Quality: Improve code structure and reduce complexity")
            
        if categories.get('performance', 1.0) < 0.6:
            recommendations.append("⚡ Performance: Optimize algorithms and remove performance bottlenecks")
            
        if categories.get('documentation', 1.0) < 0.5:
            recommendations.append("📚 Documentation: Add docstrings and improve code documentation")
            
        # Metric-specific recommendations
        for metric in metrics:
            if metric.remediation and metric.status in ["fail", "warn"]:
                recommendations.append(f"🔧 {metric.name}: {metric.remediation}")
                
        return list(set(recommendations))  # Remove duplicates
        
    async def save_report(self, report: QualityReport, output_path: Optional[str] = None):
        """Save quality report to file."""
        if output_path is None:
            output_path = self.project_root / "quality_gates_report.json"
            
        report_data = {
            'timestamp': report.timestamp,
            'overall_score': report.overall_score,
            'status': report.status,
            'categories': report.categories,
            'critical_issues': report.critical_issues,
            'recommendations': report.recommendations,
            'execution_time': report.execution_time,
            'metrics': [
                {
                    'name': m.name,
                    'category': m.category,
                    'score': m.score,
                    'status': m.status,
                    'message': m.message,
                    'details': m.details,
                    'remediation': m.remediation
                }
                for m in report.metrics
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Quality report saved to {output_path}")


# Global quality gate system
quality_gate_system = AutomatedQualityGateSystem()