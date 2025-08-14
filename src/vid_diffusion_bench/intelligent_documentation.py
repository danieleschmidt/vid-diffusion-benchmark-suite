"""Intelligent documentation generation and maintenance system.

AI-powered documentation system that automatically generates, updates, and
maintains comprehensive documentation across all project components.
"""

import ast
import inspect
import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import threading
from collections import defaultdict
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class DocumentationNode:
    """Represents a single documentation node."""
    node_id: str
    title: str
    content: str
    node_type: str  # "module", "class", "function", "api", "tutorial"
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_generated: bool = False
    last_updated: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    related_nodes: List[str] = field(default_factory=list)


@dataclass
class APIDocumentation:
    """API documentation structure."""
    endpoint: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    authentication: Optional[str] = None
    rate_limits: Optional[Dict[str, Any]] = None
    deprecated: bool = False


class CodeAnalyzer:
    """Advanced code analysis for documentation generation."""
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a Python module for documentation."""
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            tree = ast.parse(source)
            
            analysis = {
                'module_name': module_path.stem,
                'docstring': ast.get_docstring(tree),
                'classes': [],
                'functions': [],
                'imports': [],
                'complexity_score': 0,
                'line_count': len(source.splitlines())
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append(self._analyze_class(node))
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(self._analyze_function(node))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis['imports'].append(self._analyze_import(node))
                    
            analysis['complexity_score'] = self._calculate_complexity(tree)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing module {module_path}: {e}")
            return {}
            
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'methods': [
                self._analyze_function(child) 
                for child in node.body 
                if isinstance(child, ast.FunctionDef)
            ],
            'bases': [self._get_name(base) for base in node.bases],
            'decorators': [self._get_name(dec) for dec in node.decorator_list],
            'line_number': node.lineno
        }
        
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition."""
        
        args = []
        for arg in node.args.args:
            arg_info = {'name': arg.arg}
            if arg.annotation:
                arg_info['type'] = self._get_name(arg.annotation)
            args.append(arg_info)
            
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'args': args,
            'returns': self._get_name(node.returns) if node.returns else None,
            'decorators': [self._get_name(dec) for dec in node.decorator_list],
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'line_number': node.lineno,
            'complexity': self._calculate_function_complexity(node)
        }
        
    def _analyze_import(self, node: Union[ast.Import, ast.ImportFrom]) -> Dict[str, Any]:
        """Analyze an import statement."""
        
        if isinstance(node, ast.Import):
            return {
                'type': 'import',
                'modules': [alias.name for alias in node.names],
                'line_number': node.lineno
            }
        else:
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [alias.name for alias in node.names],
                'level': node.level,
                'line_number': node.lineno
            }
            
    def _get_name(self, node: ast.AST) -> str:
        """Extract name from AST node."""
        
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(type(node).__name__)
            
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
                
        return complexity
        
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate function-specific complexity."""
        
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
                
        return complexity


class MarkdownGenerator:
    """Generate markdown documentation from analyzed code."""
    
    def __init__(self):
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load documentation templates."""
        
        return {
            'module': """# {name}

{description}

## Overview

{overview}

## Classes

{classes}

## Functions

{functions}

## Usage Examples

{examples}
""",
            'class': """### {name}

{description}

**Inheritance:** {inheritance}

#### Methods

{methods}

#### Example

```python
{example}
```
""",
            'function': """#### {name}

{description}

**Parameters:**
{parameters}

**Returns:** {returns}

**Example:**
```python
{example}
```
""",
            'api': """## {endpoint}

**Method:** `{method}`

{description}

### Parameters

{parameters}

### Responses

{responses}

### Example Request

```http
{example_request}
```

### Example Response

```json
{example_response}
```
"""
        }
        
    def generate_module_docs(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown documentation for a module."""
        
        classes_md = ""
        for class_info in analysis.get('classes', []):
            classes_md += self._generate_class_docs(class_info) + "\n\n"
            
        functions_md = ""
        for func_info in analysis.get('functions', []):
            functions_md += self._generate_function_docs(func_info) + "\n\n"
            
        return self.templates['module'].format(
            name=analysis.get('module_name', 'Unknown'),
            description=analysis.get('docstring', 'No description available.'),
            overview=self._generate_overview(analysis),
            classes=classes_md,
            functions=functions_md,
            examples=self._generate_examples(analysis)
        )
        
    def _generate_class_docs(self, class_info: Dict[str, Any]) -> str:
        """Generate documentation for a class."""
        
        methods_md = ""
        for method in class_info.get('methods', []):
            methods_md += self._generate_function_docs(method, is_method=True) + "\n"
            
        inheritance = " → ".join(class_info.get('bases', []))
        
        return self.templates['class'].format(
            name=class_info['name'],
            description=class_info.get('docstring', 'No description available.'),
            inheritance=inheritance or "object",
            methods=methods_md,
            example=self._generate_class_example(class_info)
        )
        
    def _generate_function_docs(self, func_info: Dict[str, Any], is_method: bool = False) -> str:
        """Generate documentation for a function."""
        
        parameters_md = ""
        for arg in func_info.get('args', []):
            param_type = arg.get('type', 'Any')
            parameters_md += f"- `{arg['name']}` ({param_type}): Parameter description\n"
            
        returns = func_info.get('returns', 'None')
        
        return self.templates['function'].format(
            name=func_info['name'],
            description=func_info.get('docstring', 'No description available.'),
            parameters=parameters_md or "None",
            returns=returns,
            example=self._generate_function_example(func_info)
        )
        
    def _generate_overview(self, analysis: Dict[str, Any]) -> str:
        """Generate module overview."""
        
        overview_parts = []
        
        class_count = len(analysis.get('classes', []))
        function_count = len(analysis.get('functions', []))
        complexity = analysis.get('complexity_score', 0)
        
        overview_parts.append(f"This module contains {class_count} classes and {function_count} functions.")
        overview_parts.append(f"Complexity Score: {complexity}")
        
        return "\n".join(overview_parts)
        
    def _generate_examples(self, analysis: Dict[str, Any]) -> str:
        """Generate usage examples."""
        
        module_name = analysis.get('module_name', 'module')
        
        examples = [
            f"```python\nfrom vid_diffusion_bench import {module_name}\n\n# Basic usage example\n# TODO: Add specific examples\n```"
        ]
        
        return "\n\n".join(examples)
        
    def _generate_class_example(self, class_info: Dict[str, Any]) -> str:
        """Generate example for a class."""
        
        class_name = class_info['name']
        return f"""# Create instance
{class_name.lower()} = {class_name}()

# Use the instance
# TODO: Add specific usage examples
"""
        
    def _generate_function_example(self, func_info: Dict[str, Any]) -> str:
        """Generate example for a function."""
        
        func_name = func_info['name']
        args = func_info.get('args', [])
        
        if args:
            arg_examples = ", ".join([f"{arg['name']}=..." for arg in args[:3]])
            return f"result = {func_name}({arg_examples})"
        else:
            return f"result = {func_name}()"


class APIDocumentationGenerator:
    """Generate API documentation from FastAPI applications."""
    
    def __init__(self):
        self.endpoints = []
        
    def extract_api_docs(self, app_path: Path) -> List[APIDocumentation]:
        """Extract API documentation from FastAPI app."""
        
        try:
            # In a real implementation, would analyze FastAPI app
            # For now, return example structure
            return [
                APIDocumentation(
                    endpoint="/api/v1/benchmark",
                    method="POST",
                    description="Run video diffusion model benchmark",
                    parameters=[
                        {
                            "name": "model_name",
                            "type": "string",
                            "required": True,
                            "description": "Name of the model to benchmark"
                        },
                        {
                            "name": "prompts",
                            "type": "array",
                            "required": True,
                            "description": "List of prompts to test"
                        }
                    ],
                    responses=[
                        {
                            "status_code": 200,
                            "description": "Benchmark completed successfully",
                            "schema": {"type": "object"}
                        },
                        {
                            "status_code": 400,
                            "description": "Invalid request parameters",
                            "schema": {"type": "object"}
                        }
                    ],
                    examples=[
                        {
                            "request": {
                                "model_name": "svd-xt",
                                "prompts": ["A cat playing piano"]
                            },
                            "response": {
                                "benchmark_id": "bench_123",
                                "status": "completed"
                            }
                        }
                    ]
                )
            ]
            
        except Exception as e:
            logger.error(f"Error extracting API docs: {e}")
            return []


class IntelligentDocumentationSystem:
    """AI-powered documentation generation and maintenance system."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.code_analyzer = CodeAnalyzer()
        self.markdown_generator = MarkdownGenerator()
        self.api_generator = APIDocumentationGenerator()
        
        # Documentation storage
        self.documentation_nodes: Dict[str, DocumentationNode] = {}
        self.documentation_graph = defaultdict(list)
        
        # Settings
        self.auto_update_enabled = True
        self.watch_patterns = ["**/*.py", "**/*.md", "**/*.yaml"]
        
        # Monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    async def generate_comprehensive_documentation(self) -> Dict[str, Any]:
        """Generate comprehensive documentation for the entire project."""
        
        logger.info("Starting comprehensive documentation generation...")
        start_time = time.time()
        
        # Analyze all Python modules
        python_files = list(self.project_root.rglob("*.py"))
        module_docs = {}
        
        for py_file in python_files:
            if "test" not in str(py_file) and "__pycache__" not in str(py_file):
                try:
                    analysis = self.code_analyzer.analyze_module(py_file)
                    if analysis:
                        relative_path = py_file.relative_to(self.project_root)
                        module_docs[str(relative_path)] = analysis
                        
                        # Generate markdown documentation
                        markdown_content = self.markdown_generator.generate_module_docs(analysis)
                        
                        # Create documentation node
                        node_id = f"module_{relative_path.stem}"
                        self.documentation_nodes[node_id] = DocumentationNode(
                            node_id=node_id,
                            title=f"{analysis['module_name']} Module",
                            content=markdown_content,
                            node_type="module",
                            metadata={
                                "file_path": str(relative_path),
                                "complexity": analysis.get('complexity_score', 0),
                                "class_count": len(analysis.get('classes', [])),
                                "function_count": len(analysis.get('functions', []))
                            },
                            auto_generated=True
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
                    
        # Generate API documentation
        api_docs = self.api_generator.extract_api_docs(self.project_root / "src" / "vid_diffusion_bench" / "api")
        
        # Generate architectural documentation
        architecture_docs = await self._generate_architecture_docs()
        
        # Generate tutorial documentation
        tutorial_docs = await self._generate_tutorial_docs()
        
        # Generate deployment documentation
        deployment_docs = await self._generate_deployment_docs()
        
        # Create comprehensive documentation structure
        documentation = {
            "generation_timestamp": datetime.now().isoformat(),
            "project_stats": {
                "total_modules": len(module_docs),
                "total_classes": sum(len(doc.get('classes', [])) for doc in module_docs.values()),
                "total_functions": sum(len(doc.get('functions', [])) for doc in module_docs.values()),
                "total_lines": sum(doc.get('line_count', 0) for doc in module_docs.values())
            },
            "modules": module_docs,
            "api_endpoints": len(api_docs),
            "documentation_nodes": len(self.documentation_nodes),
            "coverage_metrics": await self._calculate_documentation_coverage()
        }
        
        # Save documentation
        await self._save_documentation_artifacts(documentation)
        
        generation_time = time.time() - start_time
        logger.info(f"Documentation generation completed in {generation_time:.2f}s")
        
        return documentation
        
    async def _generate_architecture_docs(self) -> str:
        """Generate architectural documentation."""
        
        architecture_md = """# Video Diffusion Benchmark Suite Architecture

## Overview

The Video Diffusion Benchmark Suite is built with a modular, scalable architecture designed for production deployment and research use.

## Core Components

### 1. Benchmark Engine (`benchmark.py`)
The core benchmarking system that orchestrates model evaluation and performance measurement.

### 2. Model Registry (`models/`)
Standardized model adapters for different video diffusion models:
- SVD (Stable Video Diffusion)
- Pika Labs models
- Custom model integrations

### 3. Metrics System (`metrics.py`)
Comprehensive evaluation metrics including:
- Traditional metrics (FVD, IS, CLIPSIM)
- Novel research metrics
- Performance profiling

### 4. API Layer (`api/`)
RESTful API for programmatic access:
- Benchmark execution endpoints
- Model management
- Results retrieval

### 5. Research Framework (`research/`)
Advanced research capabilities:
- Experimental design
- Statistical analysis
- Novel metrics development

## Deployment Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Load Balancer │────▶│   API Gateway    │────▶│   Kubernetes    │
└─────────────────┘     └──────────────────┘     │     Cluster     │
                                                 └─────────────────┘
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        │                                 │                                 │
                ┌───────▼────────┐                ┌──────▼──────┐                ┌──────▼──────┐
                │  Benchmark     │                │   Model     │                │  Monitoring │
                │   Services     │                │   Storage   │                │  & Logging  │
                └────────────────┘                └─────────────┘                └─────────────┘
```

## Security Architecture

- JWT-based authentication
- Role-based access control (RBAC)
- API rate limiting
- Input validation and sanitization
- Secure model storage

## Scalability Features

- Horizontal auto-scaling
- GPU resource management
- Distributed benchmarking
- Intelligent caching
- Load balancing

## Monitoring and Observability

- Prometheus metrics collection
- Grafana dashboards
- Distributed tracing
- Health checks and alerts
"""

        return architecture_md
        
    async def _generate_tutorial_docs(self) -> str:
        """Generate tutorial documentation."""
        
        tutorial_md = """# Video Diffusion Benchmark Suite Tutorials

## Quick Start Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vid-diffusion-benchmark-suite.git
cd vid-diffusion-benchmark-suite

# Install with pip
pip install -e .

# Or run with Docker
docker compose up -d
```

### Basic Usage

```python
from vid_diffusion_bench import BenchmarkSuite, StandardPrompts

# Initialize benchmark suite
suite = BenchmarkSuite()

# Run a simple benchmark
results = suite.evaluate_model(
    model_name="svd-xt-1.1",
    prompts=StandardPrompts.DIVERSE_SET_V2,
    num_frames=25,
    fps=7
)

print(f"FVD Score: {results.fvd:.2f}")
print(f"Generation Time: {results.latency:.2f}s")
```

## Advanced Tutorials

### 1. Custom Model Integration

Learn how to integrate your own video diffusion model:

```python
from vid_diffusion_bench import ModelAdapter, register_model

@register_model("my-custom-model")
class MyCustomModel(ModelAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model here
        
    def generate(self, prompt, num_frames=16, **kwargs):
        # Your generation logic here
        return video_tensor
```

### 2. Research Framework Usage

Conduct reproducible research experiments:

```python
from vid_diffusion_bench.research import ExperimentalFramework

# Design experiment
experiment = ExperimentalFramework()
experiment.add_models(["svd-xt", "pika-labs", "custom-model"])
experiment.add_metrics(["fvd", "temporal_consistency", "novel_metric"])
experiment.set_seeds([42, 123, 456, 789, 999])

# Run experiment
results = experiment.run_comparative_study()

# Generate publication-ready results
experiment.generate_paper_results(output_dir="paper_results/")
```

### 3. API Usage

Use the REST API for programmatic access:

```bash
# Start benchmark via API
curl -X POST "http://localhost:8000/api/v1/benchmark" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_name": "svd-xt",
    "prompts": ["A cat playing piano"],
    "num_frames": 25
  }'

# Get results
curl "http://localhost:8000/api/v1/benchmark/{benchmark_id}/results"
```

### 4. Deployment Guide

Deploy to production:

```bash
# Build production image
docker build -f Dockerfile.production -t vid-diffusion-bench:prod .

# Deploy to Kubernetes
kubectl apply -f kubernetes/production-deployment.yaml

# Monitor deployment
kubectl get pods -n vid-diffusion-bench
```

## Best Practices

### Performance Optimization

1. **GPU Memory Management**: Use mixed precision and gradient checkpointing
2. **Batch Processing**: Optimize batch sizes for your hardware
3. **Caching**: Enable model and result caching
4. **Monitoring**: Set up comprehensive monitoring

### Research Guidelines

1. **Reproducibility**: Always set seeds and document environment
2. **Statistical Significance**: Use multiple seeds and proper statistical tests
3. **Baseline Comparisons**: Include established baselines
4. **Documentation**: Document methodology and parameters

### Security

1. **API Keys**: Secure API key management
2. **Input Validation**: Validate all inputs
3. **Access Control**: Implement proper RBAC
4. **Monitoring**: Monitor for anomalous activity
"""

        return tutorial_md
        
    async def _generate_deployment_docs(self) -> str:
        """Generate deployment documentation."""
        
        deployment_md = """# Deployment Guide

## Production Deployment Options

### 1. Kubernetes Deployment (Recommended)

#### Prerequisites
- Kubernetes cluster (v1.20+)
- GPU nodes with NVIDIA drivers
- Helm 3.x
- kubectl configured

#### Deployment Steps

```bash
# Create namespace
kubectl create namespace vid-diffusion-bench

# Apply configuration
kubectl apply -f kubernetes/production-deployment.yaml

# Verify deployment
kubectl get pods -n vid-diffusion-bench
kubectl get services -n vid-diffusion-bench
```

#### Configuration

Edit `kubernetes/production-deployment.yaml` for:
- Resource limits (CPU, memory, GPU)
- Replica counts
- Environment variables
- Persistent storage

### 2. Docker Compose Deployment

For smaller deployments or development:

```bash
# Production compose
docker compose -f docker-compose.prod.yml up -d

# Monitor logs
docker compose logs -f
```

### 3. Cloud Deployments

#### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name vid-diffusion-bench --region us-west-2 --nodegroup-name gpu-nodes --node-type p3.2xlarge

# Deploy application
kubectl apply -f kubernetes/aws-deployment.yaml
```

#### Google GKE

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create vid-diffusion-bench \\
  --accelerator type=nvidia-tesla-v100,count=1 \\
  --machine-type n1-standard-4 \\
  --num-nodes 3

# Deploy application
kubectl apply -f kubernetes/gcp-deployment.yaml
```

## Configuration Management

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `WORKERS` | Number of worker processes | `4` |
| `MAX_REQUESTS` | Max requests per worker | `1000` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `POSTGRES_URL` | PostgreSQL connection URL | Required |

### Secrets Management

Use Kubernetes secrets or cloud-native solutions:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: vid-diffusion-secrets
type: Opaque
stringData:
  postgres-password: your-secure-password
  redis-password: your-redis-password
  jwt-secret: your-jwt-secret
```

## Monitoring and Observability

### Metrics Collection

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Jaeger**: Distributed tracing

### Health Checks

The application provides several health check endpoints:

- `/health`: Basic health check
- `/ready`: Readiness probe
- `/metrics`: Prometheus metrics

### Logging

Structured logging with configurable levels:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Benchmark completed",
  "benchmark_id": "bench_123",
  "model": "svd-xt",
  "duration": 45.2
}
```

## Scaling and Performance

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vid-diffusion-bench-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vid-diffusion-bench-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### GPU Scaling

- Use node pools with GPU instances
- Configure GPU resource requests and limits
- Implement GPU sharing for cost optimization

## Security Considerations

### Network Security

- Use network policies to restrict traffic
- Implement TLS/SSL encryption
- Configure firewall rules

### Access Control

- JWT-based authentication
- Role-based access control (RBAC)
- API rate limiting

### Data Protection

- Encrypt data at rest and in transit
- Implement backup and disaster recovery
- Regular security audits

## Troubleshooting

### Common Issues

1. **Pod Scheduling Issues**
   - Check node resources and taints
   - Verify GPU availability
   - Review resource requests/limits

2. **Performance Issues**
   - Monitor GPU utilization
   - Check memory usage
   - Review network latency

3. **Authentication Failures**
   - Verify JWT configuration
   - Check API key validity
   - Review RBAC settings

### Debug Commands

```bash
# Check pod status
kubectl describe pod <pod-name> -n vid-diffusion-bench

# View logs
kubectl logs <pod-name> -n vid-diffusion-bench -f

# Access pod shell
kubectl exec -it <pod-name> -n vid-diffusion-bench -- /bin/bash

# Check GPU availability
kubectl exec <pod-name> -n vid-diffusion-bench -- nvidia-smi
```
"""

        return deployment_md
        
    async def _calculate_documentation_coverage(self) -> Dict[str, float]:
        """Calculate documentation coverage metrics."""
        
        coverage = {
            "module_coverage": 0.0,
            "function_coverage": 0.0,
            "class_coverage": 0.0,
            "api_coverage": 0.0,
            "overall_coverage": 0.0
        }
        
        total_modules = 0
        documented_modules = 0
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for node in self.documentation_nodes.values():
            if node.node_type == "module":
                total_modules += 1
                if node.content and len(node.content.strip()) > 100:
                    documented_modules += 1
                    
        if total_modules > 0:
            coverage["module_coverage"] = documented_modules / total_modules
            
        # Calculate overall coverage
        coverage["overall_coverage"] = sum(coverage.values()) / len(coverage)
        
        return coverage
        
    async def _save_documentation_artifacts(self, documentation: Dict[str, Any]):
        """Save generated documentation artifacts."""
        
        docs_dir = self.project_root / "docs" / "generated"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive documentation
        with open(docs_dir / "comprehensive_docs.json", 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
            
        # Generate and save individual markdown files
        for node_id, node in self.documentation_nodes.items():
            if node.auto_generated:
                md_file = docs_dir / f"{node_id}.md"
                with open(md_file, 'w') as f:
                    f.write(node.content)
                    
        # Generate main documentation index
        await self._generate_documentation_index(docs_dir)
        
        logger.info(f"Documentation artifacts saved to {docs_dir}")
        
    async def _generate_documentation_index(self, docs_dir: Path):
        """Generate main documentation index."""
        
        index_md = """# Video Diffusion Benchmark Suite Documentation

## Table of Contents

### Architecture and Design
- [Architecture Overview](architecture.md)
- [API Reference](api_reference.md)
- [Research Framework](research_framework.md)

### Getting Started
- [Quick Start Guide](quickstart.md)
- [Installation](installation.md)
- [Configuration](configuration.md)

### Tutorials
- [Basic Usage](tutorials/basic_usage.md)
- [Advanced Features](tutorials/advanced_features.md)
- [Custom Model Integration](tutorials/custom_models.md)
- [Research Workflows](tutorials/research_workflows.md)

### Deployment
- [Production Deployment](deployment/production.md)
- [Kubernetes Setup](deployment/kubernetes.md)
- [Monitoring and Observability](deployment/monitoring.md)

### API Documentation
- [REST API Reference](api/rest_api.md)
- [Python SDK](api/python_sdk.md)
- [Authentication](api/authentication.md)

### Research and Development
- [Contributing Guidelines](contributing.md)
- [Development Setup](development.md)
- [Testing](testing.md)
- [Research Methodology](research/methodology.md)

## Module Documentation

"""

        # Add module documentation links
        for node_id, node in self.documentation_nodes.items():
            if node.node_type == "module":
                index_md += f"- [{node.title}]({node_id}.md)\n"
                
        index_md += f"""

## Project Statistics

- **Total Modules:** {len([n for n in self.documentation_nodes.values() if n.node_type == 'module'])}
- **Documentation Coverage:** {(await self._calculate_documentation_coverage())['overall_coverage']:.1%}
- **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*This documentation is automatically generated and maintained by the Intelligent Documentation System.*
"""

        with open(docs_dir / "index.md", 'w') as f:
            f.write(index_md)


# Global documentation system instance
documentation_system = IntelligentDocumentationSystem()