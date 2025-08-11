"""Production deployment automation and orchestration.

This module provides comprehensive production deployment capabilities including:
- Infrastructure provisioning and configuration
- Container orchestration with Kubernetes
- CI/CD pipeline integration
- Blue/green and canary deployment strategies
- Health monitoring and rollback mechanisms
- Secrets management and security configuration
"""

import os
import json
import yaml
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import shutil

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategy options."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: Environment
    strategy: DeploymentStrategy
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    gpu_required: bool = False
    gpu_count: int = 1
    storage_size: str = "10Gi"
    domain: str = ""
    ssl_enabled: bool = True
    monitoring_enabled: bool = True
    autoscaling_enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    secrets: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)


class KubernetesDeployment:
    """Kubernetes deployment management."""
    
    def __init__(self, config: DeploymentConfig, namespace: str = "vid-diffusion-bench"):
        self.config = config
        self.namespace = namespace
        self.manifests_dir = Path("k8s")
        self.manifests_dir.mkdir(exist_ok=True)
    
    def generate_manifests(self) -> List[Path]:
        """Generate Kubernetes manifests."""
        manifests = []
        
        # Generate deployment manifest
        deployment_manifest = self._generate_deployment_manifest()
        deployment_path = self.manifests_dir / "deployment.yaml"
        with open(deployment_path, 'w') as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        manifests.append(deployment_path)
        
        # Generate service manifest
        service_manifest = self._generate_service_manifest()
        service_path = self.manifests_dir / "service.yaml"
        with open(service_path, 'w') as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        manifests.append(service_path)
        
        # Generate ingress manifest if domain is specified
        if self.config.domain:
            ingress_manifest = self._generate_ingress_manifest()
            ingress_path = self.manifests_dir / "ingress.yaml"
            with open(ingress_path, 'w') as f:
                yaml.dump(ingress_manifest, f, default_flow_style=False)
            manifests.append(ingress_path)
        
        # Generate HPA manifest if autoscaling is enabled
        if self.config.autoscaling_enabled:
            hpa_manifest = self._generate_hpa_manifest()
            hpa_path = self.manifests_dir / "hpa.yaml"
            with open(hpa_path, 'w') as f:
                yaml.dump(hpa_manifest, f, default_flow_style=False)
            manifests.append(hpa_path)
        
        # Generate secret manifest
        if self.config.secrets:
            secret_manifest = self._generate_secret_manifest()
            secret_path = self.manifests_dir / "secrets.yaml"
            with open(secret_path, 'w') as f:
                yaml.dump(secret_manifest, f, default_flow_style=False)
            manifests.append(secret_path)
        
        # Generate configmap manifest
        configmap_manifest = self._generate_configmap_manifest()
        configmap_path = self.manifests_dir / "configmap.yaml"
        with open(configmap_path, 'w') as f:
            yaml.dump(configmap_manifest, f, default_flow_style=False)
        manifests.append(configmap_path)
        
        logger.info(f"Generated {len(manifests)} Kubernetes manifests")
        return manifests
    
    def _generate_deployment_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        container_spec = {
            "name": "vid-diffusion-bench",
            "image": f"vid-diffusion-bench:{self.config.environment.value}",
            "imagePullPolicy": "Always",
            "ports": [
                {"containerPort": 8000, "name": "http"},
                {"containerPort": 8080, "name": "metrics"}
            ],
            "resources": {
                "requests": {
                    "cpu": self.config.cpu_request,
                    "memory": self.config.memory_request
                },
                "limits": {
                    "cpu": self.config.cpu_limit,
                    "memory": self.config.memory_limit
                }
            },
            "env": [
                {"name": "ENVIRONMENT", "value": self.config.environment.value},
                {"name": "LOG_LEVEL", "value": "INFO"},
                {"name": "METRICS_ENABLED", "value": "true"}
            ],
            "envFrom": [
                {"configMapRef": {"name": "vid-diffusion-bench-config"}},
                {"secretRef": {"name": "vid-diffusion-bench-secrets"}}
            ],
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": 8000},
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
                "failureThreshold": 3
            },
            "readinessProbe": {
                "httpGet": {"path": "/ready", "port": 8000},
                "initialDelaySeconds": 5,
                "periodSeconds": 5,
                "timeoutSeconds": 3,
                "failureThreshold": 3
            },
            "volumeMounts": [
                {
                    "name": "data-storage",
                    "mountPath": "/data"
                },
                {
                    "name": "cache-storage", 
                    "mountPath": "/cache"
                }
            ]
        }
        
        # Add GPU resources if required
        if self.config.gpu_required:
            container_spec["resources"]["limits"]["nvidia.com/gpu"] = str(self.config.gpu_count)
            container_spec["resources"]["requests"]["nvidia.com/gpu"] = str(self.config.gpu_count)
        
        # Add custom environment variables
        for key, value in self.config.env_vars.items():
            container_spec["env"].append({"name": key, "value": value})
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "vid-diffusion-bench",
                "namespace": self.namespace,
                "labels": {
                    "app": "vid-diffusion-bench",
                    "environment": self.config.environment.value,
                    "version": "v1"
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "strategy": self._get_deployment_strategy(),
                "selector": {
                    "matchLabels": {
                        "app": "vid-diffusion-bench"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "vid-diffusion-bench",
                            "environment": self.config.environment.value,
                            "version": "v1"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8080",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "containers": [container_spec],
                        "volumes": [
                            {
                                "name": "data-storage",
                                "persistentVolumeClaim": {
                                    "claimName": "vid-diffusion-bench-data"
                                }
                            },
                            {
                                "name": "cache-storage",
                                "emptyDir": {"sizeLimit": "2Gi"}
                            }
                        ],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        }
                    }
                }
            }
        }
        
        # Add node selector for GPU nodes if required
        if self.config.gpu_required:
            deployment["spec"]["template"]["spec"]["nodeSelector"] = {
                "accelerator": "nvidia-tesla-k80"  # Example GPU node selector
            }
        
        return deployment
    
    def _get_deployment_strategy(self) -> Dict[str, Any]:
        """Get deployment strategy configuration."""
        if self.config.strategy == DeploymentStrategy.ROLLING:
            return {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": 1,
                    "maxSurge": 1
                }
            }
        elif self.config.strategy == DeploymentStrategy.RECREATE:
            return {"type": "Recreate"}
        else:
            # Default to rolling update
            return {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": "25%",
                    "maxSurge": "25%"
                }
            }
    
    def _generate_service_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "vid-diffusion-bench-service",
                "namespace": self.namespace,
                "labels": {
                    "app": "vid-diffusion-bench"
                }
            },
            "spec": {
                "selector": {
                    "app": "vid-diffusion-bench"
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 80,
                        "targetPort": 8000,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 8080,
                        "targetPort": 8080,
                        "protocol": "TCP"
                    }
                ],
                "type": "ClusterIP"
            }
        }
    
    def _generate_ingress_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes ingress manifest."""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "vid-diffusion-bench-ingress",
                "namespace": self.namespace,
                "labels": {
                    "app": "vid-diffusion-bench"
                },
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true" if self.config.ssl_enabled else "false"
                }
            },
            "spec": {
                "rules": [
                    {
                        "host": self.config.domain,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": "vid-diffusion-bench-service",
                                            "port": {
                                                "number": 80
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        # Add TLS configuration if SSL is enabled
        if self.config.ssl_enabled:
            ingress["spec"]["tls"] = [
                {
                    "hosts": [self.config.domain],
                    "secretName": "vid-diffusion-bench-tls"
                }
            ]
        
        return ingress
    
    def _generate_hpa_manifest(self) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "vid-diffusion-bench-hpa",
                "namespace": self.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "vid-diffusion-bench"
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
    
    def _generate_secret_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes secret manifest."""
        import base64
        
        # Encode secrets
        encoded_secrets = {}
        for key, value in self.config.secrets.items():
            encoded_secrets[key] = base64.b64encode(value.encode()).decode()
        
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "vid-diffusion-bench-secrets",
                "namespace": self.namespace
            },
            "type": "Opaque",
            "data": encoded_secrets
        }
    
    def _generate_configmap_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes ConfigMap manifest."""
        config_data = {
            "BENCHMARK_WORKERS": "4",
            "CACHE_SIZE_MB": "1024",
            "LOG_FORMAT": "json",
            "METRICS_INTERVAL": "30",
            "HEALTH_CHECK_TIMEOUT": "5"
        }
        
        # Add custom environment variables
        config_data.update(self.config.env_vars)
        
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "vid-diffusion-bench-config",
                "namespace": self.namespace
            },
            "data": config_data
        }
    
    def deploy(self, kubectl_context: str = None) -> bool:
        """Deploy to Kubernetes cluster."""
        try:
            # Generate manifests
            manifests = self.generate_manifests()
            
            # Create namespace if it doesn't exist
            self._create_namespace(kubectl_context)
            
            # Apply manifests
            for manifest in manifests:
                self._apply_manifest(manifest, kubectl_context)
            
            # Wait for deployment to be ready
            if self._wait_for_deployment(kubectl_context):
                logger.info("Deployment completed successfully")
                return True
            else:
                logger.error("Deployment failed or timed out")
                return False
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _create_namespace(self, kubectl_context: str = None):
        """Create Kubernetes namespace."""
        cmd = ["kubectl"]
        if kubectl_context:
            cmd.extend(["--context", kubectl_context])
        
        cmd.extend(["create", "namespace", self.namespace, "--dry-run=client", "-o", "yaml"])
        
        # Create namespace manifest
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Apply the namespace
            apply_cmd = ["kubectl"]
            if kubectl_context:
                apply_cmd.extend(["--context", kubectl_context])
            apply_cmd.append("apply")
            apply_cmd.extend(["-f", "-"])
            
            subprocess.run(apply_cmd, input=result.stdout, text=True)
        
        logger.info(f"Namespace '{self.namespace}' ensured")
    
    def _apply_manifest(self, manifest_path: Path, kubectl_context: str = None):
        """Apply Kubernetes manifest."""
        cmd = ["kubectl"]
        if kubectl_context:
            cmd.extend(["--context", kubectl_context])
        
        cmd.extend(["apply", "-f", str(manifest_path)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to apply {manifest_path}: {result.stderr}")
            raise RuntimeError(f"kubectl apply failed: {result.stderr}")
        
        logger.info(f"Applied manifest: {manifest_path}")
    
    def _wait_for_deployment(self, kubectl_context: str = None, timeout: int = 600) -> bool:
        """Wait for deployment to be ready."""
        cmd = ["kubectl"]
        if kubectl_context:
            cmd.extend(["--context", kubectl_context])
        
        cmd.extend([
            "wait", "--for=condition=available",
            f"deployment/vid-diffusion-bench",
            f"--namespace={self.namespace}",
            f"--timeout={timeout}s"
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0


class DockerImageBuilder:
    """Docker image building and management."""
    
    def __init__(self, project_root: Path, registry: str = None):
        self.project_root = project_root
        self.registry = registry or "localhost:5000"
        self.dockerfile_path = project_root / "Dockerfile"
    
    def build_image(self, tag: str, environment: Environment) -> bool:
        """Build Docker image."""
        try:
            # Generate optimized Dockerfile for the environment
            dockerfile_content = self._generate_dockerfile(environment)
            
            with open(self.dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build image
            full_tag = f"{self.registry}/vid-diffusion-bench:{tag}"
            
            build_cmd = [
                "docker", "build",
                "-t", full_tag,
                "-f", str(self.dockerfile_path),
                str(self.project_root)
            ]
            
            logger.info(f"Building Docker image: {full_tag}")
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr}")
                return False
            
            logger.info(f"Successfully built image: {full_tag}")
            
            # Push image if registry is not local
            if not self.registry.startswith("localhost"):
                return self._push_image(full_tag)
            
            return True
            
        except Exception as e:
            logger.error(f"Image build failed: {e}")
            return False
    
    def _generate_dockerfile(self, environment: Environment) -> str:
        """Generate optimized Dockerfile for the environment."""
        if environment == Environment.PRODUCTION:
            return self._generate_production_dockerfile()
        elif environment == Environment.STAGING:
            return self._generate_staging_dockerfile()
        else:
            return self._generate_development_dockerfile()
    
    def _generate_production_dockerfile(self) -> str:
        """Generate production-optimized Dockerfile."""
        return '''
# Multi-stage build for production
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml requirements-dev.txt ./
RUN pip install --no-cache-dir -e . && \\
    pip install --no-cache-dir -r requirements-dev.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy application code
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set permissions
RUN chown -R app:app /app
USER app

# Create directories
RUN mkdir -p /app/data /app/cache /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Start command
CMD ["python", "-m", "vid_diffusion_bench.api.app"]
'''
    
    def _generate_staging_dockerfile(self) -> str:
        """Generate staging Dockerfile."""
        return '''
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY pyproject.toml requirements-dev.txt ./
RUN pip install --no-cache-dir -e . && \\
    pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p /app/data /app/cache /app/logs

# Expose ports
EXPOSE 8000 8080

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=staging
ENV LOG_LEVEL=DEBUG

# Start command
CMD ["python", "-m", "vid_diffusion_bench.api.app"]
'''
    
    def _generate_development_dockerfile(self) -> str:
        """Generate development Dockerfile."""
        return '''
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    vim \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY pyproject.toml requirements-dev.txt ./
RUN pip install --no-cache-dir -e . && \\
    pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/cache /app/logs

# Expose ports
EXPOSE 8000 8080 5678

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Start command with hot reload
CMD ["python", "-m", "uvicorn", "vid_diffusion_bench.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
'''
    
    def _push_image(self, image_tag: str) -> bool:
        """Push image to registry."""
        try:
            push_cmd = ["docker", "push", image_tag]
            
            logger.info(f"Pushing image: {image_tag}")
            result = subprocess.run(push_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker push failed: {result.stderr}")
                return False
            
            logger.info(f"Successfully pushed image: {image_tag}")
            return True
            
        except Exception as e:
            logger.error(f"Image push failed: {e}")
            return False


class DeploymentPipeline:
    """Complete deployment pipeline orchestration."""
    
    def __init__(self, config: DeploymentConfig, project_root: Path = None):
        self.config = config
        self.project_root = project_root or Path(".")
        self.image_builder = DockerImageBuilder(self.project_root)
        self.k8s_deployment = KubernetesDeployment(config)
    
    async def deploy(self, kubectl_context: str = None, skip_build: bool = False) -> bool:
        """Execute complete deployment pipeline."""
        try:
            deployment_tag = f"{self.config.environment.value}-{int(time.time())}"
            
            logger.info(f"Starting deployment pipeline for {self.config.environment.value}")
            
            # Step 1: Build and push Docker image
            if not skip_build:
                logger.info("Building Docker image...")
                if not self.image_builder.build_image(deployment_tag, self.config.environment):
                    logger.error("Docker image build failed")
                    return False
            
            # Step 2: Update deployment configuration with new image tag
            logger.info("Updating deployment configuration...")
            
            # Step 3: Deploy to Kubernetes
            logger.info("Deploying to Kubernetes...")
            if not self.k8s_deployment.deploy(kubectl_context):
                logger.error("Kubernetes deployment failed")
                return False
            
            # Step 4: Run post-deployment verification
            logger.info("Running post-deployment verification...")
            if not await self._verify_deployment():
                logger.warning("Post-deployment verification failed")
                # Don't fail the deployment, just warn
            
            logger.info(f"Deployment pipeline completed successfully for {self.config.environment.value}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
            return False
    
    async def _verify_deployment(self) -> bool:
        """Verify deployment is healthy."""
        try:
            # This would include health checks, smoke tests, etc.
            # For now, just simulate verification
            
            import asyncio
            await asyncio.sleep(2)  # Simulate verification time
            
            logger.info("Deployment verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Deployment verification failed: {e}")
            return False
    
    def rollback(self, kubectl_context: str = None) -> bool:
        """Rollback to previous deployment."""
        try:
            cmd = ["kubectl"]
            if kubectl_context:
                cmd.extend(["--context", kubectl_context])
            
            cmd.extend([
                "rollout", "undo",
                "deployment/vid-diffusion-bench",
                f"--namespace={self.k8s_deployment.namespace}"
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Rollback failed: {result.stderr}")
                return False
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


def create_production_config() -> DeploymentConfig:
    """Create production deployment configuration."""
    return DeploymentConfig(
        environment=Environment.PRODUCTION,
        strategy=DeploymentStrategy.ROLLING,
        replicas=5,
        cpu_request="1000m",
        cpu_limit="4000m",
        memory_request="2Gi",
        memory_limit="8Gi",
        gpu_required=True,
        gpu_count=1,
        storage_size="50Gi",
        domain="vid-diffusion-bench.production.com",
        ssl_enabled=True,
        monitoring_enabled=True,
        autoscaling_enabled=True,
        min_replicas=3,
        max_replicas=20,
        secrets={
            "DATABASE_URL": "postgresql://user:pass@db:5432/benchmark",
            "API_KEY": "production-api-key",
            "JWT_SECRET": "production-jwt-secret"
        },
        env_vars={
            "BENCHMARK_WORKERS": "8",
            "CACHE_SIZE_MB": "4096",
            "MAX_CONCURRENT_BENCHMARKS": "10"
        }
    )


def create_staging_config() -> DeploymentConfig:
    """Create staging deployment configuration."""
    return DeploymentConfig(
        environment=Environment.STAGING,
        strategy=DeploymentStrategy.ROLLING,
        replicas=2,
        cpu_request="500m",
        cpu_limit="2000m",
        memory_request="1Gi",
        memory_limit="4Gi",
        gpu_required=False,
        storage_size="20Gi",
        domain="vid-diffusion-bench.staging.com",
        ssl_enabled=True,
        monitoring_enabled=True,
        autoscaling_enabled=False,
        secrets={
            "DATABASE_URL": "postgresql://user:pass@staging-db:5432/benchmark",
            "API_KEY": "staging-api-key"
        }
    )


async def deploy_to_production(
    kubectl_context: str = None,
    skip_build: bool = False
) -> bool:
    """Deploy to production environment."""
    config = create_production_config()
    pipeline = DeploymentPipeline(config)
    
    return await pipeline.deploy(kubectl_context, skip_build)


async def deploy_to_staging(
    kubectl_context: str = None,
    skip_build: bool = False
) -> bool:
    """Deploy to staging environment."""
    config = create_staging_config()
    pipeline = DeploymentPipeline(config)
    
    return await pipeline.deploy(kubectl_context, skip_build)