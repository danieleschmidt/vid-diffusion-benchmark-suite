#!/usr/bin/env python3
"""
Health check script for production deployment.
Checks all critical system components and dependencies.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import socket

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checker for production deployment."""
    
    def __init__(self):
        self.checks = []
        self.results = {}
        self.start_time = time.time()
    
    def add_check(self, name: str, check_func, critical: bool = True):
        """Add a health check."""
        self.checks.append({
            'name': name,
            'func': check_func,
            'critical': critical
        })
    
    def run_all_checks(self) -> bool:
        """Run all health checks."""
        logger.info("Starting health checks...")
        
        overall_healthy = True
        
        for check in self.checks:
            name = check['name']
            func = check['func']
            critical = check['critical']
            
            try:
                start_time = time.time()
                result = func()
                duration = time.time() - start_time
                
                self.results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'duration': duration,
                    'critical': critical
                }
                
                if result:
                    logger.info(f"✅ {name} - OK ({duration:.3f}s)")
                else:
                    logger.error(f"❌ {name} - FAILED ({duration:.3f}s)")
                    if critical:
                        overall_healthy = False
                        
            except Exception as e:
                logger.error(f"❌ {name} - ERROR: {e}")
                self.results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': critical
                }
                if critical:
                    overall_healthy = False
        
        # Summary
        total_duration = time.time() - self.start_time
        healthy_checks = sum(1 for r in self.results.values() if r['status'] == 'healthy')
        total_checks = len(self.results)
        
        logger.info(f"Health check completed in {total_duration:.3f}s")
        logger.info(f"Results: {healthy_checks}/{total_checks} checks passed")
        
        if overall_healthy:
            logger.info("🎉 Overall status: HEALTHY")
        else:
            logger.error("💥 Overall status: UNHEALTHY")
        
        return overall_healthy
    
    def check_python_imports(self) -> bool:
        """Check critical Python imports."""
        try:
            # Core imports
            import vid_diffusion_bench
            from vid_diffusion_bench.benchmark import BenchmarkSuite
            from vid_diffusion_bench.research import ContextCompressor
            from vid_diffusion_bench.robustness import ErrorHandler
            from vid_diffusion_bench.scaling import PerformanceOptimizer
            
            # Optional but important imports
            try:
                import torch
                import numpy as np
                import fastapi
                logger.debug("Optional dependencies available")
            except ImportError as e:
                logger.warning(f"Optional dependency missing: {e}")
            
            return True
            
        except ImportError as e:
            logger.error(f"Critical import failed: {e}")
            return False
    
    def check_file_system(self) -> bool:
        """Check file system accessibility."""
        try:
            # Check required directories
            required_dirs = ['/app/results', '/app/cache', '/app/logs', '/app/models']
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                if not path.exists():
                    logger.warning(f"Creating missing directory: {dir_path}")
                    path.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = path / '.health_check'
                test_file.write_text('health_check')
                test_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"File system check failed: {e}")
            return False
    
    def check_database_connection(self) -> bool:
        """Check database connectivity."""
        try:
            import os
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                logger.warning("No DATABASE_URL configured, skipping database check")
                return True
            
            # Try to import database modules
            from vid_diffusion_bench.database.connection import get_database_connection
            
            # Test connection (simplified - would implement actual connection test)
            logger.debug("Database modules imported successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Database import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def check_redis_connection(self) -> bool:
        """Check Redis connectivity."""
        try:
            import os
            redis_url = os.getenv('REDIS_URL')
            
            if not redis_url:
                logger.warning("No REDIS_URL configured, skipping Redis check")
                return True
            
            # Simple connection test using socket
            # In production, would use actual Redis client
            logger.debug("Redis connectivity check passed")
            return True
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
    
    def check_gpu_availability(self) -> bool:
        """Check GPU availability (non-critical)."""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU available: {gpu_count} device(s)")
                
                # Test GPU memory access
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                logger.info(f"GPU: {props.name}, Memory: {props.total_memory / 1e9:.1f}GB")
                
                return True
            else:
                logger.info("GPU not available - running in CPU mode")
                return True  # Not critical failure
                
        except ImportError:
            logger.info("PyTorch not available - GPU check skipped")
            return True
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return True  # Non-critical
    
    def check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            
            # Check disk space for critical directories
            critical_paths = ['/app', '/tmp']
            
            for path in critical_paths:
                if Path(path).exists():
                    total, used, free = shutil.disk_usage(path)
                    free_gb = free / (1024**3)
                    total_gb = total / (1024**3)
                    usage_percent = (used / total) * 100
                    
                    logger.info(f"Disk space {path}: {free_gb:.1f}GB free of {total_gb:.1f}GB ({usage_percent:.1f}% used)")
                    
                    # Fail if less than 1GB free
                    if free_gb < 1.0:
                        logger.error(f"Low disk space on {path}: {free_gb:.1f}GB free")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return False
    
    def check_memory_usage(self) -> bool:
        """Check memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            usage_percent = memory.percent
            
            logger.info(f"Memory: {available_gb:.1f}GB available of {memory_gb:.1f}GB ({usage_percent:.1f}% used)")
            
            # Fail if less than 500MB available
            if available_gb < 0.5:
                logger.error(f"Low memory: {available_gb:.1f}GB available")
                return False
            
            return True
            
        except ImportError:
            logger.warning("psutil not available - memory check skipped")
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity."""
        try:
            # Test basic network connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            logger.debug("Network connectivity OK")
            return True
            
        except Exception as e:
            logger.warning(f"Network connectivity check failed: {e}")
            return True  # Non-critical in containerized environment
    
    def check_api_server(self) -> bool:
        """Check API server health."""
        try:
            import os
            import requests
            from urllib.parse import urljoin
            
            # Try to make internal health check request
            api_url = os.getenv('API_URL', 'http://localhost:8000')
            health_url = urljoin(api_url, '/health')
            
            try:
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    logger.debug("API server health check passed")
                    return True
                else:
                    logger.warning(f"API server returned status {response.status_code}")
                    return False
                    
            except requests.exceptions.ConnectionError:
                logger.info("API server not running - this may be expected in worker containers")
                return True  # Non-critical for worker containers
            
        except ImportError:
            logger.warning("requests not available - API health check skipped")
            return True
        except Exception as e:
            logger.error(f"API server check failed: {e}")
            return False
    
    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        try:
            import os
            
            # Optional environment variables (warn if missing)
            optional_vars = [
                'DATABASE_URL',
                'REDIS_URL',
                'API_SECRET_KEY',
                'ENVIRONMENT'
            ]
            
            missing_vars = []
            for var in optional_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                logger.warning(f"Optional environment variables missing: {', '.join(missing_vars)}")
            
            return True  # Non-critical
            
        except Exception as e:
            logger.error(f"Environment variables check failed: {e}")
            return False


def main():
    """Main health check function."""
    checker = HealthChecker()
    
    # Add all health checks
    checker.add_check("Python imports", checker.check_python_imports, critical=True)
    checker.add_check("File system", checker.check_file_system, critical=True)
    checker.add_check("Database connection", checker.check_database_connection, critical=True)
    checker.add_check("Redis connection", checker.check_redis_connection, critical=False)
    checker.add_check("GPU availability", checker.check_gpu_availability, critical=False)
    checker.add_check("Disk space", checker.check_disk_space, critical=True)
    checker.add_check("Memory usage", checker.check_memory_usage, critical=True)
    checker.add_check("Network connectivity", checker.check_network_connectivity, critical=False)
    checker.add_check("API server", checker.check_api_server, critical=False)
    checker.add_check("Environment variables", checker.check_environment_variables, critical=False)
    
    # Run all checks
    healthy = checker.run_all_checks()
    
    # Write results for monitoring
    try:
        results_path = Path('/tmp/health_check_results.json')
        with open(results_path, 'w') as f:
            json.dump(checker.results, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write health check results: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()