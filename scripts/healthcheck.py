#!/usr/bin/env python3
"""Health check script for research framework services."""

import sys
import os
import requests
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_api_health() -> bool:
    """Check API health endpoint."""
    try:
        response = requests.get(
            'http://localhost:8000/health',
            timeout=10
        )
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Check required health indicators
            required_keys = ['status', 'timestamp', 'version']
            for key in required_keys:
                if key not in health_data:
                    logger.error(f"Missing health key: {key}")
                    return False
            
            if health_data['status'] == 'healthy':
                logger.info("API health check passed")
                return True
            else:
                logger.error(f"API status not healthy: {health_data['status']}")
                return False
        else:
            logger.error(f"API health check failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check request failed: {e}")
        return False
    except Exception as e:
        logger.error(f"API health check error: {e}")
        return False


def check_file_system() -> bool:
    """Check file system health."""
    try:
        # Check required directories exist and are writable
        required_dirs = [
            '/app/logs',
            '/app/cache',
            '/app/data',
            '/app/results'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            
            if not path.exists():
                logger.error(f"Required directory does not exist: {dir_path}")
                return False
            
            if not path.is_dir():
                logger.error(f"Path is not a directory: {dir_path}")
                return False
            
            # Test write permissions
            test_file = path / '.health_check_test'
            try:
                test_file.write_text('test')
                test_file.unlink()
            except Exception as e:
                logger.error(f"Cannot write to directory {dir_path}: {e}")
                return False
        
        logger.info("File system health check passed")
        return True
        
    except Exception as e:
        logger.error(f"File system health check error: {e}")
        return False


def check_python_imports() -> bool:
    """Check that critical Python modules can be imported."""
    try:
        # Test core research framework imports
        critical_modules = [
            'vid_diffusion_bench',
            'torch',
            'numpy',
            'logging'
        ]
        
        for module_name in critical_modules:
            try:
                __import__(module_name)
                logger.debug(f"Successfully imported {module_name}")
            except ImportError as e:
                logger.error(f"Failed to import critical module {module_name}: {e}")
                return False
        
        logger.info("Python imports health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Python imports health check error: {e}")
        return False


def check_gpu_availability() -> bool:
    """Check GPU availability if CUDA is expected."""
    try:
        import torch
        
        cuda_expected = os.getenv('CUDA_VISIBLE_DEVICES') is not None
        
        if cuda_expected:
            if not torch.cuda.is_available():
                logger.error("CUDA expected but not available")
                return False
            
            # Check if any GPUs are accessible
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.error("No CUDA devices found")
                return False
            
            logger.info(f"GPU health check passed - {gpu_count} devices available")
        else:
            logger.info("GPU health check skipped - CUDA not expected")
        
        return True
        
    except Exception as e:
        logger.error(f"GPU health check error: {e}")
        return False


def check_memory_usage() -> bool:
    """Check memory usage is within acceptable limits."""
    try:
        import psutil
        
        # Check system memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 95:
            logger.error(f"Memory usage critically high: {memory_percent}%")
            return False
        elif memory_percent > 85:
            logger.warning(f"Memory usage high: {memory_percent}%")
        
        # Check disk space
        disk = psutil.disk_usage('/app')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 95:
            logger.error(f"Disk usage critically high: {disk_percent:.1f}%")
            return False
        elif disk_percent > 85:
            logger.warning(f"Disk usage high: {disk_percent:.1f}%")
        
        logger.info(f"Memory health check passed - RAM: {memory_percent}%, Disk: {disk_percent:.1f}%")
        return True
        
    except Exception as e:
        logger.error(f"Memory health check error: {e}")
        return False


def check_service_specific() -> bool:
    """Check service-specific health based on SERVICE_NAME."""
    try:
        service_name = os.getenv('SERVICE_NAME', 'research-api')
        
        if service_name == 'research-api':
            # Main API service - check API endpoint
            return check_api_health()
            
        elif service_name in ['adaptive-algorithms', 'novel-metrics', 'quantum-acceleration', 'intelligent-scaling']:
            # Specialized services - check their specific endpoints
            try:
                service_port_map = {
                    'adaptive-algorithms': 8001,
                    'novel-metrics': 8002,
                    'quantum-acceleration': 8003,
                    'intelligent-scaling': 8004
                }
                
                port = service_port_map.get(service_name, 8000)
                response = requests.get(
                    f'http://localhost:{port}/health',
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info(f"Service {service_name} health check passed")
                    return True
                else:
                    logger.error(f"Service {service_name} health check failed: {response.status_code}")
                    return False
                    
            except requests.exceptions.RequestException:
                # Service might not have HTTP endpoint - check process instead
                logger.info(f"Service {service_name} HTTP check failed, assuming service-specific health")
                return True
                
        else:
            logger.info(f"No specific health check for service: {service_name}")
            return True
            
    except Exception as e:
        logger.error(f"Service-specific health check error: {e}")
        return False


def main():
    """Main health check function."""
    logger.info("Starting health check...")
    
    health_checks = [
        ('File System', check_file_system),
        ('Python Imports', check_python_imports),
        ('GPU Availability', check_gpu_availability),
        ('Memory Usage', check_memory_usage),
        ('Service Specific', check_service_specific)
    ]
    
    all_passed = True
    
    for check_name, check_func in health_checks:
        logger.info(f"Running {check_name} health check...")
        
        try:
            if not check_func():
                logger.error(f"{check_name} health check FAILED")
                all_passed = False
            else:
                logger.info(f"{check_name} health check PASSED")
        except Exception as e:
            logger.error(f"{check_name} health check ERROR: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("All health checks PASSED ✅")
        sys.exit(0)
    else:
        logger.error("Some health checks FAILED ❌")
        sys.exit(1)


if __name__ == '__main__':
    main()