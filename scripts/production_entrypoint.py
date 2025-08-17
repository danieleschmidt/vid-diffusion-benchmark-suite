#!/usr/bin/env python3
"""Production entrypoint script for research framework deployment."""

import os
import sys
import logging
import signal
import time
import subprocess
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/production.log')
    ]
)

logger = logging.getLogger(__name__)


class ProductionServer:
    """Production server manager for research framework."""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
    def _get_server_command(self) -> List[str]:
        """Get server command based on environment configuration."""
        
        # Default configuration
        workers = int(os.getenv('WORKERS', '4'))
        worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')
        worker_connections = int(os.getenv('WORKER_CONNECTIONS', '1000'))
        max_requests = int(os.getenv('MAX_REQUESTS', '1000'))
        max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', '100'))
        timeout = int(os.getenv('TIMEOUT', '120'))
        keepalive = int(os.getenv('KEEPALIVE', '5'))
        
        # Build command
        cmd = [
            'gunicorn',
            'vid_diffusion_bench.api.app:app',
            '--bind', '0.0.0.0:8000',
            '--workers', str(workers),
            '--worker-class', worker_class,
            '--worker-connections', str(worker_connections),
            '--max-requests', str(max_requests),
            '--max-requests-jitter', str(max_requests_jitter),
            '--timeout', str(timeout),
            '--keepalive', str(keepalive),
            '--preload',
            '--access-logfile', '/app/logs/access.log',
            '--error-logfile', '/app/logs/error.log',
            '--log-level', 'info',
            '--capture-output'
        ]
        
        return cmd
    
    def _prepare_environment(self):
        """Prepare production environment."""
        logger.info("Preparing production environment...")
        
        # Create log directories
        log_dir = Path('/app/logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create cache directories
        cache_dirs = [
            '/app/cache/models',
            '/app/cache/metrics',
            '/app/cache/adaptive',
            '/app/cache/quantum'
        ]
        
        for cache_dir in cache_dirs:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ.setdefault('PYTHONPATH', '/app/src')
        os.environ.setdefault('TORCH_HOME', '/app/cache/models')
        os.environ.setdefault('HF_HOME', '/app/cache/models')
        
        # Log configuration
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
        logger.info(f"Workers: {os.getenv('WORKERS', '4')}")
        logger.info(f"Worker Class: {os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')}")
        
    def _health_check(self) -> bool:
        """Perform health check."""
        try:
            # Import and test core modules
            from vid_diffusion_bench.research.adaptive_algorithms import AdaptiveDiffusionOptimizer
            from vid_diffusion_bench.research.validation_framework import ComprehensiveValidator
            from vid_diffusion_bench.robustness.advanced_error_handling import AdvancedErrorHandler
            
            logger.info("Core modules imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def start_server(self):
        """Start the production server."""
        logger.info("Starting production server...")
        
        # Prepare environment
        self._prepare_environment()
        
        # Health check
        if not self._health_check():
            logger.error("Health check failed, aborting startup")
            sys.exit(1)
        
        # Get server command
        cmd = self._get_server_command()
        logger.info(f"Server command: {' '.join(cmd)}")
        
        try:
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"Server started with PID: {self.server_process.pid}")
            
            # Monitor server output
            for line in iter(self.server_process.stdout.readline, ''):
                if line:
                    logger.info(f"Server: {line.strip()}")
                
                # Check if shutdown requested
                if self.shutdown_requested:
                    break
                    
                # Check if process is still running
                if self.server_process.poll() is not None:
                    break
            
            # Wait for process to complete
            return_code = self.server_process.wait()
            logger.info(f"Server process exited with code: {return_code}")
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            sys.exit(1)
    
    def stop_server(self):
        """Stop the production server gracefully."""
        if self.server_process:
            logger.info("Stopping server gracefully...")
            
            try:
                # Send SIGTERM for graceful shutdown
                self.server_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.server_process.wait(timeout=30)
                    logger.info("Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown timeout, forcing termination")
                    self.server_process.kill()
                    self.server_process.wait()
                    logger.info("Server terminated forcefully")
                    
            except Exception as e:
                logger.error(f"Error stopping server: {e}")


def main():
    """Main entrypoint function."""
    logger.info("Starting production entrypoint...")
    
    # Get service type
    service_name = os.getenv('SERVICE_NAME', 'research-api')
    
    if service_name == 'research-api':
        # Start main API server
        server = ProductionServer()
        
        try:
            server.start_server()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            server.stop_server()
            
    elif service_name == 'adaptive-algorithms':
        # Start adaptive algorithms service
        from scripts.adaptive_service_entrypoint import main as adaptive_main
        adaptive_main()
        
    elif service_name == 'novel-metrics':
        # Start metrics service
        from scripts.metrics_service_entrypoint import main as metrics_main
        metrics_main()
        
    elif service_name == 'quantum-acceleration':
        # Start quantum service
        from scripts.quantum_service_entrypoint import main as quantum_main
        quantum_main()
        
    elif service_name == 'intelligent-scaling':
        # Start scaling service
        from scripts.scaling_service_entrypoint import main as scaling_main
        scaling_main()
        
    else:
        logger.error(f"Unknown service name: {service_name}")
        sys.exit(1)
    
    logger.info("Production entrypoint completed")


if __name__ == '__main__':
    main()