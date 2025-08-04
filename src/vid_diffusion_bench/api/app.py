"""FastAPI application for Video Diffusion Benchmark Suite."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn

from ..database.connection import db_manager
from ..database.services import initialize_models_from_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Video Diffusion Benchmark API")
    
    # Initialize database
    try:
        db_manager.create_all_tables()
        initialize_models_from_registry()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        
    yield
    
    # Shutdown
    logger.info("Shutting down Video Diffusion Benchmark API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Video Diffusion Benchmark Suite API",
        description="REST API for benchmarking video diffusion models",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    from .routers import models, benchmarks, metrics, health
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
    app.include_router(benchmarks.router, prefix="/api/v1/benchmarks", tags=["benchmarks"])
    app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to docs."""
        return RedirectResponse(url="/docs")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "vid_diffusion_bench.api.app:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )