"""Health check endpoints."""

import logging
from fastapi import APIRouter

from ...database.connection import db_manager
from ...models.registry import list_models
from ..schemas import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthStatus)
async def health_check():
    """Get API health status."""
    # Check database health
    db_health = db_manager.health_check()
    
    # Check available models
    try:
        available_models = len(list_models())
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        available_models = 0
    
    # Determine overall status
    overall_status = "healthy" if db_health["status"] == "healthy" and available_models > 0 else "degraded"
    
    return HealthStatus(
        status=overall_status,
        database=db_health,
        models_available=available_models,
        version="0.1.0"
    )


@router.get("/readiness")
async def readiness_check():
    """Kubernetes readiness check."""
    db_health = db_manager.health_check()
    if db_health["status"] != "healthy":
        return {"ready": False, "reason": "Database not ready"}
    
    return {"ready": True}


@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness check."""
    return {"alive": True}