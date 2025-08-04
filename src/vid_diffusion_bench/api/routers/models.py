"""Model management endpoints."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from ...database.services import ModelService
from ...models.registry import list_models as list_registered_models
from ..schemas import Model, ModelCreate, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=List[Model])
async def list_models(
    active_only: bool = Query(True, description="Only return active models"),
    model_type: Optional[str] = Query(None, description="Filter by model type")
):
    """List all models."""
    try:
        models = ModelService.list_models(active_only=active_only, model_type=model_type)
        return [Model.model_validate(model) for model in models]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@router.get("/registered", response_model=List[str])
async def list_registered_models_endpoint():
    """List models registered in the model registry."""
    try:
        return list_registered_models()
    except Exception as e:
        logger.error(f"Failed to list registered models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve registered models")


@router.get("/{model_name}", response_model=Model)
async def get_model(model_name: str):
    """Get model by name."""
    model = ModelService.get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return Model.model_validate(model)


@router.post("/", response_model=Model, status_code=201)
async def create_model(model_data: ModelCreate):
    """Create a new model."""
    try:
        # Check if model already exists
        existing = ModelService.get_model_by_name(model_data.name)
        if existing:
            raise HTTPException(
                status_code=409, 
                detail=f"Model '{model_data.name}' already exists"
            )
        
        # Create model
        model = ModelService.create_model(
            name=model_data.name,
            display_name=model_data.display_name,
            model_type=model_data.model_type,
            requirements=model_data.requirements.model_dump(),
            description=model_data.description,
            version=model_data.version,
            author=model_data.author,
            paper_url=model_data.paper_url,
            code_url=model_data.code_url,
            huggingface_id=model_data.huggingface_id
        )
        
        return Model.model_validate(model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail="Failed to create model")


@router.put("/{model_name}", response_model=Model)
async def update_model(model_name: str, updates: dict):
    """Update model information."""
    try:
        model = ModelService.get_model_by_name(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        updated_model = ModelService.update_model(model.id, **updates)
        if not updated_model:
            raise HTTPException(status_code=500, detail="Failed to update model")
            
        return Model.model_validate(updated_model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model: {e}")
        raise HTTPException(status_code=500, detail="Failed to update model")


@router.delete("/{model_name}")
async def delete_model(model_name: str):
    """Deactivate a model (soft delete)."""
    try:
        model = ModelService.get_model_by_name(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        success = ModelService.delete_model(model.id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to deactivate model")
            
        return {"message": f"Model '{model_name}' deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail="Failed to deactivate model")


@router.get("/{model_name}/requirements")
async def get_model_requirements(model_name: str):
    """Get model hardware and software requirements."""
    try:
        from ...models.registry import get_model
        
        # Try to get requirements from adapter
        try:
            adapter = get_model(model_name, device="cpu")
            requirements = adapter.requirements
            return {
                "model_name": model_name,
                "requirements": requirements,
                "source": "adapter"
            }
        except Exception:
            # Fallback to database
            model = ModelService.get_model_by_name(model_name)
            if model:
                return {
                    "model_name": model_name,
                    "requirements": model.requirements_dict,
                    "source": "database"
                }
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model requirements: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model requirements")