"""Model registry for managing available models."""

from typing import Dict, List, Type, Optional
from .base import ModelAdapter

_MODEL_REGISTRY: Dict[str, Type[ModelAdapter]] = {}


def register_model(name: str):
    """Decorator to register a model adapter.
    
    Args:
        name: Unique name for the model
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[ModelAdapter]):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> ModelAdapter:
    """Get model instance by name.
    
    Args:
        name: Registered model name
        **kwargs: Model configuration parameters
        
    Returns:
        Instantiated model adapter
        
    Raises:
        KeyError: If model not found in registry
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found. Available: {list_models()}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> List[str]:
    """List all registered model names.
    
    Returns:
        List of available model names
    """
    return list(_MODEL_REGISTRY.keys())