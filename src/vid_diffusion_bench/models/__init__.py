"""Model adapters and registry."""

from .base import ModelAdapter
from .registry import register_model, get_model, list_models

# Import model adapters to register them
from . import svd_adapter
from . import mock_adapters

__all__ = ["ModelAdapter", "register_model", "get_model", "list_models"]