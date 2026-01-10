"""Bag of Concepts (BoC) - Multimodal concept-based generation."""

__version__ = '0.1.0'

from .models import BoCModel
from .training import get_small_config, get_base_config, get_large_config

__all__ = [
    'BoCModel',
    'get_small_config',
    'get_base_config',
    'get_large_config',
]
