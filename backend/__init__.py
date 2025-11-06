# Backend module for the Multi-Modal Assistant
# Contains model loading and utilities for image and text processing

from .model import get_model_response
from .utils import init_session_state

__all__ = ['get_model_response', 'init_session_state']