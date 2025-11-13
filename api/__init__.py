"""
API Package - Football AI Prediction Service
"""

from .main import app
from .prediction_service import PredictionService
from .models import (
    MatchInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    'app',
    'PredictionService',
    'MatchInput',
    'PredictionResponse',
    'HealthResponse',
    'ErrorResponse'
]
