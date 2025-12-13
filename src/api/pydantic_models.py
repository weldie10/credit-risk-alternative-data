"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and values",
        example={"Amount": 1000.0, "Value": 1000.0, "transaction_frequency": 5.0}
    )
    
    @validator('features')
    def validate_features_not_empty(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries",
        min_items=1
    )
    
    @validator('features')
    def validate_features_list_not_empty(cls, v):
        if not v:
            raise ValueError("Features list cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(
        ...,
        description="Predicted class (0 or 1)",
        ge=0,
        le=1
    )
    probability: float = Field(
        ...,
        description="Predicted probability of positive class",
        ge=0.0,
        le=1.0
    )
    risk_score: int = Field(
        ...,
        description="Credit risk score (0-1000)",
        ge=0,
        le=1000
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )
    total_processed: int = Field(
        ...,
        description="Total number of predictions processed"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="0.1.0", description="API version")
    model_loaded: bool = Field(default=False, description="Whether model is loaded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Current timestamp")
