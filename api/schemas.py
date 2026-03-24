"""Pydantic schemas for API."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    class_name: str = Field(..., description="Predicted class name")
    class_index: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")


class PredictionDetailResponse(PredictionResponse):
    """Response with detailed probabilities."""
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction."""
    image_path: Optional[str] = Field(None, description="Image path if available")
    image_index: int = Field(..., description="Index in batch")
    prediction: PredictionResponse


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[BatchPredictionItem] = Field(..., description="List of predictions")
    total_images: int = Field(..., description="Total number of images")
    processing_time_ms: float = Field(..., description="Total processing time in ms")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used (cuda/cpu)")
    model_name: Optional[str] = Field(None, description="Model architecture name")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model architecture name")
    num_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="List of class names")
    image_size: int = Field(..., description="Expected input image size")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")