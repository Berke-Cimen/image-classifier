"""Health check routes."""
from fastapi import APIRouter

from ..schemas import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from ..main import app

    return HealthResponse(
        status="healthy",
        model_loaded=app.state.model is not None,
        device=str(app.state.device),
        model_name=getattr(app.state, "model_name", None),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    from ..main import app

    if app.state.model is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=getattr(app.state, "model_name", "unknown"),
        num_classes=app.state.num_classes,
        class_names=app.state.class_names,
        image_size=app.state.image_size,
    )