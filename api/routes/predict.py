"""Prediction routes."""
import time
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    ErrorResponse,
    PredictionDetailResponse,
    PredictionResponse,
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionDetailResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)):
    """Predict class for uploaded image.

    Args:
        file: Image file upload

    Returns:
        Prediction result with probabilities
    """
    from ..main import app

    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = app.state.predictor.predict(image, return_probs=True)

        return PredictionDetailResponse(
            class_name=result["class_name"],
            class_index=result["class_index"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_batch(
    files: List[UploadFile] = File(...),
):
    """Predict classes for multiple uploaded images.

    Args:
        files: List of image files

    Returns:
        Batch prediction results
    """
    from ..main import app

    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        images = []
        for file in files:
            contents = await file.read()
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)

        results = app.state.predictor.predict_batch(images, return_probs=True)

        prediction_items = [
            BatchPredictionItem(
                image_path=None,
                image_index=i,
                prediction=PredictionResponse(
                    class_name=r["class_name"],
                    class_index=r["class_index"],
                    confidence=r["confidence"],
                ),
            )
            for i, r in enumerate(results)
        ]

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=prediction_items,
            total_images=len(images),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/predict/{class_name}", response_model=PredictionResponse)
async def predict_by_name(class_name: str):
    """Check if a class exists in the model.

    Args:
        class_name: Class name to check

    Returns:
        Class information
    """
    from ..main import app

    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if class_name not in app.state.class_names:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found")

    class_index = app.state.class_names.index(class_name)

    return PredictionResponse(
        class_name=class_name,
        class_index=class_index,
        confidence=1.0,
    )