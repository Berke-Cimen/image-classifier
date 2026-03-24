"""FastAPI application for image classification."""
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health, predict


def create_app(
    model=None,
    class_names=None,
    predictor=None,
    device=None,
    model_name=None,
    num_classes=None,
    image_size=None,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        model: Trained model
        class_names: List of class names
        predictor: Predictor instance
        device: torch device
        model_name: Model architecture name
        num_classes: Number of classes
        image_size: Input image size

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Image Classification API",
        description="Production-ready image classification service with PyTorch",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.model = model
    app.state.class_names = class_names or []
    app.state.predictor = predictor
    app.state.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model_name = model_name
    app.state.num_classes = num_classes or 0
    app.state.image_size = image_size or 224

    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/api", tags=["Prediction"])

    @app.on_event("startup")
    async def startup_event():
        """Run on application startup."""
        if app.state.predictor:
            print(f"Model loaded on {app.state.device}")
            print(f"Model: {app.state.model_name}")
            print(f"Classes: {len(app.state.class_names)}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Run on application shutdown."""
        pass

    return app


app = create_app()