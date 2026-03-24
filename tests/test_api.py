"""Tests for API modules."""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health endpoints."""

    def test_health_check(self):
        """Test health check endpoint."""
        from api.main import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_prediction_response(self):
        """Test PredictionResponse schema."""
        from api.schemas import PredictionResponse

        response = PredictionResponse(
            class_name="cat",
            class_index=0,
            confidence=0.95,
        )
        assert response.class_name == "cat"
        assert response.class_index == 0
        assert response.confidence == 0.95

    def test_prediction_detail_response(self):
        """Test PredictionDetailResponse schema."""
        from api.schemas import PredictionDetailResponse

        response = PredictionDetailResponse(
            class_name="cat",
            class_index=0,
            confidence=0.95,
            probabilities={"cat": 0.95, "dog": 0.05},
        )
        assert len(response.probabilities) == 2
