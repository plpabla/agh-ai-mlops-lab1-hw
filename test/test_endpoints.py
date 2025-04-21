from fastapi.testclient import TestClient

from app import app
from data_models.api import PredictRequest, PredictResponse


client = TestClient(app)


def test_can_import_app():
    assert app is not None


def test_predict_accepts_json():
    req = PredictRequest(text="test")
    response = client.post("/predict", json=req.model_dump())

    assert response.status_code == 200


def test_predict_invalid_json_returns_422():
    req = PredictRequest(text="test")
    response = client.post("/predict", json=req.model_dump(exclude={"text"}))

    assert response.status_code == 422


def test_predict_negative_response():
    req = PredictRequest(text="I'm very disappointed with the service.")
    response = client.post("/predict", json=req.model_dump())

    assert response.status_code == 200
    assert response.json() == PredictResponse(prediction="negative").model_dump()


def test_predict_positive_response():
    req = PredictRequest(text="Very good service, I am happy with the product.")
    response = client.post("/predict", json=req.model_dump())

    assert response.status_code == 200
    assert response.json() == PredictResponse(prediction="positive").model_dump()
