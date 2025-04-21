from fastapi.testclient import TestClient

from app import app
from models.api import PredictRequest, PredictResponse


client = TestClient(app)


def test_can_import_app():
    assert app is not None


def test_predict_accepts_json():
    req = PredictRequest(text="test")
    response = client.post("/predict", json=req.model_dump())

    assert response.status_code == 200


def test_predict_response():
    req = PredictRequest(text="test")
    response = client.post("/predict", json=req.model_dump())

    assert response.status_code == 200
    assert response.json() == PredictResponse(prediction="neutral").model_dump()
