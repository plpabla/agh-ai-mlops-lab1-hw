from fastapi import FastAPI

from data_models.api import PredictRequest, PredictResponse
from model import inference

app = FastAPI(version="0.0.1")


@app.post("/predict")
async def predict(data: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction=inference.predict(data.text))
