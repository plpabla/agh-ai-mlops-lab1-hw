from fastapi import FastAPI

from models.api import PredictRequest, PredictResponse

app = FastAPI(version="0.0.1")


@app.post("/predict")
async def predict(data: PredictRequest) -> PredictResponse:
    """
    Dummy prediction endpoint.
    """
    # Simulate a prediction process
    return PredictResponse(prediction="neutral")
