import uuid
import logging
from pydantic import BaseModel
from fastapi import FastAPI, Request
from prodml.config import Settings
from prodml.logging_conf import (
    configure_logging,
    correlation_id_var,
)
from prodml.predict import DurationPredictor


class PredictionRequest(BaseModel):
    trip_distance: float
    PULocationID: int
    DOLocationID: int


configure_logging()

logger = logging.getLogger(__name__)
settings = Settings()
app = FastAPI()
predictor = DurationPredictor(settings.model_path)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    logger.info(
        "Request received | method=%s | path=%s",
        request.method,
        request.url.path,
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id

    return response


@app.get("/")
async def home():
    logger.info("Hello World")
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(
        f"Received prediction request with trip_distance={request.trip_distance}, PULocationID={request.PULocationID}, DOLocationID={request.DOLocationID}"
    )

    features = {
        "trip_distance": request.trip_distance,
        "PU_DO": f"{request.PULocationID}_{request.DOLocationID}",
    }

    prediction = predictor.predict_one(features)
    logger.info(f"Prediction result: {prediction}")
    return {"predicted_duration": prediction}


@app.post("/predict_batch")
async def predict_batch(requests: list[PredictionRequest]):
    logger.info(f"Received batch prediction request with {len(requests)} requests")

    features_list = [
        {
            "trip_distance": req.trip_distance,
            "PU_DO": f"{req.PULocationID}_{req.DOLocationID}",
        }
        for req in requests
    ]

    predictions = predictor.predict_batch(features_list)
    logger.info(f"Batch prediction results: {predictions}")
    return {"predicted_durations": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
