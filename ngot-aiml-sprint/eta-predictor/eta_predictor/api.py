from fastapi import FastAPI, APIRouter
from eta_predictor.schemas import ETARequest, ETAResponse, HealthResponse
from eta_predictor.predictor import Predictor
from eta_predictor.main import app as _app

app = _app  

predictor = Predictor()  # will use heuristic unless model saved


def format_eta(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m" if h else f"{m}m"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        api_version="1.0.0",
    )


@app.post("/predict", response_model=ETAResponse)
async def predict_eta(request: ETARequest):
    features = request.to_feature_vector()
    eta, ci_low, ci_high = predictor.predict(features)
    return ETAResponse(
        eta_minutes=eta,
        eta_human_readable=format_eta(eta),
        distance_km=request.distance_km,
        confidence_low=ci_low,
        confidence_high=ci_high,
        is_rush_hour=(7 <= request.hour_of_day <= 9) or (17 <= request.hour_of_day <= 19),
    )
