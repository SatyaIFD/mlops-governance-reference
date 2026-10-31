"""
Production FastAPI Application - Project 1: Credit Card Fraud Detection
Exposes the registered fraud detection model via a high-performance REST API.
"""

import math
import os
import secrets
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from compliance.dora_circuit_breaker import MLSystemCircuitBreaker

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*protected namespace.*"
)

EXPECTED_FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + [
    "scaled_amount",
    "scaled_time",
]

model = None
circuit_breaker = MLSystemCircuitBreaker(failure_threshold=3, recovery_timeout_s=30.0)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _sanitize_non_finite_floats(obj: Any) -> Any:
    """Recursively converts NaN/Inf floats in validation error payloads into strings for strict JSON serialization."""
    if isinstance(obj, float):
        return str(obj) if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_non_finite_floats(v) for v in obj]
    return obj


def verify_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("EXPECTED_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=500, detail="Server authentication configuration error."
        )
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _validate_loaded_model_schema(loaded_model) -> None:
    """Verifies that the loaded model's expected feature schema matches the canonical 30 features."""
    model_features = None
    if hasattr(loaded_model, "get_booster"):
        model_features = loaded_model.get_booster().feature_names
    elif hasattr(loaded_model, "feature_names_in_"):
        model_features = list(loaded_model.feature_names_in_)

    if model_features is not None and list(model_features) != EXPECTED_FEATURE_NAMES:
        raise ValueError(
            f"Loaded model feature schema mismatch. Expected {EXPECTED_FEATURE_NAMES}, got {model_features}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    project_root = Path(__file__).resolve().parents[2]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        db_path = project_root.parent / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient()
        model_name = "fraud_detection_xgb"

        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]

        if prod_versions:
            latest_version = max(int(v.version) for v in prod_versions)
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/{latest_version}")
        elif versions:
            latest_version = max(int(v.version) for v in versions)
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/{latest_version}")
        else:
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/1")

        loaded = mlflow.xgboost.load_model(model_uri)
        _validate_loaded_model_schema(loaded)
        model = loaded
        print(f"🟢 Production model loaded and verified via URI: {model_uri}")
    except Exception as reg_err:
        print(f"⚠️ Model registry load or schema verification failed ({reg_err}).")
        model = None

    yield
    print("🛑 Microservice context tearing down...")


app = FastAPI(
    title="Enterprise Credit Card Fraud Risk Gateway",
    description="Real-time transactional inference API backed by MLflow Registry.",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api/v1/fraud",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    sanitized_errors = _sanitize_non_finite_floats(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class TransactionData(BaseModel):
    features: List[FiniteFloat] = Field(
        ...,
        description="Array of V1-V28 PCA features, scaled_amount, and scaled_time",
        min_length=30,
        max_length=30,
    )


class InferenceResponse(BaseModel):
    prediction: int
    risk_score: float
    status: str = "success"


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is uninitialized")
    return {"status": "HEALTHY", "model_loaded": True}


@app.post(
    "/predict",
    response_model=InferenceResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict_fraud(payload: TransactionData):
    if model is None:
        raise HTTPException(
            status_code=503, detail="Inference engine is uninitialized."
        )

    df = pd.DataFrame([payload.features], columns=EXPECTED_FEATURE_NAMES)

    def _execute_model_logic():
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
        return pred, prob

    breaker_response = circuit_breaker.execute_inference(_execute_model_logic)

    if breaker_response["status"] == "fallback":
        raise HTTPException(
            status_code=503,
            detail="Service unavailable due to internal circuit breaker fallback.",
        )

    prediction, proba = breaker_response["prediction"]
    return InferenceResponse(prediction=prediction, risk_score=proba)
