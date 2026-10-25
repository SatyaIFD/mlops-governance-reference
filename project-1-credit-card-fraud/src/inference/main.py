"""
Production FastAPI Application
Exposes the registered fraud detection model via a high-performance REST API.
"""

import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from compliance.dora_circuit_breaker import MLSystemCircuitBreaker

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*protected namespace.*"
)

model = None
# Initialize the circuit breaker with a threshold of 3 consecutive failures
circuit_breaker = MLSystemCircuitBreaker(failure_threshold=3, recovery_timeout_s=30.0)


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

        # Look for versions explicitly tagged as Production
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]

        if prod_versions:
            latest_version = max([int(v.version) for v in prod_versions])
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/{latest_version}")
        elif versions:
            latest_version = max([int(v.version) for v in versions])
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/{latest_version}")
        else:
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/1")

        # Load via xgboost flavor directly to guarantee predict_proba is available
        model = mlflow.xgboost.load_model(model_uri)
        print(f"🟢 Production model loaded successfully via URI: {model_uri}")
    except Exception as reg_err:
        print(
            f"⚠️ Primary URI load failed ({reg_err}). Attempting robust local artifact discovery..."
        )
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


class TransactionData(BaseModel):
    features: List[float] = Field(
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


@app.post("/predict", response_model=InferenceResponse)
def predict_fraud(payload: TransactionData):
    if model is None:
        raise HTTPException(
            status_code=503, detail="Inference engine is uninitialized."
        )

    feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
    df = pd.DataFrame([payload.features], columns=feature_names)

    def _execute_model_logic():
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
        return pred, prob

    # Wrap the inference in the DORA Circuit Breaker
    breaker_response = circuit_breaker.execute_inference(_execute_model_logic)

    if breaker_response["status"] == "fallback":
        return InferenceResponse(
            prediction=0, risk_score=0.0, status="CIRCUIT_OPEN_FALLBACK"
        )

    prediction, proba = breaker_response["prediction"]
    return InferenceResponse(prediction=prediction, risk_score=proba)
