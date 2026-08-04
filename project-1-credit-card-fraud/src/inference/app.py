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
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore", category=UserWarning, message=".*protected namespace.*")

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles production microservice startup and teardown routines."""
    global model
    
    project_root = Path(__file__).resolve().parents[2]
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        db_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        # Dynamically load the latest registered version of the model
        client = MlflowClient()
        latest_versions = client.get_latest_versions("credit_card_fraud_model")
        if latest_versions:
            latest_version = latest_versions[-1].version
            model_uri = f"models:/credit_card_fraud_model/{latest_version}"
        else:
            model_uri = "models:/credit_card_fraud_model/1"
            
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"🟢 Production model cached successfully from: {model_uri}")
    except Exception as e:
        print(f"❌ Critical error loading model from MLflow: {str(e)}")
        model = None
        
    yield
    print("🛑 Microservice context tearing down...")

app = FastAPI(
    title="Enterprise Credit Card Fraud Risk Gateway",
    description="Real-time transactional inference API backed by MLflow Registry.",
    version="1.0.0",
    lifespan=lifespan
)

class TransactionData(BaseModel):
    features: List[float] = Field(
        ..., 
        description="Array of V1-V28 PCA features, scaled_amount, and scaled_time", 
        min_length=30,
        max_length=30
    )

class InferenceResponse(BaseModel):
    prediction: int
    risk_score: float

@app.get("/health")
def health_check():
    """Liveness probe endpoint."""
    return {"status": "HEALTHY", "model_loaded": model is not None}

@app.post("/predict", response_model=InferenceResponse)
def predict_fraud(payload: TransactionData):
    if model is None:
        raise HTTPException(status_code=503, detail="Inference engine is uninitialized.")
        
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame([payload.features], columns=feature_names)
    
    try:
        prediction = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.0
        return InferenceResponse(prediction=prediction, risk_score=proba)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference computation crash: {str(e)}")