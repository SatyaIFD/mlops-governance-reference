"""
Production FastAPI Application
Exposes the registered fraud detection model via a high-performance REST API.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import mlflow.pyfunc

# Suppress the minor internal Pydantic namespace warning for production logs
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*protected namespace.*")

# Global variable to cache our inference engine in memory
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles production microservice startup and teardown routines."""
    global model
    BASE_DIR = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
    
    try:
        # Resolve model registry target tracking URI path
        model_uri = "models:/credit_card_fraud_model/1"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"🟢 Production model cached successfully from: {model_uri}")
    except Exception as e:
        print(f"❌ Critical error loading model from MLflow: {str(e)}")
        raise RuntimeError("Inference server halted: Model Registry unreachable.")
        
    yield
    # Clean up tasks or close active database connections on shutdown go here
    print("🛑 Microservice context tearing down...")

app = FastAPI(
    title="Enterprise Credit Card Fraud Risk Gateway",
    description="Real-time transactional inference API backed by MLflow Registry.",
    version="1.0.0",
    lifespan=lifespan
)

class TransactionData(BaseModel):
    # Enforce exact features data contract schema via Pydantic type signatures
    features: List[float] = Field(
        ..., 
        description="Array of V1-V28 PCA features, scaled_amount, and scaled_time", 
        min_items=30, 
        max_items=30
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
    """Processes real-time transaction strings and applies predictive weights."""
    if model is None:
        raise HTTPException(status_code=503, detail="Inference engine is uninitialized.")
        
    # FIX: Change 'scaled_amount' and 'scaled_time' back to the model's expected 'Amount' and 'Time'
    # Change the structural alignment sequence to place 'Time' first and 'Amount' last to match training data positions
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # Format incoming telemetry matrix safely into an analytical dataframe
    df = pd.DataFrame([payload.features], columns=feature_names)
    
    try:
        prediction = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.0
        
        return InferenceResponse(prediction=prediction, risk_score=proba)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference computation crash: {str(e)}")