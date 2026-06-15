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
    
    # Smart path resolver: look for mlflow.db locally, then search parent directories
    current_dir = Path(__file__).resolve().parent
    db_path = None
    
    # Traverse upwards up to 4 levels to find mlflow.db
    for parent in [current_dir] + list(current_dir.parents)[:4]:
        if (parent / "mlflow.db").exists():
            db_path = parent / "mlflow.db"
            break
            
    if not db_path:
        # Fallback to absolute system path reference if traversal fails
        db_path = Path("/media/storage/mlops-governance-reference/mlflow.db")
        
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    try:
        model_uri = "models:/credit_card_fraud_model/1"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"🟢 Production model cached successfully from: {model_uri}")
    except Exception as e:
        print(f"❌ Critical error loading model from MLflow: {str(e)}")
        # Don't crash lifespan initialization completely during test discovery 
        # unless it is an unrecoverable production error environment
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
    # Enforce exact features data contract schema via Pydantic V2 signatures
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
    """Processes real-time transaction strings and applies predictive weights."""
    if model is None:
        raise HTTPException(status_code=503, detail="Inference engine is uninitialized.")
        
    # EXACT MODEL TRAINING DATA CONTRACT PARSING
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame([payload.features], columns=feature_names)
    
    try:
        prediction = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.0
        
        return InferenceResponse(prediction=prediction, risk_score=proba)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference computation crash: {str(e)}")