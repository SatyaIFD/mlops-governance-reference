"""
Inference API Service - Project 2: Loan Default Prediction
Exposes the fair, sample-reweighted XGBoost model as a production-grade FastAPI endpoint.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

model = None
feature_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles microservice startup and teardown routines."""
    global model, feature_columns
    print("🚀 Initializing inference context...")
    
    project_root = Path(__file__).resolve().parents[2]
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        db_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        client = MlflowClient()
        model_versions = client.search_model_versions("name='loan_default_production_model'")
        if model_versions:
            latest_version = max([int(mv.version) for mv in model_versions])
            model_uri = os.getenv("MODEL_URI", f"models:/loan_default_production_model/{latest_version}")
        else:
            model_uri = os.getenv("MODEL_URI", "models:/loan_default_production_model/1")
            
        # Natively load XGBoost model to bypass strict PyFunc schema enforcement
        model = mlflow.xgboost.load_model(model_uri)
        print(f"🟢 Production model loaded successfully via URI: {model_uri}")
    except Exception as reg_err:
        print(f"⚠️ Registry URI load failed ({reg_err}). Attempting local artifact discovery in mlruns...")
        try:
            mlmodel_files = list((project_root / "mlruns").rglob("MLmodel"))
            if mlmodel_files:
                model_dir = mlmodel_files[0].parent
                model = mlflow.xgboost.load_model(str(model_dir))
                print(f"🟢 Production model loaded successfully from fallback artifact path: {model_dir}")
            else:
                raise FileNotFoundError(f"No MLmodel files found under {project_root / 'mlruns'}")
        except Exception as e:
            print(f"❌ Critical error loading model from MLflow: {str(e)}")
            model = None

    if model is not None:
        try:
            if hasattr(model, "get_booster"):
                feature_columns = model.get_booster().feature_names
            elif hasattr(model, "feature_names_in_"):
                feature_columns = list(model.feature_names_in_)
        except Exception as schema_err:
            print(f"⚠️ Schema extraction warning: {schema_err}")

    yield
    print("🛑 Microservice context tearing down...")

app = FastAPI(
    title="Fair Loan Default Prediction Service",
    description="Responsible AI Risk Scoring Engine using a Sample-Reweighted XGBoost Model.",
    version="1.0.0",
    lifespan=lifespan
)

class LoanApplicationPayload(BaseModel):
    Age: int = Field(..., ge=18, le=120, description="Age of the applicant")
    Income: int = Field(..., ge=0, description="Annual income of the applicant")
    CreditScore: int = Field(..., ge=300, le=850, description="Financial credit score")
    LoanAmount: int = Field(..., ge=0, description="Requested loan amount")
    DTIRatio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-Income ratio")
    MonthsEmployed: int = Field(..., ge=0, description="Total months of employment history")
    InterestRate: float = Field(..., ge=0.0, description="Offered loan interest rate")
    NumCreditLines: int = Field(..., ge=0, description="Number of open credit lines")
    LoanTerm: int = Field(..., ge=1, description="Term of the loan in months")
    Education: str = Field(..., description="Highest education tier attained")
    EmploymentType: str = Field(..., description="Employment nature")
    MaritalStatus: str = Field(..., description="Marital status context")
    HasMortgage: str = Field(..., description="Mortgage ownership indicator ('Yes' or 'No')")
    HasDependents: str = Field(..., description="Dependents visibility indicator ('Yes' or 'No')")
    LoanPurpose: str = Field(..., description="Core purpose behind the loan request")
    HasCoSigner: str = Field(..., description="Co-signer presence indicator ('Yes' or 'No')")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 25,
                "Income": 55000,
                "CreditScore": 680,
                "LoanAmount": 15000,
                "DTIRatio": 0.35,
                "MonthsEmployed": 24,
                "InterestRate": 5.4,
                "NumCreditLines": 3,
                "LoanTerm": 36,
                "Education": "Bachelor's",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Single",
                "HasMortgage": "No",
                "HasDependents": "No",
                "LoanPurpose": "Auto",
                "HasCoSigner": "Yes"
            }
        }
    )

@app.post("/predict", tags=["Inference"])
async def predict_loan_risk(payload: LoanApplicationPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable or uninitialized.")
        
    try:
        raw_data = pd.DataFrame([payload.model_dump()])
        raw_data['is_young'] = (raw_data['Age'] < 30).astype(int)
        
        categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
        transformed_df = pd.get_dummies(raw_data, columns=categorical_cols, drop_first=True)
        
        if feature_columns:
            for col in feature_columns:
                if col not in transformed_df.columns:
                    transformed_df[col] = 0
            final_payload = transformed_df[feature_columns]
        else:
            final_payload = transformed_df

        bool_cols = final_payload.select_dtypes(include=['bool']).columns
        if not bool_cols.empty:
            final_payload[bool_cols] = final_payload[bool_cols].astype('int8')
            
        preds = model.predict(final_payload)
        prob_default = float(model.predict_proba(final_payload)[0][1]) if hasattr(model, "predict_proba") else float(preds[0])
        prediction_label = int(preds[0]) if not hasattr(model, "predict_proba") else int(prob_default >= 0.5)
        
        credit_decision = "DENIED" if prediction_label == 1 else "APPROVED"
        
        return {
            "risk_metrics": {
                "default_probability": round(prob_default, 4),
                "risk_score_tier": "HIGH RISK" if prob_default >= 0.50 else "STANDARD RISK"
            },
            "governance": {
                "underwriting_decision": credit_decision,
                "fairness_demographic_proxy_flag": "Protected Young Cohort" if payload.Age < 30 else "Mature Group"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference pipeline execution failure: {str(e)}")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {"status": "HEALTHY", "model_loaded": model is not None}
