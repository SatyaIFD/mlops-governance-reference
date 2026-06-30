"""
Inference API Service - Project 2: Loan Default Prediction
Exposes the fair, sample-reweighted XGBoost model as a production-grade FastAPI endpoint.
"""

import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow

app = FastAPI(
    title="Fair Loan Default Prediction Service",
    description="Responsible AI Risk Scoring Engine using a Sample-Reweighted XGBoost Model.",
    version="1.0.0"
)

# Global variables to hold our production model artifact and feature schema layout
model = None
feature_columns = None

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
    Education: str = Field(..., description="Highest education tier attained (e.g., High School, Bachelor's, Master's, PhD)")
    EmploymentType: str = Field(..., description="Employment nature (e.g., Full-time, Part-time, Self-employed, Unemployed)")
    MaritalStatus: str = Field(..., description="Marital status context")
    HasMortgage: str = Field(..., description="Mortgage ownership indicator ('Yes' or 'No')")
    HasDependents: str = Field(..., description="Dependents visibility indicator ('Yes' or 'No')")
    LoanPurpose: str = Field(..., description="Core purpose behind the loan request")
    HasCoSigner: str = Field(..., description="Co-signer presence indicator ('Yes' or 'No')")

    class Config:
        json_schema_extra = {
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

@app.on_event("startup")
def load_production_model():
    """Resolves and caches the latest active fair model from the local MLflow registry."""
    global model, feature_columns
    print("🚀 Initializing inference context...")
    
    # Locate the active MLflow tracking directory
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    
    # EXPLICIT ALIGNMENT: Point to the local SQLite DB asset 
    db_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    experiment = mlflow.get_experiment_by_name("loan_default_risk_governance")
    if not experiment:
        raise RuntimeError("❌ Cannot initialize API: 'loan_default_risk_governance' experiment does not exist.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"])
    if runs.empty:
        raise RuntimeError("❌ Cannot initialize API: No logged model runs found.")
        
    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/fair_loan_model"
    
    print(f"🌲 Loading fair model binary from run ID: {latest_run_id}")
    model = mlflow.xgboost.load_model(model_uri)
    
    # Capture exact feature mapping structure expected by the frozen XGBoost model
    feature_columns = model.get_booster().feature_names
    print("✅ Model successfully cached. Service ready for incoming payloads.")

@app.on_event("startup")
def verify_training_features():
    """Ensures feature_columns is initialized properly."""
    global feature_columns
    if model and not feature_columns:
        feature_columns = model.get_booster().feature_names

@app.post("/predict", tags=["Inference"])
async def predict_loan_risk(payload: LoanApplicationPayload):
    """Processes incoming data packages, maps fair schema transforms, and returns default visibility evaluations."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable or uninitialized.")
        
    try:
        # Convert Pydantic request body into a raw input DataFrame
        raw_data = pd.DataFrame([payload.dict()])
        
        # Apply the exact structural transformations our training layer executed
        raw_data['is_young'] = (raw_data['Age'] < 30).astype(int)
        
        # Drop categorical columns and expand using pandas get_dummies
        categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
        transformed_df = pd.get_dummies(raw_data, columns=categorical_cols, drop_first=True)
        
        # Dynamically align columns to ensure identical layout as training feature space
        for col in feature_columns:
            if col not in transformed_df.columns:
                transformed_df[col] = 0
                
        # Handle structural conversions (e.g. converting booleans to integers)
        bool_cols = transformed_df.select_dtypes(include=['bool']).columns
        if not bool_cols.empty:
            transformed_df[bool_cols] = transformed_df[bool_cols].astype('int8')
            
        # Ensure identical column alignment order
        final_payload = transformed_df[feature_columns]
        
        # Calculate real-time probabilities and risk scores
        prob_default = float(model.predict_proba(final_payload)[0][1])
        prediction_label = int(model.predict(final_payload)[0])
        
        # Map corporate banking responses
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
    """Simple API status check endpoint."""
    return {"status": "HEALTHY", "model_loaded": model is not None}