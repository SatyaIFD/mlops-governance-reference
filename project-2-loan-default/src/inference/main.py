"""
Inference API Service - Project 2: Loan Default Prediction
Exposes the fair, sample-reweighted XGBoost model as a production-grade FastAPI endpoint.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from compliance.eu_ai_act_hitl_router import enforce_human_in_the_loop
from compliance.dora_circuit_breaker import MLSystemCircuitBreaker

model = None
feature_columns = None
circuit_breaker = MLSystemCircuitBreaker(failure_threshold=3, recovery_timeout_s=30.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles microservice startup and teardown routines."""
    global model, feature_columns
    print("🚀 Initializing inference context...")

    project_root = Path(__file__).resolve().parents[2]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        db_path = project_root.parent / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient()
        model_name = "loan_default_production_model"
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

        model = mlflow.xgboost.load_model(model_uri)
        print(f"🟢 Production model loaded successfully via URI: {model_uri}")
    except Exception as reg_err:
        print(
            f"⚠️ Registry URI load failed ({reg_err}). Attempting local artifact discovery..."
        )
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
    lifespan=lifespan,
    root_path="/api/v1/loan",
)


class LoanApplicationPayload(BaseModel):
    Age: int = Field(..., ge=18, le=120)
    Income: int = Field(..., ge=0)
    CreditScore: int = Field(..., ge=300, le=850)
    LoanAmount: int = Field(..., ge=0)
    DTIRatio: float = Field(..., ge=0.0, le=1.0)
    MonthsEmployed: int = Field(..., ge=0)
    InterestRate: float = Field(..., ge=0.0)
    NumCreditLines: int = Field(..., ge=0)
    LoanTerm: int = Field(..., ge=1)
    Education: str = Field(...)
    EmploymentType: str = Field(...)
    MaritalStatus: str = Field(...)
    HasMortgage: str = Field(...)
    HasDependents: str = Field(...)
    LoanPurpose: str = Field(...)
    HasCoSigner: str = Field(...)

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
                "HasCoSigner": "Yes",
            }
        }
    )


@app.get("/health", tags=["Monitoring"])
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is uninitialized")
    return {"status": "HEALTHY", "model_loaded": True}


# Removed 'async' to run CPU-bound pandas operations in a separate thread pool
@app.post("/predict", tags=["Inference"])
def predict_loan_risk(payload: LoanApplicationPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is uninitialized.")

    def _execute_model_logic():
        raw_data = pd.DataFrame([payload.model_dump()])
        raw_data["is_young"] = (raw_data["Age"] < 30).astype(int)

        categorical_cols = [
            "Education",
            "EmploymentType",
            "MaritalStatus",
            "HasMortgage",
            "HasDependents",
            "LoanPurpose",
            "HasCoSigner",
        ]

        transformed_df = pd.get_dummies(
            raw_data, columns=categorical_cols, drop_first=True
        )

        if feature_columns:
            for col in feature_columns:
                if col not in transformed_df.columns:
                    transformed_df[col] = 0
            final_payload = transformed_df[feature_columns]
        else:
            final_payload = transformed_df

        bool_cols = final_payload.select_dtypes(include=["bool"]).columns
        if not bool_cols.empty:
            final_payload[bool_cols] = final_payload[bool_cols].astype("int8")

        prob_default = float(model.predict_proba(final_payload)[0][1])
        return prob_default

    breaker_response = circuit_breaker.execute_inference(_execute_model_logic)

    if breaker_response["status"] == "fallback":
        return {
            "risk_metrics": {"default_probability": 0.0},
            "governance": {"action": "CIRCUIT_OPEN_FALLBACK"},
        }

    prob_default = breaker_response["prediction"]

    # EU AI Act HITL Routing
    decision_type = "CREDIT_DENY" if prob_default >= 0.50 else "CREDIT_APPROVE"
    routed_decision = enforce_human_in_the_loop(
        prob_default, decision_type=decision_type
    )

    return {
        "risk_metrics": {"default_probability": round(prob_default, 4)},
        "governance": routed_decision,
    }
