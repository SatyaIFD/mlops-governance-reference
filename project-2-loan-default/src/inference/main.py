"""
Inference API Service - Project 2: Loan Default Prediction
Exposes the fair, sample-reweighted XGBoost model as a production-grade FastAPI endpoint.
"""

import math
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from compliance.eu_ai_act_hitl_router import enforce_human_in_the_loop
from compliance.dora_circuit_breaker import MLSystemCircuitBreaker
from src.monitoring.observability import ModelObservabilityLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

model = None
feature_columns = None
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


def _extract_feature_columns(loaded_model) -> list[str] | None:
    if hasattr(loaded_model, "get_booster"):
        names = loaded_model.get_booster().feature_names
        if names:
            return list(names)
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles microservice startup and teardown routines."""
    global model, feature_columns
    print("🚀 Initializing inference context...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        db_path = PROJECT_ROOT.parent / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient()
        model_name = "loan_default_production_model"
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
        feature_columns = _extract_feature_columns(loaded)
        model = loaded
        print(f"🟢 Production model loaded successfully via URI: {model_uri}")
    except Exception as reg_err:
        print(f"⚠️ Registry URI load failed ({reg_err}).")
        model = None
        feature_columns = None

    yield
    print("🛑 Microservice context tearing down...")


app = FastAPI(
    title="Fair Loan Default Prediction Service",
    description="Responsible AI Risk Scoring Engine using a Sample-Reweighted XGBoost Model.",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api/v1/loan",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    sanitized_errors = _sanitize_non_finite_floats(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


class LoanApplicationPayload(BaseModel):
    Age: int = Field(..., ge=18, le=120)
    Income: int = Field(..., ge=0)
    CreditScore: int = Field(..., ge=300, le=850)
    LoanAmount: int = Field(..., ge=0)
    DTIRatio: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    MonthsEmployed: int = Field(..., ge=0)
    InterestRate: float = Field(..., ge=0.0, allow_inf_nan=False)
    NumCreditLines: int = Field(..., ge=0)
    LoanTerm: int = Field(..., ge=1)

    Education: Literal[
        "High School", "Bachelors", "Bachelor's", "Masters", "Master's", "PhD"
    ] = Field(...)
    EmploymentType: Literal["Full-time", "Part-time", "Self-employed", "Unemployed"] = (
        Field(...)
    )
    MaritalStatus: Literal["Single", "Married", "Divorced", "Widowed"] = Field(...)
    HasMortgage: Literal["Yes", "No"] = Field(...)
    HasDependents: Literal["Yes", "No"] = Field(...)
    LoanPurpose: Literal[
        "Auto",
        "Home",
        "Personal",
        "Debt Consolidation",
        "Business",
        "Education",
        "Other",
    ] = Field(...)
    HasCoSigner: Literal["Yes", "No"] = Field(...)

    model_config = ConfigDict(
        extra="forbid",
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
                "Education": "Bachelors",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Single",
                "HasMortgage": "No",
                "HasDependents": "No",
                "LoanPurpose": "Auto",
                "HasCoSigner": "Yes",
            }
        },
    )


def prepare_single_row_features(
    payload_dict: dict[str, Any], expected_cols: list[str] | None
) -> pd.DataFrame:
    """
    Encodes a single-row application payload for the registered XGBoost model.
    Uses drop_first=False before reindexing to expected_cols so that single-row inputs
    do not drop their sole category level, while reference categories naturally become 0.
    """
    raw_data = pd.DataFrame([payload_dict])
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
        raw_data, columns=categorical_cols, drop_first=False
    )

    if expected_cols:
        final_payload = transformed_df.reindex(columns=expected_cols, fill_value=0)
    else:
        final_payload = transformed_df

    bool_cols = final_payload.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        final_payload[bool_cols] = final_payload[bool_cols].astype("int8")

    return final_payload


@app.get("/health", tags=["Monitoring"])
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is uninitialized")
    return {"status": "HEALTHY", "model_loaded": True}


@app.post("/predict", tags=["Inference"], dependencies=[Depends(verify_api_key)])
def predict_loan_risk(payload: LoanApplicationPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is uninitialized.")

    start_time = time.perf_counter()
    payload_dict = payload.model_dump()

    def _execute_model_logic():
        active_cols = feature_columns or _extract_feature_columns(model)
        final_payload = prepare_single_row_features(payload_dict, active_cols)
        return float(model.predict_proba(final_payload)[0][1])

    breaker_response = circuit_breaker.execute_inference(_execute_model_logic)

    if breaker_response["status"] == "fallback":
        raise HTTPException(
            status_code=503,
            detail="Service unavailable due to internal circuit breaker fallback.",
        )

    prob_default = breaker_response["prediction"]
    decision_type = "CREDIT_DENY" if prob_default >= 0.50 else "CREDIT_APPROVE"
    routed_decision = enforce_human_in_the_loop(
        prob_default, decision_type=decision_type
    )

    model_recommendation = "DECLINE" if prob_default >= 0.50 else "APPROVE"
    age_group = "UNDER_30" if payload.Age < 30 else "30_AND_OVER"

    governance_block = {
        **routed_decision,
        "model_recommendation": model_recommendation,
        "review_status": "PENDING_HUMAN_REVIEW"
        if routed_decision.get("action") == "MANUAL_REVIEW_REQUIRED"
        else "NOT_REQUIRED",
        "final_decision": None,
        "age_group": age_group,
    }

    response_body = {
        "risk_metrics": {"default_probability": round(prob_default, 4)},
        "governance": governance_block,
    }

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    try:
        log_path = Path(
            os.getenv(
                "LOAN_TELEMETRY_LOG_PATH",
                str(PROJECT_ROOT / "artifacts" / "telemetry_logs.jsonl"),
            )
        )
        obs_logger = ModelObservabilityLogger(telemetry_log_path=log_path)
        obs_logger.log_inference_transaction(
            request_payload=payload_dict,
            response_data=response_body,
            latency_ms=latency_ms,
            status_code=200,
        )
    except Exception as log_err:
        print(f"⚠️ Telemetry logging warning: {log_err}")

    return response_body
