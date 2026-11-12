"""
Inference API Service - Project 2: Loan Default Prediction
Exposes the fair, sample-reweighted XGBoost model with a single validation-calibrated,
cohort-blind decision threshold loaded once at startup and surfaced in /health and telemetry.
"""

import json
import math
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field
from src.monitoring.observability import ModelObservabilityLogger

from compliance.dora_circuit_breaker import MLSystemCircuitBreaker
from compliance.eu_ai_act_hitl_router import enforce_human_in_the_loop

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_TRAFFIC_SOURCES = {
    "live_production_traffic",
    "synthetic_representative_simulation",
    "synthetic_stress_simulation",
}

EXPECTED_LOAN_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "is_young",
    "Education_High School",
    "Education_Master's",
    "Education_PhD",
    "EmploymentType_Part-time",
    "EmploymentType_Self-employed",
    "EmploymentType_Unemployed",
    "HasMortgage_Yes",
    "HasDependents_Yes",
    "LoanPurpose_Business",
    "LoanPurpose_Education",
    "LoanPurpose_Home",
    "LoanPurpose_Other",
    "HasCoSigner_Yes",
]

model = None
feature_columns = None
decision_threshold: float | None = None
loaded_model_version: str | None = None
loaded_threshold_source: str | None = None
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
    if not api_key or not secrets.compare_digest(
        api_key.encode("utf-8"), expected_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _extract_feature_columns(loaded_model) -> list[str] | None:
    if hasattr(loaded_model, "get_booster"):
        names = loaded_model.get_booster().feature_names
        if names:
            return list(names)
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)
    return None


def _resolve_threshold_file_with_source(
    expected_run_id: str | None,
) -> tuple[Path, str]:
    """Resolves preprocessing/decision_thresholds.json from container mlruns, MLflow artifacts, or local artifacts/."""
    if expected_run_id:
        mlruns_root = Path("/app/mlruns")
        if mlruns_root.exists():
            for cand in mlruns_root.rglob("decision_thresholds.json"):
                if expected_run_id in str(cand):
                    return cand, "container_mlruns"
        try:
            dl_path = mlflow.artifacts.download_artifacts(
                run_id=expected_run_id,
                artifact_path="preprocessing/decision_thresholds.json",
            )
            if Path(dl_path).is_file():
                return Path(dl_path), "mlflow_artifacts"
        except Exception as dl_err:
            print(f"ℹ️ MLflow threshold artifact lookup fallback: {dl_err}", flush=True)
    local_path = PROJECT_ROOT / "artifacts" / "decision_thresholds.json"
    return local_path, "local_artifacts"


def _resolve_threshold_file(expected_run_id: str | None) -> Path:
    path, _ = _resolve_threshold_file_with_source(expected_run_id)
    return path


def _load_startup_decision_threshold(expected_run_id: str | None = None) -> float:
    """Loads the validation-calibrated decision threshold once at startup and strictly verifies its mlflow_run_id."""
    global loaded_threshold_source
    env_thresh = os.getenv("LOAN_DECISION_THRESHOLD", "").strip()
    if env_thresh:
        cutoff = float(env_thresh)
        if not (0.0 < cutoff < 1.0):
            raise ValueError(f"Invalid LOAN_DECISION_THRESHOLD: {cutoff}")
        loaded_threshold_source = "env_override"
        print(
            f"⚠️ WARNING: Using LOAN_DECISION_THRESHOLD={cutoff} environment override (bypassing MLflow artifact lookup).",
            flush=True,
        )
        return cutoff

    thresh_file, loaded_threshold_source = _resolve_threshold_file_with_source(
        expected_run_id
    )
    if not thresh_file.is_file():
        raise FileNotFoundError(
            f"Required threshold configuration missing at {thresh_file}"
        )
    data = json.loads(thresh_file.read_text(encoding="utf-8"))
    if "decision_threshold" not in data:
        raise ValueError(f"Missing 'decision_threshold' key in {thresh_file}: {data}")
    file_run_id = data.get("mlflow_run_id")
    if not file_run_id:
        raise ValueError(f"Missing 'mlflow_run_id' lineage key in {thresh_file}")
    if expected_run_id and str(file_run_id) != str(expected_run_id):
        raise ValueError(
            f"Threshold artifact run_id mismatch: {thresh_file} has {file_run_id}, "
            f"but loaded model version belongs to run {expected_run_id}"
        )
    cutoff = float(data["decision_threshold"])
    if not (0.0 < cutoff < 1.0):
        raise ValueError(f"Out-of-range decision_threshold in {thresh_file}: {cutoff}")
    return cutoff


def _load_expected_loan_features() -> list[str]:
    feat_file = PROJECT_ROOT / "artifacts" / "feature_columns.json"
    if feat_file.is_file():
        with open(feat_file, "r", encoding="utf-8") as f:
            loaded_cols = json.load(f)
        if isinstance(loaded_cols, list) and loaded_cols:
            return [str(c) for c in loaded_cols]
    return list(EXPECTED_LOAN_FEATURES)


def _validate_loaded_model_schema(loaded_model) -> list[str]:
    """Verifies that the loaded model's feature set matches the canonical loan features."""
    cols = _extract_feature_columns(loaded_model)
    expected = _load_expected_loan_features()
    if not cols or set(cols) != set(expected) or len(cols) != len(expected):
        raise ValueError(
            f"Loaded model feature schema mismatch. Expected {expected}, got {cols}"
        )
    return cols


def _load_xgboost_model_portable(
    client: MlflowClient, model_name: str, model_uri: str, version_num: int
) -> tuple[Any, str]:
    """Loads XGBoost model from MLflow URI, falling back strictly to token-matched container paths."""
    try:
        return mlflow.xgboost.load_model(model_uri), model_uri
    except Exception:
        mv = client.get_model_version(model_name, str(version_num))
        tokens = []
        for raw_loc in (
            getattr(mv, "storage_location", None),
            getattr(mv, "source", None),
        ):
            if not raw_loc:
                continue
            if "models:/m-" in raw_loc:
                tokens.append(raw_loc.split("models:/", 1)[1].strip("/"))
            elif "/models/m-" in raw_loc:
                tokens.append("m-" + raw_loc.split("/models/m-", 1)[1].split("/")[0])
        if getattr(mv, "run_id", None):
            tokens.append(mv.run_id)

        mlruns_root = Path("/app/mlruns")
        if mlruns_root.exists() and tokens:
            matched_dirs = []
            for p in mlruns_root.rglob("MLmodel"):
                meta_txt = p.read_text(encoding="utf-8", errors="ignore")
                if any(tok and (tok in str(p) or tok in meta_txt) for tok in tokens):
                    matched_dirs.append(p.parent)
            last_err = None
            for cand_dir in matched_dirs:
                try:
                    cand_model = mlflow.xgboost.load_model(str(cand_dir))
                    _validate_loaded_model_schema(cand_model)
                    return cand_model, str(cand_dir)
                except Exception as err:
                    last_err = err
            if last_err is not None:
                raise last_err
        raise FileNotFoundError(
            f"No token-matched MLmodel artifact found under {mlruns_root} for {model_name} v{version_num} (tokens={tokens})"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles microservice startup and teardown routines."""
    global model, feature_columns, decision_threshold, loaded_model_version
    print("🚀 Initializing inference context...", flush=True)

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
            selected_mv = max(prod_versions, key=lambda v: int(v.version))
            latest_version = int(selected_mv.version)
            expected_run_id = getattr(selected_mv, "run_id", None)
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/{latest_version}")
        elif versions:
            selected_mv = max(versions, key=lambda v: int(v.version))
            latest_version = int(selected_mv.version)
            expected_run_id = getattr(selected_mv, "run_id", None)
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/{latest_version}")
        else:
            latest_version = 1
            expected_run_id = None
            model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/1")

        target_ver = latest_version
        loaded, resolved_src = _load_xgboost_model_portable(
            client, model_name, model_uri, target_ver
        )
        feature_columns = _validate_loaded_model_schema(loaded)
        decision_threshold = _load_startup_decision_threshold(
            expected_run_id=expected_run_id
        )
        loaded_model_version = f"v{target_ver}"
        model = loaded
        print(
            f"🟢 Production model loaded and verified via URI: {model_uri} "
            f"(source={resolved_src}, version={loaded_model_version}, threshold={decision_threshold}, threshold_source={loaded_threshold_source})",
            flush=True,
        )
    except Exception as reg_err:
        print(f"⚠️ Registry URI or threshold load failed ({reg_err}).", flush=True)
        model = None
        feature_columns = None
        decision_threshold = None
        loaded_model_version = None

    yield
    print("🛑 Microservice context tearing down...", flush=True)


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

    Education: Literal["Bachelor's", "High School", "Master's", "PhD"] = Field(...)
    EmploymentType: Literal["Full-time", "Part-time", "Self-employed", "Unemployed"] = (
        Field(...)
    )
    MaritalStatus: Literal["Divorced", "Married", "Single"] = Field(...)
    HasMortgage: Literal["No", "Yes"] = Field(...)
    HasDependents: Literal["No", "Yes"] = Field(...)
    LoanPurpose: Literal["Auto", "Business", "Education", "Home", "Other"] = Field(...)
    HasCoSigner: Literal["No", "Yes"] = Field(...)

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
                "Education": "Bachelor's",
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
    # ECOA / Reg B (12 CFR 1002.5(d)(1)): drop MaritalStatus before model encoding
    if "MaritalStatus" in raw_data.columns:
        raw_data.drop(columns=["MaritalStatus"], inplace=True)

    categorical_cols = [
        "Education",
        "EmploymentType",
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
    if model is None or decision_threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model or decision threshold policy is uninitialized",
        )
    return {
        "status": "HEALTHY",
        "model_loaded": True,
        "model_version": loaded_model_version,
        "decision_threshold": decision_threshold,
        "threshold_source": loaded_threshold_source or "unknown",
    }


@app.post("/predict", tags=["Inference"], dependencies=[Depends(verify_api_key)])
def predict_loan_risk(payload: LoanApplicationPayload, request: Request):
    if model is None or decision_threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model or decision threshold policy is uninitialized.",
        )

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
    active_cutoff = float(decision_threshold)
    decision_type = "CREDIT_DENY" if prob_default >= active_cutoff else "CREDIT_APPROVE"
    routed_decision = enforce_human_in_the_loop(
        prob_default, decision_type=decision_type
    )

    model_recommendation = "DECLINE" if prob_default >= active_cutoff else "APPROVE"
    age_group = "UNDER_30" if payload.Age < 30 else "30_AND_OVER"

    governance_block = {
        **routed_decision,
        "model_recommendation": model_recommendation,
        "decision_threshold": active_cutoff,
        "model_version": loaded_model_version or "unknown",
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
        raw_src = request.headers.get(
            "X-Traffic-Source", "live_production_traffic"
        ).strip()
        validated_src = (
            raw_src
            if raw_src in ALLOWED_TRAFFIC_SOURCES
            else "unverified_client_traffic"
        )
        obs_logger.log_inference_transaction(
            request_payload=payload_dict,
            response_data=response_body,
            latency_ms=latency_ms,
            status_code=200,
            traffic_source=validated_src,
        )
    except Exception as log_err:
        print(f"⚠️ Telemetry logging warning: {log_err}", flush=True)

    return response_body
