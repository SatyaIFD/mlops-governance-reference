"""
Production FastAPI Application - Project 1: Credit Card Fraud Detection
Exposes the registered fraud detection model via a high-performance REST API.
"""

import json
import math
import os
import secrets
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from compliance.dora_circuit_breaker import MLSystemCircuitBreaker

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*protected namespace.*"
)

EXPECTED_FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + [
    "scaled_amount",
    "scaled_time",
]

model = None
loaded_model_version: str | None = None
segment_policy: dict[str, Any] = {
    "low_value_threshold": 0.50,
    "high_value_threshold": 0.50,
    "scaled_amount_q80": 0.8415,
}
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


def _validate_loaded_model_schema(loaded_model) -> None:
    """Verifies that the loaded model's expected feature schema matches the canonical 30 features."""
    model_features = None
    if hasattr(loaded_model, "get_booster"):
        model_features = loaded_model.get_booster().feature_names
    elif hasattr(loaded_model, "feature_names_in_"):
        model_features = list(loaded_model.feature_names_in_)

    if not model_features or list(model_features) != EXPECTED_FEATURE_NAMES:
        raise ValueError(
            f"Loaded model feature schema mismatch. Expected {EXPECTED_FEATURE_NAMES}, got {model_features}"
        )


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
    global model, loaded_model_version, segment_policy
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

        target_ver = latest_version if (prod_versions or versions) else 1
        loaded, resolved_src = _load_xgboost_model_portable(
            client, model_name, model_uri, target_ver
        )
        _validate_loaded_model_schema(loaded)
        model = loaded
        loaded_model_version = f"v{target_ver}"
        seg_file = project_root / "artifacts" / "segment_thresholds.json"
        if seg_file.is_file():
            segment_policy = json.loads(seg_file.read_text(encoding="utf-8"))
        print(
            f"🟢 Production model loaded and verified via URI: {model_uri} (source={resolved_src})",
            flush=True,
        )
    except Exception as reg_err:
        print(
            f"⚠️ Model registry load or schema verification failed ({reg_err}).",
            flush=True,
        )
        model = None
        loaded_model_version = None

    yield
    print("🛑 Microservice context tearing down...", flush=True)


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
    features: list[FiniteFloat] = Field(
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
    return {
        "status": "HEALTHY",
        "model_loaded": True,
        "model_version": loaded_model_version or "unknown",
    }


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
        prob = float(model.predict_proba(df)[0][1])
        scaled_amt = float(df["scaled_amount"].iloc[0])
        q80 = float(segment_policy.get("scaled_amount_q80", 0.8415))
        cutoff = float(
            segment_policy.get("high_value_threshold", 0.50)
            if scaled_amt >= q80
            else segment_policy.get("low_value_threshold", 0.50)
        )
        pred = 1 if prob >= cutoff else 0
        return pred, prob

    breaker_response = circuit_breaker.execute_inference(_execute_model_logic)

    if breaker_response["status"] == "fallback":
        raise HTTPException(
            status_code=503,
            detail="Service unavailable due to internal circuit breaker fallback.",
        )

    prediction, proba = breaker_response["prediction"]
    return InferenceResponse(prediction=prediction, risk_score=proba)
