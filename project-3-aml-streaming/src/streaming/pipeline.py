import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from src.models.anomaly_detector import AMLAnomalyDetector
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.streaming.generator import transaction_stream_generator


def resolve_runtime_paths() -> tuple[Path, Path, Path, Path]:
    """
    Resolves data, DLQ, report, and model paths consistently across both
    local monorepo execution and container (/app) execution.
    """
    src_dir = Path(__file__).resolve().parents[1]
    service_root = src_dir.parent
    repo_root = (
        service_root.parent
        if (service_root.parent / "docker-compose.yml").exists()
        else service_root
    )

    env_data = os.getenv("AML_DATA_PATH", "").strip()
    data_candidates = ([Path(env_data)] if env_data else []) + [
        service_root / "data" / "SAML-D.csv",
        repo_root / "data" / "SAML-D.csv",
    ]
    data_path = next(
        (p for p in data_candidates if p.is_file()),
        service_root / "data" / "SAML-D.csv",
    )

    artifacts_base = (
        service_root / "artifacts"
        if os.access(service_root, os.W_OK)
        else repo_root / "artifacts"
    )
    dlq_dir = artifacts_base / "dlq"
    report_path = artifacts_base / "pipeline_execution_report.json"

    env_model = os.getenv("AML_MODEL_PATH", "").strip()
    model_candidates = ([Path(env_model)] if env_model else []) + [
        service_root / "aml_streaming_model",
        repo_root / "aml_streaming_model",
    ]
    model_path = next(
        (p for p in model_candidates if p.is_file()),
        repo_root / "aml_streaming_model",
    )

    return data_path, dlq_dir, report_path, model_path


def run_production_stream_pipeline(
    max_events: int | None = None, delay_seconds: float = 0.01
) -> dict:
    data_path, dlq_dir, report_path, model_path = resolve_runtime_paths()

    print("🚀 Initializing AML Streaming Pipeline Components...")
    ingestor = StreamingIngestionEngine(dlq_output_dir=dlq_dir)
    state_manager = StreamingStateManager()
    detector = AMLAnomalyDetector()

    try:
        detector.load(model_path)
        print(f"✅ Successfully loaded verified skops AML model from {model_path}.")
    except Exception as e:
        print(f"❌ FATAL: Failed to load AML model artifact at {model_path}: {e}")
        sys.exit(1)

    print(f"📡 Listening for stream events from {data_path}...")
    processed_count = 0
    anomaly_count = 0
    actual_laundering = 0
    laundering_intercepted = 0

    try:
        for raw_tx in transaction_stream_generator(
            data_path, delay_seconds=delay_seconds
        ):
            try:
                is_valid, clean_tx = ingestor.validate_and_route(raw_tx)
                if not is_valid or clean_tx is None:
                    continue

                enriched_tx = state_manager.update_and_enrich(clean_tx)
                prob, is_anomaly = detector.score_transaction(enriched_tx)
                processed_count += 1

                raw_label = clean_tx.get(
                    "Is_laundering", clean_tx.get("Is_Laundering", 0)
                )
                is_true_laundering = (
                    int(float(raw_label)) == 1 if raw_label not in (None, "") else False
                )
                if is_true_laundering:
                    actual_laundering += 1

                if is_anomaly:
                    anomaly_count += 1
                    if is_true_laundering:
                        laundering_intercepted += 1

                if processed_count % 500 == 0:
                    state_manager._evict_dormant_accounts()

                if max_events is not None and processed_count >= max_events:
                    break

            except Exception as event_err:
                ingestor._route_to_dlq(
                    raw_tx if isinstance(raw_tx, dict) else {"raw": str(raw_tx)},
                    [f"ERR_PROCESSING_FAILURE: {event_err}"],
                )
                continue

    except KeyboardInterrupt:
        print("\n🛑 Pipeline gracefully terminated by user.")
        sys.exit(0)
    except Exception as fatal_err:
        print(f"\n❌ Stream fatally interrupted: {fatal_err}")
        sys.exit(1)

    quality_report = ingestor.get_data_quality_report()
    recall_pct = (
        round((laundering_intercepted / actual_laundering) * 100.0, 2)
        if actual_laundering > 0
        else 0.0
    )

    execution_summary = {
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_governance": quality_report,
        "streaming_performance": {
            "total_processed": processed_count,
            "anomalies_flagged": anomaly_count,
            "actual_laundering_events": actual_laundering,
            "laundering_intercepted": laundering_intercepted,
            "recall_percentage": recall_pct,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(execution_summary, f, indent=2)

    print(
        f"\n📊 Session Summary: {processed_count} scored, {anomaly_count} anomalies detected."
    )
    print(
        f"🛡️ Data Quality: {quality_report['data_quality_pct']}% clean ({quality_report['dlq_events']} quarantined to DLQ)."
    )
    return execution_summary


if __name__ == "__main__":
    daemon_mode = os.getenv("AML_DAEMON_MODE", "0") == "1"
    while True:
        run_production_stream_pipeline()
        if not daemon_mode:
            break
        print(
            "🔄 Stream pass complete in daemon mode; sleeping 30s before next window..."
        )
        time.sleep(30)
