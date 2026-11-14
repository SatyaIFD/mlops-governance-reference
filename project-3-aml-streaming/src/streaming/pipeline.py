import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.features.velocity import StreamingStateManager
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.models.anomaly_detector import AMLAnomalyDetector
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
        artifacts_base / "aml_streaming_model",
        service_root / "aml_streaming_model",
        repo_root / "aml_streaming_model",
    ]
    model_path = next(
        (p for p in model_candidates if p.is_file()),
        artifacts_base / "aml_streaming_model",
    )

    return data_path, dlq_dir, report_path, model_path


def _write_heartbeat(
    artifacts_dir: Path, processed_count: int, status: str = "HEALTHY"
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    hb_path = artifacts_dir / "heartbeat.json"
    with open(hb_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "processed_count": processed_count,
                "status": status,
            },
            f,
        )


def _write_report(
    report_path: Path,
    quality_report: dict,
    model_path: Path,
    detector: AMLAnomalyDetector,
    warmup_events: int,
    processed_count: int,
    anomaly_count: int,
    actual_laundering: int,
    laundering_intercepted: int,
    status: str = "COMPLETE",
    replayed_alerts_preserved: int = 0,
) -> dict:
    recall_pct = (
        round((laundering_intercepted / actual_laundering) * 100.0, 2)
        if actual_laundering > 0
        else 0.0
    )
    precision_pct = (
        round((laundering_intercepted / anomaly_count) * 100.0, 2)
        if anomaly_count > 0
        else 0.0
    )
    execution_summary = {
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "data_governance": quality_report,
        "model_metadata": {
            "artifact_path": str(model_path.name),
            "calibrated_threshold": detector.threshold,
            "model_sha256": detector.model_sha256,
            "mlflow_run_id": detector.mlflow_run_id,
            "features_used": detector._get_active_feature_cols(),
            "training_window_end_row": detector.training_window_end_row,
            "warmup_events_count": warmup_events,
        },
        "streaming_performance": {
            "total_processed": processed_count,
            "anomalies_flagged": anomaly_count,
            "replayed_alerts_preserved": replayed_alerts_preserved,
            "total_alerts_in_file": anomaly_count + replayed_alerts_preserved,
            "actual_laundering_events": actual_laundering,
            "laundering_intercepted": laundering_intercepted,
            "recall_percentage": recall_pct,
            "precision_percentage": precision_pct,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = report_path.with_name(f"{report_path.name}.{os.getpid()}.tmp")
    with open(tmp_report, "w", encoding="utf-8") as f:
        json.dump(execution_summary, f, indent=2)
    os.replace(tmp_report, report_path)
    return execution_summary


def run_production_stream_pipeline(
    max_events: int | None = None,
    delay_seconds: float | None = None,
    warmup_events: int | None = None,
) -> dict:
    data_path, dlq_dir, report_path, model_path = resolve_runtime_paths()
    artifacts_dir = report_path.parent
    alerts_path = artifacts_dir / "aml_alerts.jsonl"

    if max_events is None:
        max_events = int(os.getenv("AML_MAX_EVENTS", "100000"))
    if delay_seconds is None:
        delay_seconds = float(os.getenv("AML_STREAM_DELAY_SECONDS", "0.0"))
    if warmup_events is None:
        warmup_events = int(os.getenv("AML_WARMUP_EVENTS", "100000"))

    pseudonymize = os.getenv("AML_PSEUDONYMIZE_ACCOUNTS", "1") == "1"

    print("🚀 Initializing AML Streaming Pipeline Components...")
    dlq_dir.mkdir(parents=True, exist_ok=True)

    ingestor = StreamingIngestionEngine(
        dlq_output_dir=dlq_dir, pseudonymize_accounts=pseudonymize
    )
    state_manager = StreamingStateManager()
    detector = AMLAnomalyDetector()
    tmp_alerts_path = alerts_path.with_name(f"{alerts_path.name}.{os.getpid()}.tmp")

    warmup_count = 0
    processed_count = 0
    anomaly_count = 0
    actual_laundering = 0
    laundering_intercepted = 0

    def _safe_record_terminal_state(status_label: str) -> None:
        try:
            if tmp_alerts_path.exists():
                tmp_alerts_path.unlink()
        except Exception as cleanup_err:
            print(f"⚠️ Failed to clean up temporary alerts file: {cleanup_err}")
        try:
            _write_heartbeat(artifacts_dir, processed_count, status=status_label)
            _write_report(
                report_path,
                ingestor.get_data_quality_report(),
                model_path,
                detector,
                warmup_count,
                processed_count,
                anomaly_count,
                actual_laundering,
                laundering_intercepted,
                status=status_label,
            )
        except Exception as rep_err:
            print(f"⚠️ Failed to write {status_label} report state: {rep_err}")

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt("SIGTERM received")

    prev_sigterm = None
    try:
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except ValueError:
        prev_sigterm = None

    try:
        detector.load(model_path)
        if warmup_events < detector.training_window_end_row:
            raise ValueError(
                f"Train/serve window overlap detected: warmup_events ({warmup_events}) < "
                f"training_window_end_row ({detector.training_window_end_row})"
            )
        print(
            f"✅ Loaded verified skops AML model from {model_path} (threshold={detector.threshold})."
        )
    except Exception as e:
        _safe_record_terminal_state("FAILED_MODEL_LOAD")
        print(f"❌ FATAL: Failed to load AML model artifact at {model_path}: {e}")
        sys.exit(1)

    _write_heartbeat(artifacts_dir, 0, status="WARMING_UP")
    if tmp_alerts_path.exists():
        tmp_alerts_path.unlink()

    print(
        f"📡 Streaming from {data_path} (warmup={warmup_events}, out-of-sample window={max_events})..."
    )
    pending_batch: list[dict] = []
    batch_size = 5000

    def _record_scored_item(
        tx_item: dict, prob: float, is_anomaly: int, alert_file
    ) -> None:
        nonlocal \
            processed_count, \
            anomaly_count, \
            actual_laundering, \
            laundering_intercepted
        raw_label = tx_item.get("Is_laundering", tx_item.get("Is_Laundering", 0))
        is_true_laundering = (
            int(float(raw_label)) == 1 if raw_label not in (None, "") else False
        )
        if is_anomaly:
            alert_entry = {
                "event_id": tx_item.get("event_id"),
                "timestamp": str(tx_item.get("Timestamp")),
                "sender_account": tx_item.get("Sender_account"),
                "receiver_account": tx_item.get("Receiver_account"),
                "amount": tx_item.get("Amount"),
                "anomaly_probability": prob,
                "threshold": detector.threshold,
            }
            alert_file.write(json.dumps(alert_entry) + "\n")

        processed_count += 1
        if is_true_laundering:
            actual_laundering += 1
        if is_anomaly:
            anomaly_count += 1
            if is_true_laundering:
                laundering_intercepted += 1

    def _flush_batch(alert_file) -> None:
        if not pending_batch:
            return
        batch_items = list(pending_batch)
        pending_batch.clear()
        scored_pairs: list[tuple[dict, tuple[float, int]]] = []
        try:
            batch_scores = detector.score_batch(batch_items)
            scored_pairs = list(zip(batch_items, batch_scores))
        except Exception:
            for tx_item in batch_items:
                try:
                    res = detector.score_transaction(tx_item)
                    scored_pairs.append((tx_item, res))
                except Exception as item_err:
                    ingestor.record_processing_failure(
                        tx_item, [f"ERR_SCORING_FAILURE: {item_err}"]
                    )
        for tx_item, (prob, is_anomaly) in scored_pairs:
            try:
                _record_scored_item(tx_item, prob, is_anomaly, alert_file)
            except Exception as rec_err:
                ingestor.record_processing_failure(
                    tx_item, [f"ERR_RECORDING_FAILURE: {rec_err}"]
                )

    preserved_replay_alerts: list[str] = []
    seen_replay_ids: set[str] = set()
    if alerts_path.is_file():
        with open(alerts_path, "r", encoding="utf-8") as existing_af:
            for line in existing_af:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("source") == "dlq_replay":
                        ev_id = str(rec.get("event_id") or "")
                        if not ev_id or ev_id not in seen_replay_ids:
                            preserved_replay_alerts.append(json.dumps(rec))
                            if ev_id:
                                seen_replay_ids.add(ev_id)
                except json.JSONDecodeError:
                    continue

    try:
        with open(tmp_alerts_path, "w", encoding="utf-8") as alert_file:
            for replay_line in preserved_replay_alerts:
                alert_file.write(replay_line + "\n")
            for raw_tx in transaction_stream_generator(
                data_path, delay_seconds=delay_seconds
            ):
                is_valid, clean_tx = ingestor.validate_and_route(raw_tx)
                if not is_valid or clean_tx is None:
                    continue

                try:
                    enriched_tx = state_manager.update_and_enrich(clean_tx)
                except Exception as enrich_err:
                    ingestor.record_processing_failure(
                        clean_tx, [f"ERR_ENRICHMENT_FAILURE: {enrich_err}"]
                    )
                    continue

                if warmup_count < warmup_events:
                    warmup_count += 1
                    if warmup_count % 5000 == 0:
                        _write_heartbeat(artifacts_dir, 0, status="WARMING_UP")
                    if warmup_count % 25000 == 0:
                        state_manager._evict_dormant_accounts()
                        print(
                            f"   ⏳ Warmup progress: {warmup_count:,}/{warmup_events:,} events..."
                        )
                    continue

                pending_batch.append(enriched_tx)
                if len(pending_batch) >= batch_size:
                    _flush_batch(alert_file)
                    _write_heartbeat(artifacts_dir, processed_count, status="HEALTHY")
                    if processed_count % 25000 == 0:
                        state_manager._evict_dormant_accounts()
                        _write_report(
                            report_path,
                            ingestor.get_data_quality_report(),
                            model_path,
                            detector,
                            warmup_count,
                            processed_count,
                            anomaly_count,
                            actual_laundering,
                            laundering_intercepted,
                            status="IN_PROGRESS",
                        )
                        print(
                            f"   ⚡ Scored progress: {processed_count:,}/{max_events:,} events..."
                        )

                if (
                    max_events > 0
                    and (processed_count + len(pending_batch)) >= max_events
                ):
                    _flush_batch(alert_file)
                    break

            _flush_batch(alert_file)

        os.replace(tmp_alerts_path, alerts_path)

    except KeyboardInterrupt:
        _safe_record_terminal_state("INTERRUPTED")
        print("\n🛑 Pipeline gracefully terminated by user.")
        sys.exit(130)
    except Exception as fatal_err:
        _safe_record_terminal_state("FAILED")
        print(f"\n❌ Stream fatally interrupted: {fatal_err}")
        sys.exit(1)
    finally:
        if prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, prev_sigterm)
            except ValueError:
                pass

    if processed_count == 0:
        _safe_record_terminal_state("FAILED_ZERO_SCORED")
        print("❌ FATAL: Stream finished with 0 out-of-sample events scored.")
        sys.exit(1)

    _write_heartbeat(artifacts_dir, processed_count, status="HEALTHY")
    execution_summary = _write_report(
        report_path,
        ingestor.get_data_quality_report(),
        model_path,
        detector,
        warmup_count,
        processed_count,
        anomaly_count,
        actual_laundering,
        laundering_intercepted,
        status="COMPLETE",
        replayed_alerts_preserved=len(preserved_replay_alerts),
    )

    print(
        f"\n📊 Out-of-Sample Summary: {processed_count} scored (after {warmup_count} warmup), "
        f"{anomaly_count} anomalies flagged, {laundering_intercepted}/{actual_laundering} laundering intercepted "
        f"(recall={execution_summary['streaming_performance']['recall_percentage']}%, "
        f"precision={execution_summary['streaming_performance']['precision_percentage']}%)."
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
