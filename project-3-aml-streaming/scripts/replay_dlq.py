import json
import logging
import os
import shutil
from pathlib import Path

from src.features.velocity import StreamingStateManager
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.models.anomaly_detector import AMLAnomalyDetector
from src.streaming.generator import transaction_stream_generator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def warm_up_state_from_stream(
    csv_path: Path, warmup_rows: int = 5000, pseudonymize: bool = True
) -> StreamingStateManager:
    """Warms up StreamingStateManager from the stream prefix so replayed DLQ events have account history."""
    state = StreamingStateManager()
    if not csv_path.is_file() or warmup_rows <= 0:
        return state
    tmp_dlq = csv_path.parent / ".warmup_dlq"
    ingestor = StreamingIngestionEngine(
        dlq_output_dir=tmp_dlq, pseudonymize_accounts=pseudonymize
    )
    try:
        for idx, raw_tx in enumerate(
            transaction_stream_generator(csv_path, delay_seconds=0.0)
        ):
            if idx >= warmup_rows:
                break
            ok, clean_tx = ingestor.validate_and_route(raw_tx)
            if ok and clean_tx is not None:
                state.update_and_enrich(clean_tx)
    finally:
        shutil.rmtree(tmp_dlq, ignore_errors=True)
    return state


def _append_and_fsync_alerts(
    alerts_output_path: Path, flagged_alerts: list[dict]
) -> int:
    """Appends deduplicated alerts to alerts_output_path and flushes/fsyncs before DLQ state is committed."""
    if not flagged_alerts:
        return 0
    alerts_output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if alerts_output_path.is_file():
        with open(alerts_output_path, "r", encoding="utf-8") as ef:
            for line in ef:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("event_id"):
                            existing_ids.add(str(rec["event_id"]))
                    except json.JSONDecodeError:
                        continue
    written = 0
    with open(alerts_output_path, "a", encoding="utf-8") as af:
        for alert in flagged_alerts:
            ev_id = str(alert.get("event_id") or "")
            if ev_id and ev_id in existing_ids:
                continue
            af.write(json.dumps(alert) + "\n")
            if ev_id:
                existing_ids.add(ev_id)
            written += 1
        af.flush()
        os.fsync(af.fileno())
    return written


def replay_dlq_events(
    dlq_dir: Path,
    target_ingestor: StreamingIngestionEngine,
    state_manager: StreamingStateManager | None = None,
    detector: AMLAnomalyDetector | None = None,
    alerts_output_path: Path | None = None,
) -> dict:
    """Replays quarantined events from dead_letter_queue.jsonl, fsyncing alerts BEFORE committing DLQ swap."""
    state_manager = state_manager or StreamingStateManager()
    jsonl_file = dlq_dir / "dead_letter_queue.jsonl"
    staging_dir = dlq_dir / ".replay_staging"
    shutil.rmtree(staging_dir, ignore_errors=True)
    legacy_files = sorted(dlq_dir.glob("dlq_event_*.json"))

    if not jsonl_file.exists() and not legacy_files:
        logging.info(f"No quarantined DLQ events found in {dlq_dir}")
        return {"replayed_clean": 0, "replayed_failed": 0, "scored_anomalies": 0}

    replayed_clean = 0
    replayed_failed = 0
    scored_anomalies = 0
    unrecovered_records: list[str] = []
    flagged_alerts: list[dict] = []
    recovered_legacy_files: list[Path] = []

    def _score_and_record(enriched_tx: dict) -> None:
        nonlocal scored_anomalies
        if detector is None:
            return
        prob, is_anom = detector.score_transaction(enriched_tx)
        if is_anom:
            scored_anomalies += 1
            flagged_alerts.append(
                {
                    "event_id": enriched_tx.get("event_id"),
                    "timestamp": str(enriched_tx.get("Timestamp")),
                    "sender_account": enriched_tx.get("Sender_account"),
                    "receiver_account": enriched_tx.get("Receiver_account"),
                    "amount": enriched_tx.get("Amount"),
                    "anomaly_probability": prob,
                    "threshold": detector.threshold,
                    "source": "dlq_replay",
                }
            )

    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        staging_dir.mkdir(parents=True, exist_ok=True)
        orig_dlq_log = target_ingestor.dlq_log_file
        target_ingestor.dlq_log_file = staging_dir / "staged_dlq.jsonl"

        try:
            for line in lines:
                try:
                    record = json.loads(line)
                    raw_tx = record.get("raw_payload", {})
                    is_valid, clean_tx = target_ingestor.validate_and_route(raw_tx)
                    if is_valid and clean_tx is not None:
                        enriched = state_manager.update_and_enrich(clean_tx)
                        _score_and_record(enriched)
                        replayed_clean += 1
                    else:
                        replayed_failed += 1
                except Exception as e:
                    logging.error(f"Error replaying JSONL record: {e}")
                    replayed_failed += 1
                    unrecovered_records.append(line)

            if target_ingestor.dlq_log_file.exists():
                with open(target_ingestor.dlq_log_file, "r", encoding="utf-8") as sf:
                    unrecovered_records.extend([ln.strip() for ln in sf if ln.strip()])
        finally:
            target_ingestor.dlq_log_file = orig_dlq_log
            shutil.rmtree(staging_dir, ignore_errors=True)

    for dlq_file in legacy_files:
        try:
            with open(dlq_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw_tx = payload.get("raw_payload", {})
            is_valid, clean_tx = target_ingestor.validate_and_route(raw_tx)
            if is_valid and clean_tx is not None:
                enriched = state_manager.update_and_enrich(clean_tx)
                _score_and_record(enriched)
                replayed_clean += 1
                recovered_legacy_files.append(dlq_file)
            else:
                replayed_failed += 1
        except Exception as e:
            logging.error(f"Error processing {dlq_file.name}: {e}")
            replayed_failed += 1

    # 1. Write and fsync alerts FIRST so a crash never drops an alert before DLQ removal
    if flagged_alerts and alerts_output_path is not None:
        _append_and_fsync_alerts(alerts_output_path, flagged_alerts)

    # 2. Commit DLQ state only after alerts are durably on disk
    if jsonl_file.exists():
        if unrecovered_records:
            tmp_dlq = jsonl_file.with_suffix(".jsonl.tmp")
            with open(tmp_dlq, "w", encoding="utf-8") as f:
                for line in unrecovered_records:
                    f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_dlq, jsonl_file)
        else:
            jsonl_file.unlink()

    for dlq_file in recovered_legacy_files:
        if dlq_file.exists():
            dlq_file.unlink()

    summary = {
        "replayed_clean": replayed_clean,
        "replayed_failed": replayed_failed,
        "scored_anomalies": scored_anomalies,
    }
    logging.info(
        f"DLQ Replay Complete | Recovered: {replayed_clean} | Still Invalid: {replayed_failed} | Anomalies Scored: {scored_anomalies}"
    )
    return summary


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = root_dir / "artifacts"
    dlq_dir = artifacts_dir / "dlq"
    model_path = artifacts_dir / "aml_streaming_model"
    alerts_path = artifacts_dir / "aml_alerts.jsonl"
    csv_path = root_dir / "data" / "SAML-D.csv"

    pseudonymize = os.getenv("AML_PSEUDONYMIZE_ACCOUNTS", "1") == "1"
    ingestor = StreamingIngestionEngine(
        dlq_output_dir=dlq_dir, pseudonymize_accounts=pseudonymize
    )
    warmed_state = warm_up_state_from_stream(
        csv_path, warmup_rows=5000, pseudonymize=pseudonymize
    )
    active_detector = None
    if model_path.is_file():
        active_detector = AMLAnomalyDetector()
        active_detector.load(model_path)

    replay_dlq_events(
        dlq_dir,
        ingestor,
        state_manager=warmed_state,
        detector=active_detector,
        alerts_output_path=alerts_path,
    )
