import json
import logging
from pathlib import Path

from src.features.velocity import StreamingStateManager
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.models.anomaly_detector import AMLAnomalyDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def replay_dlq_events(
    dlq_dir: Path,
    target_ingestor: StreamingIngestionEngine,
    state_manager: StreamingStateManager | None = None,
    detector: AMLAnomalyDetector | None = None,
    alerts_output_path: Path | None = None,
) -> dict:
    """
    Replays quarantined events from dead_letter_queue.jsonl (and any legacy dlq_event_*.json files).
    Validates, enriches through StreamingStateManager (with event_id deduplication), scores
    through AMLAnomalyDetector when provided, appends flagged alerts to alerts_output_path,
    and retains only unrecovered events in the DLQ.
    """
    state_manager = state_manager or StreamingStateManager()
    jsonl_file = dlq_dir / "dead_letter_queue.jsonl"
    legacy_files = sorted(dlq_dir.glob("dlq_event_*.json"))

    if not jsonl_file.exists() and not legacy_files:
        logging.info(f"No quarantined DLQ events found in {dlq_dir}")
        return {"replayed_clean": 0, "replayed_failed": 0, "scored_anomalies": 0}

    replayed_clean = 0
    replayed_failed = 0
    scored_anomalies = 0
    unrecovered_records = []
    flagged_alerts = []

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

        jsonl_file.unlink()

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

        if unrecovered_records:
            with open(jsonl_file, "a", encoding="utf-8") as f:
                for line in unrecovered_records:
                    f.write(line + "\n")

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
                dlq_file.unlink()
            else:
                replayed_failed += 1
        except Exception as e:
            logging.error(f"Error processing {dlq_file.name}: {e}")
            replayed_failed += 1

    if flagged_alerts and alerts_output_path is not None:
        alerts_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alerts_output_path, "a", encoding="utf-8") as af:
            for alert in flagged_alerts:
                af.write(json.dumps(alert) + "\n")

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

    ingestor = StreamingIngestionEngine(dlq_output_dir=dlq_dir)
    active_detector = None
    if model_path.is_file():
        active_detector = AMLAnomalyDetector()
        active_detector.load(model_path)

    replay_dlq_events(
        dlq_dir,
        ingestor,
        detector=active_detector,
        alerts_output_path=alerts_path,
    )
