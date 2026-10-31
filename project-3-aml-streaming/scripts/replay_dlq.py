import json
import logging
from pathlib import Path
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.models.anomaly_detector import AMLAnomalyDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def replay_dlq_events(
    dlq_dir: Path,
    target_ingestor: StreamingIngestionEngine,
    state_manager: StreamingStateManager | None = None,
    detector: AMLAnomalyDetector | None = None,
) -> dict:
    """
    Replays quarantined events from dead_letter_queue.jsonl (and any legacy dlq_event_*.json files).
    Validates, enriches through StreamingStateManager (with event_id deduplication), optionally scores
    through AMLAnomalyDetector, and retains only unrecovered events in the DLQ.
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

    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Clear the active JSONL file before re-evaluating so failed items are written cleanly once
        jsonl_file.unlink()

        for line in lines:
            try:
                record = json.loads(line)
                raw_tx = record.get("raw_payload", {})
                is_valid, clean_tx = target_ingestor.validate_and_route(raw_tx)
                if is_valid and clean_tx is not None:
                    enriched = state_manager.update_and_enrich(clean_tx)
                    if detector is not None:
                        _, is_anom = detector.score_transaction(enriched)
                        scored_anomalies += is_anom
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
                if detector is not None:
                    _, is_anom = detector.score_transaction(enriched)
                    scored_anomalies += is_anom
                replayed_clean += 1
                dlq_file.unlink()
            else:
                replayed_failed += 1
        except Exception as e:
            logging.error(f"Error processing {dlq_file.name}: {e}")
            replayed_failed += 1

    summary = {
        "replayed_clean": replayed_clean,
        "replayed_failed": replayed_failed,
        "scored_anomalies": scored_anomalies,
    }
    logging.info(
        f"DLQ Replay Complete | Recovered: {replayed_clean} | Still Invalid: {replayed_failed}"
    )
    return summary


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    dlq_dir = root_dir / "artifacts" / "dlq"
    ingestor = StreamingIngestionEngine(dlq_output_dir=dlq_dir)
    replay_dlq_events(dlq_dir, ingestor)
