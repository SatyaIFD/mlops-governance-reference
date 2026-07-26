import json
import logging
from pathlib import Path
from src.ingestion.stream_ingestor import StreamingIngestionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def replay_dlq_events(dlq_dir: Path, target_ingestor: StreamingIngestionEngine):
    """Parses quarantined JSON files in dlq_dir and re-evaluates them through the ingestion engine."""
    dlq_files = list(dlq_dir.glob("dlq_event_*.json"))
    if not dlq_files:
        logging.info(f"No quarantined DLQ events found in {dlq_dir}")
        return

    logging.info(f"Found {len(dlq_files)} quarantined events to process in DLQ...")
    replayed_clean = 0
    replayed_failed = 0

    for dlq_file in dlq_files:
        try:
            with open(dlq_file, "r") as f:
                payload = json.load(f)

            raw_tx = payload.get("raw_payload", {})
            is_valid, _ = target_ingestor.validate_and_route(raw_tx)

            if is_valid:
                replayed_clean += 1
                dlq_file.unlink()  # Remove fixed file from DLQ upon successful replay
            else:
                replayed_failed += 1
        except Exception as e:
            logging.error(f"Error processing {dlq_file.name}: {e}")

    logging.info(f"DLQ Replay Complete | Recovered: {replayed_clean} | Still Invalid: {replayed_failed}")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = root_dir / "artifacts"
    ingestor = StreamingIngestionEngine(dlq_output_dir=artifacts_dir)
    replay_dlq_events(artifacts_dir, ingestor)