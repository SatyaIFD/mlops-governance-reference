from pathlib import Path
from src.models.anomaly_detector import AMLAnomalyDetector
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.streaming.generator import transaction_stream_generator


def run_production_stream_pipeline():
    # Resolve paths relative to the monorepo root
    src_dir = Path(__file__).resolve().parents[1]
    project_root = src_dir.parents[1]
    data_path = project_root / "data" / "SAML-D.csv"
    dlq_dir = project_root / "artifacts" / "dlq"

    print("🚀 Initializing AML Streaming Pipeline Components...")
    ingestor = StreamingIngestionEngine(dlq_output_dir=dlq_dir)
    state_manager = StreamingStateManager()
    detector = AMLAnomalyDetector()

    try:
        detector.load("aml_streaming_model")
        print("✅ Successfully loaded AML model from MLflow Registry.")
    except Exception as e:
        print(f"❌ Failed to load model from registry: {e}")
        print("⚠️ ALERT: The model has not been trained or registered yet!")
        return

    print(f"📡 Listening for live events from {data_path.name}...")
    processed_count = 0
    anomaly_count = 0

    try:
        # Stream events at 100 transactions per second
        for raw_tx in transaction_stream_generator(data_path, delay_seconds=0.01):
            # 1. Validation Gateway (Send corrupt events to DLQ)
            is_valid, clean_tx = ingestor.validate_and_route(raw_tx)
            if not is_valid:
                continue

            # 2. Stateful Feature Engineering
            enriched_tx = state_manager.update_and_enrich(clean_tx)

            # 3. Anomaly Detection Inference
            prob, is_anomaly = detector.score_transaction(enriched_tx)
            processed_count += 1

            if is_anomaly:
                anomaly_count += 1
                print(
                    f"🚨 ANOMALY [Score: {prob:.4f}] | {hash(clean_tx['Sender_account']) % 10000} -> {hash(clean_tx['Receiver_account']) % 10000} | ${clean_tx['Amount']:.2f}"
                )

            if processed_count % 500 == 0:
                print(
                    f"⚡ Processed {processed_count} events... (Anomalies: {anomaly_count})"
                )

    except KeyboardInterrupt:
        print("\n🛑 Pipeline gracefully terminated by user.")
    except Exception as e:
        print(f"\n❌ Stream interrupted: {e}")

    # Final Observability Report
    report = ingestor.get_data_quality_report()
    print(
        f"\n📊 Session Summary: {processed_count} scored, {anomaly_count} anomalies detected."
    )
    print(
        f"🛡️ Data Quality: {report['data_quality_pct']}% clean ({report['dlq_events']} quarantined to DLQ)."
    )


if __name__ == "__main__":
    run_production_stream_pipeline()
