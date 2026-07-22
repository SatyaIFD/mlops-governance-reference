import json
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.models.anomaly_detector import AMLAnomalyDetector
from src.streaming.generator import transaction_stream_generator

def run_production_stream_pipeline():
    print("=== INITIALIZING UNIFIED AML STREAMING OBSERVABILITY PIPELINE ===")
    
    # 1. Setup Data, Paths, and Governance Artifact Boundaries
    file_path = Path(__file__).resolve()
    project_root = file_path.parents[2]
    repo_root = file_path.parents[3]
    
    # Check all possible dataset locations dynamically
    possible_csv_paths = [
        repo_root / "data" / "SAML-D.csv",
        project_root / "data" / "SAML-D.csv",
        repo_root / "project-3-aml-streaming" / "data" / "SAML-D.csv"
    ]
    
    csv_path = next((p for p in possible_csv_paths if p.exists()), possible_csv_paths[0])
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize core modules
    state_manager = StreamingStateManager()
    ingestor = StreamingIngestionEngine(dlq_output_dir=artifacts_dir)
    detector = AMLAnomalyDetector()
    
    # 2. Spin up live stream channel
    print("\n[1/4] Connecting to live transaction stream network...")
    stream = transaction_stream_generator(csv_path, delay_seconds=0.0)
    
    # 3. PHASE A: Pure State Cache Warmup
    print("[2/4] Phase A: Warming up state manager lookback cache (50,000 events)...")
    for _ in range(50000):
        tx_raw = next(stream)
        is_valid, tx = ingestor.validate_and_route(tx_raw)
        if is_valid:
            state_manager.update_and_enrich(tx)
            
    # 4. PHASE B: High-Fidelity Training Data Collection
    print("[3/4] Phase B: Collecting 50,000 high-fidelity mature events for training...")
    warmup_records = []
    for _ in range(50000):
        tx_raw = next(stream)
        is_valid, tx = ingestor.validate_and_route(tx_raw)
        if is_valid:
            # Ensure target label is explicitly cast to integer
            tx['Is_laundering'] = int(float(tx.get('Is_laundering', 0)))
            enriched_tx = state_manager.update_and_enrich(tx)
            warmup_records.append(enriched_tx)
            
    warmup_features_df = pd.DataFrame(warmup_records)
    warmup_features_df['Is_laundering'] = warmup_features_df['Is_laundering'].astype(int)
    
    laundering_in_train = int(warmup_features_df['Is_laundering'].sum())
    print(f" -> Laundering events present in high-fidelity training set: {laundering_in_train}")
    
    # Train and persist model artifact
    detector.train(warmup_features_df)
    model_artifact_path = artifacts_dir / "aml_model.joblib"
    detector.save(model_artifact_path)
    
    # 5. Commencing live scoring on the REMAINING stream
    print("\n[4/4] Commencing real-time transaction validation and inference...")
    
    anomaly_count = 0
    laundering_caught = 0
    total_laundering_in_stream = 0
    records_to_score = 100000
    
    for _ in range(records_to_score):
        tx_raw = next(stream)
        
        is_valid, tx = ingestor.validate_and_route(tx_raw)
        if not is_valid:
            continue
            
        is_laundering_label = int(float(tx.get('Is_laundering', 0)))
        tx['Is_laundering'] = is_laundering_label
        
        if is_laundering_label == 1:
            total_laundering_in_stream += 1
            
        enriched_tx = state_manager.update_and_enrich(tx)
        score, is_anomaly = detector.score_transaction(enriched_tx)
        
        if is_anomaly == 1:
            anomaly_count += 1
            if is_laundering_label == 1:
                laundering_caught += 1
            
            print(f"🚨 AML ALERT | Account: {tx['Sender_account']} ──> {tx['Receiver_account']} | Amt: ${float(tx['Amount']):.2f} "
                  f"| Pass-Through: {enriched_tx['pass_through_ratio_1h']:.4f} "
                  f"| Rx Inflow Count: {enriched_tx['receiver_inflow_count_1h']} "
                  f"| Score: {score:.4f} | [Actual Laundering: {bool(is_laundering_label)}]")

    # 6. Session Metrics & Lineage Report
    recall_pct = (laundering_caught / total_laundering_in_stream) * 100 if total_laundering_in_stream > 0 else 0.0
    dq_report = ingestor.get_data_quality_report()

    print("\n=== REAL-TIME STREAMING INFERENCE SESSION SUMMARY ===")
    print(f"Total Stream Transactions Processed: {records_to_score}")
    print(f"Data Quality Governance Status: {dq_report}")
    print(f"Total Anomalies Flagged by Model: {anomaly_count}")
    print(f"Actual Laundering Events in Stream: {total_laundering_in_stream}")
    print(f"Laundering Events Successfully Intercepted: {laundering_caught}")
    print(f"Real-Time Streaming Recall: {recall_pct:.2f}%")

    # Generate governance metadata report
    report_path = artifacts_dir / "pipeline_execution_report.json"
    metadata = {
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_governance": dq_report,
        "model_metadata": {
            "artifact_path": str(model_artifact_path.relative_to(project_root)),
            "features_used": detector.feature_cols,
            "warmup_events_count": 50000,
            "training_events_count": len(warmup_features_df),
            "laundering_events_in_training": laundering_in_train
        },
        "streaming_performance": {
            "total_processed": records_to_score,
            "anomalies_flagged": anomaly_count,
            "actual_laundering_events": total_laundering_in_stream,
            "laundering_intercepted": laundering_caught,
            "recall_percentage": round(recall_pct, 2)
        }
    }
    
    with open(report_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✅ Governance metadata report generated at: {report_path}")

if __name__ == '__main__':
    run_production_stream_pipeline()