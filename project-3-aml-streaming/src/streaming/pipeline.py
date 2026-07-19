import pandas as pd
from pathlib import Path
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.models.anomaly_detector import AMLAnomalyDetector
from src.streaming.generator import transaction_stream_generator

def run_production_stream_pipeline():
    print("=== INITIALIZING UNIFIED AML STREAMING OBSERVABILITY PIPELINE ===")
    
    # 1. Setup Data, Paths, and Governance Artifact Boundaries
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "data" / "SAML-D.csv"
    artifacts_dir = root_dir / "artifacts"
    
    # Initialize core modules
    state_manager = StreamingStateManager()
    ingestor = StreamingIngestionEngine(dlq_output_dir=artifacts_dir)
    detector = AMLAnomalyDetector()
    
    # 2. Spin up the single live stream channel
    print("\n[1/4] Connecting to live transaction stream network...")
    stream = transaction_stream_generator(csv_path, delay_seconds=0.0)
    
    # 3. PHASE A: Pure State Cache Warmup
    print("[2/4] Phase A: Warming up state manager lookback cache (50,000 events)...")
    for _ in range(50000):
        tx_raw = next(stream)
        is_valid, tx = ingestor.validate_and_route(tx_raw)
        if is_valid:
            state_manager.update_and_enrich(tx)
            
    # 4. PHASE B: High-Fidelity Training Data Collection (Mature Cache Only)
    print("[3/4] Phase B: Collecting 50,000 high-fidelity mature events for training...")
    warmup_records = []
    for _ in range(50000):
        tx_raw = next(stream)
        is_valid, tx = ingestor.validate_and_route(tx_raw)
        if is_valid:
            enriched_tx = state_manager.update_and_enrich(tx)
            warmup_records.append(enriched_tx)
            
    warmup_features_df = pd.DataFrame(warmup_records)
    
    # Print out safety audit to ensure laundering events are caught in this slice
    laundering_in_train = warmup_features_df['Is_laundering'].sum()
    print(f" -> Laundering events present in high-fidelity training set: {laundering_in_train}")
    
    detector.train(warmup_features_df)
    
    # 5. Commencing live scoring immediately on the REMAINING active stream
    print("\n[4/4] Commencing real-time transaction validation and inference...")
    
    anomaly_count = 0
    laundering_caught = 0
    total_laundering_in_stream = 0
    records_to_score = 100000
    
    for _ in range(records_to_score):
        tx_raw = next(stream)
        
        # Ingestion Validation Gate
        is_valid, tx = ingestor.validate_and_route(tx_raw)
        if not is_valid:
            continue
            
        if tx['Is_laundering'] == 1:
            total_laundering_in_stream += 1
            
        # Real-time stateful feature calculations (State is completely warm and connected!)
        enriched_tx = state_manager.update_and_enrich(tx)
        
        # Query the supervised brain
        score, is_anomaly = detector.score_transaction(enriched_tx)
        
        if is_anomaly == 1:
            anomaly_count += 1
            if tx['Is_laundering'] == 1:
                laundering_caught += 1
            
            # Print alert logs for structural anomalies intercepted by our new relational features
            print(f"🚨 AML ALERT | Account: {tx['Sender_account']} ──> {tx['Receiver_account']} | Amt: ${tx['Amount']:.2f} "
                  f"| Pass-Through: {enriched_tx['pass_through_ratio_1h']:.4f} "
                  f"| Rx Inflow Count: {enriched_tx['receiver_inflow_count_1h']} "
                  f"| Score: {score:.4f} | [Actual Laundering: {bool(tx['Is_laundering'])}]")

    # 6. Output Session Metrics
    print("\n=== REAL-TIME STREAMING INFERENCE SESSION SUMMARY ===")
    print(f"Total Stream Transactions Processed: {records_to_score}")
    print(f"Data Quality Governance Status: {ingestor.get_data_quality_report()}")
    print(f"Total Anomalies Flagged by Model: {anomaly_count}")
    print(f"Actual Laundering Events in Stream: {total_laundering_in_stream}")
    print(f"Laundering Events Successfully Intercepted: {laundering_caught}")
    if total_laundering_in_stream > 0:
        print(f"Real-Time Streaming Recall: {(laundering_caught / total_laundering_in_stream) * 100:.2f}%")
    else:
        print("Real-Time Streaming Recall: N/A (No true laundering events occurred in this slice)")

if __name__ == '__main__':
    run_production_stream_pipeline()