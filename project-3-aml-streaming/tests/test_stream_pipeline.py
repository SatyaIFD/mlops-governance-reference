import pytest
import pandas as pd
from pathlib import Path
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.models.anomaly_detector import AMLAnomalyDetector

def test_streaming_ingestion_dlq_routing(tmp_path):
    """Asserts that corrupted payloads are quarantined into the DLQ without breaking execution."""
    ingestor = StreamingIngestionEngine(dlq_output_dir=tmp_path)
    
    # 1. Test clean wire payload
    clean_payload = {'Timestamp': '2026-07-19 12:00:00', 'Sender_account': 'A', 'Receiver_account': 'B', 'Amount': 500.0}
    is_valid, clean_res = ingestor.validate_and_route(clean_payload)
    assert is_valid is True
    assert clean_res['Amount'] == 500.0

    # 2. Test poison payload (Negative monetary volume)
    poison_payload = {'Timestamp': '2026-07-19 12:01:00', 'Sender_account': 'C', 'Receiver_account': 'D', 'Amount': -100.0}
    is_valid, poison_res = ingestor.validate_and_route(poison_payload)
    assert is_valid is False
    assert poison_res is None
    
    # Verify report updates metrics correctly
    report = ingestor.get_data_quality_report()
    assert report['dlq_events'] == 1
    assert report['clean_events'] == 1

def test_stateful_pass_through_calculation():
    """Verifies that rapid layering loops trigger high pass-through velocities."""
    state_manager = StreamingStateManager()
    
    # Account receives money
    inflow = {'Timestamp': '2026-07-19 10:00:00', 'Sender_account': 'EXTERNAL', 'Receiver_account': 'MULE_NODE', 'Amount': 10000.0}
    state_manager.update_and_enrich(inflow)
    
    # Account immediately disperses money within the 1-hour window
    outflow = {'Timestamp': '2026-07-19 10:15:00', 'Sender_account': 'MULE_NODE', 'Receiver_account': 'TARGET', 'Amount': 9900.0}
    enriched_outflow = state_manager.update_and_enrich(outflow)
    
    # Symmetrical flow should force the pass-through ratio near 1.0
    assert enriched_outflow['pass_through_ratio_1h'] > 0.90

def test_anomaly_detector_preprocessing_shapes():
    """Ensures preprocessing log-scaling returns consistent vector dimensions."""
    detector = AMLAnomalyDetector()
    
    mock_record = {
        'Amount': 1000.0, 'tx_count_1h': 2, 'tx_amount_sum_1h': 2000.0,
        'tx_count_24h': 5, 'beneficiary_diversity_24h': 0.8, 'pass_through_ratio_1h': 0.5,
        'receiver_inflow_count_1h': 1, 'receiver_inflow_sum_1h': 1000.0
    }
    
    processed_df = detector._preprocess(mock_record)
    
    # Confirm shape rules match Sklearn specifications
    assert processed_df.shape == (1, 8)
    assert processed_df['Amount'].iloc[0] < 1000.0 # Verify log compression fired