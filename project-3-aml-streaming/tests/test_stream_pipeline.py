import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.models.anomaly_detector import AMLAnomalyDetector


def test_stream_ingestor_valid_and_dlq(tmp_path):
    """Verifies that clean payloads pass validation while malformed records route to DLQ."""
    ingestor = StreamingIngestionEngine(dlq_output_dir=tmp_path)

    # Valid event
    valid_raw = {
        'Timestamp': '2026-07-26 12:00:00',
        'Sender_account': 'ACC_A',
        'Receiver_account': 'ACC_B',
        'Amount': '9500.00'
    }
    is_valid, clean_tx = ingestor.validate_and_route(valid_raw)
    assert is_valid is True
    assert isinstance(clean_tx['Timestamp'], datetime)
    assert clean_tx['Amount'] == 9500.00

    # Invalid event (negative amount)
    invalid_raw = {
        'Timestamp': '2026-07-26 12:00:00',
        'Sender_account': 'ACC_A',
        'Receiver_account': 'ACC_B',
        'Amount': '-500.00'
    }
    is_valid_inv, _ = ingestor.validate_and_route(invalid_raw)
    assert is_valid_inv is False
    assert ingestor.dlq_event_count == 1


def test_state_manager_feature_calculation_and_eviction():
    """Verifies 7-feature state calculations including structuring flags and TTL key eviction."""
    state_manager = StreamingStateManager()
    now = datetime.now()

    # Inbound transfer to ACC_MULE
    tx_in = {
        'Timestamp': now - timedelta(minutes=30),
        'Sender_account': 'ACC_SOURCE',
        'Receiver_account': 'ACC_MULE',
        'Amount': 10000.00
    }
    state_manager.update_and_enrich(tx_in)

    # Structuring transfer out from ACC_MULE ($9,000)
    tx_out = {
        'Timestamp': now,
        'Sender_account': 'ACC_MULE',
        'Receiver_account': 'ACC_DEST',
        'Amount': 9000.00
    }
    enriched = state_manager.update_and_enrich(tx_out)

    assert enriched['is_structuring'] == 1
    assert enriched['pass_through_ratio_1h'] > 0.80
    assert enriched['fan_out_count_24h'] == 1
    assert 'velocity_acceleration' in enriched

    # Test TTL Eviction: Advance time by 30 hours
    future_tx = {
        'Timestamp': now + timedelta(hours=30),
        'Sender_account': 'ACC_NEW_1',
        'Receiver_account': 'ACC_NEW_2',
        'Amount': 500.00
    }
    state_manager.update_and_enrich(future_tx)

    # ACC_SOURCE and ACC_MULE should have been evicted due to inactivity > 24h
    assert 'ACC_SOURCE' not in state_manager.state


def test_anomaly_detector_scoring_and_persistence(tmp_path):
    """Verifies scoring and serialization across all 7 feature columns."""
    detector = AMLAnomalyDetector()

    # Fit model on dummy training set prior to scoring
    dummy_data = {
        'Amount': [100.0, 9000.0, 50.0, 9500.0],
        'pass_through_ratio_1h': [0.0, 0.95, 0.1, 0.98],
        'is_structuring': [0, 1, 0, 1],
        'velocity_acceleration': [1.0, 4.0, 0.5, 5.0],
        'fan_out_count_24h': [1, 5, 1, 6],
        'receiver_inflow_count_1h': [1, 3, 1, 4],
        'receiver_inflow_amount_1h': [100.0, 27000.0, 50.0, 30000.0],
        'Is_laundering': [0, 1, 0, 1]
    }
    train_df = pd.DataFrame(dummy_data)
    detector.train(train_df)

    sample_enriched = {
        'Amount': 9000.00,
        'pass_through_ratio_1h': 0.95,
        'is_structuring': 1,
        'velocity_acceleration': 4.0,
        'fan_out_count_24h': 5,
        'receiver_inflow_count_1h': 3,
        'receiver_inflow_amount_1h': 27000.00
    }

    # Verify single scoring
    prob, is_anomaly = detector.score_transaction(sample_enriched)
    assert 0.0 <= prob <= 1.0

    # Test model save and reload
    model_path = tmp_path / "test_model.joblib"
    detector.save(model_path)
    assert model_path.exists()

    reloaded_detector = AMLAnomalyDetector()
    reloaded_detector.load(model_path)
    prob_reloaded, _ = reloaded_detector.score_transaction(sample_enriched)
    assert prob == prob_reloaded