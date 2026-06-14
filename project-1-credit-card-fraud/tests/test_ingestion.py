import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError
from src.ingestion.schemas import TransactionInputSchema
from src.ingestion.utils import clean_and_transform_features

def test_clean_and_transform_features_drops_duplicates():
    """Verify that exact duplicate row payloads are completely purged during feature execution."""
    # Arrange: Create dummy matrix containing explicit duplicates
    mock_row = {f"V{i}": 0.0 for i in range(1, 29)}
    mock_row.update({"Time": 10.0, "Amount": 100.0, "Class": 0})
    
    # Generate identical duplicate entry points
    mock_df = pd.DataFrame([mock_row, mock_row, mock_row])
    
    # Act
    processed_df = clean_and_transform_features(mock_df)
    
    # Assert: Should purge the 2 duplicates, leaving exactly 1 unique row
    assert len(processed_df) == 1
    assert "scaled_amount" in processed_df.columns
    assert "scaled_time" in processed_df.columns
    assert "Amount" not in processed_df.columns
    assert "Time" not in processed_df.columns

def test_transaction_input_schema_catches_unauthorized_extra_fields():
    """Verify that data validation contracts actively block malicious or extra fields."""
    valid_payload = {f"V{i}": 0.1 for i in range(1, 29)}
    valid_payload.update({"Time": 12.5, "Amount": 50.0})
    
    # Inject an illegal property not listed in the schema contract
    malicious_payload = valid_payload.copy()
    malicious_payload["unauthorized_hacker_column"] = 999.9
    
    # Assert: Pydantic should raise a ValidationError because extra fields are forbidden
    with pytest.raises(ValidationError):
        TransactionInputSchema(**malicious_payload)