import sys
from pathlib import Path
import pytest
import pandas as pd
from pydantic import ValidationError

# Dynamically inject Project 1 root into sys.path to resolve the local 'src' namespace
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.ingestion.schemas import TransactionInputSchema  # noqa: E402
from src.ingestion.utils import clean_and_transform_features  # noqa: E402


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
    assert len(processed_df) == 1  # nosec B101
    assert "scaled_amount" in processed_df.columns  # nosec B101
    assert "scaled_time" in processed_df.columns  # nosec B101
    assert "Amount" not in processed_df.columns  # nosec B101
    assert "Time" not in processed_df.columns  # nosec B101


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
