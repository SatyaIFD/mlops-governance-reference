"""
Unit Test Suite - Project 2 Ingestion Layer
Validates data contract compliance, column integrity, and type validation rules.
"""

import pandas as pd
import sys
from pathlib import Path

# Ensure project source root is on path context for importing
tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.ingestion.ingestion import validate_raw_dataset  # noqa: E402
from src.ingestion.schemas import RAW_LOAN_DATA_SCHEMA  # noqa: E402


def test_schema_contract_valid_data():
    """Asserts that a perfectly structured dataframe passes our ingestion schema rules."""
    valid_mock_data = {}

    # Dynamically build mock data that perfectly respects the data contract boundaries
    for col in RAW_LOAN_DATA_SCHEMA["expected_columns"]:
        if col in RAW_LOAN_DATA_SCHEMA.get("categorical_columns", []):
            allowed_vals = RAW_LOAN_DATA_SCHEMA.get("categorical_allowed_values", {})
            if col in allowed_vals:
                # Pick the first valid category string from the schema (e.g., 'Yes')
                valid_mock_data[col] = [allowed_vals[col][0]]
            else:
                # For categorical columns without explicit boundaries, use a standard string
                valid_mock_data[col] = ["MockString"]
        else:
            # Default to 0 for numerical columns
            valid_mock_data[col] = [0]

    # Explicitly enforce correct numerical data types
    for col, dtype in RAW_LOAN_DATA_SCHEMA.get("numerical_types", {}).items():
        valid_mock_data[col] = pd.Series(valid_mock_data.get(col, [0]), dtype=dtype)

    df = pd.DataFrame(valid_mock_data)
    assert validate_raw_dataset(df) is True  # nosec B101


def test_schema_contract_missing_column():
    """Asserts that the ingestion pipeline catches and flags a missing column violation."""
    # Missing the critical target column 'Default'
    invalid_mock_data = {"LoanID": [101], "Age": [30], "Income": [50000]}
    df = pd.DataFrame(invalid_mock_data)
    assert validate_raw_dataset(df) is False  # nosec B101
