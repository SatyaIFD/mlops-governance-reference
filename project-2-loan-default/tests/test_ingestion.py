"""
Unit Test Suite - Project 2 Ingestion Layer
Validates data contract compliance, column integrity, and type validation rules.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Ensure project source root is on path context for importing
tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.ingestion.ingestion import validate_raw_dataset
from src.ingestion.schemas import RAW_LOAN_DATA_SCHEMA

def test_schema_contract_valid_data():
    """Asserts that a perfectly structured dataframe passes our ingestion schema rules."""
    # Build a mock valid row matching all columns and data types in our contract
    valid_mock_data = {col: [0] for col in RAW_LOAN_DATA_SCHEMA["expected_columns"]}
    
    # Explicitly enforce correct data types
    for col, dtype in RAW_LOAN_DATA_SCHEMA["numerical_types"].items():
        valid_mock_data[col] = pd.Series([0], dtype=dtype)
        
    df = pd.DataFrame(valid_mock_data)
    assert validate_raw_dataset(df) is True

def test_schema_contract_missing_column():
    """Asserts that the ingestion pipeline catches and flags a missing column violation."""
    # Missing the critical target column 'Default'
    invalid_mock_data = {
        "LoanID": [101],
        "Age": [30],
        "Income": [50000]
    }
    df = pd.DataFrame(invalid_mock_data)
    assert validate_raw_dataset(df) is False