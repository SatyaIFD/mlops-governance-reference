"""
Ingestion Utilities - Project 2: Loan Default Prediction
Handles directory initialization and clean filesystem path resolutions.
"""

import os
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the root directory of project 2."""
    return Path(__file__).resolve().parents[2]

def initialize_directories():
    """Ensures raw and processed data directories exist on the filesystem."""
    root = get_project_root()
    dirs = [
        root / "data",
        root / "data" / "processed",
        root / "artifacts"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)