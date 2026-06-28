"""
Unit Test Suite - Project 2 Governance Layer
Validates automated compliance generation mechanics.
"""

import pytest
import sys
from pathlib import Path

# Ensure project source root is on path context for importing
tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.governance.audit import compile_compliance_audit

def test_audit_report_generation_execution(tmp_path, monkeypatch):
    """Asserts that the compliance auditing compilation completes without fatal exceptions."""
    # Redirect PROJECT_ROOT using monkeypatch to avoid polluting production artifacts during unit testing
    monkeypatch.setattr("src.governance.audit.PROJECT_ROOT", tmp_path)
    
    artifacts_dir = tmp_path / "artifacts"
    docs_dir = tmp_path / "docs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Run audit on empty mock outputs to test safe fallback parsing logic
    try:
        compile_compliance_audit()
        execution_passed = True
    except Exception as e:
        print(f"Audit execution crashed: {e}")
        execution_passed = False
        
    assert execution_passed is True
    assert (docs_dir / "model_card.md").exists()