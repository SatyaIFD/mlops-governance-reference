import os
import pandas as pd
from compliance.eu_ai_act_hitl_router import enforce_human_in_the_loop
from compliance.dora_circuit_breaker import MLSystemCircuitBreaker
from compliance.gdpr_purge import execute_right_to_be_forgotten

# --- 1. EU AI ACT AUTOMATED TESTS ---
def test_eu_ai_act_auto_reject():
    result = enforce_human_in_the_loop(0.85)
    assert result["action"] == "AUTO_REJECT"

def test_eu_ai_act_hitl_routing():
    result = enforce_human_in_the_loop(0.60)
    assert result["action"] == "MANUAL_REVIEW_REQUIRED"

def test_eu_ai_act_auto_approve():
    result = enforce_human_in_the_loop(0.20)
    assert result["action"] == "AUTO_APPROVE"

# --- 2. DORA CIRCUIT BREAKER AUTOMATED TESTS ---
def test_dora_circuit_breaker_trips():
    breaker = MLSystemCircuitBreaker(failure_threshold=2)
    
    # First failure: Circuit should stay CLOSED
    breaker.record_failure()
    assert breaker.state == "CLOSED"
    
    # Second failure: Circuit should TRIP to OPEN
    breaker.record_failure()
    assert breaker.state == "OPEN"
    
    # Successful run should reset it to CLOSED
    breaker.record_success()
    assert breaker.state == "CLOSED"

# --- 3. GDPR PURGE AUTOMATED TESTS ---
def test_gdpr_purge_removes_user(tmp_path):
    # tmp_path is a built-in pytest fixture that creates a temporary directory
    csv_file = tmp_path / "mock_data.csv"
    
    # Create a mock dataset with two users
    df = pd.DataFrame({"user_id": ["USR-1", "USR-2"], "value": [100, 200]})
    df.to_csv(csv_file, index=False)
    
    # Execute the automated purge for USR-1
    execute_right_to_be_forgotten("USR-1", "fake.db", str(csv_file))
    
    # Load the data back and verify USR-1 is completely gone
    purged_df = pd.read_csv(csv_file)
    assert "USR-1" not in purged_df["user_id"].values
    assert "USR-2" in purged_df["user_id"].values
    assert len(purged_df) == 1
