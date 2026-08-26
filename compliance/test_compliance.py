import pandas as pd
from hypothesis import given, strategies as st
from faker import Faker
from compliance.eu_ai_act_hitl_router import enforce_human_in_the_loop
from compliance.dora_circuit_breaker import MLSystemCircuitBreaker
from compliance.gdpr_purge import execute_right_to_be_forgotten

fake = Faker()

# --- 1. EU AI ACT DYNAMIC TESTS ---
# Tests 100 random floats between 0.80 and 1.0
@given(st.floats(min_value=0.80, max_value=1.0))
def test_eu_ai_act_auto_reject_dynamic(proba):
    result = enforce_human_in_the_loop(proba)
    assert result["action"] == "AUTO_REJECT"

# Tests 100 random floats in the exact HITL grey area
@given(st.floats(min_value=0.45, max_value=0.7999))
def test_eu_ai_act_hitl_routing_dynamic(proba):
    result = enforce_human_in_the_loop(proba)
    assert result["action"] == "MANUAL_REVIEW_REQUIRED"

# Tests 100 random floats below the review floor
@given(st.floats(min_value=0.0, max_value=0.4499))
def test_eu_ai_act_auto_approve_dynamic(proba):
    result = enforce_human_in_the_loop(proba)
    assert result["action"] == "AUTO_APPROVE"

# --- 2. DORA CIRCUIT BREAKER DYNAMIC TESTS ---
# Tests random threshold limits between 1 and 100 failures
@given(st.integers(min_value=1, max_value=100))
def test_dora_circuit_breaker_dynamic(threshold):
    breaker = MLSystemCircuitBreaker(failure_threshold=threshold)
    
    # Simulate (threshold - 1) failures: Circuit should stay CLOSED
    for _ in range(threshold - 1):
        breaker.record_failure()
    assert breaker.state == "CLOSED"
    
    # One more failure trips the circuit
    breaker.record_failure()
    assert breaker.state == "OPEN"
    
    # A success resets it instantly
    breaker.record_success()
    assert breaker.state == "CLOSED"

# --- 3. GDPR PURGE DYNAMIC TESTS ---
# Generates dynamic datasets with random lists of 2 to 50 unique user IDs
@given(st.lists(st.text(min_size=5, max_size=10), min_size=2, max_size=50, unique=True))
def test_gdpr_purge_dynamic(tmp_path, user_ids):
    csv_file = tmp_path / "mock_dynamic_data.csv"
    
    # Target the very first randomly generated user for deletion
    target_user = user_ids[0]
    
    # Create a dynamic dataframe with Fake values
    df = pd.DataFrame({
        "user_id": user_ids, 
        "transaction_value": [fake.pyfloat(positive=True) for _ in range(len(user_ids))]
    })
    df.to_csv(csv_file, index=False)
    
    initial_length = len(df)
    
    # Execute the automated purge for the target user
    execute_right_to_be_forgotten(target_user, "fake.db", str(csv_file))
    
    # Verify target user is completely scrubbed, and NO ONE else was touched
    purged_df = pd.read_csv(csv_file)
    assert target_user not in purged_df["user_id"].values
    assert len(purged_df) == initial_length - 1
