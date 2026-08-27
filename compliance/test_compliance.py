import pandas as pd
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st
from faker import Faker
from compliance.eu_ai_act_hitl_router import enforce_human_in_the_loop
from compliance.dora_circuit_breaker import MLSystemCircuitBreaker
from compliance.gdpr_purge import execute_right_to_be_forgotten

fake = Faker()

# --- 1. EU AI ACT DYNAMIC TESTS ---
@given(st.floats(min_value=0.80, max_value=1.0))
def test_eu_ai_act_auto_reject_dynamic(proba):
    result = enforce_human_in_the_loop(proba)
    assert result["action"] == "AUTO_REJECT"

@given(st.floats(min_value=0.45, max_value=0.7999))
def test_eu_ai_act_hitl_routing_dynamic(proba):
    result = enforce_human_in_the_loop(proba)
    assert result["action"] == "MANUAL_REVIEW_REQUIRED"

@given(st.floats(min_value=0.0, max_value=0.4499))
def test_eu_ai_act_auto_approve_dynamic(proba):
    result = enforce_human_in_the_loop(proba)
    assert result["action"] == "AUTO_APPROVE"

# --- 2. DORA CIRCUIT BREAKER DYNAMIC TESTS ---
@given(st.integers(min_value=1, max_value=100))
def test_dora_circuit_breaker_dynamic(threshold):
    breaker = MLSystemCircuitBreaker(failure_threshold=threshold)
    
    for _ in range(threshold - 1):
        breaker.record_failure()
    assert breaker.state == "CLOSED"
    
    breaker.record_failure()
    assert breaker.state == "OPEN"
    
    breaker.record_success()
    assert breaker.state == "CLOSED"

# --- 3. GDPR PURGE DYNAMIC TESTS ---
# We restrict the alphabet to alphanumeric to prevent malformed CSVs
@given(st.lists(
    st.text(alphabet=st.characters(whitelist_categories=('L', 'N')), min_size=5, max_size=10), 
    min_size=2, max_size=50, unique=True
))
def test_gdpr_purge_dynamic(user_ids):
    # FIX: Use Python's tempfile inside the function instead of Pytest's tmp_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "mock_dynamic_data.csv"
        
        target_user = user_ids[0]
        
        df = pd.DataFrame({
            "user_id": user_ids, 
            "transaction_value": [fake.pyfloat(positive=True) for _ in range(len(user_ids))]
        })
        df.to_csv(csv_file, index=False)
        
        initial_length = len(df)
        
        execute_right_to_be_forgotten(target_user, "fake.db", str(csv_file))
        
        purged_df = pd.read_csv(csv_file)
        assert target_user not in purged_df["user_id"].values
        assert len(purged_df) == initial_length - 1
