import pandas as pd
import tempfile
import time
from pathlib import Path
from hypothesis import given, strategies as st
from faker import Faker

from compliance.eu_ai_act_hitl_router import (
    enforce_human_in_the_loop,
    HIGH_RISK_ACTIONS,
)
from compliance.dora_circuit_breaker import MLSystemCircuitBreaker
from compliance.gdpr_purge import execute_right_to_be_forgotten, PurgeError

fake = Faker()


# --- 1. EU AI ACT DYNAMIC TESTS ---
@given(
    st.floats(min_value=0.0, max_value=1.0), st.sampled_from(list(HIGH_RISK_ACTIONS))
)
def test_eu_ai_act_high_risk_routing(proba, action):
    result = enforce_human_in_the_loop(proba, decision_type=action)
    assert result["action"] == "MANUAL_REVIEW_REQUIRED"  # nosec


@given(st.floats(min_value=0.0, max_value=1.0))
def test_eu_ai_act_low_risk_routing(proba):
    result = enforce_human_in_the_loop(proba, decision_type="INTERNAL_LOGGING")
    assert result["action"] == "SCORE_ONLY"  # nosec


# --- 2. DORA CIRCUIT BREAKER DYNAMIC TESTS ---
@given(st.integers(min_value=1, max_value=10))
def test_dora_circuit_breaker_dynamic(threshold):
    breaker = MLSystemCircuitBreaker(
        failure_threshold=threshold, recovery_timeout_s=0.1
    )

    def failing_model():
        raise ValueError("Simulated Failure")

    # Fail just under the threshold
    for _ in range(threshold - 1):
        breaker.execute_inference(failing_model)
    assert breaker.state == "CLOSED"  # nosec

    # Cross the threshold
    breaker.execute_inference(failing_model)
    assert breaker.state == "OPEN"  # nosec

    # Test Recovery Timeout
    time.sleep(0.2)

    # Should transition to HALF_OPEN on next call
    def successful_model():
        return "SUCCESS"

    res = breaker.execute_inference(successful_model)
    assert res["status"] == "success"  # nosec
    assert breaker.state == "CLOSED"  # nosec


# --- 3. GDPR PURGE DYNAMIC TESTS ---
@given(
    st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=5,
            max_size=10,
        ),
        min_size=2,
        max_size=10,
        unique=True,
    )
)
def test_gdpr_purge_dynamic(user_ids):
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "mock_dynamic_data.csv"
        target_user = user_ids[0]

        df = pd.DataFrame(
            {
                "user_id": user_ids,
                "transaction_value": [
                    fake.pyfloat(positive=True) for _ in range(len(user_ids))
                ],
            }
        )
        df.to_csv(csv_file, index=False)

        # Should raise PurgeError because fake.db doesn't exist
        try:
            execute_right_to_be_forgotten(
                target_user, str(Path(tmp_dir) / "fake.db"), str(csv_file)
            )
            assert False, "Should have raised PurgeError for missing DB"  # nosec
        except PurgeError:
            pass
