import json
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
import pandas as pd
import pytest
from faker import Faker
from hypothesis import given, settings, strategies as st

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
    assert result["action"] == "MANUAL_REVIEW_REQUIRED"  # nosec B101


@given(st.floats(min_value=0.0, max_value=1.0))
def test_eu_ai_act_low_risk_routing(proba):
    result = enforce_human_in_the_loop(proba, decision_type="INTERNAL_LOGGING")
    assert result["action"] == "SCORE_ONLY"  # nosec B101


# --- 2. DORA CIRCUIT BREAKER DYNAMIC & CONCURRENCY TESTS ---
@settings(deadline=None)
@given(st.integers(min_value=1, max_value=10))
def test_dora_circuit_breaker_dynamic(threshold):
    breaker = MLSystemCircuitBreaker(
        failure_threshold=threshold, recovery_timeout_s=0.1
    )

    def failing_model():
        raise ValueError("Simulated Failure")

    for _ in range(threshold - 1):
        breaker.execute_inference(failing_model)
    assert breaker.state == "CLOSED"  # nosec B101

    breaker.execute_inference(failing_model)
    assert breaker.state == "OPEN"  # nosec B101

    time.sleep(0.15)

    def successful_model():
        return "SUCCESS"

    res = breaker.execute_inference(successful_model)
    assert res["status"] == "success"  # nosec B101
    assert breaker.state == "CLOSED"  # nosec B101


def test_dora_circuit_breaker_half_open_single_probe_exclusivity():
    """Verifies that only one thread executes the probe when transitioning to HALF_OPEN."""
    breaker = MLSystemCircuitBreaker(failure_threshold=1, recovery_timeout_s=0.05)
    breaker.execute_inference(lambda: (_ for _ in ()).throw(RuntimeError("trip")))
    assert breaker.state == "OPEN"  # nosec B101

    time.sleep(0.08)

    probe_entries = 0
    probe_lock = threading.Lock()
    barrier = threading.Event()

    def slow_recovering_model():
        nonlocal probe_entries
        with probe_lock:
            probe_entries += 1
        barrier.wait(timeout=1.0)
        return "RECOVERED"

    results = []

    def worker():
        results.append(breaker.execute_inference(slow_recovering_model))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()

    time.sleep(0.05)
    barrier.set()
    for t in threads:
        t.join()

    assert probe_entries == 1  # nosec B101
    successes = [r for r in results if r["status"] == "success"]
    fallbacks = [r for r in results if r["status"] == "fallback"]
    assert len(successes) == 1  # nosec B101
    assert len(fallbacks) == 4  # nosec B101
    assert breaker.state == "CLOSED"  # nosec B101


# --- 3. GDPR PURGE DYNAMIC, END-TO-END, & FAILURE-INJECTION TESTS ---
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

        with pytest.raises(PurgeError):
            execute_right_to_be_forgotten(
                target_user, str(Path(tmp_dir) / "fake.db"), str(csv_file)
            )


def _init_mock_sqlite_db(db_path: Path, target_user: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE params (key TEXT, value TEXT)")
    conn.execute("CREATE TABLE tags (key TEXT, value TEXT)")
    conn.execute("INSERT INTO params VALUES ('user_id', ?)", (target_user,))
    conn.execute("INSERT INTO params VALUES ('user_id', 'KEEP_USER')")
    conn.execute("INSERT INTO tags VALUES ('user_id', ?)", (target_user,))
    # Unrelated tag key sharing the same value must NOT be deleted
    conn.execute("INSERT INTO tags VALUES ('experiment_label', ?)", (target_user,))
    conn.commit()
    conn.close()


def test_gdpr_purge_end_to_end_parquet_and_scoped_tags(tmp_path):
    """Verifies positive Parquet streaming deletion and key='user_id' scoped SQLite deletion."""
    db_file = tmp_path / "mlflow.db"
    parquet_file = tmp_path / "train.parquet"
    target_user = "USER_999"

    _init_mock_sqlite_db(db_file, target_user)

    df = pd.DataFrame(
        {
            "user_id": [target_user, "USER_111", target_user, "USER_222"],
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df.to_parquet(parquet_file, index=False)

    res = execute_right_to_be_forgotten(target_user, str(db_file), str(parquet_file))
    assert res["db_rows_deleted"] == 2  # nosec B101
    assert res["file_rows_removed"] == 2  # nosec B101

    remaining_df = pd.read_parquet(parquet_file)
    assert list(remaining_df["user_id"]) == ["USER_111", "USER_222"]  # nosec B101

    conn = sqlite3.connect(db_file)
    unrelated_tag_count = conn.execute(
        "SELECT COUNT(*) FROM tags WHERE key='experiment_label' AND value=?",
        (target_user,),
    ).fetchone()[0]
    conn.close()
    assert unrelated_tag_count == 1  # nosec B101


def test_gdpr_purge_rollback_on_db_failure(tmp_path):
    """Verifies that a failure during the SQLite phase rolls back the swapped Parquet file."""
    db_file = tmp_path / "mlflow.db"
    parquet_file = tmp_path / "train.parquet"
    target_user = "USER_ROLLBACK"

    _init_mock_sqlite_db(db_file, target_user)
    df = pd.DataFrame({"user_id": [target_user, "USER_KEEP"], "amount": [100.0, 200.0]})
    df.to_parquet(parquet_file, index=False)

    with pytest.raises(PurgeError, match="rolled back"):
        execute_right_to_be_forgotten(
            target_user, str(db_file), str(parquet_file), _inject_db_failure=True
        )

    # Verify Parquet file was restored to its pre-purge state and no temp/backup files remain
    restored_df = pd.read_parquet(parquet_file)
    assert len(restored_df) == 2  # nosec B101
    assert target_user in set(restored_df["user_id"])  # nosec B101
    assert not (tmp_path / "train.parquet.bak").exists()  # nosec B101
    assert not (tmp_path / "train.parquet.purge_journal.json").exists()  # nosec B101


def test_gdpr_purge_crash_recovery_from_journal(tmp_path):
    """Simulates a crash right after FILE_SWAPPED and verifies the next retry recovers cleanly."""
    db_file = tmp_path / "mlflow.db"
    parquet_file = tmp_path / "train.parquet"
    bak_file = tmp_path / "train.parquet.bak"
    journal_file = tmp_path / "train.parquet.purge_journal.json"
    target_user = "USER_CRASH"

    _init_mock_sqlite_db(db_file, target_user)

    full_df = pd.DataFrame(
        {"user_id": [target_user, "USER_SAFE"], "amount": [50.0, 60.0]}
    )
    partial_df = pd.DataFrame({"user_id": ["USER_SAFE"], "amount": [60.0]})

    # Simulate crashed state: data_file has already been swapped to partial_df, .bak holds full_df
    partial_df.to_parquet(parquet_file, index=False)
    shutil.copy2(parquet_file, bak_file)
    full_df.to_parquet(bak_file, index=False)
    journal_file.write_text(
        json.dumps({"state": "FILE_SWAPPED", "user_id": target_user})
    )

    # Retry should recover .bak -> parquet_file and then execute the full purge cleanly
    res = execute_right_to_be_forgotten(target_user, str(db_file), str(parquet_file))
    assert res["file_rows_removed"] == 1  # nosec B101
    assert res["db_rows_deleted"] == 2  # nosec B101
    assert not bak_file.exists()  # nosec B101
    assert not journal_file.exists()  # nosec B101
