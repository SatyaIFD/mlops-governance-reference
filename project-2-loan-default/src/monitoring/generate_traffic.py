"""
Traffic Simulation Utility - Project 2: Loan Default Prediction
Samples representative population traffic without replacement across runs from the
holdout test split (stratified by age cohort), excluding already-logged applicants
using full 16-field payload hashes and raising if the unseen pool is exhausted.
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import httpx
import pandas as pd
from src.monitoring.observability import (
    compute_payload_hash,
    iter_telemetry_log_files,
    stream_telemetry_records,
)

DEFAULT_URL = os.getenv("LOAN_API_URL", "http://127.0.0.1:8081/api/v1/loan-sim/predict")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"
TELEMETRY_LOG_PATH = PROJECT_ROOT / "artifacts" / "telemetry_logs.jsonl"

PAYLOAD_COLUMNS = [
    "Age",
    "Income",
    "CreditScore",
    "LoanAmount",
    "DTIRatio",
    "MonthsEmployed",
    "InterestRate",
    "NumCreditLines",
    "LoanTerm",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]


def _normalize_payload_types(row: dict) -> dict:
    return {
        "Age": int(row["Age"]),
        "Income": int(row["Income"]),
        "CreditScore": int(row["CreditScore"]),
        "LoanAmount": int(row["LoanAmount"]),
        "DTIRatio": round(float(row["DTIRatio"]), 4),
        "MonthsEmployed": int(row["MonthsEmployed"]),
        "InterestRate": round(float(row["InterestRate"]), 4),
        "NumCreditLines": int(row["NumCreditLines"]),
        "LoanTerm": int(row["LoanTerm"]),
        "Education": str(row["Education"]),
        "EmploymentType": str(row["EmploymentType"]),
        "MaritalStatus": str(row["MaritalStatus"]),
        "HasMortgage": str(row["HasMortgage"]),
        "HasDependents": str(row["HasDependents"]),
        "LoanPurpose": str(row["LoanPurpose"]),
        "HasCoSigner": str(row["HasCoSigner"]),
    }


def _snapshot_key(row: dict) -> tuple:
    bucket = (
        str(row["age_bucket"])
        if "age_bucket" in row
        else ("UNDER_30" if int(row["Age"]) < 30 else "30_AND_OVER")
    )
    return (
        bucket,
        int(row["Income"]),
        int(row["CreditScore"]),
        int(row["LoanAmount"]),
        round(float(row["DTIRatio"]), 4),
    )


def _load_seen_identifiers(
    log_path: Path = TELEMETRY_LOG_PATH,
) -> tuple[set[str], set[tuple]]:
    seen_hashes: set[str] = set()
    seen_snapshots: set[tuple] = set()
    if not iter_telemetry_log_files(log_path):
        return seen_hashes, seen_snapshots
    for rec in stream_telemetry_records(log_path):
        try:
            if rec.get("payload_hash"):
                seen_hashes.add(str(rec["payload_hash"]))
            snap = rec.get("payload_snapshot")
            if isinstance(snap, dict) and (
                {"Income", "CreditScore", "LoanAmount", "DTIRatio"}.issubset(snap)
                and ("age_bucket" in snap or "Age" in snap)
            ):
                seen_snapshots.add(_snapshot_key(snap))
        except (KeyError, TypeError, ValueError) as exc:
            logging.warning(
                "Skipping malformed telemetry record in %s: %s", log_path.name, exc
            )
    return seen_hashes, seen_snapshots


def _load_holdout_cohorts(
    required_per_cohort: int,
    seed: int | None = None,
    seen_hashes: set[str] | None = None,
    seen_snapshots: set[tuple] | None = None,
) -> tuple[list[dict], list[dict]] | None:
    """Loads holdout records partitioned into (<30) and (>=30) cohorts, excluding already-logged applicants."""
    if not TEST_PARQUET_PATH.is_file():
        return None
    df = pd.read_parquet(TEST_PARQUET_PATH)
    if not set(PAYLOAD_COLUMNS).issubset(df.columns):
        return None

    active_seed = int(time.time_ns() % (2**31 - 1)) if seed is None else int(seed)
    young_raw = (
        df[df["Age"] < 30][PAYLOAD_COLUMNS]
        .sample(frac=1.0, random_state=active_seed)
        .to_dict(orient="records")
    )
    mature_raw = (
        df[df["Age"] >= 30][PAYLOAD_COLUMNS]
        .sample(frac=1.0, random_state=active_seed)
        .to_dict(orient="records")
    )

    seen_hashes = seen_hashes or set()
    seen_snapshots = seen_snapshots or set()

    def _filter_unseen(records: list[dict]) -> list[dict]:
        unseen = []
        used_snaps = set(seen_snapshots)
        for r in records:
            norm = _normalize_payload_types(r)
            h = compute_payload_hash(norm)
            s_key = _snapshot_key(norm)
            if h not in seen_hashes and s_key not in used_snaps:
                unseen.append(norm)
                used_snaps.add(s_key)
        return unseen

    young_records = _filter_unseen(young_raw)
    mature_records = _filter_unseen(mature_raw)

    if (
        len(young_records) < required_per_cohort
        or len(mature_records) < required_per_cohort
    ):
        raise RuntimeError(
            f"Unseen holdout applicant pool exhausted (needed {required_per_cohort} per cohort; "
            f"available young={len(young_records)}, mature={len(mature_records)})."
        )
    return young_records, mature_records


def generate_representative_applicant(is_young: bool, high_risk: bool = False) -> dict:
    """Fallback synthetic generator matching empirical training bounds when test.parquet is absent."""
    age = random.randint(18, 29) if is_young else random.randint(30, 69)  # nosec B311
    if high_risk:
        return {
            "Age": age,
            "Income": random.randint(15000, 45000),  # nosec B311
            "CreditScore": random.randint(300, 580),  # nosec B311
            "LoanAmount": random.randint(120000, 249999),  # nosec B311
            "DTIRatio": round(random.uniform(0.60, 0.90), 2),  # nosec B311
            "MonthsEmployed": random.randint(0, 24),  # nosec B311
            "InterestRate": round(random.uniform(16.0, 25.0), 2),  # nosec B311
            "NumCreditLines": random.randint(1, 4),  # nosec B311
            "LoanTerm": random.choice([36, 48, 60]),  # nosec B311
            "Education": random.choice(["High School", "Bachelor's"]),  # nosec B311
            "EmploymentType": random.choice(["Unemployed", "Part-time"]),  # nosec B311
            "MaritalStatus": random.choice(["Single", "Divorced"]),  # nosec B311
            "HasMortgage": "No",
            "HasDependents": random.choice(["Yes", "No"]),  # nosec B311
            "LoanPurpose": random.choice(["Auto", "Other", "Business"]),  # nosec B311
            "HasCoSigner": "No",
        }
    return {
        "Age": age,
        "Income": random.randint(15000, 149999),  # nosec B311
        "CreditScore": random.randint(300, 849),  # nosec B311
        "LoanAmount": random.randint(5000, 249999),  # nosec B311
        "DTIRatio": round(random.uniform(0.10, 0.90), 2),  # nosec B311
        "MonthsEmployed": random.randint(0, 119),  # nosec B311
        "InterestRate": round(random.uniform(2.0, 25.0), 2),  # nosec B311
        "NumCreditLines": random.randint(1, 4),  # nosec B311
        "LoanTerm": random.choice([12, 24, 36, 48, 60]),  # nosec B311
        "Education": random.choice(["Bachelor's", "High School", "Master's", "PhD"]),  # nosec B311
        "EmploymentType": random.choice(
            ["Full-time", "Part-time", "Self-employed", "Unemployed"]
        ),  # nosec B311
        "MaritalStatus": random.choice(["Divorced", "Married", "Single"]),  # nosec B311
        "HasMortgage": random.choice(["Yes", "No"]),  # nosec B311
        "HasDependents": random.choice(["Yes", "No"]),  # nosec B311
        "LoanPurpose": random.choice(
            ["Auto", "Business", "Education", "Home", "Other"]
        ),  # nosec B311
        "HasCoSigner": random.choice(["Yes", "No"]),  # nosec B311
    }


def generate_stress_applicant(is_young: bool) -> dict:
    """Generates severe financial distress profiles to stress-test high-risk decline routing."""
    age = random.randint(18, 29) if is_young else random.randint(30, 65)  # nosec B311
    return {
        "Age": age,
        "Income": random.randint(15000, 22000),  # nosec B311
        "CreditScore": random.randint(300, 380),  # nosec B311
        "LoanAmount": random.randint(150000, 245000),  # nosec B311
        "DTIRatio": round(random.uniform(0.82, 0.90), 2),  # nosec B311
        "MonthsEmployed": random.randint(0, 2),  # nosec B311
        "InterestRate": round(random.uniform(21.0, 24.9), 2),  # nosec B311
        "NumCreditLines": 4,
        "LoanTerm": 60,
        "Education": "High School",
        "EmploymentType": "Unemployed",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Other",
        "HasCoSigner": "No",
    }


def run_traffic_simulation(
    url: str = DEFAULT_URL,
    count: int = 80,
    mode: str = "representative",
    seed: int | None = None,
) -> int:
    api_key = os.getenv("EXPECTED_API_KEY", "")
    headers = {"X-Traffic-Source": f"synthetic_{mode}_simulation"}
    if api_key:
        headers["X-API-Key"] = api_key

    required_per_cohort = (count + 1) // 2
    seen_hashes, seen_snapshots = (
        _load_seen_identifiers() if mode == "representative" else (set(), set())
    )
    holdout_cohorts = (
        _load_holdout_cohorts(
            required_per_cohort=required_per_cohort,
            seed=seed,
            seen_hashes=seen_hashes,
            seen_snapshots=seen_snapshots,
        )
        if mode == "representative"
        else None
    )

    print(
        f"🚀 Booting {mode.upper()} traffic simulation stream ({count} requests) -> {url}"
    )
    succeeded = 0

    with httpx.Client(timeout=10.0) as client:
        for i in range(count):
            is_young = i % 2 == 0
            if mode == "stress":
                payload = generate_stress_applicant(is_young)
            elif holdout_cohorts is not None:
                young_pool, mature_pool = holdout_cohorts
                pool = young_pool if is_young else mature_pool
                payload = pool[i // 2]
            else:
                high_risk = (i // 2) % 4 == 0
                payload = generate_representative_applicant(
                    is_young, high_risk=high_risk
                )

            try:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    print(
                        f"⚠️ [{i + 1:02d}/{count}] HTTP {response.status_code}: skipping invalid response."
                    )
                    continue

                res_data = response.json()
                gov = res_data.get("governance", {})
                rec = gov.get("model_recommendation", "UNKNOWN")
                review = gov.get("review_status", "UNKNOWN")
                succeeded += 1
                print(
                    f"🔹 [{i + 1:02d}/{count}] Age: {payload['Age']} | Score: {payload['CreditScore']} | Rec: {rec} | Review: {review}"
                )
            except Exception as e:
                print(f"❌ Transmission failure at packet {i + 1}: {e}")

            time.sleep(0.02)

    print(f"🏁 Traffic stream completed ({succeeded}/{count} successful responses).")
    return succeeded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument(
        "--mode", choices=["representative", "stress"], default="representative"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_traffic_simulation(
        url=args.url, count=args.count, mode=args.mode, seed=args.seed
    )
