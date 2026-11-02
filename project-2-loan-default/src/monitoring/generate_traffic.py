"""
Traffic Simulation Utility - Project 2: Loan Default Prediction
Supports both representative population traffic (--mode representative) and
extreme stress-test traffic (--mode stress) against the authenticated API gateway.
"""

import argparse
import os
import random
import time

import httpx

DEFAULT_URL = os.getenv("LOAN_API_URL", "http://localhost/api/v1/loan/predict")


def generate_representative_applicant(is_young: bool, high_risk: bool = False) -> dict:
    """Generates applicant profiles matching the training distribution (including both approvals and declines)."""
    age = random.randint(21, 29) if is_young else random.randint(30, 68)  # nosec B311
    if high_risk:
        return {
            "Age": age,
            "Income": random.randint(15000, 32000),  # nosec B311
            "CreditScore": random.randint(320, 520),  # nosec B311
            "LoanAmount": random.randint(95000, 220000),  # nosec B311
            "DTIRatio": round(random.uniform(0.65, 0.89), 2),  # nosec B311
            "MonthsEmployed": random.randint(0, 6),  # nosec B311
            "InterestRate": round(random.uniform(19.0, 24.9), 2),  # nosec B311
            "NumCreditLines": random.randint(1, 4),  # nosec B311
            "LoanTerm": random.choice([48, 60]),  # nosec B311
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
        "Income": random.randint(35000, 145000),  # nosec B311
        "CreditScore": random.randint(560, 840),  # nosec B311
        "LoanAmount": random.randint(10000, 180000),  # nosec B311
        "DTIRatio": round(random.uniform(0.12, 0.65), 2),  # nosec B311
        "MonthsEmployed": random.randint(12, 118),  # nosec B311
        "InterestRate": round(random.uniform(4.0, 18.5), 2),  # nosec B311
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
) -> int:
    api_key = os.getenv("EXPECTED_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}

    print(
        f"🚀 Booting {mode.upper()} traffic simulation stream ({count} requests) -> {url}"
    )
    succeeded = 0

    with httpx.Client(timeout=10.0) as client:
        for i in range(count):
            is_young = i % 2 == 0
            high_risk = i % 4 == 0
            payload = (
                generate_stress_applicant(is_young)
                if mode == "stress"
                else generate_representative_applicant(is_young, high_risk=high_risk)
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
    args = parser.parse_args()
    run_traffic_simulation(url=args.url, count=args.count, mode=args.mode)
