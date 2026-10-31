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


def generate_representative_applicant(is_young: bool) -> dict:
    """Generates realistic applicant profiles spanning creditworthy and higher-risk segments."""
    age = random.randint(21, 29) if is_young else random.randint(30, 65)  # nosec B311
    return {
        "Age": age,
        "Income": random.randint(45000, 140000),  # nosec B311
        "CreditScore": random.randint(620, 820),  # nosec B311
        "LoanAmount": random.randint(8000, 45000),  # nosec B311
        "DTIRatio": round(random.uniform(0.12, 0.45), 2),  # nosec B311
        "MonthsEmployed": random.randint(12, 120),  # nosec B311
        "InterestRate": round(random.uniform(4.5, 11.5), 1),  # nosec B311
        "NumCreditLines": random.randint(2, 6),  # nosec B311
        "LoanTerm": random.choice([24, 36, 48, 60]),  # nosec B311
        "Education": random.choice(["High School", "Bachelors", "Masters", "PhD"]),  # nosec B311
        "EmploymentType": random.choice(["Full-time", "Part-time", "Self-employed"]),  # nosec B311
        "MaritalStatus": random.choice(["Single", "Married", "Divorced"]),  # nosec B311
        "HasMortgage": random.choice(["Yes", "No"]),  # nosec B311
        "HasDependents": random.choice(["Yes", "No"]),  # nosec B311
        "LoanPurpose": random.choice(["Auto", "Home", "Personal", "Business"]),  # nosec B311
        "HasCoSigner": random.choice(["Yes", "No"]),  # nosec B311
    }


def generate_stress_applicant(is_young: bool) -> dict:
    """Generates severe financial distress profiles to stress-test high-risk decline routing."""
    age = random.randint(18, 29) if is_young else random.randint(30, 65)  # nosec B311
    return {
        "Age": age,
        "Income": random.randint(4000, 9000),  # nosec B311
        "CreditScore": random.randint(300, 400),  # nosec B311
        "LoanAmount": random.randint(85000, 150000),  # nosec B311
        "DTIRatio": round(random.uniform(0.85, 0.99), 2),  # nosec B311
        "MonthsEmployed": random.randint(0, 2),  # nosec B311
        "InterestRate": round(random.uniform(18.0, 26.5), 1),  # nosec B311
        "NumCreditLines": random.randint(10, 15),  # nosec B311
        "LoanTerm": 60,
        "Education": "High School",
        "EmploymentType": "Unemployed",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Personal",
        "HasCoSigner": "No",
    }


def run_traffic_simulation(
    url: str = DEFAULT_URL,
    count: int = 40,
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
            payload = (
                generate_stress_applicant(is_young)
                if mode == "stress"
                else generate_representative_applicant(is_young)
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

            time.sleep(0.05)

    print(f"🏁 Traffic stream completed ({succeeded}/{count} successful responses).")
    return succeeded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument(
        "--mode", choices=["representative", "stress"], default="representative"
    )
    args = parser.parse_args()
    run_traffic_simulation(url=args.url, count=args.count, mode=args.mode)
