import httpx
import time
import random
import json
from pathlib import Path

URL = "http://localhost:8000/predict"
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "telemetry_logs.jsonl"

def generate_catastrophic_applicant(is_young: bool):
    """Generates severe financial distress profiles to force negative underwriting decisions."""
    age = random.randint(18, 29) if is_young else random.randint(30, 65)
    
    # Ultra-high-risk profiles: near-zero income, bottom credit scores, extreme debt
    return {
        "Age": age,
        "Income": random.randint(4000, 9000), 
        "CreditScore": random.randint(300, 400), 
        "LoanAmount": random.randint(85000, 150000), 
        "DTIRatio": round(random.uniform(0.85, 0.99), 2), 
        "MonthsEmployed": random.randint(0, 2),
        "InterestRate": round(random.uniform(18.0, 26.5), 1),
        "NumCreditLines": random.randint(10, 15),
        "LoanTerm": 60,
        "Education": "High School",
        "EmploymentType": "Unemployed",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Personal",
        "HasCoSigner": "No"
    }

print("🚀 Booting ULTRA-HIGH-RISK production traffic simulation stream...")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

with httpx.Client() as client:
    for i in range(40):
        is_young = random.random() < 0.40
        payload = generate_catastrophic_applicant(is_young)
        
        try:
            response = client.post(URL, json=payload)
            res_data = response.json()
            decision = res_data.get("governance", {}).get("underwriting_decision", "UNKNOWN")
            cohort = res_data.get("governance", {}).get("fairness_demographic_proxy_flag", "UNKNOWN")
            
            print(f"🔹 [{i+1:02d}/40] Age: {payload['Age']} | Credit: {payload['CreditScore']} | Decision: {decision}")
            
            telemetry_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "status_code": response.status_code,
                "latency_ms": round(response.elapsed.total_seconds() * 1000, 2),
                "payload_snapshot": {
                    "Age": payload["Age"],
                    "Income": payload["Income"],
                    "CreditScore": payload["CreditScore"],
                    "LoanAmount": payload["LoanAmount"],
                    "DTIRatio": payload["DTIRatio"]
                },
                "inference_output": {
                    "default_probability": res_data.get("risk_metrics", {}).get("default_probability"),
                    "underwriting_decision": decision,
                    "demographic_cohort": cohort
                }
            }
            
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(telemetry_entry) + "\n")
                
        except Exception as e:
            print(f"❌ Transmission failure at packet {i+1}: {e}")
            
        time.sleep(0.1)

print("🏁 Ultra-high-risk traffic stream execution completed.")
