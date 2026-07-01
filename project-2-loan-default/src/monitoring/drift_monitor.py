import json
from pathlib import Path
import pandas as pd

def run_production_audit():
    print("🕵️‍♂️ Initializing real-time drift & governance audit scanner...")
    
    # Locate the artifacts folder identically to your observability engine
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]  # Points directly to project-2-loan-default
    log_path = BASE_DIR / "artifacts" / "telemetry_logs.jsonl"
    
    if not log_path.exists():
        print(f"⚠️ Telemetry log ledger not found at: {log_path}")
        print("Please send payloads to the FastAPI endpoint first to populate transactions.")
        return

    # Parse JSON Lines telemetry data
    records = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
                
    df = pd.DataFrame(records)
    if df.empty:
        print("❌ Telemetry log file is empty. Monitoring idle.")
        return

    print(f"📊 Analyzing {len(df)} live production transaction logs...")
    
    # Flatten the payload snapshot and inference outputs
    payloads = pd.json_normalize(df['payload_snapshot'])
    outputs = pd.json_normalize(df['inference_output'])
    audit_df = pd.concat([payloads, outputs], axis=1)
    
    # Establish governance cohorts
    audit_df['is_young'] = (audit_df['Age'] < 30).astype(int)
    audit_df['is_approved'] = (audit_df['underwriting_decision'] == 'APPROVED').astype(int)

    # Calculate live Disparate Impact (DI) Ratio
    grouped = audit_df.groupby('is_young')['is_approved'].mean()
    approval_protected = grouped.get(1, 0.0)
    approval_baseline = grouped.get(0, 0.0)
    
    di_ratio = approval_protected / approval_baseline if approval_baseline > 0 else 1.0
    
    print("\n================== LIVE GOVERNANCE REPORT ==================")
    print(f"👶 Protected Young Cohort (<30) Approval Rate : {approval_protected:.2%}")
    print(f"🧓 Baseline Mature Cohort (>=30) Approval Rate: {approval_baseline:.2%}")
    print(f"⚖️ Rolling Production Disparate Impact Ratio   : {di_ratio:.4f}")
    
    # Validate against the regulatory Four-Fifths Rule boundary (0.80 - 1.25)
    if di_ratio < 0.80 or di_ratio > 1.25:
        print("🚨 ALERT: Adverse Regulatory Fairness Breach Detected!")
        print("👉 Action Required: Trigger bias mitigation pipeline retraining.")
    else:
        print("✅ Governance Status: COMPLIANT. Underwriting operates within fair boundaries.")
    print("============================================================\n")

if __name__ == "__main__":
    run_production_audit()