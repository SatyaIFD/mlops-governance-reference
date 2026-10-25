import json
import argparse
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HIGH_RISK_ACTIONS = {"CREDIT_APPROVE", "CREDIT_DENY", "FRAUD_BLOCK"}


def enforce_human_in_the_loop(
    prediction_proba: float, decision_type: str = "CREDIT_DENY"
) -> dict:
    """
    EU AI Act Compliance (Art 14): Enforces human oversight for high-risk decisions.
    """
    if not math.isfinite(prediction_proba) or not 0.0 <= prediction_proba <= 1.0:
        raise ValueError(
            "prediction_proba must be a finite probability between 0.0 and 1.0"
        )

    if decision_type in HIGH_RISK_ACTIONS:
        decision = {
            "action": "MANUAL_REVIEW_REQUIRED",
            "confidence": round(prediction_proba, 4),
            "compliance_status": "EU AI Act Art. 14: high-risk decision queued for human oversight.",
        }
    else:
        decision = {
            "action": "SCORE_ONLY",
            "confidence": round(prediction_proba, 4),
            "compliance_status": "Score emitted; no automated legal/credit effect.",
        }

    logging.info(
        f"Compliance Audit Log - Action: {decision['action']} | Status: {decision['compliance_status']}"
    )
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EU AI Act HITL Decision Router")
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--decision_type", type=str, default="CREDIT_DENY")
    args = parser.parse_args()

    result = enforce_human_in_the_loop(args.probability, args.decision_type)
    print(json.dumps(result, indent=2))
