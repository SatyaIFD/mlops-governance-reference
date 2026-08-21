import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def enforce_human_in_the_loop(prediction_proba: float, auto_reject_threshold: float = 0.80, manual_review_floor: float = 0.45) -> dict:
    """
    EU AI Act Compliance: Routes borderline predictions to a human auditor queue.
    Prevents High-Risk AI from autonomously making marginal, high-impact decisions.
    """
    if prediction_proba >= auto_reject_threshold:
        decision = {
            "action": "AUTO_REJECT",
            "confidence": prediction_proba,
            "compliance_status": "Automated action executed safely above threshold."
        }
    elif manual_review_floor <= prediction_proba < auto_reject_threshold:
        decision = {
            "action": "MANUAL_REVIEW_REQUIRED",
            "confidence": prediction_proba,
            "compliance_status": "EU AI Act HITL Triggered. Decision routed to human auditor."
        }
    else:
        decision = {
            "action": "AUTO_APPROVE",
            "confidence": prediction_proba,
            "compliance_status": "Automated action executed safely below threshold."
        }
    
    logging.info(f"Compliance Audit Log - Action: {decision['action']} | Status: {decision['compliance_status']}")
    return decision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EU AI Act HITL Decision Router")
    parser.add_argument("--probability", type=float, required=True, help="Model's predicted probability (0.0 to 1.0)")
    args = parser.parse_args()
    
    result = enforce_human_in_the_loop(args.probability)
    print(json.dumps(result, indent=2))
