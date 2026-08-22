import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

class MLSystemCircuitBreaker:
    """
    DORA Compliance: Operational Resilience & Fault Tolerance.
    Implements a Circuit Breaker pattern to prevent cascading failures 
    during model inference degradation or data stream outages.
    """
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.state = "CLOSED"  # CLOSED = Healthy, OPEN = Tripped/Failing

    def record_failure(self):
        self.consecutive_failures += 1
        logging.warning(f"Inference failure recorded. Consecutive failures: {self.consecutive_failures}")
        
        if self.consecutive_failures >= self.failure_threshold and self.state == "CLOSED":
            self.trip_circuit()

    def record_success(self):
        if self.state == "OPEN":
            logging.info("System stabilized. Resetting circuit breaker to CLOSED.")
        self.consecutive_failures = 0
        self.state = "CLOSED"

    def trip_circuit(self):
        self.state = "OPEN"
        logging.error("🚨 DORA CIRCUIT BREAKER TRIPPED! Cascading failure prevented.")
        logging.info("Action: Rerouting traffic to rules-based fallback system.")

    def execute_inference(self, model_function, *args, **kwargs):
        """Wrapper to safely execute ML predictions."""
        if self.state == "OPEN":
            logging.warning("Circuit is OPEN. Executing fallback heuristic instead of ML model.")
            return {"status": "fallback", "prediction": "DEFAULT_SAFE_VALUE"}
            
        try:
            result = model_function(*args, **kwargs)
            self.record_success()
            return {"status": "success", "prediction": result}
        except Exception as e:
            self.record_failure()
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DORA Operational Resilience Simulator")
    parser.add_argument("--simulate_failures", type=int, required=True, help="Number of consecutive failures to simulate")
    args = parser.parse_args()
    
    breaker = MLSystemCircuitBreaker(failure_threshold=3)
    
    for i in range(args.simulate_failures):
        try:
            # Simulating a failing model function
            breaker.execute_inference(lambda: int("trigger_error"))
        except:
            pass
            
    if breaker.state == "OPEN":
        print("\n✅ DORA Compliance Check Passed: System successfully prevented cascading failure.")
