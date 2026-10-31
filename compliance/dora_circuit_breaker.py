import threading
import time
import logging
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)


class MLSystemCircuitBreaker:
    """
    DORA Compliance: Operational Resilience & Fault Tolerance.
    Implements a Thread-Safe Circuit Breaker pattern with exclusive HALF_OPEN probing.
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout_s: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self._lock = threading.Lock()
        self.consecutive_failures = 0
        self.state = "CLOSED"
        self.opened_at = 0.0
        self._half_open_probe_in_flight = False

    def _fallback(self):
        return {"status": "fallback", "prediction": None, "reason": "circuit_open"}

    def execute_inference(self, model_function, *args, **kwargs):
        """Wrapper to safely execute ML predictions with exclusive half-open concurrency locks."""
        with self._lock:
            if self.state == "OPEN":
                if time.monotonic() - self.opened_at < self.recovery_timeout_s:
                    logging.warning("Circuit is OPEN. Executing fallback heuristic.")
                    return self._fallback()
                # Timeout expired: transition to HALF_OPEN and reserve the single probe slot
                self.state = "HALF_OPEN"
                self._half_open_probe_in_flight = True
            elif self.state == "HALF_OPEN":
                if self._half_open_probe_in_flight:
                    logging.warning(
                        "Circuit is HALF_OPEN with an active probe in flight. Rejecting concurrent request."
                    )
                    return self._fallback()
                self._half_open_probe_in_flight = True

        try:
            result = model_function(*args, **kwargs)
        except Exception:
            with self._lock:
                self.consecutive_failures += 1
                self._half_open_probe_in_flight = False
                if (
                    self.consecutive_failures >= self.failure_threshold
                    or self.state == "HALF_OPEN"
                ):
                    if self.state != "OPEN":
                        logging.error("🚨 DORA CIRCUIT BREAKER TRIPPED!")
                    self.state = "OPEN"
                    self.opened_at = time.monotonic()
            return self._fallback()

        # Probe or normal execution succeeded: reset circuit completely
        with self._lock:
            if self.state != "CLOSED":
                logging.info("System stabilized. Resetting circuit breaker to CLOSED.")
            self.consecutive_failures = 0
            self._half_open_probe_in_flight = False
            self.state = "CLOSED"

        return {"status": "success", "prediction": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DORA Operational Resilience Simulator"
    )
    parser.add_argument("--simulate_failures", type=int, required=True)
    args = parser.parse_args()

    breaker = MLSystemCircuitBreaker(failure_threshold=3)
    for _ in range(args.simulate_failures):
        breaker.execute_inference(lambda: int("trigger_error"))

    if breaker.state == "OPEN":
        print(
            "\n✅ DORA Compliance Check Passed: System successfully prevented cascading failure."
        )
