import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
AML_DIR = ROOT_DIR / "project-3-aml-streaming"
if str(AML_DIR) not in sys.path:
    sys.path.insert(0, str(AML_DIR))

from src.models.anomaly_detector import AMLAnomalyDetector  # noqa: E402

print("Training baseline 7-feature Random Forest...")
df = pd.DataFrame(
    {
        "Amount": [50.0, 9500.0, 10.0, 25000.0, 120.0, 8900.0],
        "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0, 0.05, 0.92],
        "is_structuring": [0, 1, 0, 0, 0, 1],
        "velocity_acceleration": [1.0, 12.5, 0.8, 18.0, 1.1, 9.5],
        "fan_out_count_24h": [1, 8, 1, 14, 2, 6],
        "receiver_inflow_count_1h": [1, 5, 1, 15, 1, 7],
        "receiver_inflow_amount_1h": [50.0, 45000.0, 10.0, 150000.0, 120.0, 62000.0],
        "Is_laundering": [0, 1, 0, 1, 0, 1],
    }
)

detector = AMLAnomalyDetector(threshold=0.40)
detector.train(df)

print("Serializing secure skops artifact...")
detector.save(ROOT_DIR / "aml_streaming_model")
print("✅ Secure 7-feature model 'aml_streaming_model' generated successfully!")
