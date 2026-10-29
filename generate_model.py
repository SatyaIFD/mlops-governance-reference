import pandas as pd
from pathlib import Path
from src.models.anomaly_detector import AMLAnomalyDetector

print("Training baseline Random Forest...")
# Create baseline training data matching the feature schema
df = pd.DataFrame(
    {
        "Amount": [50.0, 9500.0, 10.0, 25000.0],
        "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0],
        "receiver_inflow_count_1h": [1, 5, 1, 15],
        "receiver_inflow_amount_1h": [50.0, 45000.0, 10.0, 150000.0],
        "Is_laundering": [0, 1, 0, 1],
    }
)

detector = AMLAnomalyDetector()
detector.train(df)

print("Serializing secure skops artifact...")
# Save it to the root directory where the Docker container expects it
detector.save(Path("aml_streaming_model"))
print("✅ Secure model 'aml_streaming_model' generated successfully!")
