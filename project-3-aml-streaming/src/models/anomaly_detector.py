import skops.io as sio
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier


class AMLAnomalyDetector:
    """Supervised anomaly detection classifier tuned for streaming AML detection."""

    def __init__(self):
        self.feature_cols: List[str] = [
            "Amount",
            "pass_through_ratio_1h",
            "receiver_inflow_count_1h",
            "receiver_inflow_amount_1h",
        ]
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=50, max_depth=10, class_weight="balanced", random_state=42
        )

    def train(self, df: pd.DataFrame) -> None:
        X = df[self.feature_cols]
        y = df["Is_laundering"].astype(int)
        self.model.fit(X, y)

    def score_transaction(self, enriched_tx: Dict[str, Any]) -> Tuple[float, int]:
        feature_dict = {
            col: [float(enriched_tx.get(col, 0.0))] for col in self.feature_cols
        }
        X_single = pd.DataFrame(feature_dict)
        prob = float(self.model.predict_proba(X_single)[0][1])
        is_anomaly = 1 if prob >= 0.5 else 0
        return round(prob, 4), is_anomaly

    def save(self, artifact_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(self.model, artifact_path)

    def load(self, artifact_path: Path) -> None:
        # CVE-2024-37065 Fix: Dynamically read unknown types instead of trusted=True
        untrusted = sio.get_untrusted_types(file=artifact_path)
        self.model = sio.load(artifact_path, trusted=untrusted)
