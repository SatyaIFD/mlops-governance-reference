from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier

from src.features.velocity import CANONICAL_AML_FEATURES

TRUSTED_SKOPS_TYPES = [
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.tree._tree.Tree",
]


class AMLAnomalyDetector:
    """Supervised anomaly detection classifier tuned for streaming AML detection."""

    def __init__(self, threshold: float = 0.40):
        self.feature_cols: List[str] = list(CANONICAL_AML_FEATURES)
        self.threshold: float = float(threshold)
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=50, max_depth=10, class_weight="balanced", random_state=42
        )

    def _align_features(
        self, df_or_dict: pd.DataFrame | Dict[str, Any]
    ) -> pd.DataFrame:
        active_cols = (
            list(self.model.feature_names_in_)
            if hasattr(self.model, "feature_names_in_")
            else self.feature_cols
        )
        if isinstance(df_or_dict, dict):
            row = {col: [float(df_or_dict.get(col, 0.0))] for col in active_cols}
            return pd.DataFrame(row)
        aligned = df_or_dict.copy()
        for col in active_cols:
            if col not in aligned.columns:
                aligned[col] = 0.0
        return aligned[active_cols]

    def train(self, df: pd.DataFrame) -> None:
        X = self._align_features(df)
        target_col = "Is_laundering" if "Is_laundering" in df.columns else "Target"
        y = df[target_col].astype(int)
        self.model.fit(X, y)

    def score_transaction(self, enriched_tx: Dict[str, Any]) -> Tuple[float, int]:
        X_single = self._align_features(enriched_tx)
        prob = float(self.model.predict_proba(X_single)[0][1])
        is_anomaly = 1 if prob >= self.threshold else 0
        return round(prob, 4), is_anomaly

    def save(self, artifact_path: Path | str) -> None:
        target = Path(artifact_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(self.model, target)

    def load(self, artifact_path: Path | str) -> None:
        target = (
            Path(artifact_path)
            if not isinstance(artifact_path, Path)
            else artifact_path
        )
        self.model = sio.load(
            target,
            trusted=TRUSTED_SKOPS_TYPES,
        )
