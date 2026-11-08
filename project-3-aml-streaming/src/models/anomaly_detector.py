import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
import skops
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
        self.feature_cols: list[str] = list(CANONICAL_AML_FEATURES)
        self.threshold: float = float(threshold)
        self.model_sha256: str | None = None
        self.mlflow_run_id: str | None = None
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )

    @staticmethod
    def _meta_path_for(artifact_path: Path) -> Path:
        return artifact_path.with_name(artifact_path.name + ".meta.json")

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_active_feature_cols(self) -> list[str]:
        if hasattr(self.model, "feature_names_in_"):
            return [str(c) for c in self.model.feature_names_in_]
        return list(self.feature_cols)

    def _align_features(
        self, df_or_dict: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]
    ) -> pd.DataFrame:
        active_cols = self._get_active_feature_cols()
        if isinstance(df_or_dict, dict):
            missing = [col for col in active_cols if col not in df_or_dict]
            if missing:
                raise KeyError(f"Missing required AML features: {missing}")
            row = {col: [float(df_or_dict[col])] for col in active_cols}
            return pd.DataFrame(row, columns=active_cols)
        if isinstance(df_or_dict, list):
            rows = []
            for item in df_or_dict:
                missing = [col for col in active_cols if col not in item]
                if missing:
                    raise KeyError(f"Missing required AML features: {missing}")
                rows.append([float(item[col]) for col in active_cols])
            return pd.DataFrame(rows, columns=active_cols)
        missing = [col for col in active_cols if col not in df_or_dict.columns]
        if missing:
            raise KeyError(f"Missing required AML features: {missing}")
        return df_or_dict[active_cols].copy()

    def train(self, df: pd.DataFrame) -> None:
        X = self._align_features(df)
        target_col = "Is_laundering" if "Is_laundering" in df.columns else "Target"
        y = df[target_col].astype(int)
        self.model.fit(X, y)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

    def score_transaction(self, enriched_tx: dict[str, Any]) -> tuple[float, int]:
        X_single = self._align_features(enriched_tx)
        prob = float(self.model.predict_proba(X_single)[0][1])
        is_anomaly = 1 if prob >= self.threshold else 0
        return round(prob, 4), is_anomaly

    def score_batch(
        self, enriched_batch: list[dict[str, Any]]
    ) -> list[tuple[float, int]]:
        if not enriched_batch:
            return []
        X_batch = self._align_features(enriched_batch)
        probs = self.model.predict_proba(X_batch)[:, 1]
        thresh = self.threshold
        return [(round(float(p), 4), 1 if p >= thresh else 0) for p in probs]

    def save(self, artifact_path: Path | str, mlflow_run_id: str | None = None) -> None:
        target = Path(artifact_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        sio.dump(self.model, target)
        self.model_sha256 = self._compute_sha256(target)
        self.mlflow_run_id = mlflow_run_id
        active_cols = self._get_active_feature_cols()
        meta_path = self._meta_path_for(target)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold": self.threshold,
                    "feature_cols": active_cols,
                    "sha256": self.model_sha256,
                    "mlflow_run_id": self.mlflow_run_id,
                    "sklearn_version": sklearn.__version__,
                    "skops_version": skops.__version__,
                },
                f,
                indent=2,
            )

    def load(self, artifact_path: Path | str) -> None:
        target = (
            Path(artifact_path)
            if not isinstance(artifact_path, Path)
            else artifact_path
        )
        meta_path = self._meta_path_for(target)
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Required model metadata sidecar missing: {meta_path}"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "threshold" not in meta:
            raise ValueError(f"Missing 'threshold' key in {meta_path}")
        expected_sha256 = meta.get("sha256")
        if not expected_sha256:
            raise ValueError(f"Missing 'sha256' integrity digest in {meta_path}")
        meta_sklearn = meta.get("sklearn_version")
        if meta_sklearn and meta_sklearn != sklearn.__version__:
            raise ValueError(
                f"scikit-learn version mismatch for {target}: artifact trained with {meta_sklearn}, runtime is {sklearn.__version__}"
            )
        meta_skops = meta.get("skops_version")
        if meta_skops and meta_skops != skops.__version__:
            raise ValueError(
                f"skops version mismatch for {target}: artifact saved with {meta_skops}, runtime is {skops.__version__}"
            )

        actual_sha256 = self._compute_sha256(target)
        if expected_sha256 != actual_sha256:
            raise ValueError(
                f"Model SHA-256 mismatch for {target}: expected {expected_sha256}, got {actual_sha256}"
            )

        self.model = sio.load(
            target,
            trusted=TRUSTED_SKOPS_TYPES,
        )
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.threshold = float(meta["threshold"])
        if "feature_cols" in meta and isinstance(meta["feature_cols"], list):
            self.feature_cols = [str(c) for c in meta["feature_cols"]]
        self.model_sha256 = actual_sha256
        self.mlflow_run_id = meta.get("mlflow_run_id")
