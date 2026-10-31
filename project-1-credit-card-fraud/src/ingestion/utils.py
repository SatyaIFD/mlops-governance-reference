import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

CANONICAL_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + [
    "scaled_amount",
    "scaled_time",
]


def fit_scalers_on_training_split(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits RobustScaler (Amount) and StandardScaler (Time) strictly on the training split
    to prevent train/test preprocessing leakage, returning JSON-serializable parameters.
    """
    robust_scaler = RobustScaler()
    standard_scaler = StandardScaler()

    robust_scaler.fit(train_df["Amount"].values.reshape(-1, 1))
    standard_scaler.fit(train_df["Time"].values.reshape(-1, 1))

    return {
        "amount_center": float(robust_scaler.center_[0]),
        "amount_scale": float(robust_scaler.scale_[0])
        if float(robust_scaler.scale_[0]) != 0.0
        else 1.0,
        "time_mean": float(standard_scaler.mean_[0]),
        "time_scale": float(standard_scaler.scale_[0])
        if float(standard_scaler.scale_[0]) != 0.0
        else 1.0,
    }


def save_scaler_params(scaler_params: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)


def load_scaler_params(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_and_transform_features(
    df: pd.DataFrame,
    scaler_params: Dict[str, float] | None = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Applies scaling adjustments to raw transactional inputs.
    If scaler_params is provided, uses pre-fitted training statistics to prevent test leakage.
    """
    processed_df = df.copy()

    if drop_duplicates:
        initial_rows = len(processed_df)
        processed_df.drop_duplicates(inplace=True)
        purged_rows = initial_rows - len(processed_df)
        if purged_rows > 0:
            print(
                f"🧹 Purged {purged_rows:,} exact duplicate entries from execution block."
            )

    if scaler_params is None:
        scaler_params = fit_scalers_on_training_split(processed_df)

    amount_scale = scaler_params["amount_scale"] or 1.0
    time_scale = scaler_params["time_scale"] or 1.0

    processed_df["scaled_amount"] = (
        processed_df["Amount"] - scaler_params["amount_center"]
    ) / amount_scale
    processed_df["scaled_time"] = (
        processed_df["Time"] - scaler_params["time_mean"]
    ) / time_scale

    processed_df.drop(["Time", "Amount"], axis=1, inplace=True)

    ordered_cols = list(CANONICAL_FEATURE_COLS)
    if "Class" in processed_df.columns:
        ordered_cols.append("Class")

    return processed_df[ordered_cols]


def split_and_scale_without_leakage(
    raw_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Deduplicates raw data, performs a stratified train/test split FIRST, fits scalers
    exclusively on the training split, and transforms both splits without leakage.
    """
    from sklearn.model_selection import train_test_split

    dedup_df = raw_df.drop_duplicates().reset_index(drop=True)
    stratify_col = dedup_df["Class"] if "Class" in dedup_df.columns else None

    train_raw, test_raw = train_test_split(
        dedup_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    scaler_params = fit_scalers_on_training_split(train_raw)
    train_scaled = clean_and_transform_features(
        train_raw, scaler_params=scaler_params, drop_duplicates=False
    )
    test_scaled = clean_and_transform_features(
        test_raw, scaler_params=scaler_params, drop_duplicates=False
    )
    return train_scaled, test_scaled, scaler_params
