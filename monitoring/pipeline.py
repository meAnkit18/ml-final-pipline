"""Shared data loading, validation, preprocessing and model construction."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from monitoring.config import FEATURE_COLUMNS, TARGET_COLUMN


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build the exact preprocessing + estimator pipeline used for every model."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_dataset(
    frame: pd.DataFrame,
    *,
    labelled: bool = True,
    require_unique_ids: bool = False,
) -> None:
    """Fail fast on missing columns/values, invalid measurements and duplicates."""
    required = set(FEATURE_COLUMNS)
    if labelled:
        required.add(TARGET_COLUMN)
    if require_unique_ids:
        required.add("record_id")

    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset is empty")

    checked_columns = FEATURE_COLUMNS + ([TARGET_COLUMN] if labelled else [])
    null_counts = frame[checked_columns].isna().sum()
    missing_values = {key: int(value) for key, value in null_counts.items() if value}
    if missing_values:
        raise ValueError(f"Missing values found: {missing_values}")

    if require_unique_ids:
        if frame["record_id"].isna().any():
            raise ValueError("record_id contains missing values")
        if frame["record_id"].duplicated().any():
            raise ValueError("Duplicate record_id values found")

    duplicates = int(frame.duplicated(subset=checked_columns).sum())
    if duplicates:
        raise ValueError(f"Duplicate data records found: {duplicates}")

    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Feature '{column}' must be numeric")
        if (frame[column] <= 0).any():
            raise ValueError(f"Feature '{column}' must contain positive values")

    if labelled:
        if not pd.api.types.is_integer_dtype(frame[TARGET_COLUMN]):
            raise ValueError("target must contain integer class labels 0, 1 or 2")
        invalid = sorted(set(frame[TARGET_COLUMN].unique()) - {0, 1, 2})
        if invalid:
            raise ValueError(f"Unsupported target labels: {invalid}")


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run the project's training step first."
        )
    return joblib.load(path)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
