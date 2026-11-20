import pandas as pd
import pytest

from monitoring.pipeline import build_pipeline, validate_dataset
from monitoring.config import FEATURE_COLUMNS, TARGET_COLUMN


def valid_frame():
    return pd.DataFrame({
        "sepal_length": [5.1, 6.2, 6.3],
        "sepal_width": [3.5, 2.8, 2.9],
        "petal_length": [1.4, 4.8, 5.0],
        "petal_width": [0.2, 1.8, 1.9],
        "target": [0, 1, 2],
        "record_id": ["a", "b", "c"],
    })


def test_validation_accepts_valid_labelled_batch():
    validate_dataset(valid_frame(), require_unique_ids=True)


def test_validation_rejects_missing_values():
    frame = valid_frame()
    frame.loc[0, "sepal_length"] = None
    with pytest.raises(ValueError, match="Missing values"):
        validate_dataset(frame)


def test_validation_rejects_duplicate_ids():
    frame = valid_frame()
    frame.loc[1, "record_id"] = "a"
    with pytest.raises(ValueError, match="Duplicate record_id"):
        validate_dataset(frame, require_unique_ids=True)


def test_pipeline_fits_and_predicts_three_classes():
    frame = valid_frame()
    model = build_pipeline()
    model.fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN])
    assert len(model.predict(frame[FEATURE_COLUMNS])) == 3
