"""Create a deterministic labelled incoming batch from the Iris dataset.

The first 120 Iris rows represent the original training corpus. The remaining
30 rows are treated as a later incoming labelled batch for this practical.
This gives the demonstration a reproducible, held-out monitoring dataset.
"""
import pandas as pd
from sklearn.datasets import load_iris

from monitoring.config import DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN


def main() -> None:
    iris = load_iris(as_frame=True)
    frame = iris.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": TARGET_COLUMN,
        }
    )
    # Stable split: the new batch is separate from the original training rows.
    original = frame.iloc[:120].copy()
    incoming = frame.iloc[120:].copy()
    incoming.insert(0, "record_id", [f"iris-new-{i:03d}" for i in range(1, len(incoming) + 1)])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    original[FEATURE_COLUMNS + [TARGET_COLUMN]].to_csv(
        DATA_DIR / "original_train.csv", index=False
    )
    incoming.to_csv(DATA_DIR / "incoming_batch.csv", index=False)

    print(f"Original training rows: {len(original)}")
    print(f"Incoming labelled rows: {len(incoming)}")
    print("Incoming target distribution:")
    print(incoming[TARGET_COLUMN].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
