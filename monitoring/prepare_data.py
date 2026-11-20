"""Create a reproducible incoming batch balanced across all Iris classes.

The dataset is stratified by class before splitting: each class contributes
40 historical rows and 10 incoming labelled rows. The new batch therefore
contains 30 rows with class labels 0, 1 and 2 represented equally.
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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

    original, incoming = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame[TARGET_COLUMN],
    )
    incoming = incoming.copy().reset_index(drop=True)
    original = original.copy().reset_index(drop=True)
    incoming.insert(
        0,
        "record_id",
        [f"iris-new-{i:03d}" for i in range(1, len(incoming) + 1)],
    )

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
