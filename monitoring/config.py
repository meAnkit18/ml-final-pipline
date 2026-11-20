"""Configuration shared by the Iris monitoring and retraining workflow."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "target"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Promotion uses macro F1 so every Iris class contributes equally.
MIN_MACRO_F1 = 0.90
MIN_ACCURACY = 0.90
