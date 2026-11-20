"""Train the first local champion when the repository has no compatible artifact."""
from monitoring.config import MODEL_DIR, DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN
from monitoring.pipeline import build_pipeline, load_csv, save_model, validate_dataset

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split


def main() -> None:
    frame = load_csv(DATA_DIR / "original_train.csv")
    validate_dataset(frame)
    X, y = frame[FEATURE_COLUMNS], frame[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    # Use the untouched holdout only for baseline validation.
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }

    save_model(model, MODEL_DIR / "champion.joblib")
    mlflow.set_experiment("iris-model-monitoring")
    with mlflow.start_run(run_name="initial-champion") as run:
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "random_state": 42,
            "training_rows": len(X_train),
        })
        mlflow.log_metrics({
            key: metrics[key]
            for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        })
        mlflow.sklearn.log_model(model, name="iris-champion")
        print(f"MLflow run ID: {run.info.run_id}")

    print("Baseline champion trained and saved:", MODEL_DIR / "champion.joblib")
    print({key: metrics[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")})


if __name__ == "__main__":
    main()
