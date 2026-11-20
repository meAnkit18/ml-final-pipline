"""Evaluate the champion, retrain a candidate, and record a quality-gate decision."""
import argparse
import json
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from monitoring.config import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    DATA_DIR,
    FEATURE_COLUMNS,
    MIN_ACCURACY,
    MIN_MACRO_F1,
    MODEL_DIR,
    REPORT_DIR,
    TARGET_COLUMN,
)
from monitoring.pipeline import (
    build_pipeline,
    load_csv,
    load_model,
    save_model,
    validate_dataset,
)


def calculate_metrics(model, frame: pd.DataFrame) -> dict:
    predictions = model.predict(frame[FEATURE_COLUMNS])
    truth = frame[TARGET_COLUMN]
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_macro": float(precision_score(truth, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            truth, predictions, labels=[0, 1, 2], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", default=str(DATA_DIR / "incoming_batch.csv"))
    parser.add_argument("--original", default=str(DATA_DIR / "original_train.csv"))
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("iris-model-monitoring")

    incoming = load_csv(__import__("pathlib").Path(args.incoming))
    validate_dataset(incoming, require_unique_ids="record_id" in incoming.columns)
    original = load_csv(__import__("pathlib").Path(args.original))
    validate_dataset(original)

    print(f"Incoming dataset shape: {incoming.shape}")
    print(f"Duplicate rows: {int(incoming.duplicated(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).sum())}")
    print("Target distribution:")
    print(incoming[TARGET_COLUMN].value_counts().sort_index().to_string())

    # Task B: Evaluate the currently deployed champion on the newly labelled batch.
    champion_path = MODEL_DIR / "champion.joblib"
    champion = load_model(champion_path)
    champion_metrics = calculate_metrics(champion, incoming)

    # Task C: Retrain from combined historical + newly labelled data.
    combined = pd.concat(
        [original[FEATURE_COLUMNS + [TARGET_COLUMN]], incoming[FEATURE_COLUMNS + [TARGET_COLUMN]]],
        ignore_index=True,
    )
    validate_dataset(combined)
    candidate = build_pipeline()
    candidate.fit(combined[FEATURE_COLUMNS], combined[TARGET_COLUMN])
    candidate_path = ARTIFACT_DIR / "candidate.joblib"
    save_model(candidate, candidate_path)
    candidate_metrics = calculate_metrics(candidate, incoming)

    # Task D: A candidate must satisfy both thresholds and not regress against
    # the champion on the primary metric (macro F1). Candidate remains separate.
    eligible = (
        candidate_metrics["f1_macro"] >= MIN_MACRO_F1
        and candidate_metrics["accuracy"] >= MIN_ACCURACY
        and candidate_metrics["f1_macro"] >= champion_metrics["f1_macro"]
    )
    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "champion_artifact": str(champion_path),
        "candidate_artifact": str(candidate_path),
        "eligible_for_promotion": bool(eligible),
        "decision": "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAIN_CHAMPION",
        "criteria": {
            "primary_metric": "f1_macro",
            "minimum_macro_f1": MIN_MACRO_F1,
            "minimum_accuracy": MIN_ACCURACY,
            "must_not_underperform_champion_on_macro_f1": True,
        },
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "training_rows": int(len(combined)),
        "incoming_rows": int(len(incoming)),
        "note": "Eligibility is recorded only. Champion is never overwritten automatically.",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "evaluation_report.json"
    report_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Track a complete run with both models and the quality-gate decision.
    with mlflow.start_run(run_name="monitor-and-retrain") as run:
        mlflow.log_params({
            "training_rows": len(combined),
            "incoming_rows": len(incoming),
            "model_type": "RandomForestClassifier",
            "gate_primary_metric": "f1_macro",
        })
        mlflow.log_metrics({
            "champion_accuracy": champion_metrics["accuracy"],
            "champion_f1_macro": champion_metrics["f1_macro"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "candidate_f1_macro": candidate_metrics["f1_macro"],
        })
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(candidate_path))
        mlflow.sklearn.log_model(candidate, name="candidate-model")
        mlflow.set_tag("quality_gate_decision", decision["decision"])
        mlflow.set_tag("candidate_eligible", str(eligible).lower())
        decision["mlflow_run_id"] = run.info.run_id

    # Re-write report after adding the MLflow run ID. Still do not promote.
    report_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\n=== MODEL QUALITY REPORT ===")
    print(f"Champion: accuracy={champion_metrics['accuracy']:.4f}, macro-F1={champion_metrics['f1_macro']:.4f}")
    print(f"Candidate: accuracy={candidate_metrics['accuracy']:.4f}, macro-F1={candidate_metrics['f1_macro']:.4f}")
    print(f"Decision: {decision['decision']}")
    print(f"Candidate artifact: {candidate_path}")
    print(f"Evaluation report: {report_path}")
    print(f"MLflow run ID: {decision['mlflow_run_id']}")


if __name__ == "__main__":
    main()
