# Iris model monitoring and retraining practical

This extension implements the 5 assessment tasks using the repository's Iris
classification project, Random Forest model, and MLflow experiment tracking.

## Workflow

1. **Prepare data** — the Iris dataset is split reproducibly into 120 original
   training rows and 30 labelled incoming rows. The incoming CSV carries unique
   `record_id` values. The workflow checks required columns, nulls, duplicate
   rows/IDs, numeric positive measurements, row count and target distribution.
2. **Evaluate champion** — loads `models/champion.joblib` and evaluates it on
   the labelled incoming batch. It records accuracy, macro precision, macro
   recall, macro F1, confusion matrix and per-class report.
3. **Retrain candidate** — combines original training rows with incoming rows
   and trains a separate pipeline. Median imputation and standard scaling are
   inside the scikit-learn Pipeline; the Random Forest follows them. The same
   pipeline is used for baseline and candidate.
4. **Quality gate** — requires candidate macro-F1 >= 0.90, accuracy >= 0.90,
   and candidate macro-F1 no worse than champion macro-F1 on the same incoming
   batch. The report records both metric sets and the decision.
5. **Manual promotion** — candidate is never copied over champion by the
   monitoring command. Promotion requires a passing gate and an explicit
   `--confirm`; the previous champion is backed up and an audit JSON is written.

### Why macro-F1?

The target has three Iris classes. Macro-F1 computes F1 per class and averages
them equally, so performance on a smaller or weaker class is not hidden by
the other classes. Accuracy is also used as a minimum guardrail, while precision
and recall are reported to make error trade-offs visible.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-monitoring.txt

# Task A: create the two CSV files
python -m monitoring.prepare_data

# Start MLflow tracking server in another terminal
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 6001

# Create initial champion artifact and tracked baseline
python -m monitoring.train_initial

# Tasks A-E: evaluate, retrain, report and record quality gate
python -m monitoring.run_workflow --tracking-uri http://127.0.0.1:6001
```

Outputs:
- `data/original_train.csv` — historical labelled data
- `data/incoming_batch.csv` — new labelled batch
- `models/champion.joblib` — existing champion used by the API
- `artifacts/candidate.joblib` — newly trained candidate
- `reports/evaluation_report.json` — full metrics and quality-gate decision
- MLflow experiment `iris-model-monitoring` — parameters, metrics, candidate
  artifact, report and gate tags

### Optional explicit promotion

Review the report first. Only if it says `ELIGIBLE_FOR_PROMOTION`, deliberately run:

```bash
python -m monitoring.promote_candidate --confirm
```

The command backs up the old champion to
`models/champion.previous.joblib` and records
`reports/promotion_audit.json`. The monitoring run itself never promotes.

## Assessment evidence

Capture terminal output showing:
- incoming batch row count and target distribution;
- champion and candidate metrics;
- the quality-gate decision;
- the candidate file and evaluation report;
- the MLflow experiment run.

Open the report with:

```bash
python -m json.tool reports/evaluation_report.json
```

## Notes

The repository's pre-existing `models/iris_model.joblib` is an older standalone estimator and is not the champion artifact expected by this monitoring extension. Create a champion pipeline with the baseline-training command below before running monitoring.\n\nThe 30-row incoming batch is a reproducible demonstration batch, not a claim
that these records arrived from a live production stream. For a real deployment,
replace `data/incoming_batch.csv` with a new, independently labelled batch
using the same four feature columns and target encoding (0=setosa, 1=versicolor,
2=virginica). The workflow intentionally fails validation rather than silently
dropping missing or duplicate data.
