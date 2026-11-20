"""Explicit, auditable manual promotion after reviewing the quality report."""
import argparse
import json
from datetime import datetime, timezone

from monitoring.config import ARTIFACT_DIR, MODEL_DIR, REPORT_DIR
from monitoring.pipeline import load_model, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Required to copy the candidate over champion.")
    args = parser.parse_args()

    report_path = REPORT_DIR / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Run monitoring first; evaluation report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("eligible_for_promotion"):
        raise SystemExit("Promotion blocked: quality gate did not pass.")
    if not args.confirm:
        raise SystemExit("Candidate is eligible, but champion was not changed. Re-run with --confirm after review.")

    candidate = load_model(ARTIFACT_DIR / "candidate.joblib")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = MODEL_DIR / "champion.previous.joblib"
    if (MODEL_DIR / "champion.joblib").exists():
        from shutil import copy2
        copy2(MODEL_DIR / "champion.joblib", backup_path)
    save_model(candidate, MODEL_DIR / "champion.joblib")

    audit = {
        "promotion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PROMOTED_WITH_EXPLICIT_CONFIRMATION",
        "previous_champion_backup": str(backup_path) if backup_path.exists() else None,
        "new_champion": str(MODEL_DIR / "champion.joblib"),
        "quality_report": str(report_path),
    }
    (REPORT_DIR / "promotion_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
