from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.predict_health.artifacts import blend_probability_artifacts, save_probability_artifacts
from src.predict_health.metrics import optimize_class_multipliers, predict_with_multipliers
from src.predict_health.submission import make_submission, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend saved probability artifacts from previous runs.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Run directories containing saved .npy artifacts.")
    parser.add_argument("--weights", default=None, help="Comma-separated blend weights. Defaults to equal weights.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multiplier-rounds", type=int, default=1500)
    return parser.parse_args()


def parse_weights(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"blend-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    weights = parse_weights(args.weights)
    blended = blend_probability_artifacts(args.runs, weights=weights)

    oof_proba = blended["oof_proba"]
    test_proba = blended["test_proba"]
    y_true = blended["y_true"]
    classes = blended["classes"]
    test_ids = blended["test_ids"]
    normalized_weights = blended["weights"]

    raw_oof_pred = classes[np.argmax(oof_proba, axis=1)]
    raw_score = float(balanced_accuracy_score(y_true, raw_oof_pred))
    multipliers, tuned_score = optimize_class_multipliers(
        y_true,
        oof_proba,
        classes,
        rounds=args.multiplier_rounds,
        random_state=args.seed,
    )
    tuned_oof_pred = predict_with_multipliers(oof_proba, classes, multipliers)
    tuned_score = float(balanced_accuracy_score(y_true, tuned_oof_pred))
    test_pred = predict_with_multipliers(test_proba, classes, multipliers)

    submission = make_submission(test_ids, test_pred)
    validate_submission(submission, sample)
    submission_path = run_dir / "submission.csv"
    metadata_path = run_dir / "metadata.json"
    submission.to_csv(submission_path, index=False)
    artifact_paths = save_probability_artifacts(run_dir, oof_proba, test_proba, y_true, classes, test_ids)

    metadata = {
        "run_name": run_name,
        "source_runs": [str(path) for path in args.runs],
        "weights": normalized_weights.tolist(),
        "classes": classes.tolist(),
        "raw_oof_balanced_accuracy": raw_score,
        "tuned_oof_balanced_accuracy": tuned_score,
        "multipliers": multipliers,
        "classification_report": classification_report(y_true, tuned_oof_pred, output_dict=True),
        "artifacts": artifact_paths,
        "submission_path": str(submission_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nBlend complete", flush=True)
    print(f"raw_oof_balanced_accuracy={raw_score:.6f}", flush=True)
    print(f"tuned_oof_balanced_accuracy={tuned_score:.6f}", flush=True)
    print(f"weights={normalized_weights.tolist()}", flush=True)
    print(f"multipliers={multipliers}", flush=True)
    print(f"submission={submission_path}", flush=True)
    print(f"metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
