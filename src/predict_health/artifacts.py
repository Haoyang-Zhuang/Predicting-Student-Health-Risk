from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

ARTIFACT_FILENAMES = {
    "oof_proba": "oof_proba.npy",
    "test_proba": "test_proba.npy",
    "y_true": "y_true.npy",
    "classes": "classes.npy",
    "test_ids": "test_ids.npy",
}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=float).copy()
    row_sums = normalized.sum(axis=1, keepdims=True)
    np.divide(normalized, row_sums, out=normalized, where=row_sums > 0)
    return normalized


def save_probability_artifacts(
    run_dir: Path,
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    y_true: Sequence,
    classes: Sequence,
    test_ids: Sequence,
) -> dict[str, str]:
    """Save reusable OOF/test probability artifacts for later blending."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "oof_proba": np.asarray(oof_proba, dtype=float),
        "test_proba": np.asarray(test_proba, dtype=float),
        "y_true": np.asarray(y_true, dtype=str),
        "classes": np.asarray(classes, dtype=str),
        "test_ids": np.asarray(test_ids),
    }
    paths = {}
    for key, filename in ARTIFACT_FILENAMES.items():
        path = run_dir / filename
        np.save(path, payload[key])
        paths[key] = str(path)
    return paths


def load_probability_artifacts(run_dir: Path) -> dict[str, np.ndarray]:
    """Load OOF/test probability artifacts from a previous training run."""

    run_dir = Path(run_dir)
    missing = [name for name in ARTIFACT_FILENAMES.values() if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing artifact files in {run_dir}: {missing}")
    return {
        key: np.load(run_dir / filename, allow_pickle=False)
        for key, filename in ARTIFACT_FILENAMES.items()
    }


def blend_probability_artifacts(
    run_dirs: Sequence[Path],
    weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Blend saved run probabilities after verifying class and row alignment."""

    if not run_dirs:
        raise ValueError("at least one run directory is required")
    loaded = [load_probability_artifacts(Path(run_dir)) for run_dir in run_dirs]
    first = loaded[0]

    for item in loaded[1:]:
        for key in ["y_true", "classes", "test_ids"]:
            if not np.array_equal(first[key], item[key]):
                raise ValueError(f"cannot blend runs with mismatched {key}")
        if first["oof_proba"].shape != item["oof_proba"].shape:
            raise ValueError("cannot blend runs with mismatched OOF probability shapes")
        if first["test_proba"].shape != item["test_proba"].shape:
            raise ValueError("cannot blend runs with mismatched test probability shapes")

    if weights is None:
        weights_array = np.ones(len(loaded), dtype=float)
    else:
        weights_array = np.asarray(weights, dtype=float)
    if len(weights_array) != len(loaded):
        raise ValueError("number of weights must match number of run directories")
    if np.any(weights_array < 0):
        raise ValueError("weights must be non-negative")
    if weights_array.sum() <= 0:
        raise ValueError("at least one weight must be positive")
    weights_array = weights_array / weights_array.sum()

    oof_stack = np.stack([item["oof_proba"] for item in loaded], axis=0)
    test_stack = np.stack([item["test_proba"] for item in loaded], axis=0)
    oof_proba = _normalize_rows(np.tensordot(weights_array, oof_stack, axes=(0, 0)))
    test_proba = _normalize_rows(np.tensordot(weights_array, test_stack, axes=(0, 0)))

    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "y_true": first["y_true"],
        "classes": first["classes"],
        "test_ids": first["test_ids"],
        "weights": weights_array,
    }
