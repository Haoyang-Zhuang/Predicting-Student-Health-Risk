from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def predict_with_multipliers(
    proba: np.ndarray,
    classes: np.ndarray,
    multipliers: Mapping[str, float],
) -> np.ndarray:
    """Apply class-specific multipliers before argmax."""

    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    adjusted = proba.copy()
    for idx, cls in enumerate(classes):
        adjusted[:, idx] *= float(multipliers.get(str(cls), 1.0))
    return classes[np.argmax(adjusted, axis=1)]


@dataclass(frozen=True)
class MultiplierSearchResult:
    multipliers: dict[str, float]
    score: float


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def optimize_class_multipliers(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    rounds: int = 200,
    random_state: int = 42,
) -> tuple[dict[str, float], float]:
    """Search for class multipliers that maximize balanced accuracy on OOF data."""

    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    rng = np.random.default_rng(random_state)

    def eval_multipliers(multipliers: dict[str, float]) -> float:
        pred = predict_with_multipliers(proba, classes, multipliers)
        return _score(y_true, pred)

    best = {str(cls): 1.0 for cls in classes}
    best_score = eval_multipliers(best)

    grid = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
    for _ in range(3):
        improved = False
        for cls in classes:
            cls_key = str(cls)
            local_best = best[cls_key]
            for value in grid:
                cand = dict(best)
                cand[cls_key] = float(value)
                score = eval_multipliers(cand)
                if score > best_score + 1e-12:
                    best = cand
                    best_score = score
                    local_best = value
                    improved = True
            best[cls_key] = float(local_best)
        if not improved:
            break

    for _ in range(max(0, rounds)):
        cand = dict(best)
        for cls in classes:
            cls_key = str(cls)
            spread = 0.18
            if cls_key != str(classes[0]):
                spread = 0.28
            cand[cls_key] = float(np.clip(best[cls_key] * np.exp(rng.normal(0.0, spread)), 0.2, 5.0))
        score = eval_multipliers(cand)
        if score > best_score + 1e-12:
            best = cand
            best_score = score

    return best, best_score

