from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def class_weight_map(y: Sequence[Any]) -> dict[Any, float]:
    """Return sklearn-style balanced class weights."""

    labels, counts = np.unique(np.asarray(y), return_counts=True)
    n_samples = counts.sum()
    n_classes = len(labels)
    return {label.item() if hasattr(label, "item") else label: float(n_samples / (n_classes * count)) for label, count in zip(labels, counts)}


def align_probabilities(
    proba: np.ndarray,
    model_classes: Sequence[Any],
    global_classes: Sequence[Any],
) -> np.ndarray:
    """Align model probability columns to the global class order."""

    proba = np.asarray(proba, dtype=float)
    model_classes = list(model_classes)
    global_classes = list(global_classes)
    class_to_index = {cls: idx for idx, cls in enumerate(model_classes)}
    aligned = np.zeros((proba.shape[0], len(global_classes)), dtype=float)
    for target_idx, cls in enumerate(global_classes):
        if cls in class_to_index:
            aligned[:, target_idx] = proba[:, class_to_index[cls]]
    row_sums = aligned.sum(axis=1, keepdims=True)
    np.divide(aligned, row_sums, out=aligned, where=row_sums > 0)
    return aligned


def average_probabilities(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    """Average probability matrices and normalize each row."""

    if not probabilities:
        raise ValueError("at least one probability matrix is required")
    stacked = np.stack([np.asarray(p, dtype=float) for p in probabilities], axis=0)
    averaged = stacked.mean(axis=0)
    row_sums = averaged.sum(axis=1, keepdims=True)
    np.divide(averaged, row_sums, out=averaged, where=row_sums > 0)
    return averaged

