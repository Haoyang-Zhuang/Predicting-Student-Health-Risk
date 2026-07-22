from __future__ import annotations

from typing import Iterable

import pandas as pd


def make_submission(ids: Iterable, labels: Iterable) -> pd.DataFrame:
    return pd.DataFrame({"id": list(ids), "health_condition": list(labels)})


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> bool:
    expected_columns = ["id", "health_condition"]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"submission columns must be {expected_columns}")
    if sample_submission.columns.tolist() != expected_columns:
        raise ValueError(f"sample_submission columns must be {expected_columns}")
    if len(submission) != len(sample_submission):
        raise ValueError("submission row count does not match sample_submission")
    if submission["id"].tolist() != sample_submission["id"].tolist():
        raise ValueError("submission ids do not match sample_submission order")
    if submission.isna().any().any():
        raise ValueError("submission contains missing values")
    return True

