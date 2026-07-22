from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_COLUMNS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

DERIVED_CATEGORICAL_COLUMNS = [
    "stress_activity",
    "sleep_stress",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact, model-friendly tabular features."""

    out = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in out.columns:
            out[f"{column}_missing"] = out[column].isna().astype(np.int8)

    for column in CATEGORICAL_COLUMNS:
        if column in out.columns:
            out[column] = out[column].fillna("__missing__").astype(str)

    if {"stress_level", "physical_activity_level"}.issubset(out.columns):
        out["stress_activity"] = (
            out["stress_level"].astype(str) + "__" + out["physical_activity_level"].astype(str)
        )
    else:
        out["stress_activity"] = "__missing__"

    if {"sleep_quality", "stress_level"}.issubset(out.columns):
        out["sleep_stress"] = out["sleep_quality"].astype(str) + "__" + out["stress_level"].astype(str)
    else:
        out["sleep_stress"] = "__missing__"

    if "sleep_duration" in out.columns:
        out["sleep_duration_from_7"] = (out["sleep_duration"].astype(float) - 7.0).abs()
    else:
        out["sleep_duration_from_7"] = np.nan

    if {"exercise_duration", "step_count"}.issubset(out.columns):
        out["exercise_per_1k_steps"] = _safe_divide(out["exercise_duration"], out["step_count"] / 1000.0)
    else:
        out["exercise_per_1k_steps"] = np.nan

    if {"calorie_expenditure", "step_count"}.issubset(out.columns):
        out["calories_per_step"] = _safe_divide(out["calorie_expenditure"], out["step_count"])
    else:
        out["calories_per_step"] = np.nan

    if {"water_intake", "bmi"}.issubset(out.columns):
        out["water_per_bmi"] = _safe_divide(out["water_intake"], out["bmi"])
    else:
        out["water_per_bmi"] = np.nan

    if {"heart_rate", "sleep_duration"}.issubset(out.columns):
        out["heart_sleep_ratio"] = _safe_divide(out["heart_rate"], out["sleep_duration"])
    else:
        out["heart_sleep_ratio"] = np.nan

    if {"exercise_duration", "sleep_duration"}.issubset(out.columns):
        out["exercise_sleep_ratio"] = _safe_divide(out["exercise_duration"], out["sleep_duration"])
    else:
        out["exercise_sleep_ratio"] = np.nan

    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature lists after feature engineering."""

    columns = list(df.columns)
    numeric = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(df[column]) or column.endswith("_missing")
    ]
    categorical = [column for column in columns if column not in numeric]
    return numeric, categorical

