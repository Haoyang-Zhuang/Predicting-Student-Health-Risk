import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_health.features import CATEGORICAL_COLUMNS, add_features


class FeatureTests(unittest.TestCase):
    def test_add_features_preserves_rows_and_adds_missing_indicators(self):
        df = pd.DataFrame(
            {
                "sleep_duration": [7.5, np.nan],
                "heart_rate": [72.0, 81.0],
                "bmi": [22.0, np.nan],
                "calorie_expenditure": [2300.0, 1900.0],
                "step_count": [10000.0, 0.0],
                "exercise_duration": [45.0, 10.0],
                "water_intake": [2.0, 1.5],
                "diet_type": ["balanced", None],
                "stress_level": ["low", "high"],
                "sleep_quality": ["good", None],
                "physical_activity_level": ["active", "sedentary"],
                "smoking_alcohol": ["no", None],
                "gender": ["female", None],
            }
        )

        out = add_features(df)

        self.assertEqual(len(out), len(df))
        self.assertIn("sleep_duration_missing", out.columns)
        self.assertEqual(out.loc[0, "sleep_duration_missing"], 0)
        self.assertEqual(out.loc[1, "sleep_duration_missing"], 1)
        self.assertIn("bmi_missing", out.columns)
        self.assertEqual(out.loc[1, "bmi_missing"], 1)

    def test_add_features_keeps_categorical_values_as_strings(self):
        df = pd.DataFrame(
            {
                "sleep_duration": [7.5],
                "heart_rate": [72.0],
                "bmi": [22.0],
                "calorie_expenditure": [2300.0],
                "step_count": [10000.0],
                "exercise_duration": [45.0],
                "water_intake": [2.0],
                "diet_type": [None],
                "stress_level": ["low"],
                "sleep_quality": ["good"],
                "physical_activity_level": ["active"],
                "smoking_alcohol": ["no"],
                "gender": ["female"],
            }
        )

        out = add_features(df)

        for column in CATEGORICAL_COLUMNS + ["stress_activity", "sleep_stress"]:
            self.assertIn(column, out.columns)
            self.assertIsInstance(out.loc[0, column], str)
        self.assertEqual(out.loc[0, "diet_type"], "__missing__")

    def test_ratio_features_do_not_create_infinite_values(self):
        df = pd.DataFrame(
            {
                "sleep_duration": [np.nan],
                "heart_rate": [72.0],
                "bmi": [0.0],
                "calorie_expenditure": [2300.0],
                "step_count": [0.0],
                "exercise_duration": [45.0],
                "water_intake": [2.0],
                "diet_type": ["balanced"],
                "stress_level": ["low"],
                "sleep_quality": ["good"],
                "physical_activity_level": ["active"],
                "smoking_alcohol": ["no"],
                "gender": ["female"],
            }
        )

        out = add_features(df)
        ratio_columns = ["exercise_per_1k_steps", "calories_per_step", "water_per_bmi"]
        values = out[ratio_columns].to_numpy(dtype=float)
        self.assertFalse(np.isinf(values).any())


if __name__ == "__main__":
    unittest.main()
