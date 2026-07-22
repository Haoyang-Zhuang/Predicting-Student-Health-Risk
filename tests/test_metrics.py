import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_health.metrics import optimize_class_multipliers, predict_with_multipliers


class MetricTests(unittest.TestCase):
    def test_multiplier_search_improves_minority_recall_case(self):
        classes = np.array(["at-risk", "fit"])
        y_true = np.array(["at-risk", "at-risk", "fit", "fit"])
        proba = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.55, 0.45],
                [0.60, 0.40],
            ]
        )
        baseline = balanced_accuracy_score(y_true, classes[np.argmax(proba, axis=1)])

        multipliers, tuned_score = optimize_class_multipliers(
            y_true,
            proba,
            classes,
            rounds=150,
            random_state=7,
        )
        tuned_pred = predict_with_multipliers(proba, classes, multipliers)

        self.assertGreaterEqual(tuned_score, baseline)
        self.assertGreaterEqual(balanced_accuracy_score(y_true, tuned_pred), baseline)
        self.assertEqual(set(multipliers), set(classes))

    def test_predict_with_multipliers_keeps_input_probabilities_unchanged(self):
        classes = np.array(["at-risk", "fit", "unhealthy"])
        proba = np.array([[0.45, 0.40, 0.15]])
        original = proba.copy()

        pred = predict_with_multipliers(
            proba,
            classes,
            {"at-risk": 1.0, "fit": 2.0, "unhealthy": 1.0},
        )

        np.testing.assert_allclose(proba, original)
        self.assertEqual(pred.tolist(), ["fit"])


if __name__ == "__main__":
    unittest.main()
