import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_health.modeling import align_probabilities, average_probabilities, class_weight_map


class ModelingTests(unittest.TestCase):
    def test_align_probabilities_reorders_columns_to_global_classes(self):
        raw = np.array([[0.2, 0.7, 0.1]])
        model_classes = np.array(["fit", "unhealthy", "at-risk"])
        global_classes = np.array(["at-risk", "fit", "unhealthy"])

        aligned = align_probabilities(raw, model_classes, global_classes)

        np.testing.assert_allclose(aligned, np.array([[0.1, 0.2, 0.7]]))

    def test_average_probabilities_returns_row_normalized_mean(self):
        first = np.array([[0.7, 0.2, 0.1]])
        second = np.array([[0.1, 0.8, 0.1]])

        averaged = average_probabilities([first, second])

        np.testing.assert_allclose(averaged, np.array([[0.4, 0.5, 0.1]]))
        np.testing.assert_allclose(averaged.sum(axis=1), np.array([1.0]))

    def test_class_weight_map_gives_larger_weights_to_rare_classes(self):
        y = np.array(["at-risk"] * 8 + ["fit"] * 2)

        weights = class_weight_map(y)

        self.assertGreater(weights["fit"], weights["at-risk"])


if __name__ == "__main__":
    unittest.main()
