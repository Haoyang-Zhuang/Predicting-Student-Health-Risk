import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_health.submission import make_submission, validate_submission


class SubmissionTests(unittest.TestCase):
    def test_make_submission_uses_required_columns(self):
        submission = make_submission([10, 11], ["fit", "at-risk"])

        self.assertEqual(submission.columns.tolist(), ["id", "health_condition"])
        self.assertEqual(submission["id"].tolist(), [10, 11])
        self.assertEqual(submission["health_condition"].tolist(), ["fit", "at-risk"])

    def test_validate_submission_rejects_wrong_id_order(self):
        sample = pd.DataFrame({"id": [10, 11], "health_condition": ["at-risk", "at-risk"]})
        submission = pd.DataFrame({"id": [11, 10], "health_condition": ["fit", "at-risk"]})

        with self.assertRaises(ValueError):
            validate_submission(submission, sample)

    def test_validate_submission_accepts_matching_shape_and_ids(self):
        sample = pd.DataFrame({"id": [10, 11], "health_condition": ["at-risk", "at-risk"]})
        submission = pd.DataFrame({"id": [10, 11], "health_condition": ["fit", "at-risk"]})

        self.assertTrue(validate_submission(submission, sample))


if __name__ == "__main__":
    unittest.main()
