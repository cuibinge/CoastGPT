import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.coastline_eval_tools import (
    aggregate_metric_records,
    choose_best_thresholds,
    parse_threshold_values,
)


class CoastlineEvalToolsTest(unittest.TestCase):
    def test_parse_threshold_values_accepts_ranges_and_lists(self):
        self.assertEqual(parse_threshold_values("0.1,0.25,0.5"), [0.1, 0.25, 0.5])
        self.assertEqual(parse_threshold_values("0.1:0.3:0.1"), [0.1, 0.2, 0.3])

    def test_choose_best_thresholds_can_select_per_category(self):
        records = [
            {"category": "生物岸线", "threshold": 0.2, "buffered_f1_3px": 0.70},
            {"category": "生物岸线", "threshold": 0.3, "buffered_f1_3px": 0.74},
            {"category": "盐田围堤", "threshold": 0.2, "buffered_f1_3px": 0.40},
            {"category": "盐田围堤", "threshold": 0.1, "buffered_f1_3px": 0.62},
        ]

        best = choose_best_thresholds(
            records,
            metric="buffered_f1_3px",
            group_key="category",
        )

        self.assertEqual(best["生物岸线"]["threshold"], 0.3)
        self.assertEqual(best["盐田围堤"]["threshold"], 0.1)

    def test_aggregate_metric_records_ignores_non_numeric_fields(self):
        records = [
            {"sample_id": "a", "category": "x", "buffered_f1_3px": 0.5, "num_features": 2},
            {"sample_id": "b", "category": "x", "buffered_f1_3px": 0.7, "num_features": 4},
        ]

        aggregate = aggregate_metric_records(records)

        self.assertEqual(aggregate["buffered_f1_3px"], 0.6)
        self.assertEqual(aggregate["num_features"], 3.0)
        self.assertNotIn("sample_id", aggregate)


if __name__ == "__main__":
    unittest.main()
