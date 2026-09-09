import unittest

import pandas as pd

from qfemp.folds import build_panel_a_folds, build_panel_b_folds


class FoldTests(unittest.TestCase):
    def test_boundaries_are_strict_and_non_overlapping(self):
        dates = pd.bdate_range("2010-01-01", periods=1200)
        folds = build_panel_a_folds(dates)
        self.assertGreater(len(folds), 0)
        for fold in folds:
            self.assertLess(fold.selection_end, fold.certification_start)
            self.assertLess(fold.certification_end, fold.test_start)
        for left, right in zip(folds, folds[1:]):
            self.assertLess(left.test_end, right.test_start)

    def test_rejects_unsorted_dates(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-01"])
        with self.assertRaises(ValueError):
            build_panel_a_folds(dates, 1, 1, 1)

    def test_panel_b_training_precedes_nonoverlapping_test(self):
        dates = pd.bdate_range("2020-01-01", periods=600)
        folds = build_panel_b_folds(dates, training_days=252, test_days=126)
        self.assertTrue(folds)
        for fold in folds:
            self.assertLess(fold.training_end, fold.test_start)
        for left, right in zip(folds, folds[1:]):
            self.assertLess(left.test_end, right.test_start)


if __name__ == "__main__":
    unittest.main()
