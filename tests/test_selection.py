import unittest

import numpy as np

from qfemp.selection import fit_and_residualize, greedy_select, residual_metrics


class SelectionTests(unittest.TestCase):
    def test_common_driver_is_selected_deterministically(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(500, 4))
        load = np.array([[1.0, 0.8, -0.5]])
        y = x[:, [0]] @ load + rng.normal(scale=0.3, size=(500, 3))
        result = greedy_select(y, x, penalty=0.001, max_drivers=2)
        self.assertIn(0, result.selected)
        self.assertLess(result.final_score, result.baseline_score)

    def test_residual_metric_zero_for_independent_identity_sample(self):
        x = np.vstack([np.eye(3), -np.eye(3)])
        self.assertAlmostEqual(residual_metrics(x)["s_f"], 0.0, places=12)

    def test_cross_product_score_matches_direct_residualization(self):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(300, 5))
        y = x[:, [2]] @ np.array([[1.0, -0.8, 0.5]]) + rng.normal(
            scale=0.5, size=(300, 3)
        )
        result = greedy_select(y, x, penalty=0.0, max_drivers=1)
        residuals, _ = fit_and_residualize(y, x[:, result.selected])
        self.assertAlmostEqual(
            result.final_score, residual_metrics(residuals)["s_f"], places=11
        )


if __name__ == "__main__":
    unittest.main()
