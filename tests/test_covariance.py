import unittest

import numpy as np

from qfemp.covariance import (
    gmv_weights, long_only_gmv_weights, poet_covariance,
    quadratic_inverse_shrinkage_covariance, structured_covariances,
)


class CovarianceTests(unittest.TestCase):
    def test_gmv_is_fully_invested(self):
        covariance = np.array([[2.0, 0.2], [0.2, 1.0]])
        weights, _ = gmv_weights(covariance)
        self.assertAlmostEqual(weights.sum(), 1.0, places=12)

    def test_long_only_gmv_is_feasible(self):
        covariance = np.array([[2.0, 0.2], [0.2, 1.0]])
        weights, _ = long_only_gmv_weights(covariance)
        self.assertAlmostEqual(weights.sum(), 1.0, places=12)
        self.assertTrue(np.all(weights >= 0.0))

    def test_structured_matrices_are_symmetric(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(300, 2))
        y = z @ rng.normal(size=(2, 5)) + rng.normal(scale=0.4, size=(300, 5))
        q0, qfull = structured_covariances(y, z)
        self.assertTrue(np.allclose(q0, q0.T))
        self.assertTrue(np.allclose(qfull, qfull.T))

    def test_poet_is_symmetric_and_preserves_sample_diagonal(self):
        rng = np.random.default_rng(19)
        returns = rng.normal(size=(252, 40))
        covariance = poet_covariance(returns, 3)
        sample = np.cov(returns, rowvar=False, ddof=0)
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertTrue(np.allclose(np.diag(covariance), np.diag(sample)))

    def test_qis_is_positive_and_trace_preserving(self):
        rng = np.random.default_rng(31)
        returns = rng.normal(size=(252, 40))
        covariance = quadratic_inverse_shrinkage_covariance(returns)
        sample = np.cov(returns, rowvar=False, ddof=1)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.0)
        self.assertAlmostEqual(np.trace(covariance), np.trace(sample), places=10)


if __name__ == "__main__":
    unittest.main()
