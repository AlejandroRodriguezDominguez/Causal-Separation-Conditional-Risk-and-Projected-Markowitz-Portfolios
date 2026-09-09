import unittest

import numpy as np

from qfemp.bootstrap import moving_block_indices, wilson_interval


class BootstrapTests(unittest.TestCase):
    def test_moving_blocks_are_valid_and_reproducible(self):
        first = moving_block_indices(25, 6, np.random.default_rng(8))
        second = moving_block_indices(25, 6, np.random.default_rng(8))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 25)
        self.assertTrue(((first >= 0) & (first < 25)).all())

    def test_wilson_interval_contains_observed_frequency(self):
        lower, upper = wilson_interval(60, 100)
        self.assertLess(lower, 0.6)
        self.assertGreater(upper, 0.6)


if __name__ == "__main__":
    unittest.main()
