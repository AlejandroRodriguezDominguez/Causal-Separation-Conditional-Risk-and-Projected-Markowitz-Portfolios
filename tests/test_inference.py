import unittest

import numpy as np

from qfemp.inference import paired_family_inference


class InferenceTests(unittest.TestCase):
    def test_strong_negative_difference_excludes_zero(self):
        rng = np.random.default_rng(41)
        values = np.column_stack([
            rng.normal(-1.0, 0.1, 30),
            rng.normal(-0.6, 0.1, 30),
        ])
        rows = paired_family_inference(
            values, ["first", "second"],
            bootstrap_repetitions=500, randomization_repetitions=1000, seed=7,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["simultaneous_ci_upper"] < 0 for row in rows))
        self.assertTrue(all(row["sign_flip_p_familywise"] < 0.01 for row in rows))


if __name__ == "__main__":
    unittest.main()
