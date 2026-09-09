import unittest
import numpy as np

from qfemp.extended import (
    greedy_driver_selection, nonoverlap_sum, offdiag_metrics, rolling_folds,
)


class ExtendedExperimentTests(unittest.TestCase):
    def test_nonoverlap_sum(self):
        x=np.arange(12.0).reshape(6,2); d=np.arange(6)
        y,dates=nonoverlap_sum(x,d,2)
        np.testing.assert_allclose(y,np.array([[2,4],[10,12],[18,20]]))
        np.testing.assert_array_equal(dates,np.array([1,3,5]))

    def test_folds_are_temporally_ordered(self):
        folds=rolling_folds(20,8,4)
        self.assertEqual(len(folds),3)
        self.assertTrue(all(f.train.stop<=f.test.start for f in folds))

    def test_selection_finds_common_driver(self):
        rng=np.random.default_rng(5); x=rng.normal(size=(300,4))
        load=np.array([1.0,-.6,.4]); y=x[:,[2]]@load[None,:]+.2*rng.normal(size=(300,3))
        selected=greedy_driver_selection(y,x,max_drivers=2,penalty=.001)
        self.assertIn(2,selected)
        self.assertLess(offdiag_metrics(y-x[:,[2]]@load[None,:])['s_f'],offdiag_metrics(y)['s_f'])


if __name__=='__main__': unittest.main()
