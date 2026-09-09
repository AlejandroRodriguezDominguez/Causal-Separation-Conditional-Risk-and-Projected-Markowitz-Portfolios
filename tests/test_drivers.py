import unittest

import numpy as np
import pandas as pd

from qfemp.drivers import build_driver_ledger, classify_driver, transform_driver_panel


class DriverTests(unittest.TestCase):
    def test_classification_rules(self):
        self.assertEqual(classify_driver("EUR SWAP ANN (VS 6M) 10Y")["transformation"], "DIFFERENCE")
        self.assertTrue(classify_driver("Generic 1st 'CO' Future")["pilot_eligible"])
        self.assertFalse(classify_driver("US Initial Jobless Claims SA")["pilot_eligible"])
        self.assertFalse(classify_driver("Eonia Capitalization Index 7 D")["pilot_eligible"])
        self.assertEqual(classify_driver("DAX INDEX")["transformation"], "LOG_DIFFERENCE")
        self.assertEqual(classify_driver("Brazilian Real Spot")["economic_class"], "foreign_exchange")
        self.assertEqual(classify_driver("Swedish Krona Spot")["economic_class"], "foreign_exchange")
        self.assertEqual(classify_driver("DB Euro Overnight Rate")["transformation"], "DIFFERENCE")
        self.assertEqual(classify_driver("DOLLAR INDEX SPOT")["economic_class"], "currency_index")

    def test_sentinel_and_lag(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        drivers = pd.DataFrame({"CSI 300 INDEX": [100.0, -1.0, 102.0, 104.0, 108.0]}, index=dates)
        ledger = build_driver_ledger(drivers)
        z = transform_driver_panel(drivers, ledger, dates)
        self.assertTrue(np.isnan(z.loc[dates[2], "CSI 300 INDEX"]))
        self.assertAlmostEqual(z.loc[dates[4], "CSI 300 INDEX"], np.log(104.0 / 102.0))

    def test_short_local_holiday_gap_is_carried_before_transform(self):
        source_dates = pd.bdate_range("2020-01-01", periods=5)
        target_dates = source_dates
        drivers = pd.DataFrame(
            {"DAX INDEX": [100.0, np.nan, 102.0, 103.0, 104.0]},
            index=source_dates,
        )
        ledger = build_driver_ledger(drivers)
        z = transform_driver_panel(
            drivers, ledger, target_dates, availability_lag_asset_sessions=0
        )
        self.assertAlmostEqual(z.loc[target_dates[1], "DAX INDEX"], 0.0)
        self.assertAlmostEqual(
            z.loc[target_dates[2], "DAX INDEX"], np.log(102.0 / 100.0)
        )


if __name__ == "__main__":
    unittest.main()
