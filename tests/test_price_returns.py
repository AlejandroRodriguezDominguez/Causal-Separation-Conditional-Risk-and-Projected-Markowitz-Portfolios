import unittest

import numpy as np
import pandas as pd

from qfemp.price_returns import reconstruct_fixed_panel_price_returns


class PriceReturnTests(unittest.TestCase):
    def test_split_adjustment_and_no_missing_bridge(self):
        dates = pd.bdate_range("2020-01-01", periods=6)
        prices = pd.DataFrame({
            "AAA": [100.0, 102.0, 25.5, 26.0, 26.5, 27.0],
            "BBB": [50.0, 51.0, np.nan, 52.0, 53.0, 54.0],
        }, index=dates)
        result = reconstruct_fixed_panel_price_returns(
            prices, start="2020-01-01", end="2020-12-31"
        )
        # Rows touching BBB's missing observation are excluded; no bridge is formed.
        self.assertNotIn(dates[2], result.returns.index)
        self.assertNotIn(dates[3], result.returns.index)

    def test_four_for_one_split_is_neutralised(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        prices = pd.DataFrame({"AAA": [100.0, 102.0, 25.5, 26.0]}, index=dates)
        result = reconstruct_fixed_panel_price_returns(
            prices, start="2020-01-01", end="2020-12-31"
        )
        self.assertAlmostEqual(result.returns.loc[dates[2], "AAA"], 0.0, places=12)
        self.assertEqual(result.report["mechanical_split_adjustments"], 1)

    def test_verified_factor_overrides_threshold(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"AAA": [100.0, 66.0, 67.0]}, index=dates)
        ledger = pd.DataFrame([{
            "date": dates[1].date().isoformat(),
            "ticker": "AAA",
            "disposition": "APPLY_FACTOR",
            "factor": 2 / 3,
            "verification_status": "PRIMARY_SOURCE_VERIFIED",
            "event_type": "3-for-2 split",
            "source_url": "https://example.test/filing",
            "rationale": "Verified split.",
        }])
        result = reconstruct_fixed_panel_price_returns(
            prices, start="2020-01-01", end="2020-12-31", event_ledger=ledger
        )
        self.assertAlmostEqual(
            result.returns.loc[dates[1], "AAA"], np.log(0.66 / (2 / 3)), places=12
        )

    def test_verified_distribution_drops_full_synchronous_date(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"AAA": [100.0, 50.0, 51.0], "BBB": [10.0, 10.1, 10.2]}, index=dates)
        ledger = pd.DataFrame([{
            "date": dates[1].date().isoformat(),
            "ticker": "AAA",
            "disposition": "EXCLUDE_DATE",
            "factor": np.nan,
            "verification_status": "PRIMARY_SOURCE_VERIFIED",
            "event_type": "spin-off",
            "source_url": "https://example.test/filing",
            "rationale": "Distributed-asset value is unavailable.",
        }])
        result = reconstruct_fixed_panel_price_returns(
            prices, start="2020-01-01", end="2020-12-31", event_ledger=ledger
        )
        self.assertNotIn(dates[1], result.returns.index)
        self.assertNotIn(dates[1], result.unresolved_moves.get("date", []))

    def test_verified_market_move_is_retained_and_resolved(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"AAA": [100.0, 200.0, 201.0]}, index=dates)
        ledger = pd.DataFrame([{
            "date": dates[1].date().isoformat(),
            "ticker": "AAA",
            "disposition": "KEEP_OBSERVED",
            "factor": np.nan,
            "verification_status": "PRIMARY_SOURCE_VERIFIED",
            "event_type": "earnings announcement",
            "source_url": "https://example.test/filing",
            "rationale": "Economic price move rather than a split.",
        }])
        result = reconstruct_fixed_panel_price_returns(
            prices, start="2020-01-01", end="2020-12-31", event_ledger=ledger
        )
        self.assertAlmostEqual(result.returns.loc[dates[1], "AAA"], np.log(2.0), places=12)
        self.assertEqual(len(result.unresolved_moves), 0)


if __name__ == "__main__":
    unittest.main()
