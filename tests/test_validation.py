import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qfemp.validation import validate_asset_ledger


class ValidationTests(unittest.TestCase):
    def test_unknown_adjustment_status_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.csv"
            pd.DataFrame([{
                "panel": "B",
                "source_label": "AAA",
                "stable_security_id": "",
                "source_vendor": "",
                "source_field": "",
                "data_type": "raw_price_unverified",
                "corporate_action_adjusted": "UNKNOWN",
                "delisting_return_policy": "UNKNOWN",
                "confirmatory_eligible": "UNKNOWN",
                "exclusion_reason": "unresolved",
            }]).to_csv(path, index=False)
            results = validate_asset_ledger(path)
            self.assertFalse(all(item.passed for item in results))


if __name__ == "__main__":
    unittest.main()
