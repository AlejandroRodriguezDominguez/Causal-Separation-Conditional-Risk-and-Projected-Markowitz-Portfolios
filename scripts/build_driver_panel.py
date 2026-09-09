#!/usr/bin/env python3
"""Create the pilot driver ledger and lagged innovation panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qfemp.drivers import build_driver_ledger, transform_driver_panel


def parse_date(values: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values.astype(str), format="%Y%m%d", errors="raise"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2023-06-30")
    parser.add_argument("--lag", type=int, default=1)
    args = parser.parse_args()

    assets_raw = pd.read_excel(args.workbook, sheet_name="Assets")
    drivers_raw = pd.read_excel(args.workbook, sheet_name="Drivers")
    assets_raw["Date"] = parse_date(assets_raw["Date"])
    drivers_raw["Date"] = parse_date(drivers_raw["Date"])
    assets = assets_raw.set_index("Date").sort_index().loc[:args.cutoff]
    drivers = drivers_raw.set_index("Date").sort_index().loc[:args.cutoff]

    ledger = build_driver_ledger(drivers)
    if args.lag == 0:
        ledger["availability_timestamp_or_timezone"] = (
            "Contemporaneous daily close alignment; exact timestamp not supplied (pilot only)"
        )
    innovations = transform_driver_panel(
        drivers, ledger, assets.index,
        availability_lag_asset_sessions=args.lag,
    )

    asset_numeric = assets.apply(pd.to_numeric, errors="coerce")
    valid_assets = asset_numeric.notna() & asset_numeric.gt(0)
    valid_asset_pair = (valid_assets & valid_assets.shift(1, fill_value=False)).all(axis=1)
    asset_returns = np.log(asset_numeric / asset_numeric.shift(1)).loc[valid_asset_pair]

    common_index = asset_returns.index.intersection(innovations.dropna().index)
    common_asset_returns = asset_returns.loc[common_index]
    common_innovations = innovations.loc[common_index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger["pilot_applied_lag_asset_sessions"] = args.lag
    ledger.to_csv(args.output_dir / "driver_ledger_pilot.csv", index=False)
    innovations.to_csv(args.output_dir / f"driver_innovations_lag{args.lag}.csv", index_label="date")
    common_innovations.to_csv(args.output_dir / "panel_a_driver_innovations_common.csv", index_label="date")
    common_asset_returns.to_csv(args.output_dir / "panel_a_asset_log_returns_common.csv", index_label="date")

    report = {
        "source_drivers": int(drivers.shape[1]),
        "pilot_eligible_drivers": int(ledger["pilot_eligible"].sum()),
        "excluded_generic_futures": int((ledger["economic_class"] == "generic_future").sum()),
        "excluded_low_frequency_releases": int((ledger["economic_class"] == "macro_release").sum()),
        "excluded_discontinued_or_truncated": int((ledger["economic_class"] == "discontinued_or_truncated_market_series").sum()),
        "confirmatory_eligible_drivers": int(ledger["confirmatory_eligible"].sum()),
        "availability_lag_asset_sessions": args.lag,
        "asset_return_rows_before_driver_intersection": int(len(asset_returns)),
        "common_complete_rows": int(len(common_index)),
        "common_start": common_index.min().date().isoformat(),
        "common_end": common_index.max().date().isoformat(),
        "asset_count": int(common_asset_returns.shape[1]),
        "innovation_missing_cells_common": int(common_innovations.isna().sum().sum()),
        "asset_missing_cells_common": int(common_asset_returns.isna().sum().sum()),
        "nonfinite_innovation_cells_common": int((~np.isfinite(common_innovations.to_numpy())).sum()),
        "nonfinite_asset_cells_common": int((~np.isfinite(common_asset_returns.to_numpy())).sum()),
        "status": "PILOT_READY_CONFIRMATORY_PROVENANCE_UNRESOLVED",
    }
    (args.output_dir / "driver_panel_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
