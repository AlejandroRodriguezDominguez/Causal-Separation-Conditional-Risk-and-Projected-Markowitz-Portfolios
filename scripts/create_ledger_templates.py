#!/usr/bin/env python3
"""Generate review templates without asserting undocumented data properties."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_dates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if float(numeric.between(19000101, 21001231).mean()) >= 0.5:
        return pd.to_datetime(numeric.astype("Int64").astype("string"), format="%Y%m%d")
    return pd.to_datetime(series, errors="coerce")


def series_stats(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce")
    positive = x.where(x > 0)
    jump = np.log(positive).diff().abs().max(skipna=True)
    return {
        "observations": int(x.notna().sum()),
        "missing_fraction": float(x.isna().mean()),
        "nonpositive_observations": int(((x <= 0) & x.notna()).sum()),
        "max_abs_adjacent_log_change": float(jump) if pd.notna(jump) else "",
    }


def build_asset_rows(input_dir: Path) -> list[dict]:
    rows = []
    specs = [
        ("A", input_dir / "Drivers_Assets_Dataset.xlsx", "Assets", "Date"),
        ("B", input_dir / "processed_output.xlsx", "Sheet1", "date"),
    ]
    for panel, path, sheet, date_col in specs:
        frame = pd.read_excel(path, sheet_name=sheet)
        dates = parse_dates(frame[date_col])
        for label in (c for c in frame.columns if c != date_col):
            stats = series_stats(frame[label])
            rows.append({
                "panel": panel,
                "source_label": label,
                "stable_security_id": "",
                "source_file": path.name,
                "source_vendor": "",
                "source_field": "",
                "data_type": "raw_price_unverified",
                "corporate_action_adjusted": "UNKNOWN",
                "delisting_return_policy": "UNKNOWN",
                "observed_start": dates.min().date().isoformat(),
                "observed_end": dates.max().date().isoformat(),
                **stats,
                "confirmatory_eligible": "UNKNOWN",
                "exclusion_reason": "Adjustment/provenance not yet documented",
                "reviewed_by": "",
                "reviewed_on": "",
            })
    return rows


def build_driver_rows(input_dir: Path) -> list[dict]:
    path = input_dir / "Drivers_Assets_Dataset.xlsx"
    frame = pd.read_excel(path, sheet_name="Drivers")
    dates = parse_dates(frame["Date"])
    rows = []
    for label in (c for c in frame.columns if c != "Date"):
        stats = series_stats(frame[label])
        rows.append({
            "source_label": label,
            "economic_class": "",
            "suggested_transformation": "REVIEW_REQUIRED",
            "approved_transformation": "",
            "unit": "",
            "source_file": path.name,
            "source_vendor": "",
            "source_field": "",
            "availability_timestamp_or_timezone": "",
            "availability_lag_trading_days": "",
            "missing_value_rule": "",
            "sentinel_value_rule": "",
            "observed_start": dates.min().date().isoformat(),
            "observed_end": dates.max().date().isoformat(),
            **stats,
            "confirmatory_eligible": "UNKNOWN",
            "exclusion_reason": "Point-in-time metadata not yet documented",
            "reviewed_by": "",
            "reviewed_on": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(build_asset_rows(args.input_dir)).to_csv(
        args.output_dir / "asset_ledger_template.csv", index=False
    )
    pd.DataFrame(build_driver_rows(args.input_dir)).to_csv(
        args.output_dir / "driver_ledger_template.csv", index=False
    )


if __name__ == "__main__":
    main()

