#!/usr/bin/env python3
"""Build Panel B price returns and a complete corporate-action audit trail."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qfemp.price_returns import reconstruct_fixed_panel_price_returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-prices", type=Path, required=True)
    parser.add_argument("--fixed-universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-ledger", type=Path)
    parser.add_argument("--start", default="2010-07-01")
    parser.add_argument("--end", default="2023-06-30")
    args = parser.parse_args()

    universe_header = pd.read_excel(args.fixed_universe, nrows=1)
    tickers = [str(col) for col in universe_header.columns if str(col).lower() != "date"]
    raw = pd.read_excel(args.raw_prices)
    date_col = next((col for col in raw.columns if str(col).lower() == "date"), None)
    if date_col is None:
        raise ValueError("raw price workbook has no date column")
    missing_tickers = sorted(set(tickers).difference(raw.columns))
    if missing_tickers:
        raise ValueError(f"fixed-universe tickers missing from raw workbook: {missing_tickers}")
    raw[date_col] = pd.to_datetime(raw[date_col], errors="raise")
    prices = raw.set_index(date_col)[tickers]

    ledger = pd.read_csv(args.event_ledger) if args.event_ledger else None
    result = reconstruct_fixed_panel_price_returns(
        prices,
        start=args.start,
        end=args.end,
        event_ledger=ledger,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.returns.to_csv(args.output_dir / "panel_b_log_price_returns.csv", index_label="date")
    result.corporate_actions.to_csv(args.output_dir / "panel_b_corporate_action_candidates.csv", index=False)
    result.unresolved_moves.to_csv(args.output_dir / "panel_b_unresolved_large_moves.csv", index=False)
    pd.DataFrame({"ticker": tickers}).to_csv(args.output_dir / "panel_b_fixed_universe.csv", index=False)
    adjusted_event_dates = pd.to_datetime(
        result.corporate_actions.loc[
            result.corporate_actions["rule_status"].isin([
                "ADJUSTED_VERIFIED_FACTOR",
                "ADJUSTED_MECHANICAL_SPLIT_CANDIDATE",
                "EXCLUDED_VERIFIED_DISTRIBUTION_DATE",
            ]),
            "date",
        ]
    ).drop_duplicates()
    sensitivity = result.returns.drop(index=adjusted_event_dates, errors="ignore")
    result.report["event_dates_removed_in_sensitivity_panel"] = int(len(adjusted_event_dates))
    result.report["sensitivity_return_rows"] = int(len(sensitivity))
    sensitivity.to_csv(
        args.output_dir / "panel_b_event_day_exclusion_log_returns.csv",
        index_label="date",
    )
    (args.output_dir / "panel_b_construction_report.json").write_text(
        json.dumps(result.report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Panel B return reconstruction",
        "",
        "This is a fixed-universe price-return dataset. It is not CRSP `RET/DLRET`,",
        "does not include dividends, and is not a point-in-time S&P 500 membership backtest.",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, value in result.report.items():
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"| `{key}` | {value} |")
    (args.output_dir / "panel_b_construction_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
