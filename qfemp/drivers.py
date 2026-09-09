"""Conservative, auditable transformations for the supplied driver panel."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


FUTURE_PATTERN = re.compile(r"^Generic 1st .* Future$")
YIELD_LABELS = {
    "BONOS Y OBLIG DEL ESTADO",
    "BUONI POLIENNALI DEL TES",
    "FRANCE (GOVT OF)",
    "INDIA GOVERNMENT BOND",
    "NETHERLANDS GOVERNMENT",
    "UK Gilts 10 Yr",
    "UK Gilts 30 Year",
    "UK Gilts 5 Year",
    "US Generic Govt 10 Yr",
    "US Generic Govt 2 Yr",
    "US Generic Govt 5 Yr",
}
RATE_LABELS = {
    "DB Euro Overnight Rate",
}
SENTINEL_MINUS_ONE = {
    "CSI 300 INDEX",
    "Eonia Capitalization Index 7 D",
    "Russian Ruble SPOT (TOM)",
    "S&P CoreLogic Case-Shiller 20-",
    "USD-CLP RR 25D 3M",
}
LOW_FREQUENCY_RELEASES = {
    "S&P CoreLogic Case-Shiller 20-",
    "US Initial Jobless Claims SA",
}
TRUNCATED_SERIES = {
    "Eonia Capitalization Index 7 D": "Series becomes a persistent -1 sentinel after December 2022.",
    "Russian Ruble SPOT (TOM)": "Series becomes a persistent -1 sentinel after December 2022.",
}


def classify_driver(label: str) -> dict:
    """Return a frozen label-based pilot classification.

    The source is a user-supplied Bloomberg export of PX_LAST observations.
    Calendar labels identify the observation date. Predictive exercises still
    apply an asset-specific lag because European and U.S. closes are not
    simultaneous and macroeconomic releases require their own availability
    convention.
    """
    if FUTURE_PATTERN.match(label):
        return {
            "economic_class": "generic_future",
            "transformation": "LOG_DIFFERENCE",
            "unit": "contract quotation",
            "pilot_eligible": True,
            "reason": "Bloomberg Generic 1st PX_LAST; roll dates are monitored in robustness checks.",
        }
    if label in LOW_FREQUENCY_RELEASES:
        return {
            "economic_class": "macro_release",
            "transformation": "EXCLUDE",
            "unit": "published level",
            "pilot_eligible": False,
            "reason": "Release vintage and publication timestamp are unavailable.",
        }
    if label in TRUNCATED_SERIES:
        return {
            "economic_class": "discontinued_or_truncated_market_series",
            "transformation": "EXCLUDE",
            "unit": "source level",
            "pilot_eligible": False,
            "reason": TRUNCATED_SERIES[label],
        }
    if "EUR SWAP ANN" in label:
        return {
            "economic_class": "swap_rate",
            "transformation": "DIFFERENCE",
            "unit": "percentage points",
            "pilot_eligible": True,
            "reason": "",
        }
    if label in YIELD_LABELS or "GOVT BND" in label:
        return {
            "economic_class": "sovereign_yield",
            "transformation": "DIFFERENCE",
            "unit": "percentage points",
            "pilot_eligible": True,
            "reason": "",
        }
    if label in RATE_LABELS:
        return {
            "economic_class": "money_market_rate",
            "transformation": "DIFFERENCE",
            "unit": "percentage points",
            "pilot_eligible": True,
            "reason": "",
        }
    if "RR 25D" in label:
        return {
            "economic_class": "option_risk_reversal",
            "transformation": "DIFFERENCE",
            "unit": "volatility points",
            "pilot_eligible": True,
            "reason": "",
        }
    if label == "Cboe Volatility Index":
        return {
            "economic_class": "volatility_index",
            "transformation": "LOG_DIFFERENCE",
            "unit": "index level",
            "pilot_eligible": True,
            "reason": "",
        }
    fx_tokens = (
        "X-RATE", "Peso Spot", "Dollar Spot", "Pound Spot", "Euro Spot",
        "Rupee Spot", "Yen Spot", "Krone Spot", "Krona Spot", "Real Spot",
        "Zloty Spot", "Ruble SPOT",
        "Rand Spot", "Singapore Dollar Spot", "Won Spot", "Franc Spot",
        "Lira Spot",
    )
    if any(token in label for token in fx_tokens):
        return {
            "economic_class": "foreign_exchange",
            "transformation": "LOG_DIFFERENCE",
            "unit": "exchange-rate level",
            "pilot_eligible": True,
            "reason": "",
        }
    if label == "DOLLAR INDEX SPOT":
        return {
            "economic_class": "currency_index",
            "transformation": "LOG_DIFFERENCE",
            "unit": "index level",
            "pilot_eligible": True,
            "reason": "",
        }
    if "INDEX" in label or label in {"MSCI INDIA", "NIKKEI 225", "S&P/BMV IPC"}:
        return {
            "economic_class": "equity_index",
            "transformation": "LOG_DIFFERENCE",
            "unit": "index level",
            "pilot_eligible": True,
            "reason": "",
        }
    commodity_tokens = ("Commodity", "Gold Spot", "Silver Spot", "LME ", "CRB")
    if any(token in label for token in commodity_tokens):
        return {
            "economic_class": "commodity",
            "transformation": "LOG_DIFFERENCE",
            "unit": "price or index level",
            "pilot_eligible": True,
            "reason": "",
        }
    return {
        "economic_class": "bond_or_credit_index",
        "transformation": "LOG_DIFFERENCE",
        "unit": "index level",
        "pilot_eligible": True,
        "reason": "",
    }


def build_driver_ledger(drivers: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(drivers.index, pd.DatetimeIndex):
        raise TypeError("driver panel must have a DatetimeIndex")
    rows = []
    for label in drivers.columns:
        info = classify_driver(str(label))
        series = pd.to_numeric(drivers[label], errors="coerce")
        finite = series[np.isfinite(series)]
        sentinel_rule = (
            "Exact -1 is a missing-value sentinel; no fill."
            if str(label) in SENTINEL_MINUS_ONE else
            "No numeric sentinel declared."
        )
        pilot_reason = info["reason"]
        confirmatory_reason = pilot_reason if not info["pilot_eligible"] else ""
        rows.append({
            "source_label": str(label),
            "economic_class": info["economic_class"],
            "suggested_transformation": info["transformation"],
            "approved_transformation": info["transformation"],
            "unit": info["unit"],
            "source_file": "Drivers_Assets_Dataset.xlsx",
            "source_vendor": "Bloomberg",
            "source_field": "PX_LAST",
            "availability_timestamp_or_timezone": (
                "Bloomberg market close observation; predictive analyses use "
                "the predeclared asset-specific lag"
            ),
            "availability_lag_trading_days": 1,
            "missing_value_rule": (
                "Carry PX_LAST for at most three source rows across local-market "
                "closures, then transform on the target asset calendar; declared "
                "sentinels and longer outages remain missing"
            ),
            "sentinel_value_rule": sentinel_rule,
            "observed_start": drivers.index.min().date().isoformat(),
            "observed_end": drivers.index.max().date().isoformat(),
            "observations": int(series.notna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "nonpositive_observations": int((finite <= 0).sum()),
            "pilot_eligible": bool(info["pilot_eligible"]),
            "pilot_exclusion_reason": pilot_reason,
            "confirmatory_eligible": bool(info["pilot_eligible"]),
            "exclusion_reason": confirmatory_reason,
            "reviewed_by": "Rule-based audit",
            "reviewed_on": "2026-09-03",
        })
    return pd.DataFrame(rows)


def transform_driver_panel(
    drivers: pd.DataFrame,
    ledger: pd.DataFrame,
    asset_calendar: pd.DatetimeIndex,
    availability_lag_asset_sessions: int = 1,
    max_carry_source_observations: int = 3,
) -> pd.DataFrame:
    """Align PX_LAST levels, transform them, and apply an asset-session lag.

    Bloomberg market series can be absent on a local holiday even when the
    target asset market is open. Each raw level is therefore carried for at
    most ``max_carry_source_observations`` rows before it is sampled on the
    target calendar. Transformations are computed *after* that sampling, so an
    unchanged local-holiday close contributes a zero innovation. Declared
    numeric sentinels remain missing and long data outages are not bridged.
    """
    if not isinstance(drivers.index, pd.DatetimeIndex):
        raise TypeError("driver panel must have a DatetimeIndex")
    if drivers.index.has_duplicates or not drivers.index.is_monotonic_increasing:
        raise ValueError("driver dates must be unique and sorted")
    if asset_calendar.has_duplicates or not asset_calendar.is_monotonic_increasing:
        raise ValueError("asset calendar must be unique and sorted")
    if availability_lag_asset_sessions < 0:
        raise ValueError("availability lag cannot be negative")
    if max_carry_source_observations < 0:
        raise ValueError("maximum carry cannot be negative")

    eligible = ledger.loc[ledger["pilot_eligible"].eq(True)].copy()
    transformed = {}
    for row in eligible.itertuples(index=False):
        original = pd.to_numeric(drivers[row.source_label], errors="coerce")
        sentinel = pd.Series(False, index=original.index)
        if row.source_label in SENTINEL_MINUS_ONE:
            sentinel = original.eq(-1)
        x = original.mask(sentinel)

        # Fill only short, ordinary source gaps. Re-mask declared sentinel
        # dates before taking the target-calendar sample.
        union = x.index.union(asset_calendar).sort_values()
        x = x.reindex(union)
        blocked = sentinel.reindex(union, fill_value=False)
        if max_carry_source_observations:
            x = x.ffill(limit=max_carry_source_observations)
        x = x.mask(blocked).reindex(asset_calendar)
        valid_pair = x.notna() & x.shift(1).notna()
        if row.approved_transformation == "LOG_DIFFERENCE":
            valid_pair &= x.gt(0) & x.shift(1).gt(0)
            innovation = np.log(x / x.shift(1)).where(valid_pair)
        elif row.approved_transformation == "DIFFERENCE":
            innovation = x.diff().where(valid_pair)
        else:
            raise ValueError(f"unsupported transformation for {row.source_label}")
        transformed[row.source_label] = innovation

    return pd.DataFrame(transformed, index=asset_calendar).shift(
        availability_lag_asset_sessions
    )
