"""Auditable price-return reconstruction for a fixed equity panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_SPLIT_FACTORS = np.array([
    1 / 20, 1 / 10, 1 / 8, 1 / 7, 1 / 5, 1 / 4, 1 / 3, 1 / 2.5,
    1 / 2, 2, 2.5, 3, 4, 5, 7, 8, 10, 20,
], dtype=float)

ALLOWED_DISPOSITIONS = {"APPLY_FACTOR", "EXCLUDE_DATE", "KEEP_OBSERVED"}


@dataclass(frozen=True)
class ReconstructionResult:
    returns: pd.DataFrame
    corporate_actions: pd.DataFrame
    unresolved_moves: pd.DataFrame
    report: dict


def reconstruct_fixed_panel_price_returns(
    prices: pd.DataFrame,
    *,
    start: str,
    end: str,
    split_factors: np.ndarray = DEFAULT_SPLIT_FACTORS,
    candidate_log_threshold: float = 0.5,
    split_relative_tolerance: float = 0.05,
    unresolved_log_threshold: float = 0.40,
    event_ledger: pd.DataFrame | None = None,
) -> ReconstructionResult:
    """Construct synchronous log price returns without imputing missing prices.

    A return is retained only when every asset has a positive price at both the
    current and immediately preceding source date. Verified ledger decisions
    take precedence over the mechanical split rule. ``EXCLUDE_DATE`` removes
    the full synchronous date because a price-only panel cannot reconstruct the
    wealth return for a spin-off or distribution without the distributed asset.
    """
    if prices.empty:
        raise ValueError("price panel is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("price panel must have a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("price panel contains duplicate dates")
    prices = prices.sort_index().loc[start:end].copy()
    if prices.empty:
        raise ValueError("no prices remain inside the requested date range")
    if prices.columns.duplicated().any():
        raise ValueError("price panel contains duplicate ticker labels")

    numeric = prices.apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric) | (numeric <= 0)
    valid_pair = ~(invalid | invalid.shift(1, fill_value=True)).any(axis=1)
    raw_ratio = numeric / numeric.shift(1)
    raw_log = np.log(raw_ratio)
    synchronous = raw_log.loc[valid_pair].copy()
    missing_or_invalid_drop_count = len(numeric) - 1 - len(synchronous)

    ledger_columns = [
        "date", "ticker", "disposition", "factor", "verification_status",
        "event_type", "source_url", "rationale",
    ]
    if event_ledger is None:
        ledger = pd.DataFrame(columns=ledger_columns)
    else:
        missing = set(ledger_columns).difference(event_ledger.columns)
        if missing:
            raise ValueError(f"event ledger is missing columns: {sorted(missing)}")
        ledger = event_ledger[ledger_columns].copy()
        ledger["date"] = pd.to_datetime(ledger["date"], errors="raise").dt.date.astype(str)
        ledger["ticker"] = ledger["ticker"].astype(str)
        ledger["disposition"] = ledger["disposition"].astype(str)
        unknown = sorted(set(ledger["disposition"]).difference(ALLOWED_DISPOSITIONS))
        if unknown:
            raise ValueError(f"unknown event-ledger dispositions: {unknown}")
        if ledger.duplicated(["date", "ticker"]).any():
            raise ValueError("event ledger contains duplicate date/ticker keys")
        apply_rows = ledger["disposition"].eq("APPLY_FACTOR")
        factor = pd.to_numeric(ledger.loc[apply_rows, "factor"], errors="coerce")
        if factor.isna().any() or (factor <= 0).any():
            raise ValueError("APPLY_FACTOR ledger rows require a positive factor")

    ledger_lookup = {
        (row.date, row.ticker): row._asdict()
        for row in ledger.itertuples(index=False)
    }
    logged_keys: set[tuple[str, str]] = set()
    excluded_dates: set[pd.Timestamp] = set()

    action_rows: list[dict] = []
    for date, ticker in zip(*np.where(synchronous.abs().to_numpy() >= candidate_log_threshold)):
        dt = synchronous.index[date]
        label = synchronous.columns[ticker]
        ratio = float(raw_ratio.at[dt, label])
        key = (dt.date().isoformat(), str(label))
        override = ledger_lookup.get(key)
        relative_error = np.abs(ratio - split_factors) / split_factors
        match_index = int(np.argmin(relative_error))
        matched_factor = float(split_factors[match_index])
        matched_error = float(relative_error[match_index])
        accepted = matched_error <= split_relative_tolerance
        disposition = None if override is None else override["disposition"]
        if disposition == "APPLY_FACTOR":
            matched_factor = float(override["factor"])
            matched_error = abs(ratio - matched_factor) / matched_factor
            adjusted_log_return = float(np.log(ratio / matched_factor))
            status = "ADJUSTED_VERIFIED_FACTOR"
        elif disposition == "EXCLUDE_DATE":
            adjusted_log_return = float(np.log(ratio))
            excluded_dates.add(dt)
            status = "EXCLUDED_VERIFIED_DISTRIBUTION_DATE"
        elif disposition == "KEEP_OBSERVED":
            adjusted_log_return = float(np.log(ratio))
            status = "RETAINED_VERIFIED_MARKET_MOVE"
        elif accepted:
            adjusted_log_return = float(np.log(ratio / matched_factor))
            status = "ADJUSTED_MECHANICAL_SPLIT_CANDIDATE"
        else:
            adjusted_log_return = float(np.log(ratio))
            status = "UNMATCHED_LARGE_MOVE"
        if status in {"ADJUSTED_VERIFIED_FACTOR", "ADJUSTED_MECHANICAL_SPLIT_CANDIDATE"}:
            synchronous.at[dt, label] = adjusted_log_return
        prior_pos = numeric.index.get_loc(dt) - 1
        action_rows.append({
            "date": dt.date().isoformat(),
            "ticker": str(label),
            "prior_source_date": numeric.index[prior_pos].date().isoformat(),
            "prior_price": float(numeric.iloc[prior_pos][label]),
            "current_price": float(numeric.at[dt, label]),
            "raw_price_ratio": ratio,
            "raw_log_return": float(np.log(ratio)),
            "matched_split_factor": matched_factor,
            "relative_error": matched_error,
            "rule_status": status,
            "verification_status": (
                "MECHANICAL_RULE_ONLY" if override is None else override["verification_status"]
            ),
            "event_type": "" if override is None else override["event_type"],
            "source_url": "" if override is None else override["source_url"],
            "rationale": "" if override is None else override["rationale"],
            "adjusted_log_return": adjusted_log_return,
        })
        logged_keys.add(key)
    automatic_candidate_count = len(action_rows)

    # Ledgered moves below the candidate threshold must still be auditable.
    for row in ledger.itertuples(index=False):
        key = (row.date, row.ticker)
        if key in logged_keys:
            continue
        dt = pd.Timestamp(row.date)
        if dt not in synchronous.index or row.ticker not in synchronous.columns:
            continue
        ratio = float(raw_ratio.at[dt, row.ticker])
        adjusted_log_return = float(np.log(ratio))
        if row.disposition == "APPLY_FACTOR":
            adjusted_log_return = float(np.log(ratio / float(row.factor)))
            synchronous.at[dt, row.ticker] = adjusted_log_return
            status = "ADJUSTED_VERIFIED_FACTOR"
        elif row.disposition == "EXCLUDE_DATE":
            excluded_dates.add(dt)
            status = "EXCLUDED_VERIFIED_DISTRIBUTION_DATE"
        else:
            status = "RETAINED_VERIFIED_MARKET_MOVE"
        prior_pos = numeric.index.get_loc(dt) - 1
        action_rows.append({
            "date": row.date,
            "ticker": row.ticker,
            "prior_source_date": numeric.index[prior_pos].date().isoformat(),
            "prior_price": float(numeric.iloc[prior_pos][row.ticker]),
            "current_price": float(numeric.at[dt, row.ticker]),
            "raw_price_ratio": ratio,
            "raw_log_return": float(np.log(ratio)),
            "matched_split_factor": float(row.factor) if row.disposition == "APPLY_FACTOR" else np.nan,
            "relative_error": (
                abs(ratio - float(row.factor)) / float(row.factor)
                if row.disposition == "APPLY_FACTOR" else np.nan
            ),
            "rule_status": status,
            "verification_status": row.verification_status,
            "event_type": row.event_type,
            "source_url": row.source_url,
            "rationale": row.rationale,
            "adjusted_log_return": adjusted_log_return,
        })

    if excluded_dates:
        synchronous = synchronous.drop(index=sorted(excluded_dates), errors="ignore")

    corporate_actions = pd.DataFrame(action_rows)
    residual_mask = synchronous.abs() > unresolved_log_threshold
    unresolved_rows = []
    for row, col in zip(*np.where(residual_mask.to_numpy())):
        dt, label = synchronous.index[row], synchronous.columns[col]
        key = (dt.date().isoformat(), str(label))
        if key in ledger_lookup:
            continue
        unresolved_rows.append({
            "date": dt.date().isoformat(),
            "ticker": str(label),
            "adjusted_log_return": float(synchronous.iat[row, col]),
            "adjusted_simple_return": float(np.expm1(synchronous.iat[row, col])),
            "status": "REVIEW_REQUIRED",
        })
    unresolved = pd.DataFrame(unresolved_rows)

    rows_in_range = len(numeric)
    report = {
        "data_product": "fixed-universe split-adjusted log price returns",
        "not_equivalent_to": ["CRSP RET/DLRET", "total returns", "point-in-time index membership"],
        "source_rows_in_range": rows_in_range,
        "assets": int(numeric.shape[1]),
        "retained_return_rows": int(len(synchronous)),
        "dropped_return_rows_due_to_missing_or_invalid_endpoint": int(missing_or_invalid_drop_count),
        "dropped_return_rows_due_to_verified_distribution": int(len(excluded_dates)),
        "first_return_date": synchronous.index.min().date().isoformat(),
        "last_return_date": synchronous.index.max().date().isoformat(),
        "automatic_large_move_candidates": int(automatic_candidate_count),
        "event_audit_rows": int(len(corporate_actions)),
        "verified_factor_adjustments": int((corporate_actions.get("rule_status", pd.Series(dtype=str)) == "ADJUSTED_VERIFIED_FACTOR").sum()),
        "mechanical_split_adjustments": int((corporate_actions.get("rule_status", pd.Series(dtype=str)) == "ADJUSTED_MECHANICAL_SPLIT_CANDIDATE").sum()),
        "unmatched_large_candidates": int((corporate_actions.get("rule_status", pd.Series(dtype=str)) == "UNMATCHED_LARGE_MOVE").sum()),
        "verified_distribution_dates_excluded": int(len(excluded_dates)),
        "verified_market_moves_retained": int((corporate_actions.get("rule_status", pd.Series(dtype=str)) == "RETAINED_VERIFIED_MARKET_MOVE").sum()),
        "unresolved_post_adjustment_moves": int(len(unresolved)),
        "dividends_included": False,
        "missing_prices_imputed": False,
        "return_across_missing_source_observation": False,
        "survivorship_conditioned_fixed_universe": True,
        "publication_status": (
            "READY_FOR_FIXED_UNIVERSE_PRICE_RETURN_ANALYSIS"
            if len(unresolved) == 0 else "PROVISIONAL_PENDING_EVENT_REVIEW"
        ),
    }
    return ReconstructionResult(synchronous, corporate_actions, unresolved, report)
