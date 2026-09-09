"""Fail-closed validation helpers for point-in-time empirical inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRUE_VALUES = {"true", "yes", "y", "1"}
FALSE_VALUES = {"false", "no", "n", "0"}


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    detail: str


def _normalise_bool(value) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def validate_asset_ledger(path: Path) -> list[GateResult]:
    if not path.exists():
        return [GateResult("asset_ledger_present", False, f"Missing {path.name}")]
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "panel", "source_label", "stable_security_id", "source_vendor",
        "source_field", "data_type", "corporate_action_adjusted",
        "delisting_return_policy", "confirmatory_eligible", "exclusion_reason",
    }
    missing = sorted(required.difference(frame.columns))
    results = [GateResult("asset_ledger_schema", not missing,
                          "Complete" if not missing else f"Missing columns: {missing}")]
    if missing:
        return results

    eligible = frame[frame["confirmatory_eligible"].map(_normalise_bool).eq(True)]
    results.append(GateResult(
        "asset_confirmatory_universe_nonempty",
        not eligible.empty,
        f"{len(eligible)} eligible rows",
    ))
    if not eligible.empty:
        adjusted = eligible["corporate_action_adjusted"].map(_normalise_bool).eq(True).all()
        stable = eligible["stable_security_id"].astype(str).str.strip().ne("").all()
        provenance = (
            eligible["source_vendor"].astype(str).str.strip().ne("")
            & eligible["source_field"].astype(str).str.strip().ne("")
        ).all()
        results.extend([
            GateResult("asset_returns_adjusted", bool(adjusted),
                       "All eligible rows verified" if adjusted else "Unverified adjustment status"),
            GateResult("asset_stable_ids", bool(stable),
                       "All eligible rows mapped" if stable else "Stable IDs missing"),
            GateResult("asset_source_provenance", bool(provenance),
                       "All eligible rows documented" if provenance else "Vendor/field missing"),
        ])
    return results


def validate_driver_ledger(path: Path) -> list[GateResult]:
    if not path.exists():
        return [GateResult("driver_ledger_present", False, f"Missing {path.name}")]
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "source_label", "economic_class", "approved_transformation", "unit",
        "source_vendor", "source_field", "availability_timestamp_or_timezone",
        "availability_lag_trading_days", "missing_value_rule", "sentinel_value_rule",
        "confirmatory_eligible", "exclusion_reason",
    }
    missing = sorted(required.difference(frame.columns))
    results = [GateResult("driver_ledger_schema", not missing,
                          "Complete" if not missing else f"Missing columns: {missing}")]
    if missing:
        return results

    status = frame["confirmatory_eligible"].map(_normalise_bool)
    resolved = status.notna().all()
    eligible = frame[status.eq(True)]
    results.append(GateResult(
        "driver_eligibility_resolved",
        bool(resolved),
        f"{int(status.notna().sum())}/{len(frame)} rows resolved",
    ))
    results.append(GateResult(
        "driver_confirmatory_universe_nonempty",
        not eligible.empty,
        f"{len(eligible)} eligible rows",
    ))
    if not eligible.empty:
        fields = [
            "economic_class", "approved_transformation", "unit", "source_vendor",
            "source_field", "availability_timestamp_or_timezone",
            "availability_lag_trading_days", "missing_value_rule", "sentinel_value_rule",
        ]
        complete = eligible[fields].astype(str).apply(lambda c: c.str.strip().ne("")).all(axis=None)
        results.append(GateResult(
            "driver_point_in_time_metadata_complete",
            bool(complete),
            "All eligible rows documented" if complete else "Required point-in-time fields missing",
        ))
    return results

