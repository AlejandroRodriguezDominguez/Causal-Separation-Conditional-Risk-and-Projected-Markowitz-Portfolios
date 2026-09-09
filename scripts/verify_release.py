#!/usr/bin/env python3
"""Fail-closed validation of the reproducibility package for the submitted manuscript."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "equal_weight", "sample_covariance", "ridge_10pct", "ledoit_wolf",
    "oas_shrinkage", "nonlinear_shrinkage_qis", "pca_matched",
    "pca_plus_two", "poet_soft_c0_5", "structured_q0", "residual_aware_qr",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_panel_b(directory: str, expected_constraint: str) -> dict:
    path = ROOT / directory / "panel_b_fold_results.csv"
    data = pd.read_csv(path)
    require(set(data["method"]) == METHODS, f"method mismatch in {directory}")
    require(set(data["portfolio_constraint"]) == {expected_constraint}, f"constraint mismatch in {directory}")
    require(data.groupby("training_days")["fold_id"].nunique().to_dict() == {252: 23, 504: 21}, f"fold mismatch in {directory}")
    numeric = ["predicted_variance", "realized_variance", "qlike", "gross_exposure", "turnover"]
    require(np.isfinite(data[numeric].to_numpy()).all(), f"nonfinite metric in {directory}")
    require((data["predicted_variance"] > 0).all(), f"non-positive forecast variance in {directory}")
    if expected_constraint == "long_only":
        require(np.allclose(data["gross_exposure"], 1.0, atol=1e-8), "long-only portfolios do not have unit gross exposure")
    return {"rows": int(len(data)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    ledger = pd.read_csv(ROOT / "development_panel/panel_a_fold_ledger.csv")
    require(ledger["fold_id"].nunique() == 38, "development-panel ledger fold count mismatch")
    multi = pd.read_csv(ROOT / "extended_results/multiasset_fold_results.csv")
    counts = multi.groupby("panel")["fold"].nunique().to_dict()
    require(counts == {"equities_150": 21, "barclays_17": 12}, f"screening fold mismatch: {counts}")
    require((multi["delta_s_f"] < 0).all(), "screening does not improve every fold")
    inference = pd.read_csv(ROOT / "extended_results/extended_inference.csv")
    require({"multiasset", "multihorizon", "state_interaction"} <= set(inference["family"]), "inference families missing")
    panel_b = pd.read_csv(ROOT / "results_inference/panel_b_inference.csv")
    require(np.isfinite(panel_b[["mean_difference", "simultaneous_ci_lower", "simultaneous_ci_upper"]].to_numpy()).all(), "Panel B inference contains nonfinite values")
    report = {
        "status": "PASS_SUBMISSION_RELEASE",
        "development_panel_folds": 38,
        "screening_folds": counts,
        "panel_b": {
            "unconstrained": validate_panel_b("results_panel_b", "unconstrained"),
            "long_only": validate_panel_b("results_panel_b_long_only", "long_only"),
            "event_exclusion": validate_panel_b("results_panel_b_event_exclusion", "unconstrained"),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
