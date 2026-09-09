#!/usr/bin/env python3
"""Run Panel B covariance and static-GMV pilot comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qfemp.covariance import (
    drift_weights, eigen_floor, gmv_weights, ledoit_wolf_covariance,
    long_only_gmv_weights,
    linear_ridge_covariance, oas_covariance, pca_diagonal_covariance,
    poet_covariance, quadratic_inverse_shrinkage_covariance,
    sample_covariance, structured_covariances,
    tune_residual_alpha,
)
from qfemp.folds import build_panel_b_folds


ALPHA_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)
COST_GRID_BPS = (5, 10, 20)


def latest_selection(panel_a: pd.DataFrame, date: pd.Timestamp) -> tuple[int, list[str]]:
    eligible = panel_a[pd.to_datetime(panel_a["certification_end"]) <= date]
    if eligible.empty:
        raise ValueError(f"no Panel A selection is available by {date.date()}")
    row = eligible.sort_values("certification_end").iloc[-1]
    labels = [] if pd.isna(row["selected_drivers"]) else str(row["selected_drivers"]).split(" | ")
    return int(row["fold_id"]), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-returns", type=Path, required=True)
    parser.add_argument("--driver-innovations", type=Path, required=True)
    parser.add_argument("--panel-a-folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-days", type=int, nargs="+", default=[252, 504])
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--floor-scale", type=float, default=1e-6)
    parser.add_argument(
        "--portfolio-constraint", choices=("unconstrained", "long_only"),
        default="unconstrained",
    )
    args = parser.parse_args()

    returns = pd.read_csv(args.asset_returns, parse_dates=["date"]).set_index("date")
    drivers = pd.read_csv(args.driver_innovations, parse_dates=["date"]).set_index("date")
    panel_a = pd.read_csv(args.panel_a_folds)
    common = returns.index.intersection(drivers.dropna().index)
    returns, drivers = returns.loc[common], drivers.loc[common]
    if returns.isna().any().any() or drivers.isna().any().any():
        raise ValueError("Panel B inputs must be finite and complete")

    fold_rows, tuning_rows, pnl_rows = [], [], []
    previous = {}
    for training_days in args.training_days:
        folds = build_panel_b_folds(common, training_days, args.test_days)
        for fold in folds:
            train = returns.loc[fold.training_start:fold.training_end]
            test = returns.loc[fold.test_start:fold.test_end]
            ztrain_all = drivers.loc[train.index]
            panel_a_fold, selected = latest_selection(panel_a, fold.training_end)
            if not selected:
                raise ValueError(f"empty selected representation for Panel B fold {fold.fold_id}")
            ztrain = ztrain_all[selected].to_numpy()
            # The input panel stores log returns.  Convert once so estimation,
            # validation, realized risk, and portfolio drift all use the same
            # economically investable simple-return convention.
            ytrain = np.expm1(train.to_numpy())
            ytest_simple = np.expm1(test.to_numpy())
            k = len(selected)

            alpha, tuning = tune_residual_alpha(
                ytrain, ztrain, ALPHA_GRID, args.floor_scale
            )
            for item in tuning:
                tuning_rows.append({
                    "training_days": training_days, "fold_id": fold.fold_id,
                    "panel_a_fold_id": panel_a_fold, **item,
                })
            q0, qfull = structured_covariances(ytrain, ztrain)
            estimators = {
                "equal_weight": sample_covariance(ytrain),
                "sample_covariance": sample_covariance(ytrain),
                "ridge_10pct": linear_ridge_covariance(ytrain, 0.10),
                "ledoit_wolf": ledoit_wolf_covariance(ytrain),
                "oas_shrinkage": oas_covariance(ytrain),
                "nonlinear_shrinkage_qis": quadratic_inverse_shrinkage_covariance(ytrain),
                "pca_matched": pca_diagonal_covariance(ytrain, k),
                "pca_plus_two": pca_diagonal_covariance(ytrain, k + 2),
                "poet_soft_c0_5": poet_covariance(ytrain, k),
                "structured_q0": q0,
                "residual_aware_qr": (1.0 - alpha) * q0 + alpha * qfull,
            }

            for method, covariance in estimators.items():
                covariance, floor_activated = eigen_floor(covariance, args.floor_scale)
                if method == "equal_weight":
                    weights = np.repeat(1.0 / returns.shape[1], returns.shape[1])
                elif args.portfolio_constraint == "long_only":
                    weights, second_floor = long_only_gmv_weights(
                        covariance, args.floor_scale
                    )
                    floor_activated = floor_activated or second_floor
                else:
                    weights, second_floor = gmv_weights(covariance, args.floor_scale)
                    floor_activated = floor_activated or second_floor
                portfolio = ytest_simple @ weights
                realized_var = float(np.var(portfolio, ddof=1))
                predicted_var = float(weights @ covariance @ weights)
                ratio = realized_var / predicted_var if predicted_var > 0 else np.nan
                if not np.isfinite(predicted_var) or predicted_var <= 0:
                    raise ValueError(
                        f"non-positive predicted variance after eigenvalue floor: "
                        f"method={method}, fold={fold.fold_id}, value={predicted_var}"
                    )
                qlike = float(np.log(predicted_var) + realized_var / predicted_var)
                key = (training_days, method)
                if key in previous:
                    prior_weights, prior_returns = previous[key]
                    pretrade = drift_weights(prior_weights, prior_returns)
                    turnover = float(np.abs(weights - pretrade).sum())
                    initial = False
                else:
                    turnover = float(np.abs(weights).sum())
                    initial = True
                previous[key] = (weights, ytest_simple)

                row = {
                    **fold.to_dict(), "panel_a_fold_id": panel_a_fold,
                    "selected_count": k, "selected_drivers": " | ".join(selected),
                    "method": method, "chosen_alpha": alpha if method == "residual_aware_qr" else np.nan,
                    "portfolio_constraint": args.portfolio_constraint,
                    "predicted_variance": predicted_var,
                    "realized_variance": realized_var,
                    "realized_annualized_volatility": float(np.sqrt(252 * realized_var)),
                    "realized_to_predicted_ratio": ratio,
                    "qlike": qlike,
                    "gross_exposure": float(np.abs(weights).sum()),
                    "max_abs_weight": float(np.abs(weights).max()),
                    "turnover": turnover,
                    "initial_formation": initial,
                    "eigen_floor_activated": floor_activated,
                }
                for bps in COST_GRID_BPS:
                    cost = 0.0 if initial else turnover * bps / 10000.0
                    net = portfolio.copy()
                    net[0] -= cost
                    row[f"net_mean_return_{bps}bps"] = float(np.mean(net))
                    row[f"net_annualized_volatility_{bps}bps"] = float(np.std(net, ddof=1) * np.sqrt(252))
                    row[f"rebalance_cost_{bps}bps"] = cost
                fold_rows.append(row)
                for date, gross in zip(test.index, portfolio):
                    pnl_rows.append({
                        "training_days": training_days, "fold_id": fold.fold_id,
                        "date": date.date().isoformat(), "method": method,
                        "gross_return": float(gross),
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = pd.DataFrame(fold_rows)
    tuning_results = pd.DataFrame(tuning_rows)
    pnl = pd.DataFrame(pnl_rows)
    fold_results.to_csv(args.output_dir / "panel_b_fold_results.csv", index=False)
    tuning_results.to_csv(args.output_dir / "panel_b_alpha_tuning.csv", index=False)
    pnl.to_csv(args.output_dir / "panel_b_daily_pnl.csv", index=False)

    summary_rows = []
    for (training_days, method), group in fold_results.groupby(["training_days", "method"]):
        day = pnl[(pnl["training_days"] == training_days) & (pnl["method"] == method)]
        summary_rows.append({
            "training_days": training_days, "method": method,
            "folds": int(len(group)),
            "pooled_annualized_volatility": float(day["gross_return"].std(ddof=1) * np.sqrt(252)),
            "median_fold_annualized_volatility": float(group["realized_annualized_volatility"].median()),
            "median_realized_to_predicted_ratio": float(group["realized_to_predicted_ratio"].median()),
            "mean_qlike": float(group["qlike"].mean()),
            "median_gross_exposure": float(group["gross_exposure"].median()),
            "median_max_abs_weight": float(group["max_abs_weight"].median()),
            "mean_recurring_turnover": float(group.loc[~group["initial_formation"], "turnover"].mean()),
            "eigen_floor_activation_fraction": float(group["eigen_floor_activated"].mean()),
            "median_chosen_alpha": float(group["chosen_alpha"].median()) if method == "residual_aware_qr" else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "panel_b_summary.csv", index=False)

    report = {
        "common_dates": int(len(common)),
        "first_date": common.min().date().isoformat(),
        "last_date": common.max().date().isoformat(),
        "assets": int(returns.shape[1]),
        "training_designs": args.training_days,
        "portfolio_constraint": args.portfolio_constraint,
        "estimators": sorted(fold_results["method"].unique()),
        "folds_by_training_design": {
            str(k): int(v) for k, v in fold_results.groupby("training_days")["fold_id"].nunique().items()
        },
        "status": "PILOT_PANEL_B_ESTIMATION_COMPLETE",
    }
    (args.output_dir / "panel_b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
