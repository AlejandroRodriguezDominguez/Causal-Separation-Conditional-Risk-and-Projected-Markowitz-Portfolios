#!/usr/bin/env python3
"""Run deterministic, point-in-time Panel A representation selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qfemp.folds import build_panel_a_folds
from qfemp.selection import (
    baseline_residuals, fit_and_residualize, greedy_select, pca_scores,
    residual_metrics,
)


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-returns", type=Path, required=True)
    parser.add_argument("--driver-innovations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-days", type=int, default=504)
    parser.add_argument("--certification-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument("--max-drivers", type=int, default=8)
    args = parser.parse_args()

    ydf = pd.read_csv(args.asset_returns, parse_dates=["date"]).set_index("date")
    xdf = pd.read_csv(args.driver_innovations, parse_dates=["date"]).set_index("date")
    common = ydf.index.intersection(xdf.index)
    ydf, xdf = ydf.loc[common], xdf.loc[common]
    if ydf.isna().any().any() or xdf.isna().any().any():
        raise ValueError("Panel A inputs must be common and complete")
    folds = build_panel_a_folds(
        common, args.selection_days, args.certification_days, args.test_days
    )
    names = list(xdf.columns)
    observable_names = [
        name for name in [
            "Cboe Volatility Index", "DOLLAR INDEX SPOT", "U.S. Treasury",
            "Gold Spot   $/Oz", "BBG Commodity",
        ] if name in names
    ]
    result_rows, path_rows, fold_rows = [], [], []

    for fold in folds:
        sel_mask = (common >= fold.selection_start) & (common <= fold.selection_end)
        cert_mask = (common >= fold.certification_start) & (common <= fold.certification_end)
        test_mask = (common >= fold.test_start) & (common <= fold.test_end)
        y_sel, x_sel = ydf.loc[sel_mask].to_numpy(), xdf.loc[sel_mask].to_numpy()
        y_cert, x_cert = ydf.loc[cert_mask].to_numpy(), xdf.loc[cert_mask].to_numpy()
        y_test, x_test = ydf.loc[test_mask].to_numpy(), xdf.loc[test_mask].to_numpy()

        selection = greedy_select(
            y_sel, x_sel, penalty=args.penalty, max_drivers=args.max_drivers
        )
        selected = list(selection.selected)
        selected_names = [names[i] for i in selected]

        base_sel, base_cert = baseline_residuals(y_sel, y_cert)
        base_dev, base_test = baseline_residuals(
            np.vstack([y_sel, y_cert]), y_test
        )
        if selected:
            _, selected_cert = fit_and_residualize(
                y_sel, x_sel[:, selected], y_cert, x_cert[:, selected]
            )
            _, selected_test = fit_and_residualize(
                np.vstack([y_sel, y_cert]),
                np.vstack([x_sel[:, selected], x_cert[:, selected]]),
                y_test, x_test[:, selected],
            )
        else:
            selected_cert, selected_test = base_cert, base_test

        k = max(len(selected), 1)
        dev_x = np.vstack([x_sel, x_cert])
        dev_y = np.vstack([y_sel, y_cert])
        pca_k = pca_scores(dev_y, dev_x, y_test, x_test, k)
        pca_k2 = pca_scores(dev_y, dev_x, y_test, x_test, k + 2)
        obs_idx = [names.index(name) for name in observable_names]
        _, obs_test = fit_and_residualize(
            dev_y, dev_x[:, obs_idx], y_test, x_test[:, obs_idx]
        )

        row = {
            "fold_id": fold.fold_id,
            "selected_count": len(selected_names),
            "selected_drivers": " | ".join(selected_names),
            "selection_objective": selection.final_score + args.penalty * len(selected),
            **prefixed("selection_no_conditioning", residual_metrics(base_sel)),
            **prefixed("selection_selected", residual_metrics(
                base_sel if not selected else fit_and_residualize(y_sel, x_sel[:, selected])[0]
            )),
            **prefixed("certification_no_conditioning", residual_metrics(base_cert)),
            **prefixed("certification_selected_frozen", residual_metrics(selected_cert)),
            **prefixed("test_no_conditioning", residual_metrics(base_test)),
            **prefixed("test_selected_frozen", residual_metrics(selected_test)),
            **prefixed("test_pca_matched", pca_k),
            **prefixed("test_pca_plus_two", pca_k2),
            **prefixed("test_observable_set", residual_metrics(obs_test)),
        }
        result_rows.append(row)
        for step, item in enumerate(selection.path, start=1):
            path_rows.append({
                "fold_id": fold.fold_id, "step": step,
                "operation": item["operation"],
                "driver": names[item["driver_index"]],
                "model_size": item["model_size"], "s_f": item["s_f"],
                "objective": item["objective"],
            })
        fold_rows.append({
            **fold.to_dict(),
            "selected_count": len(selected_names),
            "selected_drivers": " | ".join(selected_names),
            "driver_pool_count": len(names),
            "penalty": args.penalty,
            "max_drivers": args.max_drivers,
            "selection_dates_used": int(sel_mask.sum()),
            "certification_dates_used": int(cert_mask.sum()),
            "test_dates_used": int(test_mask.sum()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    paths = pd.DataFrame(path_rows)
    fold_ledger = pd.DataFrame(fold_rows)
    results.to_csv(args.output_dir / "panel_a_fold_results.csv", index=False)
    paths.to_csv(args.output_dir / "panel_a_selection_path.csv", index=False)
    fold_ledger.to_csv(args.output_dir / "panel_a_fold_ledger.csv", index=False)

    delta = results["test_no_conditioning_s_f"] - results["test_selected_frozen_s_f"]
    report = {
        "folds": int(len(results)),
        "driver_pool": int(len(names)),
        "observable_comparator": observable_names,
        "median_selected_count": float(results["selected_count"].median()),
        "mean_selected_count": float(results["selected_count"].mean()),
        "median_test_s_f_reduction": float(delta.median()),
        "mean_test_s_f_reduction": float(delta.mean()),
        "fraction_test_folds_improved": float((delta > 0).mean()),
        "selection_window_days": args.selection_days,
        "certification_window_days": args.certification_days,
        "outer_test_days": args.test_days,
        "penalty": args.penalty,
        "max_drivers": args.max_drivers,
        "status": "PILOT_PANEL_A_ESTIMATION_COMPLETE",
    }
    (args.output_dir / "panel_a_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
