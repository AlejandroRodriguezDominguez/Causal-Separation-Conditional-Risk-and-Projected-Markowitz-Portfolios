#!/usr/bin/env python3
"""Run the extended QF experiment suite and generate publication tables/figures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qfemp.drivers import build_driver_ledger, transform_driver_panel
from qfemp.extended import (
    evaluate_representation, nonoverlap_sum,
    proxy_break_experiment, random_perturbation_experiment,
    simulate_covariance_designs, selection_similarity,
)


def read_dates(values) -> pd.DatetimeIndex:
    s = pd.Series(values).astype(str).str.replace(r"\.0$", "", regex=True)
    compact = s.str.fullmatch(r"\d{8}")
    out = pd.to_datetime(s, errors="coerce")
    out.loc[compact] = pd.to_datetime(s.loc[compact], format="%Y%m%d", errors="coerce")
    return pd.DatetimeIndex(out)


def load_drivers(raw: Path, out: Path):
    frame = pd.read_excel(raw / "Drivers_Assets_Dataset.xlsx", sheet_name="Drivers")
    frame.index = read_dates(frame.pop(frame.columns[0]))
    frame = frame.loc[~frame.index.isna()].sort_index()
    ledger = build_driver_ledger(frame)
    out.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out / "driver_ledger_bloomberg_px_last.csv", index=False)
    return frame, ledger


def load_equities(repro: Path):
    p = repro / "data" / "panel_b_log_price_returns.csv"
    x = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    return x.sort_index()


def load_indices(raw: Path, out: Path):
    x = pd.read_excel(raw / "HF_data.xlsx", sheet_name="data")
    x.index = read_dates(x.pop(x.columns[0]))
    x = x.loc[~x.index.isna()].sort_index().astype(float)
    x.to_csv(out / "barclays_17_index_returns.csv.gz", compression="gzip")
    return x


def complete_alignment(y: pd.DataFrame, x: pd.DataFrame, state: pd.Series | None = None):
    idx = y.index.intersection(x.index)
    yy, xx = y.loc[idx], x.loc[idx]
    # Restrict to the common empirical support before measuring column
    # coverage. This prevents pre-inception/post-end dates in a longer target
    # panel from mechanically excluding otherwise complete Bloomberg series.
    active = xx.notna().mean(axis=1) >= .80
    yy, xx = yy.loc[active], xx.loc[active]
    coverage = xx.notna().mean()
    xx = xx.loc[:, coverage >= .98]
    joined = pd.concat([yy, xx], axis=1).dropna()
    yy = joined.iloc[:, :y.shape[1]]
    xx = joined.iloc[:, y.shape[1]:]
    ss = None if state is None else state.reindex(joined.index).ffill().bfill()
    return yy, xx, ss


def summarize_folds(panel: str, rows: list[dict]):
    d = pd.DataFrame(rows)
    result = {
        "panel": panel, "folds": len(d),
        "assets": None,
        "mean_delta_s_f": float(d.delta_s_f.mean()),
        "median_relative_reduction_s_f": float(d.relative_reduction_s_f.median()),
        "improved_fraction": float((d.delta_s_f < 0).mean()),
        "median_selected_count": float(d.selected_count.median()),
    }
    if "state_delta_vs_constant_s_f" in d:
        result["mean_state_delta_vs_constant_s_f"] = float(d.state_delta_vs_constant_s_f.mean())
        result["state_improved_fraction"] = float((d.state_delta_vs_constant_s_f < 0).mean())
    return result


def run_multiasset(raw: Path, repro: Path, out: Path, figures: Path):
    level, ledger = load_drivers(raw, out)
    vix = level["Cboe Volatility Index"]
    equities = load_equities(repro)
    indices = load_indices(raw, out)

    # Compute innovations after sampling Bloomberg levels on each target asset
    # calendar. This preserves legitimate zero changes on local holidays.
    equity_drivers = transform_driver_panel(level, ledger, equities.index, 0)
    monthly_drivers = transform_driver_panel(level, ledger, indices.index, 0)
    equity_drivers.to_csv(
        out / "driver_innovations_equity_calendar.csv.gz", compression="gzip"
    )

    panels = []
    eq, ex, es = complete_alignment(equities, equity_drivers, vix)
    panels.append(("equities_150", eq, ex, es, 504, 126, 220))

    # Monthly innovations use month-end sampled PX_LAST levels directly.
    month_driver = monthly_drivers
    month_vix = vix.resample("ME").last()
    hi, hx, hs = complete_alignment(indices, month_driver, month_vix)
    panels.append(("barclays_17", hi, hx, hs, 120, 12, 220))

    all_rows, all_sel, summaries = [], [], []
    aligned = {}
    for name, y, x, state, train, test, limit in panels:
        rows, sel = evaluate_representation(
            y.to_numpy(), x.to_numpy(), list(x.columns), y.index.to_numpy(),
            train=train, test=test, max_drivers=6, penalty=.006,
            asset_limit=limit, state=state.to_numpy(),
        )
        for row in rows:
            row.update({"panel": name, "assets": y.shape[1], "observations": len(y)})
        for row in sel:
            row["panel"] = name
        all_rows.extend(rows); all_sel.extend(sel)
        s = summarize_folds(name, rows); s["assets"] = y.shape[1]; s["observations"] = len(y)
        summaries.append(s); aligned[name] = (y, x, state)

    pd.DataFrame(all_rows).to_csv(out / "multiasset_fold_results.csv", index=False)
    pd.DataFrame(all_sel).to_csv(out / "multiasset_driver_selections.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "multiasset_summary.csv", index=False)
    pd.DataFrame(selection_similarity(all_sel)).to_csv(out / "multiasset_selection_similarity.csv", index=False)

    # Horizon experiment, with windows expressed in aggregated observations.
    horizon_rows = []
    for panel in ("equities_150",):
        y, x, _ = aligned[panel]
        for h in (1, 5):
            ya, da = nonoverlap_sum(y.to_numpy(), y.index.to_numpy(), h)
            xa, _ = nonoverlap_sum(x.to_numpy(), x.index.to_numpy(), h)
            if h == 1:
                train, test = 504, 126
            else:
                train, test = 104, 26
            rows, _ = evaluate_representation(
                ya, xa, list(x.columns), da, train=train, test=test,
                max_drivers=6, penalty=.006, asset_limit=220,
            )
            for row in rows:
                horizon_rows.append({"panel": panel, "horizon_sessions": h, **row})
    pd.DataFrame(horizon_rows).to_csv(out / "multihorizon_fold_results.csv", index=False)
    hsumm = (pd.DataFrame(horizon_rows).groupby(["panel", "horizon_sessions"])
             .agg(folds=("fold", "count"), mean_delta_s_f=("delta_s_f", "mean"),
                  median_relative_reduction=("relative_reduction_s_f", "median"),
                  improved_fraction=("delta_s_f", lambda z: float((z < 0).mean())))
             .reset_index())
    hsumm.to_csv(out / "multihorizon_summary.csv", index=False)

    return summaries


def run_controlled(out: Path, figures: Path):
    cov = pd.DataFrame(simulate_covariance_designs())
    cov.to_csv(out / "controlled_dynamic_covariance.csv", index=False)
    covsum = (cov.groupby(["design", "method"])
              .agg(median_frobenius=("frobenius_relative", "median"),
                   median_calibration=("calibration", "median"),
                   median_realized_variance=("realized_variance", "median"))
              .reset_index())
    covsum.to_csv(out / "controlled_dynamic_covariance_summary.csv", index=False)

    pert = pd.DataFrame(random_perturbation_experiment())
    pert.to_csv(out / "random_perturbations.csv", index=False)
    perts = (pert.groupby(["rho", "channel"])
             .agg(median_weight_error=("weight_error", "median"),
                  median_first_order=("first_order", "median"),
                  coverage=("bound_covers", "mean"),
                  median_frontier_change=("frontier_change", "median"))
             .reset_index())
    perts.to_csv(out / "random_perturbations_summary.csv", index=False)

    proxy = pd.DataFrame(proxy_break_experiment())
    proxy.to_csv(out / "proxy_soft_intervention.csv", index=False)

    figures.mkdir(parents=True, exist_ok=True)
    methods = ["sample", "ledoit_wolf", "pca4", "structured_q0", "residual_qr_50"]
    designs = ["ar1_gaussian", "var2_student", "dcc_student", "state_loadings"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    pivot = covsum.pivot(index="design", columns="method", values="median_frobenius").loc[designs, methods]
    x=np.arange(len(designs)); width=.15
    for j,m in enumerate(methods): axes[0].bar(x+(j-2)*width,pivot[m],width,label=m)
    axes[0].set_xticks(x, ["AR(1)", "VAR(2)-t", "DCC-t", "State loadings"], rotation=15)
    axes[0].set_ylabel("Median relative Frobenius loss")
    cp = covsum.pivot(index="design", columns="method", values="median_calibration").loc[designs, methods]
    for j,m in enumerate(methods): axes[1].bar(x+(j-2)*width,cp[m],width,label=m)
    axes[1].axhline(1,color="black",lw=.8); axes[1].set_xticks(x,["AR(1)","VAR(2)-t","DCC-t","State loadings"],rotation=15)
    axes[1].set_ylabel("Median realized/predicted variance")
    axes[1].legend(frameon=False,fontsize=7,ncol=2); fig.tight_layout()
    fig.savefig(figures / "e9_dynamic_misspecification.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for channel,g in perts.groupby("channel"):
        axes[0].plot(g.rho,g.median_weight_error,marker="o",label=channel)
    axes[0].set_xlabel(r"Perturbation size $\rho$"); axes[0].set_ylabel("Median relative weight displacement")
    axes[0].legend(frameon=False)
    for noise,g in proxy.groupby("proxy_noise"):
        axes[1].plot(g.break_strength,g.gap,marker="o",label=f"noise={noise:.1f}")
    axes[1].set_xlabel("Soft-intervention strength"); axes[1].set_ylabel("Proxy minus state residual score")
    axes[1].legend(frameon=False); fig.tight_layout()
    fig.savefig(figures / "e10_robustness_extensions.pdf"); plt.close(fig)
    return covsum, perts, proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "extended_results")
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "extended_figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = run_multiasset(args.raw_dir, ROOT, args.output_dir, args.figures_dir)
    covsum, perts, proxy = run_controlled(args.output_dir, args.figures_dir)
    manifest = {
        "status": "complete", "random_seed": 20260903,
        "source": {"vendor": "Bloomberg", "field": "PX_LAST",
                   "equity_close": "U.S. close",
                   "equity_prices": "Tiingo daily end-of-day close", "additional_indices": "17 BarclayHedge hedge-fund strategy indices"},
        "multiasset": summaries,
        "controlled_rows": int(len(pd.read_csv(args.output_dir / "controlled_dynamic_covariance.csv"))),
    }
    (args.output_dir / "extended_experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
