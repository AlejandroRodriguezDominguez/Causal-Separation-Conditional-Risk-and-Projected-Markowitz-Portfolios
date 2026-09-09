#!/usr/bin/env python3
"""Run paired fold-level bootstrap and sign-flip inference for Panels A and B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qfemp.inference import paired_family_inference


PANEL_A_COMPARATORS = {
    "no_conditioning": "test_no_conditioning",
    "pca_matched": "test_pca_matched",
    "pca_plus_two": "test_pca_plus_two",
    "observable_set": "test_observable_set",
}


def panel_a_inference(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("s_f", "s_inf", "cov_offdiag_f", "cov_offdiag_spectral"):
        reference = data[f"test_selected_frozen_{metric}"]
        differences, labels = [], []
        for label, prefix in PANEL_A_COMPARATORS.items():
            differences.append((reference - data[f"{prefix}_{metric}"]).to_numpy())
            labels.append(label)
        matrix = pd.DataFrame(dict(zip(labels, differences))).to_numpy()
        family = paired_family_inference(matrix, labels, seed=20260902)
        for row in family:
            rows.append({"panel": "A", "metric": metric,
                         "reference": "selected_frozen", **row})
    return pd.DataFrame(rows)


def panel_b_inference(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparators = sorted(set(data["method"]) - {"residual_aware_qr"})
    for training_days, subset in data.groupby("training_days"):
        for metric in ("realized_variance", "qlike"):
            pivot = subset.pivot(index="fold_id", columns="method", values=metric)
            pivot = pivot.dropna(subset=["residual_aware_qr", *comparators])
            differences = pd.DataFrame({
                method: pivot["residual_aware_qr"] - pivot[method]
                for method in comparators
            })
            family = paired_family_inference(
                differences.to_numpy(), comparators,
                seed=20260902 + int(training_days) + (1 if metric == "qlike" else 0),
            )
            for row in family:
                rows.append({
                    "panel": "B", "training_days": int(training_days),
                    "metric": metric, "reference": "residual_aware_qr", **row,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, required=False, default=None)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel_a = panel_a_inference(pd.read_csv(args.panel_a)) if args.panel_a else None
    panel_b = panel_b_inference(pd.read_csv(args.panel_b))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if panel_a is not None:
        panel_a.to_csv(args.output_dir / "panel_a_inference.csv", index=False)
    panel_b.to_csv(args.output_dir / "panel_b_inference.csv", index=False)
    report = {
        "bootstrap_repetitions": 5000,
        "sign_flip_repetitions": 20000,
        "seed": 20260902,
        "panel_a_folds": int(pd.read_csv(args.panel_a)["fold_id"].nunique()) if args.panel_a else None,
        "panel_b_folds_by_training_design": {
            str(k): int(v) for k, v in pd.read_csv(args.panel_b).groupby("training_days")["fold_id"].nunique().items()
        },
        "difference_direction": "negative favors the reference method",
        "multiplicity": "familywise within metric and training design via max-t/sign-flip maximum",
    }
    (args.output_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
