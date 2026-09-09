#!/usr/bin/env python3
"""Emit a publication-facing go/no-go report for the empirical data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qfemp.validation import GateResult, validate_asset_ledger, validate_driver_ledger


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    expected = [
        "Drivers_Assets_Dataset.xlsx",
        "processed_output.xlsx",
        "prices.xlsx",
        "filtered_output.xlsx",
    ]
    inputs = []
    gates: list[GateResult] = []
    for name in expected:
        path = args.input_dir / name
        present = path.exists()
        gates.append(GateResult(f"input_present:{name}", present,
                                "Present" if present else "Missing"))
        if present:
            inputs.append({"name": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})

    gates.extend(validate_asset_ledger(args.ledger_dir / "asset_ledger_template.csv"))
    gates.extend(validate_driver_ledger(args.ledger_dir / "driver_ledger_template.csv"))
    passed = all(gate.passed for gate in gates)
    report = {
        "decision": "GO" if passed else "NO-GO",
        "confirmatory_run_authorized": passed,
        "inputs": inputs,
        "gates": [gate.__dict__ for gate in gates],
        "next_required_inputs": [] if passed else [
            "Corporate-action-adjusted returns (CRSP including delisting returns, Bloomberg total-return indices, or an equivalently documented field) for the fixed asset universes.",
            "A completed asset ledger with stable identifiers, source vendor/field, adjustment status, and delisting-return policy.",
            "A completed driver availability ledger with transformation, unit, timestamp/timezone, lag, missing-value rule, sentinel rule, and eligibility decision for all 127 candidates.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# QF empirical preflight",
        "",
        f"**Decision: {report['decision']}**",
        "",
        "The confirmatory run is blocked unless every gate passes. An exploratory",
        "raw-price calculation may be used only for engineering tests and may not be",
        "reported as publication evidence.",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for gate in gates:
        lines.append(f"| `{gate.gate}` | {'PASS' if gate.passed else 'FAIL'} | {gate.detail} |")
    if report["next_required_inputs"]:
        lines.extend(["", "## Required to unlock the confirmatory run", ""])
        lines.extend(f"- {item}" for item in report["next_required_inputs"])
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

