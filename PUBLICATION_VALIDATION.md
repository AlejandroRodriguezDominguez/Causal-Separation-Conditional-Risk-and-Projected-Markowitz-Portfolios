# Publication validation

Run date: 5 September 2026

- All 26 unit tests pass (`python -m unittest discover -s tests -v`).
- `scripts/verify_release.py` passes on the shipped outputs.
- `scripts/run_extended_inference.py` regenerates `extended_results/extended_inference.csv`
  deterministically (seed 20260903); its Holm-adjusted values are the ones reported in the manuscript.
- Equity and hedge-fund screening use rolling, temporally ordered folds, frozen test blocks,
  paired resampling and Holm adjustment; the covariance experiment freezes selection and
  tuning before every non-overlapping test block.
- The package contains no licensed source files and no series from which licensed data
  could be reconstructed.
