# Experiment map

## Controlled experiments

The controlled suite generates all samples internally with fixed seeds. It
covers: the two information windows; covariance estimation under a correctly
specified affine representation; separate omitted-response and residual-
screening perturbations; observationally equivalent proxies under intervention;
heavy-tailed VAR(2), Student-t DCC and state-varying-loading designs; random
perturbation orientations; and gradual proxy failure.

Primary outputs are `controlled_dynamic_covariance*.csv`,
`random_perturbations*.csv` and `proxy_soft_intervention.csv`. Figures 1--6 in
the manuscript are generated from these outputs or from the corresponding
baseline scripts. The population structure is known in every simulation, so
the exercises test implementation and theoretical mechanisms rather than
estimating causal structure from market data.

## Market screening

`run_extended_experiments.py` reads the licensed source files (Tiingo equity prices, Bloomberg driver levels and BarclayHedge index returns), applies
the field-level transformation ledger, aligns observations to the target
calendar and evaluates two target panels:

- 150 equities with 504-session training and 126-session frozen test blocks;
- 17 BarclayHedge hedge-fund strategy indices with 120-month training and 12-month
  frozen test blocks.

The selector minimizes normalized off-diagonal residual correlation plus the
fixed dimension penalty `0.006 × number of selected drivers`, subject to at most
six drivers. `multiasset_fold_results.csv`,
`multiasset_driver_selections.csv` and `multiasset_summary.csv` contain the fold
results. The prespecified equity robustness checks add VIX-state interactions
and aggregate returns to five non-overlapping sessions; their results are in
`multiasset_fold_results.csv` and `multihorizon_*.csv`.

## Equity covariance and portfolios

`run_panel_b.py` compares the sample covariance, Ledoit--Wolf, nonlinear
shrinkage, OAS, ridge, matched and expanded PCA, POET, equal weighting, the
diagonal-residual estimator and its residual-aware regularization. Training
windows contain 252 or 504 sessions and each frozen test block contains 126
sessions. The regularization weight is selected using training information
only. The conditioning drivers come from `development_panel/panel_a_fold_ledger.csv` (see README). Main results are in `results_panel_b/`; long-only and event-exclusion
checks are in the correspondingly named folders.

## Inference

`run_extended_inference.py` and `run_inference.py` use paired fold resampling
and Holm adjustment within each comparison family. The manuscript reports only
experiments that address the screening restriction or its covariance and
portfolio consequences.
