## Citation

If you use this repository, please cite:

> Rodríguez Domínguez, A. (2026). *Causal Separation, Conditional Risk, and
> Projected Markowitz Portfolios*. Zenodo. https://doi.org/10.5281/zenodo.21224465

# Reproducibility package — *Causal Separation and Projected Markowitz Portfolios*

Complete code for every experiment, figure and table of the paper. Synthetic data
only (the data-generating process is Eq. (nf) of the paper); fixed seeds; no
tuned hyperparameters; no proprietary inputs.

## Layout

```
src/causal_markowitz/
    dgp.py         # normal-form DGP:  Z_{t+1}=Phi Z_t+eta,  r=a+GZ_t+B eta+varsigma*zeta
    estimators.py  # closed-window structured estimator; sample / Ledoit-Wolf / PCA covariances
    optimizer.py   # dense KKT solve; M and w0; O(nk^2) two-stage (Woodbury); HJ-gap check
experiments/
    e1_diagonality.py   # E1: two-window factorization (Fig. 1)
    e2_oos.py           # E2: OOS estimation risk (Tables 1-2, Fig. 2)
    e3_identities.py    # E3: five machine-precision identities (Table 3)
    e4_frontier.py      # E4: realized conditional frontier (Fig. 3)
    e5_robustness.py    # E5: sensitivity bounds under injected dependence (Fig. 4)
    e6_intervention.py  # E6: causal vs correlational under do(P) + lattice selection (Fig. 5, lattice table)
    e7_scaling.py       # E7: two-stage exactness and scaling (Fig. 6)
    run_all.py          # master script
out/figures, out/tables  # written by the scripts
```

## Setup and run

```bash
python3 -m pip install -r requirements.txt
cd experiments
python3 run_all.py        # full suite, ~4-5 minutes on a laptop (E2: 250 reps x 7 estimators)
```

Each script is also runnable standalone. Seeds are hard-coded per experiment;
rerunning reproduces every number quoted in the paper (Monte Carlo standard
errors are reported in the paper where applicable).

## Map experiment -> paper

| Script | Paper object | Validates |
|---|---|---|
| e1 | Fig. 1, Sec. 6.1 | Prop. 3.1, Rem. 2.3 (two windows) |
| e2 | Tables 1-2, Fig. 2, Sec. 6.2 | Prop. 4.5 (regularization), estimation risk |
| e3 | Table 3, Sec. 6.3 | Thm. 4.1 (P1-P4), Prop. 4.5-4.8 (floor, HJ gap, gauge, two-stage) |
| e4 | Fig. 3, Sec. 6.4 | Thm. 4.2 (frontier realized) |
| e5 | Fig. 4, Sec. 6.5 | Prop. 5.1, Cor. 5.2 |
| e6 | Fig. 5 + lattice table, Sec. 6.6 | invariance under intervention (SCM declared in the paper); exhaustive selection over the 15-subset candidate lattice; turnover/gross |
| e8 | field report CSV, Sec. 6.9 | rolling greedy selection with sample splitting, holdout certification, PCA-residual reference |
| e7 | Fig. 6, Sec. 6.7 | Prop. 4.8 (exactness + scaling) |

## Notes

- `estimators.fit_causal` fits the driver dynamics **with an intercept**; this is
  what makes affine gauge invariance exact in finite samples (Prop. 4.7 / Table 3).
- The E6 intervention replaces the proxy's structural assignment by independent
  noise with the same marginal law at mid-test; training never sees
  post-intervention data. The noise decoy W of the lattice search is drawn from
  an rng stream keyed on the replication index, so headline E6 numbers are
  bit-identical with or without the lattice computation.
- Sharpe ratios in the E2 table are reported as fractions of the same-policy
  oracle Sharpe (scale-free in the simulator's premium); turnover is one-way
  daily L1 over gross exposure. See the configuration appendix of the paper.
- Premium scales per experiment are declared at the top of each script.
