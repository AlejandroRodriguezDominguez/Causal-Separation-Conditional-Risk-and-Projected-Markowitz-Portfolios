"""Pre-specified field-data evaluation protocol (paper, Sec. 6.9).

Runs on any user-supplied panel: a CSV of asset returns (dates x assets) and a
CSV of candidate driver levels or returns (dates x drivers), date-aligned on
the index column. For each rolling window the protocol
  1. splits the window into a selection segment and a disjoint holdout segment;
  2. selects a candidate set by greedy forward selection with backward
     elimination on the selection segment (objective: closed-window maximal
     absolute residual correlation, the working dependence functional);
  3. certifies eps_hat on the holdout segment (sample splitting: the certified
     value is evaluated on data not used for selection);
  4. evaluates the same functional on the residuals of a PCA factor model of
     matched dimension, on the same holdout, as the reference;
and writes a per-window report (selected set, eps_select, eps_holdout,
eps_pca_holdout) plus a stability summary (set survival across windows).

No performance metric is computed: the object of the protocol is the
certificate. See the paper for the rationale and the reporting conventions.
"""
import csv
import numpy as np
from . import estimators


def _maxcorr(resid):
    C = np.corrcoef(resid.T)
    np.fill_diagonal(C, 0)
    return float(np.abs(C).max())


def _eps(R, D, idx, sl):
    """Closed-window residual max |corr| for driver columns idx on slice sl."""
    fit = estimators.fit_causal(R[sl], D[sl.start:sl.stop + 1][:, idx])
    return _maxcorr(fit["resid"])


def greedy_select(R, D, sl, kappa=0.0):
    """Forward selection with backward elimination on slice sl.

    Returns (selected column indices, eps on sl). kappa is the optional
    cardinality penalty of the paper's selection remark.
    """
    M = D.shape[1]
    remaining, chosen = list(range(M)), []
    best = np.inf
    while remaining:
        scores = [(_eps(R, D, chosen + [j], sl) + kappa * (len(chosen) + 1), j)
                  for j in remaining]
        s, j = min(scores)
        if s >= best - 1e-12:
            break
        best = s
        chosen.append(j)
        remaining.remove(j)
    improved = True
    while improved and len(chosen) > 1:
        improved = False
        for j in list(chosen):
            trial = [c for c in chosen if c != j]
            s = _eps(R, D, trial, sl) + kappa * len(trial)
            if s <= best + 1e-12:
                chosen, best, improved = trial, min(best, s), True
                break
    return chosen, _eps(R, D, chosen, sl)


def run_protocol(returns_csv, drivers_csv, out_csv, window=756, step=126,
                 select_frac=0.6, kappa=0.01):
    """Execute the protocol; returns the list of per-window records."""
    def load(path):
        with open(path) as f:
            rows = list(csv.reader(f))
        header, dates = rows[0][1:], [r[0] for r in rows[1:]]
        X = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
        return header, dates, X

    anames, adates, R = load(returns_csv)
    dnames, ddates, D = load(drivers_csv)
    common = sorted(set(adates) & set(ddates))
    ra = {d: i for i, d in enumerate(adates)}
    rd = {d: i for i, d in enumerate(ddates)}
    R = R[[ra[d] for d in common]]
    D = D[[rd[d] for d in common]]
    T = len(common)
    records = []
    for start in range(0, T - window, step):
        cut = start + int(select_frac * window)
        sel_sl = slice(start, cut)
        hold_sl = slice(cut, start + window)
        chosen, eps_sel = greedy_select(R, D, sel_sl, kappa)
        eps_hold = _eps(R, D, chosen, hold_sl)
        Qp = estimators.cov_pca(R[hold_sl], max(len(chosen), 1))
        # PCA residual functional: residual after removing top-k PCs, same holdout
        Xc = R[hold_sl] - R[hold_sl].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = max(len(chosen), 1)
        resid_pca = Xc - Xc @ Vt[:k].T @ Vt[:k]
        records.append({
            "window_start": common[start], "window_end": common[start + window - 1],
            "selected": "+".join(dnames[j] for j in chosen),
            "eps_select": round(eps_sel, 4), "eps_holdout": round(eps_hold, 4),
            "eps_pca_holdout": round(_maxcorr(resid_pca), 4)})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    sets = [r["selected"] for r in records]
    stab = max((sets.count(s) for s in set(sets)), default=0) / max(len(sets), 1)
    print(f"windows: {len(records)} | modal-set stability: {stab:.2f} | report: {out_csv}")
    return records
