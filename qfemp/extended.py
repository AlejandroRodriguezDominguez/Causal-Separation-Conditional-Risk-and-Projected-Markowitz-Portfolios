"""Extended empirical and controlled experiments for the QF revision.

The routines in this module are deterministic, use temporally ordered outer
tests, and keep the information cutoff explicit.  They deliberately separate
contemporaneous representation tests from predictive portfolio exercises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr


def offdiag_metrics(values: np.ndarray) -> dict[str, float]:
    """Correlation diagnostics without changing the asset scale."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or y.shape[0] < 3 or y.shape[1] < 2:
        raise ValueError("values must have at least three rows and two columns")
    c = np.cov(y, rowvar=False, ddof=1)
    sd = np.sqrt(np.maximum(np.diag(c), 1e-18))
    r = c / np.outer(sd, sd)
    np.fill_diagonal(r, 0.0)
    n = r.shape[0]
    scale = np.sqrt(n * (n - 1))
    # Use the smaller time-domain spectrum when n_assets exceeds n_dates.
    yc = y - y.mean(axis=0)
    singular = np.linalg.svd(yc, compute_uv=False)
    eig = np.maximum(singular**2 / (len(y) - 1), 0.0)
    p = eig / max(eig.sum(), 1e-18)
    effective_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    return {
        "s_f": float(np.linalg.norm(r, "fro") / scale),
        "s_inf": float(np.max(np.abs(r))),
        "effective_rank": effective_rank,
    }


def fit_residuals(
    y_train: np.ndarray,
    x_train: np.ndarray,
    y_test: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    x0 = np.asarray(x_train, float)
    x1 = np.asarray(x_test, float)
    if x0.ndim == 1:
        x0, x1 = x0[:, None], x1[:, None]
    mu = x0.mean(axis=0)
    sd = x0.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0
    d0 = np.column_stack([np.ones(len(x0)), (x0 - mu) / sd])
    d1 = np.column_stack([np.ones(len(x1)), (x1 - mu) / sd])
    beta = np.linalg.lstsq(d0, np.asarray(y_train, float), rcond=None)[0]
    return np.asarray(y_test, float) - d1 @ beta


def baseline_residuals(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    return np.asarray(y_test, float) - np.asarray(y_train, float).mean(axis=0)


def _score_from_crossproducts(yy: np.ndarray, xx: np.ndarray, xy: np.ndarray,
                              indices: list[int], dof: int) -> float:
    if indices:
        gram = xx[np.ix_(indices, indices)]
        cross = xy[indices]
        yy = yy - cross.T @ np.linalg.pinv(gram, hermitian=True) @ cross
    c = yy / dof
    sd = np.sqrt(np.maximum(np.diag(c), 1e-18))
    r = c / np.outer(sd, sd)
    np.fill_diagonal(r, 0.0)
    n = r.shape[0]
    return float(np.linalg.norm(r, "fro") / np.sqrt(n * (n - 1)))


def greedy_driver_selection(
    y: np.ndarray,
    x: np.ndarray,
    *,
    max_drivers: int = 6,
    penalty: float = 0.006,
    asset_limit: int = 220,
) -> list[int]:
    """Forward selection on a deterministic asset subset.

    The subset only controls computational cost for very wide bond panels.  A
    selected representation is always refitted and evaluated on the full
    outer-test asset panel.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if y.shape[1] > asset_limit:
        idx = np.linspace(0, y.shape[1] - 1, asset_limit).round().astype(int)
        y = y[:, np.unique(idx)]
    yc = y - y.mean(axis=0)
    xc = x - x.mean(axis=0)
    yy, xx, xy = yc.T @ yc, xc.T @ xc, xc.T @ yc
    dof = len(y) - 1
    selected: list[int] = []
    current = _score_from_crossproducts(yy, xx, xy, selected, dof)
    current_obj = current
    for _ in range(max_drivers):
        candidates = []
        for j in range(x.shape[1]):
            if j in selected:
                continue
            trial = selected + [j]
            score = _score_from_crossproducts(yy, xx, xy, trial, dof)
            candidates.append((score + penalty * len(trial), j, score))
        if not candidates:
            break
        best = min(candidates)
        if best[0] >= current_obj - 1e-12:
            break
        current_obj = best[0]
        selected.append(best[1])
    return selected


@dataclass(frozen=True)
class Fold:
    fold: int
    train: slice
    test: slice


def rolling_folds(n: int, train: int, test: int, step: int | None = None) -> list[Fold]:
    step = test if step is None else step
    folds, start, k = [], 0, 0
    while start + train + test <= n:
        folds.append(Fold(k, slice(start, start + train),
                          slice(start + train, start + train + test)))
        start += step
        k += 1
    return folds


def evaluate_representation(
    y: np.ndarray,
    x: np.ndarray,
    names: list[str],
    dates: Iterable,
    *,
    train: int,
    test: int,
    max_drivers: int = 6,
    penalty: float = 0.006,
    asset_limit: int = 220,
    state: np.ndarray | None = None,
) -> tuple[list[dict], list[dict]]:
    """Frozen rolling screening with an optional state-interaction model."""
    rows, selections = [], []
    date_values = np.asarray(list(dates))
    for fold in rolling_folds(len(y), train, test):
        yt, yo = y[fold.train], y[fold.test]
        xt, xo = x[fold.train], x[fold.test]
        chosen = greedy_driver_selection(
            yt, xt, max_drivers=max_drivers, penalty=penalty,
            asset_limit=asset_limit,
        )
        base = baseline_residuals(yt, yo)
        if chosen:
            resid = fit_residuals(yt, xt[:, chosen], yo, xo[:, chosen])
        else:
            resid = base
        bm, sm = offdiag_metrics(base), offdiag_metrics(resid)
        row = {
            "fold": fold.fold,
            "train_start": str(date_values[fold.train.start])[:10],
            "train_end": str(date_values[fold.train.stop - 1])[:10],
            "test_start": str(date_values[fold.test.start])[:10],
            "test_end": str(date_values[fold.test.stop - 1])[:10],
            "selected_count": len(chosen),
            "selected_drivers": " | ".join(names[j] for j in chosen),
            **{f"baseline_{k}": v for k, v in bm.items()},
            **{f"selected_{k}": v for k, v in sm.items()},
            "delta_s_f": sm["s_f"] - bm["s_f"],
            "relative_reduction_s_f": (bm["s_f"] - sm["s_f"]) / bm["s_f"],
        }
        if state is not None and chosen:
            st, so = state[fold.train], state[fold.test]
            threshold = float(np.median(st))
            it, io = (st > threshold).astype(float), (so > threshold).astype(float)
            zt = xt[:, chosen]
            zo = xo[:, chosen]
            zt_state = np.column_stack([zt, it, zt * it[:, None]])
            zo_state = np.column_stack([zo, io, zo * io[:, None]])
            state_resid = fit_residuals(yt, zt_state, yo, zo_state)
            stm = offdiag_metrics(state_resid)
            row.update({f"state_{k}": v for k, v in stm.items()})
            row["state_delta_vs_constant_s_f"] = stm["s_f"] - sm["s_f"]
            row["state_threshold"] = threshold
        rows.append(row)
        for j in chosen:
            selections.append({"fold": fold.fold, "driver": names[j]})
    return rows, selections


def nonoverlap_sum(values: np.ndarray, dates: np.ndarray, horizon: int):
    """Non-overlapping sums ending at a common observation date."""
    n = len(values) // horizon
    if n < 1:
        return values[:0], dates[:0]
    trimmed = values[len(values) - n * horizon:]
    out = trimmed.reshape(n, horizon, *values.shape[1:]).sum(axis=1)
    d = dates[len(dates) - n * horizon:].reshape(n, horizon)[:, -1]
    return out, d


def projection_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """Mean squared cosine of principal angles between leading loading spaces."""
    _, _, va = np.linalg.svd(a - a.mean(axis=0), full_matrices=False)
    _, _, vb = np.linalg.svd(b - b.mean(axis=0), full_matrices=False)
    ka = min(k, len(va), len(vb))
    angles = subspace_angles(va[:ka].T, vb[:ka].T)
    return float(np.mean(np.cos(angles) ** 2))


def gmv(cov: np.ndarray, long_only: bool = False) -> np.ndarray:
    cov = (cov + cov.T) / 2
    eig, vec = np.linalg.eigh(cov)
    floor = max(1e-10, 1e-6 * np.trace(cov) / len(cov))
    cov = (vec * np.maximum(eig, floor)) @ vec.T
    if long_only:
        # Stable projected-gradient solver; sufficient for a robustness check.
        w = np.repeat(1 / len(cov), len(cov))
        lip = max(np.linalg.eigvalsh(cov).max(), 1e-12)
        for _ in range(4000):
            old = w.copy()
            w = _simplex_projection(w - cov @ w / lip)
            if np.linalg.norm(w - old) < 1e-11:
                break
        return w
    z = np.linalg.solve(cov, np.ones(len(cov)))
    return z / z.sum()


def _simplex_projection(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u - cssv / np.arange(1, len(v) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def random_perturbation_experiment(seed: int = 20260903) -> list[dict]:
    """Random ranks, sparse residual errors, and simultaneous channels."""
    rng = np.random.default_rng(seed)
    rows = []
    n = 40
    a = np.ones((1, n))
    nsp = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, n - 1))]))[0][:, 1:]
    mu = rng.normal(0.04, 0.02, n)
    base = np.diag(rng.uniform(0.02, 0.05, n) ** 2)
    for rho in (0.2, 0.5, 0.8):
        for channel in ("response", "screening", "joint"):
            for draw in range(120):
                rank = int(rng.integers(1, 7))
                u = rng.normal(size=(n, rank))
                omega = u @ u.T
                omega /= np.linalg.norm(omega, 2)
                mask = rng.random((n, n)) < 0.08
                e = rng.normal(size=(n, n)) * mask
                e = (e + e.T) / 2
                np.fill_diagonal(e, 0)
                if np.linalg.norm(e, 2) > 0:
                    e /= np.linalg.norm(e, 2)
                scale = rho * np.min(np.diag(base))
                r = scale * ({"response": omega, "screening": e,
                              "joint": 0.65 * omega + 0.35 * e}[channel])
                q1 = base + r
                mine = np.linalg.eigvalsh(q1).min()
                if mine <= 1e-9:
                    q1 += (1e-9 - mine) * np.eye(n)
                m0 = nsp @ np.linalg.inv(nsp.T @ base @ nsp) @ nsp.T
                m1 = nsp @ np.linalg.inv(nsp.T @ q1 @ nsp) @ nsp.T
                w0, w1 = m0 @ mu, m1 @ mu
                exact = np.linalg.norm(w1 - w0) / max(np.linalg.norm(w0), 1e-18)
                first = np.linalg.norm(-m0 @ r @ w0) / max(np.linalg.norm(w0), 1e-18)
                bound = (np.linalg.norm(m0 @ r, 2) /
                         max(1 - np.linalg.norm(m0 @ r, 2), 1e-12))
                d0, d1 = mu @ m0 @ mu, mu @ m1 @ mu
                rows.append({"rho": rho, "channel": channel, "draw": draw,
                             "rank": rank, "weight_error": exact,
                             "first_order": first, "bound": bound,
                             "frontier_change": (d1 - d0) / d0,
                             "bound_covers": bound + 1e-12 >= exact})
    return rows


def proxy_break_experiment(seed: int = 20260903) -> list[dict]:
    """Noisy proxy and gradual soft intervention."""
    rng = np.random.default_rng(seed)
    rows = []
    n, train, test = 30, 800, 400
    load = rng.normal(size=(2, n))
    for noise in (0.0, 0.1, 0.3):
        for strength in np.linspace(0, 1, 6):
            metrics_true, metrics_proxy = [], []
            for _ in range(60):
                z = np.zeros((train + test, 2))
                alt = np.zeros(train + test)
                for t in range(1, len(z)):
                    z[t] = 0.9 * z[t - 1] + rng.normal(size=2) * np.sqrt(1 - .9**2)
                    alt[t] = 0.9 * alt[t - 1] + rng.normal() * np.sqrt(1 - .9**2)
                y = z @ load + rng.normal(scale=.5, size=(len(z), n))
                p = z[:, 0] + noise * rng.normal(size=len(z))
                p[train:] = ((1 - strength) * z[train:, 0] +
                             strength * alt[train:] + noise * rng.normal(size=test))
                rt = fit_residuals(y[:train], z[:train], y[train:], z[train:])
                rp = fit_residuals(y[:train], np.column_stack([p[:train], z[:train, 1]]),
                                   y[train:], np.column_stack([p[train:], z[train:, 1]]))
                metrics_true.append(offdiag_metrics(rt)["s_f"])
                metrics_proxy.append(offdiag_metrics(rp)["s_f"])
            rows.append({"proxy_noise": noise, "break_strength": float(strength),
                         "true_s_f": float(np.mean(metrics_true)),
                         "proxy_s_f": float(np.mean(metrics_proxy)),
                         "gap": float(np.mean(metrics_proxy) - np.mean(metrics_true))})
    return rows


def simulate_covariance_designs(seed: int = 20260903) -> list[dict]:
    """Covariance estimation under four increasingly misspecified dynamics."""
    from sklearn.covariance import LedoitWolf, OAS
    from .covariance import pca_diagonal_covariance, structured_covariances

    rng = np.random.default_rng(seed)
    rows = []
    n, k, train, test, reps = 60, 4, 504, 252, 80
    for design in ("ar1_gaussian", "var2_student", "dcc_student", "state_loadings"):
        for rep in range(reps):
            total = train + test
            z = np.zeros((total, k))
            h = np.repeat(.25, k)
            qbar = .25 * np.ones((k, k)) + .75 * np.eye(k)
            q = qbar.copy()
            standardized_previous = np.zeros(k)
            innovation_previous = np.zeros(k)
            state = np.zeros(total)
            for t in range(2, total):
                if design == "var2_student":
                    a1 = .65 * np.eye(k) + .08 * (np.ones((k, k)) - np.eye(k)) / (k - 1)
                    shock = rng.standard_t(6, k) / np.sqrt(6 / 4)
                    z[t] = a1 @ z[t - 1] + .18 * z[t - 2] + .5 * shock
                elif design == "dcc_student":
                    h = .02 + .06 * innovation_previous**2 + .90 * h
                    q = .03 * np.outer(standardized_previous, standardized_previous) + .95 * q + .02 * qbar
                    d = np.sqrt(np.maximum(np.diag(q), 1e-12))
                    corr = q / np.outer(d, d)
                    eig, vec = np.linalg.eigh((corr + corr.T) / 2)
                    corr = (vec * np.maximum(eig, 1e-8)) @ vec.T
                    d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
                    standardized = rng.multivariate_normal(np.zeros(k), corr)
                    standardized *= np.sqrt(5 / rng.chisquare(5)) / np.sqrt(5 / 3)
                    innovation = np.sqrt(h) * standardized
                    z[t] = .65 * z[t - 1] + innovation
                    standardized_previous = standardized
                    innovation_previous = innovation
                else:
                    z[t] = .92 * z[t - 1] + rng.normal(size=k) * np.sqrt(1 - .92**2)
                state[t] = .96 * state[t - 1] + rng.normal() * np.sqrt(1 - .96**2)
            load = rng.normal(0, .012, size=(k, n))
            y = np.empty((total, n))
            for t in range(total):
                lt = load
                if design == "state_loadings":
                    lt = load * (1 + .55 * np.tanh(state[t]))
                scale = .007 + .007 * (1 + np.tanh(state[t])) / 2
                noise = rng.standard_t(7, n) / np.sqrt(7 / 5) if design != "ar1_gaussian" else rng.normal(size=n)
                y[t] = z[t] @ lt + scale * noise
            yt, yo, zt = y[:train], y[train:], z[:train]
            truth = np.cov(yo, rowvar=False, ddof=1)
            estimators = {
                "sample": np.cov(yt, rowvar=False, ddof=1),
                "ledoit_wolf": LedoitWolf().fit(yt).covariance_,
                "oas": OAS().fit(yt).covariance_,
                "pca4": pca_diagonal_covariance(yt, 4),
            }
            q0, qfull = structured_covariances(yt, zt)
            estimators["structured_q0"] = q0
            estimators["residual_qr_50"] = .5 * q0 + .5 * qfull
            for name, cov in estimators.items():
                eig, vec = np.linalg.eigh((cov + cov.T) / 2)
                cov = (vec * np.maximum(eig, 1e-10)) @ vec.T
                w = gmv(cov)
                realized = float(np.var(yo @ w, ddof=1))
                predicted = float(w @ cov @ w)
                rows.append({"design": design, "rep": rep, "method": name,
                             "frobenius_relative": float(np.linalg.norm(cov - truth, "fro") /
                                                         np.linalg.norm(truth, "fro")),
                             "realized_variance": realized,
                             "calibration": realized / predicted})
    return rows


def selection_similarity(selection_rows: list[dict]) -> list[dict]:
    by = {}
    for row in selection_rows:
        by.setdefault(row["panel"], set()).add(row["driver"])
    out = []
    keys = sorted(by)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            union = by[a] | by[b]
            out.append({"panel_a": a, "panel_b": b,
                        "jaccard_selected_labels": len(by[a] & by[b]) / max(len(union), 1)})
    return out


def rank_agreement(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)
