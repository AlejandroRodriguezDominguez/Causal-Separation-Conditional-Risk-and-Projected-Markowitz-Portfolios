"""Deterministic point-in-time representation selection for Panel A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def residual_metrics(residuals: np.ndarray) -> dict:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or residuals.shape[0] < 3:
        raise ValueError("residual matrix must have at least three rows")
    cov = np.cov(residuals, rowvar=False, ddof=1)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    off_corr = corr - np.diag(np.diag(corr))
    off_cov = cov - np.diag(np.diag(cov))
    n = residuals.shape[1]
    normalizer = np.sqrt(n * (n - 1)) if n > 1 else 1.0
    return {
        "s_f": float(np.linalg.norm(off_corr, ord="fro") / normalizer),
        "s_inf": float(np.max(np.abs(off_corr))) if n > 1 else 0.0,
        "cov_offdiag_f": float(np.linalg.norm(off_cov, ord="fro") / normalizer),
        "cov_offdiag_spectral": float(np.linalg.norm(off_cov, ord=2)),
    }


def _s_f_from_covariance(covariance: np.ndarray) -> float:
    """Compute the selection score directly from a residual covariance."""
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance),
        where=denominator > 0,
    )
    np.fill_diagonal(correlation, 0.0)
    n = covariance.shape[0]
    normalizer = np.sqrt(n * (n - 1)) if n > 1 else 1.0
    return float(np.linalg.norm(correlation, ord="fro") / normalizer)


def _standardize(train: np.ndarray, other: np.ndarray | None = None):
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_z = (train - mean) / scale
    other_z = None if other is None else (other - mean) / scale
    return train_z, other_z


def fit_and_residualize(
    y_train: np.ndarray,
    x_train: np.ndarray,
    y_eval: np.ndarray | None = None,
    x_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    y_train = np.asarray(y_train, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim == 1:
        x_train = x_train[:, None]
    x_train_z, x_eval_z = _standardize(x_train, x_eval)
    design = np.column_stack([np.ones(len(x_train_z)), x_train_z])
    beta = np.linalg.lstsq(design, y_train, rcond=None)[0]
    train_resid = y_train - design @ beta
    eval_resid = None
    if y_eval is not None:
        if x_eval_z is None:
            raise ValueError("x_eval is required with y_eval")
        eval_design = np.column_stack([np.ones(len(x_eval_z)), x_eval_z])
        eval_resid = np.asarray(y_eval, dtype=float) - eval_design @ beta
    return train_resid, eval_resid


def baseline_residuals(y_train: np.ndarray, y_eval: np.ndarray | None = None):
    mean = np.mean(y_train, axis=0)
    train = y_train - mean
    other = None if y_eval is None else y_eval - mean
    return train, other


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[int, ...]
    path: tuple[dict, ...]
    baseline_score: float
    final_score: float


def greedy_select(
    y: np.ndarray,
    x: np.ndarray,
    *,
    penalty: float = 0.01,
    max_drivers: int = 8,
    tolerance: float = 1e-12,
) -> SelectionResult:
    """Forward selection followed by backward deletion with stable ties."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    # Candidate scoring only depends on the regression column space. Centering
    # once and using cross-products is algebraically equivalent to repeatedly
    # fitting intercept-plus-standardized regressions, but makes the frozen
    # moving-block bootstrap computationally tractable.
    yc = y - y.mean(axis=0)
    xc = x - x.mean(axis=0)
    degrees = y.shape[0] - 1
    yy = yc.T @ yc
    xx = xc.T @ xc
    xy = xc.T @ yc
    base_score = _s_f_from_covariance(yy / degrees)

    def score_subset(indices: list[int]) -> float:
        if not indices:
            return base_score
        gram = xx[np.ix_(indices, indices)]
        cross = xy[indices]
        fitted_cross = cross.T @ np.linalg.pinv(gram, hermitian=True) @ cross
        residual_covariance = (yy - fitted_cross) / degrees
        return _s_f_from_covariance(residual_covariance)
    selected: list[int] = []
    path: list[dict] = []
    current_obj = base_score

    while len(selected) < max_drivers:
        best = None
        for candidate in range(x.shape[1]):
            if candidate in selected:
                continue
            trial = selected + [candidate]
            score = score_subset(trial)
            obj = score + penalty * len(trial)
            item = (obj, candidate, score)
            if best is None or item[:2] < best[:2]:
                best = item
        if best is None or best[0] >= current_obj - tolerance:
            break
        current_obj, candidate, score = best
        selected.append(candidate)
        path.append({
            "operation": "ADD", "driver_index": int(candidate),
            "model_size": len(selected), "s_f": float(score),
            "objective": float(current_obj),
        })

    improved = True
    while improved and selected:
        improved = False
        best = None
        for candidate in selected:
            trial = [j for j in selected if j != candidate]
            score = score_subset(trial)
            obj = score + penalty * len(trial)
            item = (obj, candidate, score, trial)
            if best is None or item[:2] < best[:2]:
                best = item
        if best is not None and best[0] < current_obj - tolerance:
            current_obj, candidate, score, selected = best
            path.append({
                "operation": "DELETE", "driver_index": int(candidate),
                "model_size": len(selected), "s_f": float(score),
                "objective": float(current_obj),
            })
            improved = True

    final_score = score_subset(selected)
    return SelectionResult(tuple(selected), tuple(path), base_score, final_score)


def pca_scores(
    y_train: np.ndarray,
    x_train: np.ndarray,
    y_eval: np.ndarray,
    x_eval: np.ndarray,
    n_components: int,
) -> dict:
    xz, xeval_z = _standardize(x_train, x_eval)
    _, _, vt = np.linalg.svd(xz, full_matrices=False)
    k = min(max(int(n_components), 1), vt.shape[0])
    train_pc = xz @ vt[:k].T
    eval_pc = xeval_z @ vt[:k].T
    _, resid = fit_and_residualize(y_train, train_pc, y_eval, eval_pc)
    return residual_metrics(resid)
