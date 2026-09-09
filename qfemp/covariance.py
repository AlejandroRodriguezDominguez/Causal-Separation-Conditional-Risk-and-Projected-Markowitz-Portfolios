"""Covariance estimators and portfolio helpers for the Panel B pilot."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(returns, dtype=float), rowvar=False, ddof=1)


def linear_ridge_covariance(returns: np.ndarray, intensity: float = 0.10) -> np.ndarray:
    s = sample_covariance(returns)
    target = np.trace(s) / s.shape[0] * np.eye(s.shape[0])
    return (1.0 - intensity) * s + intensity * target


def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    return LedoitWolf(assume_centered=False).fit(returns).covariance_


def oas_covariance(returns: np.ndarray) -> np.ndarray:
    return OAS(assume_centered=False).fit(returns).covariance_


def quadratic_inverse_shrinkage_covariance(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf quadratic-inverse nonlinear shrinkage (QIS).

    This is an independent NumPy implementation of the authors' January 2024
    reference algorithm. Observations are demeaned and the effective sample
    size is reduced by one, matching its default convention.
    """
    y = np.asarray(returns, dtype=float)
    if y.ndim != 2 or y.shape[0] < 3:
        raise ValueError("QIS requires a two-dimensional return matrix")
    observations, dimension = y.shape
    y = y - y.mean(axis=0)
    effective = observations - 1
    concentration = dimension / effective
    sample = y.T @ y / effective
    sample = (sample + sample.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    nonnull = eigenvalues[max(0, dimension - effective):]
    if np.any(nonnull <= 0):
        raise ValueError("QIS encountered a non-positive non-null eigenvalue")
    inverse = 1.0 / nonnull
    q = min(dimension, effective)
    lj = np.tile(inverse, (q, 1))
    differences = lj - lj.T
    bandwidth = min(concentration**2, concentration**-2)**0.35 / dimension**0.35
    denominator = differences**2 + bandwidth**2 * lj**2
    theta = np.mean(lj * differences / denominator, axis=1)
    hilbert_theta = np.mean(lj * (bandwidth * lj) / denominator, axis=1)
    amplitude_squared = theta**2 + hilbert_theta**2
    if dimension <= effective:
        shrunk = 1.0 / (
            (1.0 - concentration)**2 * inverse
            + 2.0 * concentration * (1.0 - concentration) * inverse * theta
            + concentration**2 * inverse * amplitude_squared
        )
    else:
        null_value = 1.0 / ((concentration - 1.0) * inverse.mean())
        shrunk = np.concatenate([
            np.repeat(null_value, dimension - effective),
            1.0 / (inverse * amplitude_squared),
        ])
    shrunk *= eigenvalues.sum() / shrunk.sum()
    covariance = (eigenvectors * shrunk) @ eigenvectors.T
    return (covariance + covariance.T) / 2


def pca_diagonal_covariance(returns: np.ndarray, components: int) -> np.ndarray:
    s = sample_covariance(returns)
    values, vectors = np.linalg.eigh(s)
    order = np.argsort(values)[::-1]
    k = min(max(int(components), 1), s.shape[0] - 1)
    vec = vectors[:, order[:k]]
    val = np.maximum(values[order[:k]], 0.0)
    factor = (vec * val) @ vec.T
    residual_diag = np.maximum(np.diag(s - factor), 0.0)
    return factor + np.diag(residual_diag)


def poet_covariance(
    returns: np.ndarray, components: int, threshold_constant: float = 0.5
) -> np.ndarray:
    """POET with soft adaptive thresholding of the residual covariance.

    The construction follows Fan, Liao, and Mincheva's POET algorithm and its
    CRAN implementation with user-specified factor count, C=0.5, soft
    thresholding, and direct residual-covariance (``vad``) thresholding.
    Positive definiteness is handled only by the common downstream floor.
    """
    y = np.asarray(returns, dtype=float)
    if y.ndim != 2 or y.shape[0] < 3:
        raise ValueError("POET requires a two-dimensional return matrix")
    y = y - y.mean(axis=0)
    observations, dimension = y.shape
    k = min(max(int(components), 1), min(observations, dimension) - 1)
    left, singular_values, right_transpose = np.linalg.svd(y, full_matrices=False)
    factors = np.sqrt(observations) * left[:, :k]
    loadings = y.T @ factors / observations
    residuals = y - factors @ loadings.T
    low_rank = loadings @ loadings.T
    residual_covariance = residuals.T @ residuals / observations
    squared = residuals**2
    product_second_moment = squared.T @ squared / observations
    product_variance = observations / (observations - 1) * np.maximum(
        product_second_moment - residual_covariance**2, 0.0
    )
    rate = 1.0 / np.sqrt(dimension) + np.sqrt(np.log(dimension) / observations)
    threshold = threshold_constant * np.sqrt(product_variance) * rate
    sparse = np.sign(residual_covariance) * np.maximum(
        np.abs(residual_covariance) - threshold, 0.0
    )
    np.fill_diagonal(sparse, np.diag(residual_covariance))
    covariance = low_rank + sparse
    return (covariance + covariance.T) / 2


def structured_covariances(returns: np.ndarray, drivers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(returns, dtype=float)
    z = np.asarray(drivers, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    yc = y - y.mean(axis=0)
    zc = z - z.mean(axis=0)
    beta = np.linalg.lstsq(zc, yc, rcond=None)[0]
    fitted = zc @ beta
    residual = yc - fitted
    driver_cov = np.atleast_2d(np.cov(zc, rowvar=False, ddof=1))
    factor_cov = beta.T @ driver_cov @ beta
    residual_cov = np.cov(residual, rowvar=False, ddof=1)
    q0 = factor_cov + np.diag(np.diag(residual_cov))
    q_full = factor_cov + residual_cov
    return q0, q_full


def eigen_floor(covariance: np.ndarray, scale: float = 1e-6) -> tuple[np.ndarray, bool]:
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    floor = scale * np.trace(covariance) / covariance.shape[0]
    floored = np.maximum(values, floor)
    activated = bool(np.any(values < floor))
    return (vectors * floored) @ vectors.T, activated


def gmv_weights(covariance: np.ndarray, floor_scale: float = 1e-6) -> tuple[np.ndarray, bool]:
    cov, activated = eigen_floor(covariance, floor_scale)
    ones = np.ones(cov.shape[0])
    solved = np.linalg.solve(cov, ones)
    denom = ones @ solved
    if not np.isfinite(denom) or abs(denom) < 1e-14:
        raise ValueError("GMV normalization is numerically invalid")
    return solved / denom, activated


def long_only_gmv_weights(
    covariance: np.ndarray, floor_scale: float = 1e-6
) -> tuple[np.ndarray, bool]:
    """Solve the fully invested, no-short-sale GMV problem."""
    cov, activated = eigen_floor(covariance, floor_scale)
    scale = float(np.median(np.diag(cov)))
    scaled = cov / scale if np.isfinite(scale) and scale > 0 else cov
    n = cov.shape[0]
    initial = np.repeat(1.0 / n, n)
    result = minimize(
        lambda weights: 0.5 * float(weights @ scaled @ weights),
        initial,
        jac=lambda weights: scaled @ weights,
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0,
                     "jac": lambda weights: np.ones(n)},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise ValueError(f"long-only GMV optimization failed: {result.message}")
    weights = np.maximum(result.x, 0.0)
    weights /= weights.sum()
    return weights, activated


def tune_residual_alpha(
    returns: np.ndarray,
    drivers: np.ndarray,
    alpha_grid: tuple[float, ...],
    floor_scale: float = 1e-6,
) -> tuple[float, list[dict]]:
    split = int(np.floor(0.75 * len(returns)))
    y_fit, y_val = returns[:split], returns[split:]
    z_fit = drivers[:split]
    q0, qfull = structured_covariances(y_fit, z_fit)
    rows = []
    for alpha in alpha_grid:
        q = (1.0 - alpha) * q0 + alpha * qfull
        weights, activated = gmv_weights(q, floor_scale)
        portfolio = y_val @ weights
        variance = float(np.var(portfolio, ddof=1))
        rows.append({
            "alpha": float(alpha), "validation_variance": variance,
            "eigen_floor_activated": activated,
        })
    best = min(rows, key=lambda row: (row["validation_variance"], row["alpha"]))
    return best["alpha"], rows


def drift_weights(weights: np.ndarray, simple_returns: np.ndarray) -> np.ndarray:
    wealth = np.prod(1.0 + simple_returns, axis=0)
    drifted = weights * wealth
    total = drifted.sum()
    if not np.isfinite(total) or abs(total) < 1e-14:
        raise ValueError("portfolio wealth invalid during drift")
    return drifted / total
