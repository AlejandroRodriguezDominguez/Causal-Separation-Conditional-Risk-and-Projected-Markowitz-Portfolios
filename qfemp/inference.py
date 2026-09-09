"""Paired fold-level inference for non-overlapping out-of-sample blocks."""

from __future__ import annotations

import numpy as np


def paired_family_inference(
    differences: np.ndarray,
    labels: list[str],
    bootstrap_repetitions: int = 5000,
    randomization_repetitions: int = 20000,
    seed: int = 20260902,
) -> list[dict]:
    """Return raw and familywise-adjusted inference for paired differences.

    Columns are comparisons and rows are non-overlapping test folds. Negative
    differences favor the reference method. Simultaneous confidence intervals
    use a bootstrap max-t critical value. P-values use paired sign-flip tests,
    with the adjusted value based on the maximum absolute t statistic within
    the supplied family.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("differences and labels have incompatible dimensions")
    if values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("paired inference requires at least three finite folds")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / np.sqrt(n)
    if np.any(standard_errors <= 0):
        raise ValueError("paired difference has zero sampling variation")

    indices = rng.integers(0, n, size=(bootstrap_repetitions, n))
    bootstrap_means = values[indices].mean(axis=1)
    bootstrap_se = bootstrap_means.std(axis=0, ddof=1)
    bootstrap_se = np.maximum(bootstrap_se, np.finfo(float).eps)
    maximum_t = np.max(
        np.abs((bootstrap_means - estimates) / bootstrap_se), axis=1
    )
    critical = float(np.quantile(maximum_t, 0.95))

    observed_t = estimates / standard_errors
    exceed_raw = np.zeros(values.shape[1], dtype=int)
    exceed_adjusted = np.zeros(values.shape[1], dtype=int)
    remaining = randomization_repetitions
    batch_size = 2000
    while remaining:
        batch = min(batch_size, remaining)
        signs = rng.choice((-1.0, 1.0), size=(batch, n, 1))
        randomized = signs * values[None, :, :]
        randomized_mean = randomized.mean(axis=1)
        randomized_se = randomized.std(axis=1, ddof=1) / np.sqrt(n)
        randomized_t = np.divide(
            randomized_mean,
            randomized_se,
            out=np.zeros_like(randomized_mean),
            where=randomized_se > 0,
        )
        absolute = np.abs(randomized_t)
        exceed_raw += np.sum(absolute >= np.abs(observed_t), axis=0)
        family_max = absolute.max(axis=1)
        exceed_adjusted += np.sum(
            family_max[:, None] >= np.abs(observed_t)[None, :], axis=0
        )
        remaining -= batch

    rows = []
    for j, label in enumerate(labels):
        lower, upper = np.quantile(bootstrap_means[:, j], [0.025, 0.975])
        rows.append({
            "comparison": label,
            "folds": int(n),
            "mean_difference": float(estimates[j]),
            "standard_error": float(standard_errors[j]),
            "bootstrap_ci_lower": float(lower),
            "bootstrap_ci_upper": float(upper),
            "simultaneous_ci_lower": float(estimates[j] - critical * bootstrap_se[j]),
            "simultaneous_ci_upper": float(estimates[j] + critical * bootstrap_se[j]),
            "sign_flip_p_raw": float((exceed_raw[j] + 1) / (randomization_repetitions + 1)),
            "sign_flip_p_familywise": float((exceed_adjusted[j] + 1) / (randomization_repetitions + 1)),
            "fraction_reference_better": float(np.mean(values[:, j] < 0)),
        })
    return rows
