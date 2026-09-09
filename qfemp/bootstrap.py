"""Moving-block bootstrap utilities for paired financial time series."""

from __future__ import annotations

import numpy as np


def moving_block_indices(
    length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample overlapping blocks and truncate to the requested length."""
    if length <= 0 or block_length <= 0 or block_length > length:
        raise ValueError("block length must lie between one and sample length")
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length - block_length + 1, size=blocks)
    offsets = np.arange(block_length)
    return (starts[:, None] + offsets).ravel()[:length]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return float(center - half), float(center + half)
