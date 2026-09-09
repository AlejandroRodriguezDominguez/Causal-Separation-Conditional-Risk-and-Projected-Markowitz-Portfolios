"""Time-series fold construction with explicit no-leakage invariants."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class PanelAFold:
    fold_id: int
    selection_start: pd.Timestamp
    selection_end: pd.Timestamp
    certification_start: pd.Timestamp
    certification_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in asdict(self).items()}


def build_panel_a_folds(
    dates,
    selection_days: int = 504,
    certification_days: int = 252,
    test_days: int = 126,
) -> list[PanelAFold]:
    index = pd.DatetimeIndex(dates)
    if index.has_duplicates:
        raise ValueError("dates contain duplicates")
    if not index.is_monotonic_increasing:
        raise ValueError("dates must be strictly increasing")
    if min(selection_days, certification_days, test_days) <= 0:
        raise ValueError("fold lengths must be positive")

    development_days = selection_days + certification_days
    folds: list[PanelAFold] = []
    for test_start in range(development_days, len(index) - test_days + 1, test_days):
        selection_start = test_start - development_days
        certification_start = test_start - certification_days
        fold = PanelAFold(
            fold_id=len(folds) + 1,
            selection_start=index[selection_start],
            selection_end=index[certification_start - 1],
            certification_start=index[certification_start],
            certification_end=index[test_start - 1],
            test_start=index[test_start],
            test_end=index[test_start + test_days - 1],
        )
        assert fold.selection_end < fold.certification_start
        assert fold.certification_end < fold.test_start
        folds.append(fold)
    return folds


@dataclass(frozen=True)
class PanelBFold:
    fold_id: int
    training_days: int
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in asdict(self).items()}


def build_panel_b_folds(dates, training_days: int, test_days: int = 126) -> list[PanelBFold]:
    index = pd.DatetimeIndex(dates)
    if index.has_duplicates:
        raise ValueError("dates contain duplicates")
    if not index.is_monotonic_increasing:
        raise ValueError("dates must be strictly increasing")
    if min(training_days, test_days) <= 0:
        raise ValueError("fold lengths must be positive")
    folds = []
    for test_start in range(training_days, len(index) - test_days + 1, test_days):
        train_start = test_start - training_days
        fold = PanelBFold(
            fold_id=len(folds) + 1,
            training_days=training_days,
            training_start=index[train_start],
            training_end=index[test_start - 1],
            test_start=index[test_start],
            test_end=index[test_start + test_days - 1],
        )
        assert fold.training_end < fold.test_start
        folds.append(fold)
    return folds
