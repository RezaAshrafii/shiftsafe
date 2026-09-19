"""Small deterministic baselines used as a minimum comparison point."""

from __future__ import annotations

import math

import pandas as pd


def regression_mean_baseline(y_train: pd.Series, n_rows: int) -> pd.Series:
    """Predict the training mean for every requested row."""

    if y_train.empty:
        raise ValueError("target_training_values_must_not_be_empty")
    if n_rows < 1:
        raise ValueError("n_rows_must_be_positive")
    numeric = pd.to_numeric(y_train, errors="coerce")
    if numeric.isna().any() or not numeric.map(lambda value: math.isfinite(float(value))).all():
        raise ValueError("regression_target_must_be_numeric")
    return pd.Series([float(numeric.mean())] * n_rows)


def classification_majority_baseline(y_train: pd.Series, n_rows: int) -> pd.Series:
    """Predict the most frequent training class, deterministically on ties."""

    if y_train.empty:
        raise ValueError("target_training_values_must_not_be_empty")
    if n_rows < 1:
        raise ValueError("n_rows_must_be_positive")
    if y_train.isna().any():
        raise ValueError("classification_target_must_not_contain_missing_values")
    counts = y_train.value_counts(dropna=False)
    if counts.empty:
        raise ValueError("classification_target_must_contain_a_value")
    # value_counts is deterministic for the same input order; sorting the
    # textual representation makes ties stable across scalar types.
    maximum = counts.max()
    candidates = [value for value, count in counts.items() if count == maximum]
    winner = min(candidates, key=lambda value: (type(value).__name__, str(value)))
    return pd.Series([winner] * n_rows)
