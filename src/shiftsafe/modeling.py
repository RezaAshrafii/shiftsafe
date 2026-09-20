"""A deliberately small, deterministic numeric regression model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import regression_metrics
from .splitting import group_split, temporal_split


def _numeric_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if not feature_columns:
        raise ValueError("feature_columns_must_not_be_empty")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("feature_columns_missing")
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("feature_values_must_be_numeric_and_finite")
    return values.to_numpy(dtype=float)


def fit_linear_regression(train: pd.DataFrame, target_column: str, feature_columns: list[str]) -> np.ndarray:
    """Fit an ordinary least-squares model and return intercept + coefficients."""

    if target_column not in train.columns:
        raise ValueError("target_column_missing")
    target = pd.to_numeric(train[target_column], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("target_values_must_be_numeric_and_finite")
    features = _numeric_matrix(train, feature_columns)
    design = np.column_stack([np.ones(len(features)), features])
    coefficients, _, _, _ = np.linalg.lstsq(design, target.to_numpy(dtype=float), rcond=None)
    return coefficients


def predict_linear_regression(frame: pd.DataFrame, coefficients: np.ndarray, feature_columns: list[str]) -> pd.Series:
    features = _numeric_matrix(frame, feature_columns)
    design = np.column_stack([np.ones(len(features)), features])
    return pd.Series(design @ coefficients, index=frame.index)


def run_linear_regression_evaluation(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    split_strategy: str,
    *,
    split_column: str,
    test_fraction: float = 0.2,
) -> dict[str, object]:
    """Train only on the split's train rows and return test metrics."""

    if target_column not in frame.columns:
        raise ValueError("target_column_missing")
    if frame[target_column].isna().any():
        raise ValueError("target_column_contains_missing_values")
    if split_strategy == "temporal":
        train, test = temporal_split(frame, split_column, test_fraction)
    elif split_strategy == "group":
        train, test = group_split(frame, split_column, test_fraction)
    else:
        raise ValueError("split_strategy_must_be_temporal_or_group")
    coefficients = fit_linear_regression(train, target_column, feature_columns)
    predictions = predict_linear_regression(test, coefficients, feature_columns)
    return {
        "model": "ordinary_least_squares",
        "split_strategy": split_strategy,
        "split_column": split_column,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "train_rows": len(train),
        "test_rows": len(test),
        "metrics": regression_metrics(test[target_column], predictions),
    }
