"""Dependency-light regression and classification metrics."""

from __future__ import annotations

import math

import pandas as pd


def _aligned(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    if len(y_true) != len(y_pred):
        raise ValueError("target_and_prediction_lengths_must_match")
    if len(y_true) == 0:
        raise ValueError("target_and_prediction_must_not_be_empty")
    true = pd.Series(y_true).reset_index(drop=True)
    pred = pd.Series(y_pred).reset_index(drop=True)
    return true, pred


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return MAE and RMSE for numeric regression outputs."""

    true, pred = _aligned(y_true, y_pred)
    true = pd.to_numeric(true, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    if true.isna().any() or pred.isna().any():
        raise ValueError("regression_targets_and_predictions_must_be_numeric")
    if not true.map(lambda value: math.isfinite(float(value))).all() or not pred.map(
        lambda value: math.isfinite(float(value))
    ).all():
        raise ValueError("regression_targets_and_predictions_must_be_finite")
    errors = true - pred
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(math.sqrt((errors**2).mean())),
    }


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return accuracy and macro balanced accuracy without sklearn."""

    true, pred = _aligned(y_true, y_pred)
    if true.isna().any() or pred.isna().any():
        raise ValueError("classification_targets_and_predictions_must_not_contain_missing_values")
    accuracy = float((true == pred).mean())
    recalls: list[float] = []
    for label in pd.unique(true):
        actual = true == label
        denominator = int(actual.sum())
        recalls.append(float(((pred[actual] == label).sum()) / denominator))
    return {"accuracy": accuracy, "balanced_accuracy": float(sum(recalls) / len(recalls))}
