"""Minimal split-conformal prediction intervals for regression."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .baselines import regression_mean_baseline
from .checks import run_quality_gate
from .contracts import DataContract
from .metrics import regression_metrics
from .modeling import fit_linear_regression, predict_linear_regression
from .splitting import group_split, temporal_split


def conformal_radius(
    y_calibration: pd.Series,
    calibration_predictions: pd.Series,
    alpha: float = 0.1,
) -> float:
    """Return a finite-sample absolute-residual conformal radius.

    The returned order statistic is calibrated on a reference calibration set.
    It does not promise coverage after distribution shift.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha_must_be_between_0_and_1")
    if len(y_calibration) != len(calibration_predictions):
        raise ValueError("calibration_lengths_must_match")
    if len(y_calibration) < 3:
        raise ValueError("calibration_data_must_have_at_least_three_rows")
    y = pd.to_numeric(pd.Series(y_calibration), errors="coerce")
    prediction = pd.to_numeric(pd.Series(calibration_predictions), errors="coerce")
    if y.isna().any() or prediction.isna().any():
        raise ValueError("calibration_values_must_be_numeric")
    if not np.isfinite(y.to_numpy()).all() or not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("calibration_values_must_be_finite")
    scores = np.sort(np.abs(y.to_numpy(dtype=float) - prediction.to_numpy(dtype=float)))
    rank = min(len(scores) - 1, max(0, math.ceil((len(scores) + 1) * (1 - alpha)) - 1))
    return float(scores[rank])


def prediction_interval(predictions: pd.Series, radius: float) -> pd.DataFrame:
    """Build symmetric lower/upper intervals around point predictions."""

    if radius < 0 or not math.isfinite(float(radius)):
        raise ValueError("radius_must_be_finite_and_nonnegative")
    values = pd.to_numeric(pd.Series(predictions), errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("predictions_must_be_numeric_and_finite")
    return pd.DataFrame(
        {"lower": values - radius, "prediction": values, "upper": values + radius},
        index=predictions.index,
    )


def interval_metrics(y_true: pd.Series, intervals: pd.DataFrame) -> dict[str, float]:
    """Return empirical coverage and mean interval width."""

    required = {"lower", "upper"}
    if not required.issubset(intervals.columns):
        raise ValueError("interval_columns_missing")
    truth = pd.to_numeric(pd.Series(y_true), errors="coerce").reset_index(drop=True)
    lower = pd.to_numeric(intervals["lower"], errors="coerce").reset_index(drop=True)
    upper = pd.to_numeric(intervals["upper"], errors="coerce").reset_index(drop=True)
    if len(truth) != len(lower):
        raise ValueError("interval_lengths_must_match")
    if truth.isna().any() or lower.isna().any() or upper.isna().any():
        raise ValueError("interval_values_must_be_numeric")
    if not np.isfinite(truth.to_numpy()).all() or not np.isfinite(lower.to_numpy()).all() or not np.isfinite(
        upper.to_numpy()
    ).all():
        raise ValueError("interval_values_must_be_finite")
    if (lower > upper).any():
        raise ValueError("interval_lower_must_not_exceed_upper")
    covered = (truth >= lower) & (truth <= upper)
    return {"coverage": float(covered.mean()), "mean_width": float((upper - lower).mean())}


def run_split_conformal_evaluation(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    split_strategy: str,
    *,
    split_column: str,
    test_fraction: float = 0.2,
    calibration_fraction: float = 0.25,
    alpha: float = 0.1,
) -> dict[str, object]:
    """Evaluate a linear model with a reference calibration interval."""

    if target_column not in frame.columns:
        raise ValueError("target_column_missing")
    if frame[target_column].isna().any():
        raise ValueError("target_column_contains_missing_values")
    if target_column in feature_columns:
        raise ValueError("target_column_must_not_be_a_feature")
    if split_column in feature_columns:
        raise ValueError("split_column_must_not_be_a_feature")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction_must_be_between_0_and_1")
    if not 0 < alpha < 1:
        raise ValueError("alpha_must_be_between_0_and_1")

    contract = DataContract(
        target=target_column,
        time_column=split_column if split_strategy == "temporal" else None,
        group_columns=[split_column] if split_strategy == "group" else [],
    )
    gate = run_quality_gate(frame, contract)
    if gate.decision.status != "CONTINUE":
        raise ValueError(f"quality_gate_{gate.decision.status.lower()}")

    if split_strategy == "temporal":
        development, test = temporal_split(frame, split_column, test_fraction)
        train, calibration = temporal_split(development, split_column, calibration_fraction)
    elif split_strategy == "group":
        development, test = group_split(frame, split_column, test_fraction)
        train, calibration = group_split(development, split_column, calibration_fraction)
    else:
        raise ValueError("split_strategy_must_be_temporal_or_group")

    coefficients = fit_linear_regression(train, target_column, feature_columns)
    calibration_predictions = predict_linear_regression(calibration, coefficients, feature_columns)
    test_predictions = predict_linear_regression(test, coefficients, feature_columns)
    radius = conformal_radius(calibration[target_column], calibration_predictions, alpha)
    intervals = prediction_interval(test_predictions, radius)
    metrics = interval_metrics(test[target_column], intervals)
    point_metrics = regression_metrics(test[target_column], test_predictions)
    baseline_predictions = regression_mean_baseline(train[target_column], len(test))
    baseline_metrics = regression_metrics(test[target_column], baseline_predictions)
    nominal_coverage = 1 - alpha
    coverage_gap = metrics["coverage"] - nominal_coverage
    interval_rows = []
    truth = test[target_column].reset_index(drop=True)
    for row_number, row in intervals.reset_index(drop=True).iterrows():
        actual = float(truth.iloc[row_number])
        interval_rows.append(
            {
                "row": row_number,
                "actual": actual,
                "prediction": float(row["prediction"]),
                "lower": float(row["lower"]),
                "upper": float(row["upper"]),
                "covered": bool(row["lower"] <= actual <= row["upper"]),
            }
        )
    return {
        "method": "split_conformal_absolute_residual",
        "alpha": alpha,
        "nominal_coverage": nominal_coverage,
        "coverage_gap": coverage_gap,
        "coverage_status": "below_nominal" if coverage_gap < 0 else "at_or_above_nominal",
        "radius": radius,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "point_metrics": point_metrics,
        "baseline": "training_mean",
        "baseline_metrics": baseline_metrics,
        "interval_metrics": metrics,
        "interval_rows": interval_rows,
        "quality_gate": gate.model_dump(mode="json"),
    }
