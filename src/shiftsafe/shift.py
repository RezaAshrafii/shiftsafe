"""Deterministic stress tests for controlled numeric distribution shift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import regression_mean_baseline
from .checks import run_quality_gate
from .contracts import DataContract
from .metrics import regression_metrics
from .modeling import fit_linear_regression, predict_linear_regression
from .splitting import group_split, temporal_split
from .uncertainty import conformal_radius, interval_metrics, prediction_interval


def apply_feature_shift(
    frame: pd.DataFrame,
    feature_columns: list[str],
    kind: str,
    magnitude: float,
) -> pd.DataFrame:
    """Return a copy with a deterministic mean or scale shift on features."""

    if not feature_columns:
        raise ValueError("feature_columns_must_not_be_empty")
    if kind not in {"mean", "scale"}:
        raise ValueError("shift_kind_must_be_mean_or_scale")
    if not np.isfinite(float(magnitude)):
        raise ValueError("shift_magnitude_must_be_finite")
    shifted = frame.copy()
    missing = [column for column in feature_columns if column not in shifted.columns]
    if missing:
        raise ValueError("feature_columns_missing")
    values = shifted[feature_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("feature_values_must_be_numeric_and_finite")
    if kind == "mean":
        shifted[feature_columns] = values + float(magnitude)
    else:
        if magnitude <= 0:
            raise ValueError("scale_magnitude_must_be_positive")
        shifted[feature_columns] = values * float(magnitude)
    return shifted


def run_shift_stress_evaluation(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    split_strategy: str,
    *,
    split_column: str,
    shift_kind: str,
    shift_magnitude: float,
    test_fraction: float = 0.2,
    calibration_fraction: float = 0.25,
    alpha: float = 0.1,
) -> dict[str, object]:
    """Compare reference and shifted test metrics using one frozen model."""

    if target_column not in frame.columns:
        raise ValueError("target_column_missing")
    if frame[target_column].isna().any():
        raise ValueError("target_column_contains_missing_values")
    if target_column in feature_columns or split_column in feature_columns:
        raise ValueError("target_or_split_column_must_not_be_a_feature")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns_must_be_unique")
    contract = DataContract(
        target=target_column,
        time_column=split_column if split_strategy == "temporal" else None,
        group_columns=[split_column] if split_strategy == "group" else [],
    )
    gate = run_quality_gate(frame, contract)
    if gate.decision.status != "CONTINUE":
        raise ValueError(f"quality_gate_{gate.decision.status.lower()}")
    if split_strategy == "temporal":
        development, reference_test = temporal_split(frame, split_column, test_fraction)
        train, calibration = temporal_split(development, split_column, calibration_fraction)
    elif split_strategy == "group":
        development, reference_test = group_split(frame, split_column, test_fraction)
        train, calibration = group_split(development, split_column, calibration_fraction)
    else:
        raise ValueError("split_strategy_must_be_temporal_or_group")

    coefficients = fit_linear_regression(train, target_column, feature_columns)
    calibration_predictions = predict_linear_regression(calibration, coefficients, feature_columns)
    radius = conformal_radius(calibration[target_column], calibration_predictions, alpha)
    reference_predictions = predict_linear_regression(reference_test, coefficients, feature_columns)
    shifted_test = apply_feature_shift(reference_test, feature_columns, shift_kind, shift_magnitude)
    shifted_predictions = predict_linear_regression(shifted_test, coefficients, feature_columns)
    reference_intervals = prediction_interval(reference_predictions, radius)
    shifted_intervals = prediction_interval(shifted_predictions, radius)
    reference_baseline = regression_mean_baseline(train[target_column], len(reference_test))
    shifted_baseline = regression_mean_baseline(train[target_column], len(shifted_test))
    nominal_coverage = 1 - alpha
    reference_interval_metrics = interval_metrics(reference_test[target_column], reference_intervals)
    shifted_interval_metrics = interval_metrics(reference_test[target_column], shifted_intervals)

    def interval_report(metrics: dict[str, float]) -> dict[str, object]:
        gap = metrics["coverage"] - nominal_coverage
        return {
            **metrics,
            "nominal_coverage": nominal_coverage,
            "coverage_gap": gap,
            "coverage_status": "below_nominal" if gap < 0 else "at_or_above_nominal",
        }

    reference_rows = reference_test[target_column].reset_index(drop=True)
    reference_intervals_reset = reference_intervals.reset_index(drop=True)
    shifted_intervals_reset = shifted_intervals.reset_index(drop=True)
    reference_predictions_reset = reference_predictions.reset_index(drop=True)
    shifted_predictions_reset = shifted_predictions.reset_index(drop=True)
    interval_rows = []
    for row_number, actual_value in reference_rows.items():
        reference_row = reference_intervals_reset.iloc[row_number]
        shifted_row = shifted_intervals_reset.iloc[row_number]
        actual = float(actual_value)
        interval_rows.append(
            {
                "row": row_number,
                "actual": actual,
                "reference_prediction": float(reference_predictions_reset.iloc[row_number]),
                "shifted_prediction": float(shifted_predictions_reset.iloc[row_number]),
                "reference_lower": float(reference_row["lower"]),
                "reference_upper": float(reference_row["upper"]),
                "shifted_lower": float(shifted_row["lower"]),
                "shifted_upper": float(shifted_row["upper"]),
                "reference_covered": bool(reference_row["lower"] <= actual <= reference_row["upper"]),
                "shifted_covered": bool(shifted_row["lower"] <= actual <= shifted_row["upper"]),
            }
        )
    return {
        "shift_kind": shift_kind,
        "shift_magnitude": shift_magnitude,
        "shift_columns": feature_columns,
        "shift_semantics": "covariate_shift_only_target_held_fixed",
        "radius_frozen_from_reference": radius,
        "reference": {
            "point_metrics": regression_metrics(reference_test[target_column], reference_predictions),
            "baseline_metrics": regression_metrics(reference_test[target_column], reference_baseline),
            "interval_metrics": interval_report(reference_interval_metrics),
        },
        "shifted": {
            "point_metrics": regression_metrics(reference_test[target_column], shifted_predictions),
            "baseline_metrics": regression_metrics(reference_test[target_column], shifted_baseline),
            "interval_metrics": interval_report(shifted_interval_metrics),
        },
        "interval_rows": interval_rows,
        "quality_gate": gate.model_dump(mode="json"),
    }
