import pandas as pd
import pytest

from shiftsafe.uncertainty import (
    conformal_radius,
    interval_metrics,
    prediction_interval,
    run_split_conformal_evaluation,
)


def test_conformal_interval_reports_coverage_and_width() -> None:
    predictions = pd.Series([1.0, 2.0, 3.0])
    radius = conformal_radius(pd.Series([1.1, 1.8, 3.4]), predictions, alpha=0.1)
    intervals = prediction_interval(pd.Series([4.0, 5.0]), radius)

    assert radius == pytest.approx(0.4)
    metrics = interval_metrics(pd.Series([4.2, 5.5]), intervals)
    assert metrics["coverage"] == 0.5
    assert metrics["mean_width"] == pytest.approx(0.8)


def test_conformal_rejects_invalid_alpha_and_empty_calibration() -> None:
    with pytest.raises(ValueError, match="alpha_must_be"):
        conformal_radius(pd.Series([1.0]), pd.Series([1.0]), alpha=1.0)
    with pytest.raises(ValueError, match="calibration_data_must_not_be_empty"):
        conformal_radius(pd.Series(dtype=float), pd.Series(dtype=float))


def test_split_conformal_pipeline_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12),
            "sensor": range(12),
            "target": [float(2 * value + (value % 2) * 0.2) for value in range(12)],
        }
    )

    first = run_split_conformal_evaluation(
        frame,
        "target",
        ["sensor"],
        "temporal",
        split_column="timestamp",
        test_fraction=0.25,
        calibration_fraction=0.25,
        alpha=0.1,
    )
    second = run_split_conformal_evaluation(
        frame,
        "target",
        ["sensor"],
        "temporal",
        split_column="timestamp",
        test_fraction=0.25,
        calibration_fraction=0.25,
        alpha=0.1,
    )

    assert first == second
    assert first["nominal_coverage"] == 0.9
    assert first["interval_metrics"]["mean_width"] >= 0


def test_split_conformal_rejects_invalid_calibration_fraction() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=4), "sensor": range(4), "target": range(4)}
    )

    with pytest.raises(ValueError, match="calibration_fraction_must_be"):
        run_split_conformal_evaluation(
            frame,
            "target",
            ["sensor"],
            "temporal",
            split_column="timestamp",
            calibration_fraction=1.0,
        )
