import pandas as pd
import pytest

from shiftsafe.shift import apply_feature_shift, run_shift_stress_evaluation


def test_apply_mean_and_scale_shift_is_deterministic() -> None:
    frame = pd.DataFrame({"sensor": [1.0, 2.0], "other": [3.0, 4.0]})

    mean_shifted = apply_feature_shift(frame, ["sensor"], "mean", 2.0)
    scale_shifted = apply_feature_shift(frame, ["sensor"], "scale", 3.0)

    assert mean_shifted["sensor"].tolist() == [3.0, 4.0]
    assert scale_shifted["sensor"].tolist() == [3.0, 6.0]
    assert frame["sensor"].tolist() == [1.0, 2.0]


def test_apply_shift_rejects_invalid_kind_and_scale() -> None:
    frame = pd.DataFrame({"sensor": [1.0, 2.0]})

    with pytest.raises(ValueError, match="shift_kind_must_be"):
        apply_feature_shift(frame, ["sensor"], "noise", 1.0)
    with pytest.raises(ValueError, match="scale_magnitude_must_be_positive"):
        apply_feature_shift(frame, ["sensor"], "scale", 0.0)


def test_shift_evaluation_reports_reference_and_shifted_metrics() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=16),
            "sensor": range(16),
            "target": [float(2 * value + (value % 2) * 0.2) for value in range(16)],
        }
    )

    result = run_shift_stress_evaluation(
        frame,
        "target",
        ["sensor"],
        "temporal",
        split_column="timestamp",
        shift_kind="mean",
        shift_magnitude=5.0,
        test_fraction=0.25,
        calibration_fraction=0.25,
    )

    assert result["shift_kind"] == "mean"
    assert result["shift_columns"] == ["sensor"]
    assert result["shift_semantics"] == "covariate_shift_only_target_held_fixed"
    assert result["radius_frozen_from_reference"] >= 0
    assert result["reference"]["interval_metrics"]["coverage"] >= 0
    assert result["shifted"]["interval_metrics"]["coverage"] >= 0
    assert len(result["interval_rows"]) == 4
    assert result["reference"]["point_metrics"] != result["shifted"]["point_metrics"]
    assert "coverage_gap" in result["shifted"]["interval_metrics"]


def test_shift_evaluation_rejects_unknown_strategy() -> None:
    frame = pd.DataFrame({"group": ["a", "b", "c", "d"], "sensor": [1, 2, 3, 4], "target": [2, 4, 6, 8]})

    with pytest.raises(ValueError, match="split_strategy_must_be"):
        run_shift_stress_evaluation(
            frame,
            "target",
            ["sensor"],
            "random",
            split_column="group",
            shift_kind="mean",
            shift_magnitude=1.0,
        )


def test_shift_evaluation_rejects_duplicate_features() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=6), "sensor": range(6), "target": range(6)}
    )

    with pytest.raises(ValueError, match="feature_columns_must_be_unique"):
        run_shift_stress_evaluation(
            frame,
            "target",
            ["sensor", "sensor"],
            "temporal",
            split_column="timestamp",
            shift_kind="mean",
            shift_magnitude=1.0,
        )
