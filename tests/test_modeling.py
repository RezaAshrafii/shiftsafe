import pandas as pd
import pytest

from shiftsafe.modeling import run_linear_regression_evaluation


def test_linear_regression_is_reproducible_and_uses_temporal_train_only() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="D"),
            "sensor": [0, 1, 2, 3, 4, 5],
            "target": [1, 3, 5, 7, 9, 11],
        }
    )

    first = run_linear_regression_evaluation(
        frame,
        "target",
        ["sensor"],
        "temporal",
        split_column="timestamp",
        test_fraction=1 / 3,
    )
    second = run_linear_regression_evaluation(
        frame,
        "target",
        ["sensor"],
        "temporal",
        split_column="timestamp",
        test_fraction=1 / 3,
    )

    assert first == second
    assert first["metrics"] == {"mae": 0.0, "rmse": 0.0}
    assert first["baseline_metrics"] == {"mae": 6.0, "rmse": 6.082762530298219}
    assert first["mae_improvement_percent"] == 100.0
    assert first["design_rank"] == 2
    assert first["quality_gate"]["decision"]["status"] == "CONTINUE"


def test_model_rejects_non_numeric_features() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=3), "sensor": ["x", "y", "z"], "target": [1, 2, 3]}
    )

    with pytest.raises(ValueError, match="feature_values_must_be_numeric"):
        run_linear_regression_evaluation(
            frame, "target", ["sensor"], "temporal", split_column="timestamp", test_fraction=1 / 3
        )


def test_model_rejects_unknown_split_strategy() -> None:
    frame = pd.DataFrame({"group": ["a", "b"], "feature": [1, 2], "target": [2, 4]})

    with pytest.raises(ValueError, match="split_strategy_must_be"):
        run_linear_regression_evaluation(frame, "target", ["feature"], "random", split_column="group")


def test_model_rejects_target_and_split_feature_leakage() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=3), "sensor": [1, 2, 3], "target": [2, 4, 6]}
    )

    with pytest.raises(ValueError, match="target_column_must_not_be_a_feature"):
        run_linear_regression_evaluation(
            frame, "target", ["target"], "temporal", split_column="timestamp", test_fraction=1 / 3
        )
    with pytest.raises(ValueError, match="split_column_must_not_be_a_feature"):
        run_linear_regression_evaluation(
            frame, "target", ["timestamp"], "temporal", split_column="timestamp", test_fraction=1 / 3
        )


def test_model_stops_when_quality_gate_fails() -> None:
    frame = pd.DataFrame(
        {"timestamp": ["2026-01-02", "2026-01-01", "2026-01-03"], "sensor": [1, 2, 3], "target": [2, 4, 6]}
    )

    with pytest.raises(ValueError, match="quality_gate_rework"):
        run_linear_regression_evaluation(
            frame, "target", ["sensor"], "temporal", split_column="timestamp", test_fraction=1 / 3
        )


def test_model_reports_nonzero_error_on_noisy_data() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8),
            "sensor": range(8),
            "target": [1.0, 3.2, 4.8, 7.1, 9.4, 10.6, 13.2, 14.7],
        }
    )

    result = run_linear_regression_evaluation(
        frame, "target", ["sensor"], "temporal", split_column="timestamp", test_fraction=0.25
    )

    assert result["metrics"]["mae"] > 0
    assert result["design_rank"] == 2
