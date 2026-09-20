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
