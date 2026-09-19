import pandas as pd
import pytest

from shiftsafe.pipeline import run_baseline_evaluation


def test_pipeline_runs_temporal_regression_end_to_end() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
            "target": [1.0, 2.0, 3.0, 10.0, 11.0],
        }
    )

    result = run_baseline_evaluation(
        frame,
        "target",
        "regression",
        "temporal",
        split_column="timestamp",
        test_fraction=0.4,
    )

    assert result["train_rows"] == 3
    assert result["test_rows"] == 2
    assert result["baseline"] == "training_mean"
    assert result["metrics"] == {"mae": 8.5, "rmse": 8.514693182963201}


def test_pipeline_runs_group_classification_end_to_end() -> None:
    frame = pd.DataFrame(
        {"site": ["a", "a", "b", "b", "c", "c"], "target": [0, 0, 1, 1, 1, 1]}
    )

    result = run_baseline_evaluation(
        frame,
        "target",
        "classification",
        "group",
        split_column="site",
        test_fraction=1 / 3,
    )

    assert result["train_rows"] == 4
    assert result["test_rows"] == 2
    assert result["baseline"] == "training_majority_class"
    assert result["metrics"] == {"accuracy": 0.0, "balanced_accuracy": 0.0}


def test_pipeline_rejects_missing_target_before_split() -> None:
    frame = pd.DataFrame({"site": ["a", "b"], "target": [1.0, None]})

    with pytest.raises(ValueError, match="target_column_contains_missing_values"):
        run_baseline_evaluation(
            frame,
            "target",
            "regression",
            "group",
            split_column="site",
        )


def test_pipeline_rejects_unknown_task_and_split() -> None:
    frame = pd.DataFrame({"site": ["a", "b"], "target": [1, 0]})

    with pytest.raises(ValueError, match="task_must_be"):
        run_baseline_evaluation(frame, "target", "forecast", "group", split_column="site")
    with pytest.raises(ValueError, match="split_strategy_must_be"):
        run_baseline_evaluation(frame, "target", "classification", "random", split_column="site")
