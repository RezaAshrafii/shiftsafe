import pandas as pd
import pytest

from shiftsafe.baselines import classification_majority_baseline, regression_mean_baseline
from shiftsafe.metrics import classification_metrics, regression_metrics


def test_regression_baseline_and_metrics() -> None:
    prediction = regression_mean_baseline(pd.Series([1.0, 3.0]), 2)

    assert prediction.tolist() == [2.0, 2.0]
    assert regression_metrics(pd.Series([1.0, 3.0]), prediction) == {"mae": 1.0, "rmse": 1.0}


def test_regression_metrics_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="must_be_finite"):
        regression_metrics(pd.Series([1.0, float("inf")]), pd.Series([1.0, 2.0]))


def test_classification_baseline_and_metrics() -> None:
    prediction = classification_majority_baseline(pd.Series(["a", "a", "b"]), 2)

    assert prediction.tolist() == ["a", "a"]
    assert classification_metrics(pd.Series(["a", "b"]), pd.Series(["a", "a"])) == {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
    }


def test_classification_metrics_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="must_not_contain_missing_values"):
        classification_metrics(pd.Series(["a", None]), pd.Series(["a", "a"]))


def test_constant_target_is_a_valid_baseline_but_not_a_model_claim() -> None:
    prediction = classification_majority_baseline(pd.Series([1, 1]), 1)

    assert prediction.tolist() == [1]


def test_classification_baseline_rejects_missing_target_values() -> None:
    with pytest.raises(ValueError, match="classification_target_must_not_contain_missing_values"):
        classification_majority_baseline(pd.Series(["a", None]), 1)


def test_regression_baseline_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="regression_target_must_be_numeric"):
        regression_mean_baseline(pd.Series([1.0, float("inf")]), 1)


@pytest.mark.parametrize("function", [regression_mean_baseline, classification_majority_baseline])
def test_baselines_reject_empty_training_target(function) -> None:
    with pytest.raises(ValueError, match="must_not_be_empty"):
        function(pd.Series(dtype=float), 1)
