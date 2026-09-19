"""Small end-to-end baseline evaluation pipeline.

This module deliberately stops at a transparent baseline.  It is the bridge
between the individual validation/split/metric helpers and a future model.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .baselines import classification_majority_baseline, regression_mean_baseline
from .metrics import classification_metrics, regression_metrics
from .splitting import group_split, temporal_split


def run_baseline_evaluation(
    frame: pd.DataFrame,
    target_column: str,
    task: Literal["regression", "classification"],
    split_strategy: Literal["temporal", "group"],
    *,
    split_column: str,
    test_fraction: float = 0.2,
) -> dict[str, object]:
    """Run an auditable baseline from raw frame to JSON-friendly results.

    The function does not fit a learned model.  It exists to make the
    evaluation contract executable before a model is introduced.
    """

    if target_column not in frame.columns:
        raise ValueError("target_column_missing")
    if frame[target_column].isna().any():
        raise ValueError("target_column_contains_missing_values")
    if task not in {"regression", "classification"}:
        raise ValueError("task_must_be_regression_or_classification")

    if split_strategy == "temporal":
        train, test = temporal_split(frame, split_column, test_fraction)
    elif split_strategy == "group":
        train, test = group_split(frame, split_column, test_fraction)
    else:
        raise ValueError("split_strategy_must_be_temporal_or_group")

    if task == "regression":
        predictions = regression_mean_baseline(train[target_column], len(test))
        metrics = regression_metrics(test[target_column], predictions)
        baseline_name = "training_mean"
    else:
        predictions = classification_majority_baseline(train[target_column], len(test))
        metrics = classification_metrics(test[target_column], predictions)
        baseline_name = "training_majority_class"

    return {
        "task": task,
        "split_strategy": split_strategy,
        "split_column": split_column,
        "target_column": target_column,
        "train_rows": len(train),
        "test_rows": len(test),
        "baseline": baseline_name,
        "metrics": metrics,
    }
