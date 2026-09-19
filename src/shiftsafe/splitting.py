"""Deterministic, leakage-aware dataset splits for ShiftSafe.

These helpers intentionally avoid model-specific behavior.  They make the
evaluation boundary explicit before any model is trained.
"""

from __future__ import annotations

import math

import pandas as pd


def _validate_common(frame: pd.DataFrame, test_fraction: float) -> None:
    """Validate arguments shared by both split strategies."""

    if frame.empty:
        raise ValueError("input_frame_must_not_be_empty")
    if not isinstance(test_fraction, (int, float)) or isinstance(test_fraction, bool):
        raise TypeError("test_fraction_must_be_numeric")
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction_must_be_between_0_and_1")
    if len(frame) < 2:
        raise ValueError("input_frame_must_have_at_least_two_rows")


def temporal_split(
    frame: pd.DataFrame,
    time_column: str,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chronologically ordered frame without looking into the future.

    The input must already be monotonic in ``time_column``.  Refusing to sort
    silently is deliberate: an accidental order change can hide a data issue.
    """

    _validate_common(frame, test_fraction)
    if time_column not in frame.columns:
        raise ValueError("time_column_missing")

    parsed = pd.to_datetime(frame[time_column], errors="coerce")
    if parsed.isna().any():
        raise ValueError("time_column_contains_unparseable_values")
    if not parsed.is_monotonic_increasing:
        raise ValueError("time_column_must_be_monotonic_increasing")

    test_size = max(1, min(len(frame) - 1, math.ceil(len(frame) * float(test_fraction))))
    cut = len(frame) - test_size
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def group_split(
    frame: pd.DataFrame,
    group_column: str,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by whole groups so one group cannot leak across train and test."""

    _validate_common(frame, test_fraction)
    if group_column not in frame.columns:
        raise ValueError("group_column_missing")
    if frame[group_column].isna().any():
        raise ValueError("group_column_contains_missing_values")

    groups = list(pd.unique(frame[group_column]))
    if len(groups) < 2:
        raise ValueError("group_column_must_contain_at_least_two_groups")

    # Sorting by a stable textual key makes the result reproducible for mixed
    # scalar group identifiers without adding a random seed or dependency.
    groups = sorted(groups, key=lambda value: (type(value).__name__, str(value)))
    test_group_count = max(1, min(len(groups) - 1, math.ceil(len(groups) * float(test_fraction))))
    test_groups = set(groups[-test_group_count:])
    test_mask = frame[group_column].isin(test_groups)
    return frame.loc[~test_mask].copy(), frame.loc[test_mask].copy()
