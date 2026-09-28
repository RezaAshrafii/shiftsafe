"""Simple width based abstention decisions for prediction intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def abstention_decisions(intervals: pd.DataFrame, max_width: float) -> pd.DataFrame:
    """Label each interval TRUST or ABSTAIN using a fixed width threshold.

    The threshold is a policy input.  This function does not claim that a
    width threshold is optimal or that it represents a real business cost.
    """

    if not math.isfinite(float(max_width)) or max_width < 0:
        raise ValueError("max_width_must_be_finite_and_nonnegative")
    required = {"lower", "upper"}
    if not required.issubset(intervals.columns):
        raise ValueError("interval_columns_missing")
    lower = pd.to_numeric(intervals["lower"], errors="coerce")
    upper = pd.to_numeric(intervals["upper"], errors="coerce")
    if lower.isna().any() or upper.isna().any():
        raise ValueError("interval_values_must_be_numeric")
    if not np.isfinite(lower.to_numpy()).all() or not np.isfinite(upper.to_numpy()).all():
        raise ValueError("interval_values_must_be_finite")
    if (lower > upper).any():
        raise ValueError("interval_lower_must_not_exceed_upper")
    widths = upper - lower
    return pd.DataFrame(
        {
            "width": widths,
            "decision": np.where(widths <= max_width, "TRUST", "ABSTAIN"),
            "reason": np.where(widths <= max_width, "width_within_threshold", "width_above_threshold"),
        },
        index=intervals.index,
    )


def abstention_summary(decisions: pd.DataFrame) -> dict[str, float]:
    """Summarize trust and abstention rates."""

    if "decision" not in decisions.columns:
        raise ValueError("decision_column_missing")
    if decisions.empty:
        raise ValueError("decisions_must_not_be_empty")
    total = len(decisions)
    abstained = int((decisions["decision"] == "ABSTAIN").sum())
    trusted = int((decisions["decision"] == "TRUST").sum())
    return {
        "total_rows": float(total),
        "trusted_rows": float(trusted),
        "abstained_rows": float(abstained),
        "trust_rate": trusted / total,
        "abstention_rate": abstained / total,
    }
