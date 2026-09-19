from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import DataContract, GateDecision, RunSummary


def _required_columns(frame: pd.DataFrame, contract: DataContract) -> list[str]:
    required = [contract.target, *contract.id_columns, *contract.group_columns]
    if contract.time_column:
        required.append(contract.time_column)
    return sorted(set(required))


def _decision(checks: dict[str, Any]) -> GateDecision:
    reasons: list[str] = []
    if checks["missing_required_columns"]:
        reasons.append("required_columns_missing")
    if checks["target_missing"] > 0:
        reasons.append("target_contains_missing_values")
    if checks["duplicate_rows"] > 0:
        reasons.append("duplicate_rows_present")
    if checks["time_parseable"] is False:
        reasons.append("time_column_is_not_parseable")
    if checks["temporal_ordered"] is False:
        reasons.append("time_column_is_not_monotonic")

    if any(reason in reasons for reason in ("required_columns_missing", "target_contains_missing_values")):
        return GateDecision(status="STOP", reasons=reasons)
    if reasons:
        return GateDecision(status="REWORK", reasons=reasons)
    return GateDecision(status="CONTINUE", reasons=[])


def run_quality_gate(frame: pd.DataFrame, contract: DataContract) -> RunSummary:
    """Run deterministic, pre-model checks on a tabular time-series frame."""

    missing_columns = [name for name in _required_columns(frame, contract) if name not in frame]
    time_parseable: bool | None = None
    temporal_ordered: bool | None = None
    if contract.time_column and contract.time_column in frame:
        parsed = pd.to_datetime(frame[contract.time_column], errors="coerce")
        time_parseable = bool(parsed.notna().all())
        temporal_ordered = bool(parsed.is_monotonic_increasing) if time_parseable else False

    checks: dict[str, Any] = {
        "missing_required_columns": missing_columns,
        "missing_columns": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_missing": int(frame[contract.target].isna().sum()) if contract.target in frame else 0,
        "time_parseable": time_parseable,
        "temporal_ordered": temporal_ordered,
        "column_types": {name: str(dtype) for name, dtype in frame.dtypes.items()},
    }
    decision = _decision(checks)
    return RunSummary(
        rows=len(frame),
        columns=len(frame.columns),
        missing_cells=checks["missing_columns"],
        duplicate_rows=checks["duplicate_rows"],
        target=contract.target,
        target_missing=checks["target_missing"],
        time_parseable=time_parseable,
        temporal_ordered=temporal_ordered,
        decision=decision,
        checks=checks,
    )
