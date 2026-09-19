from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DataContract(BaseModel):
    """Minimal, explicit contract for a first ShiftSafe run."""

    target: str = Field(min_length=1)
    time_column: str | None = None
    id_columns: list[str] = Field(default_factory=list)
    group_columns: list[str] = Field(default_factory=list)
    positive_label: Any | None = None


class GateDecision(BaseModel):
    status: Literal["CONTINUE", "REWORK", "STOP"]
    reasons: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    target: str
    target_missing: int
    time_parseable: bool | None = None
    temporal_ordered: bool | None = None
    decision: GateDecision
    checks: dict[str, Any] = Field(default_factory=dict)
