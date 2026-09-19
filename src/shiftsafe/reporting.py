from __future__ import annotations

from .contracts import RunSummary


def to_markdown(summary: RunSummary) -> str:
    decision = summary.decision
    reason_text = ", ".join(decision.reasons) if decision.reasons else "none"
    validation_errors = ", ".join(summary.checks.get("validation_errors", [])) or "none"
    return f"""# ShiftSafe quality-gate report

## Decision

`{decision.status}`

Reasons: `{reason_text}`

Validation errors: `{validation_errors}`

## Dataset summary

| Check | Value |
|---|---:|
| Rows | {summary.rows} |
| Columns | {summary.columns} |
| Missing cells | {summary.missing_cells} |
| Duplicate rows | {summary.duplicate_rows} |
| Target | `{summary.target}` |
| Missing target values | {summary.target_missing} |
| Time parseable | {summary.time_parseable} |
| Time monotonic | {summary.temporal_ordered} |

## Scope boundary

This is a deterministic data-quality gate. It is not a certification of safety, legality,
causal impact, fairness, or economic return. Model performance, calibration, uncertainty,
distribution shift, and decision cost are planned next milestones.
"""
