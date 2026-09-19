import pandas as pd

from shiftsafe.checks import run_quality_gate
from shiftsafe.contracts import DataContract


def test_clean_monotonic_frame_can_continue() -> None:
    frame = pd.DataFrame(
        {"timestamp": ["2026-01-01", "2026-01-02"], "sensor": [1.0, 1.2], "target": [0, 1]}
    )
    summary = run_quality_gate(frame, DataContract(target="target", time_column="timestamp"))
    assert summary.decision.status == "CONTINUE"


def test_missing_target_stops_run() -> None:
    frame = pd.DataFrame({"timestamp": ["2026-01-01"], "target": [None]})
    summary = run_quality_gate(frame, DataContract(target="target", time_column="timestamp"))
    assert summary.decision.status == "STOP"
    assert "target_contains_missing_values" in summary.decision.reasons


def test_non_monotonic_time_requires_rework() -> None:
    frame = pd.DataFrame(
        {"timestamp": ["2026-01-02", "2026-01-01"], "target": [0, 1]}
    )
    summary = run_quality_gate(frame, DataContract(target="target", time_column="timestamp"))
    assert summary.decision.status == "REWORK"


def test_empty_frame_stops_with_explicit_reason() -> None:
    frame = pd.DataFrame(columns=["timestamp", "target"])
    summary = run_quality_gate(frame, DataContract(target="target", time_column="timestamp"))
    assert summary.decision.status == "STOP"
    assert "dataset_is_empty" in summary.decision.reasons


def test_missing_target_column_stops_with_explicit_reason() -> None:
    frame = pd.DataFrame({"timestamp": ["2026-01-01"], "sensor": [1.0]})
    summary = run_quality_gate(frame, DataContract(target="target", time_column="timestamp"))
    assert summary.decision.status == "STOP"
    assert "target_column_missing" in summary.decision.reasons


def test_missing_time_column_stops_with_explicit_reason() -> None:
    frame = pd.DataFrame({"target": [0], "sensor": [1.0]})
    summary = run_quality_gate(frame, DataContract(target="target", time_column="timestamp"))
    assert summary.decision.status == "STOP"
    assert "time_column_missing" in summary.decision.reasons


def test_missing_required_group_column_is_reported() -> None:
    frame = pd.DataFrame({"timestamp": ["2026-01-01"], "target": [0]})
    contract = DataContract(target="target", time_column="timestamp", group_columns=["site"])
    summary = run_quality_gate(frame, contract)
    assert summary.decision.status == "STOP"
    assert summary.checks["missing_required_columns"] == ["site"]
