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
