import pandas as pd
import pytest

from shiftsafe.abstention import abstention_decisions, abstention_summary


def test_abstention_labels_intervals_by_width() -> None:
    intervals = pd.DataFrame({"lower": [0.0, 0.0], "upper": [1.0, 3.0]})

    decisions = abstention_decisions(intervals, max_width=1.0)

    assert decisions["decision"].tolist() == ["TRUST", "ABSTAIN"]
    assert decisions["reason"].tolist() == ["width_within_threshold", "width_above_threshold"]
    assert abstention_summary(decisions) == {
        "total_rows": 2.0,
        "trusted_rows": 1.0,
        "abstained_rows": 1.0,
        "trust_rate": 0.5,
        "abstention_rate": 0.5,
    }


def test_abstention_rejects_invalid_intervals_and_threshold() -> None:
    with pytest.raises(ValueError, match="max_width_must_be"):
        abstention_decisions(pd.DataFrame({"lower": [0.0], "upper": [1.0]}), -1.0)
    with pytest.raises(ValueError, match="interval_lower_must_not_exceed_upper"):
        abstention_decisions(pd.DataFrame({"lower": [2.0], "upper": [1.0]}), 1.0)


def test_abstention_summary_rejects_empty_decisions() -> None:
    with pytest.raises(ValueError, match="decisions_must_not_be_empty"):
        abstention_summary(pd.DataFrame(columns=["decision"]))
