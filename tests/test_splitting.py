import pandas as pd
import pytest

from shiftsafe.splitting import group_split, temporal_split


def test_temporal_split_keeps_future_rows_in_test_only() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
            "value": range(5),
        }
    )

    train, test = temporal_split(frame, "timestamp", test_fraction=0.4)

    assert len(train) == 3
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()


def test_temporal_split_rejects_missing_time_column() -> None:
    with pytest.raises(ValueError, match="time_column_missing"):
        temporal_split(pd.DataFrame({"value": [1, 2]}), "timestamp")


def test_temporal_split_rejects_unsorted_time_data() -> None:
    frame = pd.DataFrame({"timestamp": ["2026-01-02", "2026-01-01"], "value": [1, 2]})

    with pytest.raises(ValueError, match="monotonic"):
        temporal_split(frame, "timestamp")


def test_group_split_never_overlaps_groups() -> None:
    frame = pd.DataFrame({"site": ["a", "a", "b", "b", "c", "c"], "value": range(6)})

    train, test = group_split(frame, "site", test_fraction=1 / 3)

    assert set(train["site"]).isdisjoint(set(test["site"]))
    assert len(train) + len(test) == len(frame)


def test_group_split_rejects_missing_group_column() -> None:
    with pytest.raises(ValueError, match="group_column_missing"):
        group_split(pd.DataFrame({"value": [1, 2]}), "site")


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_splits_reject_invalid_test_fraction(fraction: float) -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2)})

    with pytest.raises(ValueError, match="test_fraction"):
        temporal_split(frame, "timestamp", fraction)


def test_splits_reject_empty_input() -> None:
    empty = pd.DataFrame(columns=["timestamp", "site"])

    with pytest.raises(ValueError, match="input_frame_must_not_be_empty"):
        temporal_split(empty, "timestamp")
    with pytest.raises(ValueError, match="input_frame_must_not_be_empty"):
        group_split(empty, "site")
