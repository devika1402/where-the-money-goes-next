"""The temporal split, and the leakage cutoff that separates features from labels.

F4 says every feature uses only transactions strictly before the label window starts.
These tests hold the window construction to that, so the boundary cannot drift once the
feature code starts consuming it.
"""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from src.definitions import SchemaError, SplitSpec, Window, build_windows, load_params

SPEC = SplitSpec(
    data_start=pd.Timestamp("2022-09-01 00:00:00"),
    data_end=pd.Timestamp("2022-09-11 00:00:00"),
    feature_window_days=4,
)
LABEL_DAYS = 2


def test_the_split_produces_three_named_windows() -> None:
    windows = build_windows(SPEC, label_window_days=LABEL_DAYS)

    assert [w.name for w in windows] == ["train", "val", "test"]


def test_each_feature_window_ends_exactly_where_its_label_window_begins() -> None:
    """The leakage cutoff is one instant, not two that happen to agree."""
    for window in build_windows(SPEC, label_window_days=LABEL_DAYS):
        assert window.feature_end == window.label_start


def test_the_windows_match_the_dates_the_decision_log_records() -> None:
    """The split boundaries. Pinned so a change shows up here rather than in a metric."""
    train, val, test = build_windows(SPEC, label_window_days=LABEL_DAYS)

    assert train.feature_start == pd.Timestamp("2022-09-01")
    assert train.label_start == pd.Timestamp("2022-09-05")
    assert train.label_end == pd.Timestamp("2022-09-07")

    assert val.feature_start == pd.Timestamp("2022-09-03")
    assert val.label_start == pd.Timestamp("2022-09-07")
    assert val.label_end == pd.Timestamp("2022-09-09")

    assert test.feature_start == pd.Timestamp("2022-09-05")
    assert test.label_start == pd.Timestamp("2022-09-09")
    assert test.label_end == pd.Timestamp("2022-09-11")


def test_label_windows_are_disjoint_and_strictly_ordered() -> None:
    """Overlapping label windows would score the same account twice on the same event."""
    windows = build_windows(SPEC, label_window_days=LABEL_DAYS)

    for earlier, later in itertools.pairwise(windows):
        assert earlier.label_end <= later.label_start


def test_every_feature_window_has_the_same_length() -> None:
    """Fixed length rather than expanding, so a feature means the same in every split.

    An expanding window would give train 3.5 days of history and test 10 days, which is a
    distribution shift written into the design.
    """
    lengths = {
        w.feature_end - w.feature_start for w in build_windows(SPEC, label_window_days=LABEL_DAYS)
    }

    assert lengths == {pd.Timedelta(days=4)}


def test_the_last_label_window_ends_at_the_end_of_the_usable_span() -> None:
    """The four day tail after 09-14 is excluded on purpose."""
    windows = build_windows(SPEC, label_window_days=LABEL_DAYS)

    assert windows[-1].label_end == SPEC.data_end


def test_a_span_too_short_for_three_windows_is_refused() -> None:
    """Silently returning two windows would leave the test split undefined."""
    too_short = SplitSpec(
        data_start=pd.Timestamp("2022-09-01"),
        data_end=pd.Timestamp("2022-09-09"),
        feature_window_days=4,
    )

    with pytest.raises(SchemaError, match="span"):
        build_windows(too_short, label_window_days=LABEL_DAYS)


def test_a_window_whose_cutoff_disagrees_with_itself_is_refused() -> None:
    """The guard that makes the cutoff identity structural rather than a convention."""
    with pytest.raises(SchemaError, match="feature_end"):
        Window(
            name="broken",
            feature_start=pd.Timestamp("2022-09-01"),
            feature_end=pd.Timestamp("2022-09-05 00:00:01"),
            label_start=pd.Timestamp("2022-09-05"),
            label_end=pd.Timestamp("2022-09-08"),
        )


def test_the_configured_split_covers_the_span_the_decision_log_approved() -> None:
    """Guards config/params.yaml against an edit that quietly widens the usable span."""
    params = load_params()

    assert params.split.data_start == pd.Timestamp("2022-09-01 00:00:00")
    assert params.split.data_end == pd.Timestamp("2022-09-11 00:00:00")
    assert params.split.feature_window_days == 4
    assert params.windows.label_window_days == 2
