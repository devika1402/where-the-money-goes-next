"""The pass-through window sweep. Section 14, D40.

The load-bearing behaviour pinned here is that an extension the feature window cannot hold is
dropped rather than allowed to raise, and that the configured value's row is checked against the
published operating point so the sweep cannot silently measure something else.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.definitions import SchemaError, Window
from src.sweep import OperatingPoint, assert_reproduces_published, feasible_hours

WINDOW = Window(
    name="test",
    feature_start=pd.Timestamp("2022-09-05"),
    feature_end=pd.Timestamp("2022-09-09"),
    label_start=pd.Timestamp("2022-09-09"),
    label_end=pd.Timestamp("2022-09-11"),
)


def test_an_extension_the_feature_window_cannot_hold_is_dropped() -> None:
    """96 minus 120 is negative, so 120 leaves no inflow window and is not swept.

    The feature window here is four days, 96 hours. 72 leaves 24 hours of inflow and is kept,
    120 would leave a negative window and is dropped, which is the arithmetic the specified
    {24, 48, 72, 120} runs into and the reason the feasible set is {24, 48, 72}.
    """
    feasible = feasible_hours(WINDOW, (24, 48, 72, 120))

    assert feasible == (24, 48, 72)


def test_the_boundary_value_that_leaves_nothing_is_dropped() -> None:
    """96 hours of extension leaves exactly zero, which the feature stage refuses."""
    assert feasible_hours(WINDOW, (95, 96, 97)) == (95,)


def test_the_configured_row_must_match_the_published_operating_point() -> None:
    """A configured value disagreeing with the published run is measuring something else."""
    points = [
        OperatingPoint(
            24,
            "xgboost",
            72,
            caught=5,
            alerts=400,
            precision_at_k=0.0125,
            threshold=0.517247,
            net_per_day=-210409.0,
            trees_kept=1,
        ),
    ]
    published = {"test/xgboost": {"caught": 5, "threshold": 0.517247}}

    assert_reproduces_published(points, published)


def test_a_configured_row_that_drifts_stops_the_sweep() -> None:
    """The regression check has teeth: a different catch count on the configured value raises."""
    points = [
        OperatingPoint(
            24,
            "xgboost",
            72,
            caught=4,
            alerts=400,
            precision_at_k=0.01,
            threshold=0.517247,
            net_per_day=-1.0,
            trees_kept=1,
        ),
    ]
    published = {"test/xgboost": {"caught": 5, "threshold": 0.517247}}

    with pytest.raises(SchemaError, match="does not reproduce"):
        assert_reproduces_published(points, published)
