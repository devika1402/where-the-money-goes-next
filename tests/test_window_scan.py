"""Tests for the G1 history-window scan.

The scan's job is to vary one thing and leave everything else identical, so these tests fix
the label window and assert that the parts which must not move do not move: the mule count in
the label window, the leakage invariant, and the triangle count under blocking.

The published four-day figures are not asserted here, because reproducing them needs the
148 MB interim parquet, which is not in the repository. That check runs inside
``src.window_scan.main`` against the real file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.definitions import SchemaError, Window
from src.window_scan import graph_stats, rehistory, scan, structure

DATA_START = pd.Timestamp("2022-09-01 00:00:00")
LABEL_START = pd.Timestamp("2022-09-09 00:00:00")
LABEL_END = pd.Timestamp("2022-09-11 00:00:00")
BLOCK_ROWS = 50_000

PUBLISHED = Window(
    name="test",
    feature_start=pd.Timestamp("2022-09-05 00:00:00"),
    feature_end=LABEL_START,
    label_start=LABEL_START,
    label_end=LABEL_END,
)


def _frame() -> pd.DataFrame:
    """Six accounts over the span, built so each history length has a known answer.

    Account 10 receives on 09-02 only, so a four-day history starting 09-05 cannot see it and
    an eight-day history starting 09-01 can. Account 20 receives on 09-06, inside every
    history length. Account 30 receives nothing before the label window at all, so it is
    unreachable at every length. Accounts 10 and 30 both receive laundering in the label
    window, account 20 receives clean money there.
    """
    rows = [
        # (timestamp, from, to, laundering)
        ("2022-09-02 09:00:00", 1, 10, 0),
        ("2022-09-06 09:00:00", 1, 20, 0),
        ("2022-09-06 10:00:00", 2, 20, 0),
        # Label window.
        ("2022-09-09 09:00:00", 3, 10, 1),
        ("2022-09-09 10:00:00", 3, 20, 0),
        ("2022-09-10 09:00:00", 4, 30, 1),
    ]
    frame = pd.DataFrame(rows, columns=["timestamp", "account_from", "account_to", "is_laundering"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["amount_in_base_received"] = 100.0
    frame["amount_in_base_paid"] = 100.0
    frame["is_laundering"] = frame["is_laundering"].astype("int8")
    return frame


def test_rehistory_keeps_the_leakage_boundary_and_the_label_window() -> None:
    """Only the history side moves. The cutoff instant and the label window stay put."""
    window = rehistory(PUBLISHED, 8, DATA_START)

    assert window.feature_start == DATA_START
    assert window.feature_end == LABEL_START
    assert window.label_start == LABEL_START
    assert window.label_end == LABEL_END
    assert window.feature_end == window.label_start


def test_rehistory_raises_outside_the_usable_span() -> None:
    """A ten-day history before 09-09 starts on 08-30, which is not in the file."""
    with pytest.raises(SchemaError, match="outside the usable span"):
        rehistory(PUBLISHED, 10, DATA_START)


def test_reachability_rises_with_history_and_the_label_count_does_not_move() -> None:
    """Account 10 is reachable at eight days and not at four. Both lengths see two mules."""
    rows = scan(_frame(), PUBLISHED, (4, 8), DATA_START, BLOCK_ROWS)
    four, eight = rows

    assert four.mules_in_label_window == 2
    assert eight.mules_in_label_window == 2

    assert four.reachable_mules == 0
    assert four.unscoreable_mules == 2

    assert eight.reachable_mules == 1
    assert eight.unscoreable_mules == 1
    assert eight.reachability == pytest.approx(0.5)


def test_population_grows_with_history() -> None:
    """Four days sees account 20 only. Eight days also sees account 10."""
    four, eight = scan(_frame(), PUBLISHED, (4, 8), DATA_START, BLOCK_ROWS)

    assert four.population == 1
    assert eight.population == 2
    assert eight.base_rate == pytest.approx(0.5)


def test_a_mule_with_no_history_at_any_length_stays_unreachable() -> None:
    """Account 30's first appearance is inside the label window, so no history reaches it."""
    for row in scan(_frame(), PUBLISHED, (4, 6, 8), DATA_START, BLOCK_ROWS):
        assert row.unscoreable_mules >= 1


def test_triangle_count_is_the_same_blocked_and_unblocked() -> None:
    """One triangle over 0-1-2 plus an isolated edge. Blocking must not change the answer."""
    size = 5
    rows = np.array([0, 1, 0, 3])
    cols = np.array([1, 2, 2, 4])

    whole = structure(rows, cols, size, size)
    blocked = structure(rows, cols, size, 2)

    assert whole == blocked
    assert whole.triangles == 1
    assert whole.accounts_in_triangle == 3
    assert whole.undirected_edges == 4


def test_graph_stats_counts_one_triangle_and_the_accounts_in_it() -> None:
    """Three accounts paying each other in a cycle make one triangle, and 40 makes none."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2022-09-06 01:00:00",
                    "2022-09-06 02:00:00",
                    "2022-09-06 03:00:00",
                    "2022-09-06 04:00:00",
                ]
            ),
            "account_from": [10, 20, 30, 10],
            "account_to": [20, 30, 10, 40],
        }
    )

    stats = graph_stats(frame, PUBLISHED, BLOCK_ROWS)

    assert stats.nodes == 4
    assert stats.directed_edges == 4
    assert stats.undirected_edges == 4
    assert stats.triangles == 1
    assert stats.accounts_in_triangle == 3
    assert stats.zero_triangle_share == pytest.approx(0.25)
    assert stats.mean_degree == pytest.approx(2.0)


def test_reciprocal_pairs_collapse_to_one_undirected_edge() -> None:
    """Two accounts paying each other are two directed edges and one undirected edge."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-09-06 01:00:00", "2022-09-06 02:00:00"]),
            "account_from": [10, 20],
            "account_to": [20, 10],
        }
    )

    stats = graph_stats(frame, PUBLISHED, BLOCK_ROWS)

    assert stats.directed_edges == 2
    assert stats.undirected_edges == 1
    assert stats.triangles == 0
