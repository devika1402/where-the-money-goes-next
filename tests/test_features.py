"""Worked-example test for F2 (pass-through ratio), plus the window rules it depends on.

The numbers come from PRD section 8 F2 and are used exactly as written there. The test
asserts the correct value and the specific wrong value that boundary censoring produces,
so a regression back to the W-bounded form is caught rather than passing quietly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.definitions import SchemaError, Window
from src.features import (
    fifo_latency_pairs,
    fifo_pass_through_latency,
    median_hours_to_outflow,
    pass_through_ratio,
    ratio_windows,
)

ACCOUNT_A = 1

# W = [day 1 00:00, day 8 00:00), taking day 1 as 2022-09-01.
W_START = pd.Timestamp("2022-09-01 00:00:00")
W_END = pd.Timestamp("2022-09-08 00:00:00")
# W extended by pass_through_window_hours = 72.
W_EXTENDED_END = pd.Timestamp("2022-09-11 00:00:00")
MIN_FLOW_EUR = 100.0
POPULATION = pd.Index([ACCOUNT_A], name="account")


def _f2_frame() -> pd.DataFrame:
    """The F2 worked example as a normalised transaction frame.

    In:  200 EUR on day 2, 300 EUR on day 7 22:00
    Out: 180 EUR on day 3, 250 EUR on day 9 06:00, the last one inside the 72h extension.

    The day 7 inflow is the one that matters. Its matching outflow falls outside W, so a
    ratio bounded by W counts the money arriving and not the money leaving.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2022-09-02 00:00:00",
                    "2022-09-07 22:00:00",
                    "2022-09-03 00:00:00",
                    "2022-09-09 06:00:00",
                ]
            ),
            "account_from": [2, 2, ACCOUNT_A, ACCOUNT_A],
            "account_to": [ACCOUNT_A, ACCOUNT_A, 3, 3],
            "amount_in_base_received": [200.0, 300.0, 180.0, 250.0],
            "amount_in_base_paid": [200.0, 300.0, 180.0, 250.0],
        }
    )


def test_f2_matches_the_worked_example() -> None:
    """inflow 200 + 300 = 500. outflow over the extension 180 + 250 = 430. 430/500 = 0.86."""
    ratio = pass_through_ratio(
        _f2_frame(),
        POPULATION,
        inflow_start=W_START,
        inflow_end=W_END,
        outflow_end=W_EXTENDED_END,
        min_flow_eur=MIN_FLOW_EUR,
    )

    assert ratio.loc[ACCOUNT_A] == pytest.approx(0.86, abs=0.001)


def test_f2_bounding_the_outflow_by_w_gives_the_censored_answer() -> None:
    """The specific wrong value. 180/500 = 0.36.

    Every account looks like a hoarder at the window edge, because money arriving late has
    not had time to leave. This single choice moves the account from looking like a mule to
    looking like a saver.
    """
    ratio = pass_through_ratio(
        _f2_frame(),
        POPULATION,
        inflow_start=W_START,
        inflow_end=W_END,
        outflow_end=W_END,
        min_flow_eur=MIN_FLOW_EUR,
    )

    assert ratio.loc[ACCOUNT_A] == pytest.approx(0.36, abs=0.001)


def test_f2_zero_inflow_is_nan_rather_than_zero() -> None:
    """Zero means received money and kept all of it, which is the opposite of no inflow."""
    outflow_only = _f2_frame().iloc[2:]

    ratio = pass_through_ratio(
        outflow_only,
        POPULATION,
        inflow_start=W_START,
        inflow_end=W_END,
        outflow_end=W_EXTENDED_END,
        min_flow_eur=MIN_FLOW_EUR,
    )

    assert pd.isna(ratio.loc[ACCOUNT_A])


def test_f2_inflow_below_the_floor_is_nan() -> None:
    """The min_flow_eur floor stops tiny accounts producing extreme ratios."""
    frame = _f2_frame()
    frame.loc[[0, 1], "amount_in_base_received"] = [40.0, 50.0]

    ratio = pass_through_ratio(
        frame,
        POPULATION,
        inflow_start=W_START,
        inflow_end=W_END,
        outflow_end=W_EXTENDED_END,
        min_flow_eur=MIN_FLOW_EUR,
    )

    assert pd.isna(ratio.loc[ACCOUNT_A])


def test_f2_an_account_absent_from_the_data_is_nan_not_missing() -> None:
    """The result is indexed by the population, so a silent account still gets a row."""
    population = pd.Index([ACCOUNT_A, 99], name="account")

    ratio = pass_through_ratio(
        _f2_frame(),
        population,
        inflow_start=W_START,
        inflow_end=W_END,
        outflow_end=W_EXTENDED_END,
        min_flow_eur=MIN_FLOW_EUR,
    )

    assert list(ratio.index) == [ACCOUNT_A, 99]
    assert pd.isna(ratio.loc[99])


# --------------------------------------------------------------------------------------
# F2 edge case (b) meets F4: the outflow extension must never cross the leakage cutoff.
# --------------------------------------------------------------------------------------

WINDOW = Window(
    name="test",
    feature_start=pd.Timestamp("2022-09-05"),
    feature_end=pd.Timestamp("2022-09-09"),
    label_start=pd.Timestamp("2022-09-09"),
    label_end=pd.Timestamp("2022-09-11"),
)


def test_the_outflow_extension_stops_at_the_leakage_cutoff() -> None:
    """The extension buys time for money to leave. It may not buy time past the cutoff.

    With a 96 hour feature window and a 72 hour extension, the inflow sub-window is the
    first 24 hours and the outflow runs to the cutoff.
    """
    inflow_start, inflow_end, outflow_end = ratio_windows(WINDOW, pass_through_window_hours=72)

    assert inflow_start == pd.Timestamp("2022-09-05")
    assert inflow_end == pd.Timestamp("2022-09-06")
    assert outflow_end == WINDOW.feature_end
    assert outflow_end == WINDOW.label_start


def test_a_shorter_extension_leaves_a_longer_inflow_window() -> None:
    """The trade-off the sweep explores. 24 hours of extension leaves 72 hours of inflow."""
    _, inflow_end, outflow_end = ratio_windows(WINDOW, pass_through_window_hours=24)

    assert inflow_end == pd.Timestamp("2022-09-08")
    assert outflow_end == WINDOW.feature_end


def test_an_extension_that_would_consume_the_whole_window_is_refused() -> None:
    """Silently returning an empty inflow window would make every ratio NaN."""
    with pytest.raises(SchemaError, match="inflow"):
        ratio_windows(WINDOW, pass_through_window_hours=96)


# --------------------------------------------------------------------------------------
# F3 proxy: median hours from an inflow to the account's next outflow. The same edge-case
# (b) argument F2 makes about the window edge applies here, and for a subtler reason.
# --------------------------------------------------------------------------------------

ACCOUNT_B = 4


def _latency_frame() -> pd.DataFrame:
    """One account, two inflows, each followed by an outflow.

    The first inflow sits early and is forwarded in 6 hours. The second sits after the
    inflow cutoff, 12 hours before the leakage boundary, and is forwarded in 1 hour.

    Near the edge only the fast forwards are observable: an inflow at that point which was
    going to be forwarded in 20 hours has no outflow inside the window to match against, so
    it contributes nothing. Counting the survivors biases measured latency downward.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2022-09-05 12:00:00",
                    "2022-09-05 18:00:00",
                    "2022-09-08 12:00:00",
                    "2022-09-08 13:00:00",
                ]
            ),
            "account_from": [2, ACCOUNT_B, 2, ACCOUNT_B],
            "account_to": [ACCOUNT_B, 3, ACCOUNT_B, 3],
            "amount_in_base_received": [500.0, 500.0, 500.0, 500.0],
            "amount_in_base_paid": [500.0, 500.0, 500.0, 500.0],
        }
    )


def test_latency_ignores_inflows_with_no_room_for_a_slow_outflow_to_appear() -> None:
    """Only the early inflow counts, so the answer is 6 hours."""
    _, inflow_end, outflow_end = ratio_windows(WINDOW, pass_through_window_hours=24)

    latency = median_hours_to_outflow(
        _latency_frame(),
        pd.Index([ACCOUNT_B], name="account"),
        inflow_start=WINDOW.feature_start,
        inflow_end=inflow_end,
        outflow_end=outflow_end,
    )

    assert latency.loc[ACCOUNT_B] == pytest.approx(6.0)


def test_latency_bounded_by_the_window_takes_the_censored_answer() -> None:
    """The specific wrong value. Counting the late inflow too gives median(6, 1) = 3.5.

    That is the account looking almost twice as fast at moving money on, produced entirely
    by which inflows the window happened to let us see the other end of.
    """
    latency = median_hours_to_outflow(
        _latency_frame(),
        pd.Index([ACCOUNT_B], name="account"),
        inflow_start=WINDOW.feature_start,
        inflow_end=WINDOW.feature_end,
        outflow_end=WINDOW.feature_end,
    )

    assert latency.loc[ACCOUNT_B] == pytest.approx(3.5)


def test_latency_is_null_for_an_account_that_never_sent_money() -> None:
    """56% of the population. Null is the observation, not a missing measurement."""
    inflow_only = _latency_frame().iloc[[0]]

    latency = median_hours_to_outflow(
        inflow_only,
        pd.Index([ACCOUNT_B], name="account"),
        inflow_start=WINDOW.feature_start,
        inflow_end=WINDOW.feature_end,
        outflow_end=WINDOW.feature_end,
    )

    assert pd.isna(latency.loc[ACCOUNT_B])


# --------------------------------------------------------------------------------------
# F3, pass-through latency under FIFO attribution. G5, D31.
#
# The PRD cut this on the grounds that FIFO matching is inherently sequential per account.
# It is not. Recast onto a cumulative-amount axis, an inflow occupies the money interval
# (A[k-1], A[k]] and an outflow occupies (B[j-1], B[j]], so FIFO matching is the overlap of
# two step functions and the pairs fall out of a merge_asof. No loop over accounts.

F3_ORIGIN = pd.Timestamp("2022-09-01 00:00:00")


def _f3_frame(account: int = ACCOUNT_A) -> pd.DataFrame:
    """The F3 worked example from PRD section 8, hours measured from F3_ORIGIN.

    In:  t=0h 100, t=10h 200, t=30h 100
    Out: t=12h 150, t=40h 200
    """
    rows = [
        (0, 2, account, 100.0),
        (10, 2, account, 200.0),
        (30, 2, account, 100.0),
        (12, account, 3, 150.0),
        (40, account, 3, 200.0),
    ]
    return pd.DataFrame(
        {
            "timestamp": [F3_ORIGIN + pd.Timedelta(hours=h) for h, _, _, _ in rows],
            "account_from": [f for _, f, _, _ in rows],
            "account_to": [t for _, _, t, _ in rows],
            "amount_in_base_received": [a for _, _, _, a in rows],
            "amount_in_base_paid": [a for _, _, _, a in rows],
        }
    )


F3_WINDOW_END = F3_ORIGIN + pd.Timedelta(hours=48)


def _f3_pairs(frame: pd.DataFrame, population: pd.Index) -> pd.DataFrame:
    """The matched pairs, so the test can assert the attribution and not only its median."""
    return fifo_latency_pairs(
        frame,
        population,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )


def _pair_list(pairs: pd.DataFrame) -> list[tuple[float, float]]:
    """Matched pairs as sorted (latency, weight) tuples, to assert against the PRD's list."""
    latencies = pairs["latency_hours"].to_numpy(dtype=float)
    weights = pairs["weight"].to_numpy(dtype=float)
    return sorted(
        (round(float(a), 6), round(float(b), 6)) for a, b in zip(latencies, weights, strict=True)
    )


def test_f3_matches_the_worked_example() -> None:
    """Weighted median latency is 12.0 hours. PRD section 8 F3."""
    latency = fifo_pass_through_latency(
        _f3_frame(),
        POPULATION,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )

    assert latency.loc[ACCOUNT_A] == pytest.approx(12.0)


def test_f3_produces_the_four_worked_example_pairs() -> None:
    """(12,100) (2,50) (30,150) (10,50), total weight 350. The attribution, not the median.

    Asserting the median alone would pass for several wrong matchings that happen to land
    on 12 hours, so the pairs the PRD writes out are asserted one for one.
    """
    pairs = _f3_pairs(_f3_frame(), POPULATION)
    got = _pair_list(pairs)

    assert got == [(2.0, 50.0), (10.0, 50.0), (12.0, 100.0), (30.0, 150.0)]
    assert pairs["weight"].sum() == pytest.approx(350.0)


def test_unmatched_inflow_at_the_window_edge_is_dropped() -> None:
    """400 arrived and 350 left, so 50 of the t=30 inflow is never attributed.

    The PRD is explicit that this is dropped rather than counted as infinite latency.
    Counting it would make every account that is still holding money look slow.
    """
    pairs = _f3_pairs(_f3_frame(), POPULATION)

    assert pairs["weight"].sum() == pytest.approx(350.0)
    assert pairs["weight"].sum() < 400.0


def test_the_naive_time_to_next_outflow_gives_a_different_answer() -> None:
    """10.0 hours against F3's 12.0. The specific wrong value, so a silent swap is caught.

    The naive form asks each inflow how long until the next outflow: 12h, 2h and 10h, whose
    unweighted median is 10. F3 asks which money left when, and answers 12.
    """
    naive = median_hours_to_outflow(
        _f3_frame(),
        POPULATION,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )
    fifo = fifo_pass_through_latency(
        _f3_frame(),
        POPULATION,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )

    assert naive.loc[ACCOUNT_A] == pytest.approx(10.0)
    assert fifo.loc[ACCOUNT_A] == pytest.approx(12.0)
    assert naive.loc[ACCOUNT_A] != fifo.loc[ACCOUNT_A]


def test_an_outflow_spanning_two_inflows_produces_two_pairs() -> None:
    """The FIFO rule the PRD states: one outflow larger than the oldest unconsumed inflow."""
    frame = pd.DataFrame(
        {
            "timestamp": [
                F3_ORIGIN,
                F3_ORIGIN + pd.Timedelta(hours=4),
                F3_ORIGIN + pd.Timedelta(hours=6),
            ],
            "account_from": [2, 2, ACCOUNT_A],
            "account_to": [ACCOUNT_A, ACCOUNT_A, 3],
            "amount_in_base_received": [100.0, 100.0, 150.0],
            "amount_in_base_paid": [100.0, 100.0, 150.0],
        }
    )

    pairs = _f3_pairs(frame, POPULATION)
    got = _pair_list(pairs)

    assert got == [(2.0, 50.0), (6.0, 100.0)]


def test_two_accounts_are_matched_independently() -> None:
    """The whole point of the money-axis form is that it runs for every account at once.

    Account B's cumulative sums must not leak into account A's matching, which is the
    failure a groupby-free cumsum would produce and which no single-account test can see.
    """
    both = pd.concat([_f3_frame(ACCOUNT_A), _f3_frame(ACCOUNT_B)], ignore_index=True)
    population = pd.Index([ACCOUNT_A, ACCOUNT_B], name="account")

    latency = fifo_pass_through_latency(
        both,
        population,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )

    assert latency.loc[ACCOUNT_A] == pytest.approx(12.0)
    assert latency.loc[ACCOUNT_B] == pytest.approx(12.0)


def test_f3_is_null_for_an_account_that_never_sent_money() -> None:
    """No outflow means no matched pair, so there is nothing to take a median of."""
    inflow_only = _f3_frame().iloc[[0, 1, 2]]

    latency = fifo_pass_through_latency(
        inflow_only,
        POPULATION,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )

    assert pd.isna(latency.loc[ACCOUNT_A])


def test_the_weighted_median_takes_the_first_latency_reaching_half_the_weight() -> None:
    """The tie convention, which the worked example cannot pin because it has no tie.

    Two pairs of equal weight: 4 hours and 14 hours, 100 each. Half of 200 is exactly 100,
    which the first pair reaches exactly. Taking the first latency whose cumulative weight
    reaches half gives 4. Requiring it to exceed half gives 14, and interpolating between
    the two gives 9. All three are defensible and the PRD fixes the first, so it is pinned
    here rather than left to whichever comparison the implementation happens to use.
    """
    frame = pd.DataFrame(
        {
            "timestamp": [
                F3_ORIGIN,
                F3_ORIGIN + pd.Timedelta(hours=6),
                F3_ORIGIN + pd.Timedelta(hours=4),
                F3_ORIGIN + pd.Timedelta(hours=20),
            ],
            "account_from": [2, 2, ACCOUNT_A, ACCOUNT_A],
            "account_to": [ACCOUNT_A, ACCOUNT_A, 3, 3],
            "amount_in_base_received": [100.0, 100.0, 100.0, 100.0],
            "amount_in_base_paid": [100.0, 100.0, 100.0, 100.0],
        }
    )

    pairs = _f3_pairs(frame, POPULATION)
    got = _pair_list(pairs)
    latency = fifo_pass_through_latency(
        frame,
        POPULATION,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )

    assert got == [(4.0, 100.0), (14.0, 100.0)]
    assert latency.loc[ACCOUNT_A] == pytest.approx(4.0)


def test_an_outflow_before_any_inflow_produces_no_pair() -> None:
    """Money leaving before it arrived is not a pass-through, and the PRD's example cannot see it.

    An account holding a balance from before the window sends money the window never saw
    arrive. On the cumulative-amount axis that outflow still lands under the first inflow
    inside the window, which matches an outflow at t=2h to an inflow at t=8h and calls the
    latency minus six hours. FIFO consumes a running balance, so an outflow can only be
    attributed to inflow that had already arrived, and this pair is dropped.

    Dropping it is the same rule the PRD states for unmatched inflow at the far edge, applied
    to the near edge: the window cannot see the money, so the window does not price it.
    """
    frame = pd.DataFrame(
        {
            "timestamp": [
                F3_ORIGIN + pd.Timedelta(hours=2),
                F3_ORIGIN + pd.Timedelta(hours=8),
                F3_ORIGIN + pd.Timedelta(hours=12),
            ],
            "account_from": [ACCOUNT_A, 2, ACCOUNT_A],
            "account_to": [3, ACCOUNT_A, 3],
            "amount_in_base_received": [100.0, 200.0, 100.0],
            "amount_in_base_paid": [100.0, 200.0, 100.0],
        }
    )

    pairs = _f3_pairs(frame, POPULATION)
    got = _pair_list(pairs)
    latency = fifo_pass_through_latency(
        frame,
        POPULATION,
        inflow_start=F3_ORIGIN,
        inflow_end=F3_WINDOW_END,
        outflow_end=F3_WINDOW_END,
    )

    # Only the t=12h outflow is attributable, against the t=8h inflow.
    assert got == [(4.0, 100.0)]
    assert latency.loc[ACCOUNT_A] == pytest.approx(4.0)
    assert all(pairs["latency_hours"] > 0)


def test_every_pair_on_the_worked_example_still_survives_the_ordering_rule() -> None:
    """The ordering rule must not quietly discard any of the four pairs F3 specifies."""
    pairs = _f3_pairs(_f3_frame(), POPULATION)

    assert len(pairs) == 4
    assert pairs["weight"].sum() == pytest.approx(350.0)
