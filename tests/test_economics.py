"""Worked-example test for F6 (precision at k and expected value recovered).

The numbers come from PRD section 8 F6 and are used exactly as written there. The second
block asserts the specific wrong value the edge case produces: pooling a two-day window
into one budget gives a different and higher precision, so the per-day rule of edge case
(a) is enforced by the test rather than by a comment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.definitions import Costs, SchemaError, queue_policy_horizon
from src.economics import (
    _write_ceiling_figures,
    apply_overflow_policy,
    break_even_curve,
    break_even_precision,
    build_candidates,
    crossing_ratio,
    distinct_mules,
    evaluate,
    optimal_volume,
    overflow_verdict,
    replace_cost,
    select_alerts,
    sensitivity,
)

# PRD section 8 F6: k = 3 per day, cost_investigation = 25, cost_false_freeze = 75,
# cost_missed = 1150, no overflow. k is analyst_capacity_per_day, so it lives in Costs.
COSTS = Costs(
    cost_missed_mule_eur=1150.0,
    cost_investigation_eur=25.0,
    cost_false_freeze_eur=75.0,
    analyst_capacity_per_day=3,
    queue_overflow_policy="same_day",
)
K = COSTS.analyst_capacity_per_day

DAY_1 = pd.Timestamp("2022-09-09")
DAY_2 = pd.Timestamp("2022-09-10")

#: Exposure is only ever read for accounts that are mules, so the clean rows carry 0.0
#: rather than the PRD's "-".
DAY_1_ROWS = [
    (1, DAY_1, 0.91, 1, 2000.0),
    (2, DAY_1, 0.88, 0, 0.0),
    (3, DAY_1, 0.74, 1, 900.0),
    (4, DAY_1, 0.51, 0, 0.0),
    (5, DAY_1, 0.30, 1, 400.0),
]

#: A second day whose only mule scores below that day's top three. Pooling the two days
#: into one budget of 3 spends the whole period's capacity on day one, which is exactly
#: what F6 edge case (a) forbids.
DAY_2_ROWS = [
    (6, DAY_2, 0.60, 0, 0.0),
    (7, DAY_2, 0.40, 0, 0.0),
    (8, DAY_2, 0.20, 0, 0.0),
    (9, DAY_2, 0.05, 1, 500.0),
]

COLUMNS = ["account", "day", "score", "is_mule", "exposure"]


def _frame(*rows: tuple[int, pd.Timestamp, float, int, float]) -> pd.DataFrame:
    """A candidate queue: one row per account per day it was eligible to be alerted."""
    return pd.DataFrame(list(rows), columns=COLUMNS)


def test_f6_matches_the_worked_example() -> None:
    """Top 3 of day one: 0.91 mule, 0.88 clean, 0.74 mule. Every figure from PRD F6."""
    outcome = evaluate(_frame(*DAY_1_ROWS), k=K, costs=COSTS)

    assert outcome.precision_at_k == pytest.approx(0.6667, abs=0.0001)
    assert outcome.threshold == pytest.approx(0.74)
    assert outcome.ev_recovered == pytest.approx(2750.0)
    assert outcome.ev_missed == pytest.approx(1150.0)
    assert outcome.net == pytest.approx(1600.0)


def test_f6_reports_the_parts_the_worked_example_sums() -> None:
    """2000 + 900 recovered, 3 investigations at 25, one clean account frozen at 75."""
    outcome = evaluate(_frame(*DAY_1_ROWS), k=K, costs=COSTS)

    assert outcome.n_alerts == 3
    assert outcome.true_positives == 2
    assert outcome.false_positives == 1
    assert outcome.exposure_recovered == pytest.approx(2900.0)
    assert outcome.investigation_cost == pytest.approx(75.0)
    assert outcome.freeze_cost == pytest.approx(75.0)
    assert outcome.missed == 1


def test_computing_k_globally_over_two_days_flatters_the_result() -> None:
    """F6 edge case (a). The specific wrong value, and it is higher.

    Collapsing both days onto one date is the mistake: it lets one budget of 3 be spent
    entirely on the better day. Per day, the same budget must also be spent on day two,
    where the only mule scores below the cut.
    """
    two_days = _frame(*DAY_1_ROWS, *DAY_2_ROWS)
    per_day = evaluate(two_days, k=K, costs=COSTS)

    pooled = evaluate(two_days.assign(day=DAY_1), k=K, costs=COSTS)

    assert per_day.precision_at_k == pytest.approx(0.3333, abs=0.0001)
    assert pooled.precision_at_k == pytest.approx(0.6667, abs=0.0001)
    assert pooled.precision_at_k > per_day.precision_at_k

    # The per-day run spends twice the budget for the same two catches.
    assert per_day.n_alerts == 6
    assert pooled.n_alerts == 3
    assert per_day.true_positives == pooled.true_positives == 2


def test_an_account_already_investigated_does_not_consume_a_second_day() -> None:
    """Capacity is same-day only. Nobody is worked twice, and no queue rolls over."""
    both_days = _frame(
        (1, DAY_1, 0.91, 1, 2000.0),
        (2, DAY_1, 0.10, 0, 0.0),
        (1, DAY_2, 0.91, 1, 2000.0),
        (3, DAY_2, 0.20, 0, 0.0),
    )

    alerted = select_alerts(both_days, k=1)

    assert list(alerted) == [True, False, False, True]


def test_a_mule_seen_on_both_days_counts_once() -> None:
    """Recall divides by distinct mule accounts, not by rows in the queue."""
    both_days = _frame(
        (1, DAY_1, 0.10, 1, 2000.0),
        (2, DAY_1, 0.90, 0, 0.0),
        (1, DAY_2, 0.10, 1, 2000.0),
        (3, DAY_2, 0.90, 0, 0.0),
    )

    outcome = evaluate(both_days, k=1, costs=COSTS)

    assert outcome.missed == 1
    assert outcome.ev_missed == pytest.approx(1150.0)
    assert outcome.recall == pytest.approx(0.0)


def test_a_day_short_of_candidates_is_only_charged_for_work_actually_done() -> None:
    """Investigation cost follows alerts raised, not the capacity the rota allowed for."""
    outcome = evaluate(_frame(*DAY_1_ROWS[:2]), k=K, costs=COSTS)

    assert outcome.n_alerts == 2
    assert outcome.investigation_cost == pytest.approx(50.0)


def test_overflow_above_the_threshold_is_capped_at_the_day_it_arrived() -> None:
    """Ten candidates, capacity 3, no rollover. The other seven are never worked."""
    busy = _frame(*[(i, DAY_1, 1.0 - i / 100, 0, 0.0) for i in range(10)])
    quiet = _frame((99, DAY_2, 0.5, 0, 0.0))

    outcome = evaluate(pd.concat([busy, quiet], ignore_index=True), k=3, costs=COSTS)

    assert outcome.n_alerts == 4


def test_ties_are_broken_by_account_so_a_reordered_queue_picks_the_same_ones() -> None:
    """A rules baseline produces heavy ties, so the pick must not depend on row order.

    The tie-break is the lowest account id. It is arbitrary, and arbitrary is the point:
    the alternative is letting parquet row order decide who gets investigated.
    """
    tied = _frame(
        (7, DAY_1, 0.5, 1, 100.0),
        (3, DAY_1, 0.5, 0, 0.0),
        (5, DAY_1, 0.5, 0, 0.0),
    )
    shuffled = tied.iloc[::-1].reset_index(drop=True)

    forwards = tied.loc[select_alerts(tied, k=2), "account"]
    backwards = shuffled.loc[select_alerts(shuffled, k=2), "account"]

    assert sorted(forwards) == [3, 5]
    assert sorted(backwards) == [3, 5]


# --------------------------------------------------------------------------------------
# F6 edge case (c), queue overflow. G3, D29.
#
# The policy is a transformation of the queue rather than a second selection rule: under
# rollover an unreached candidate keeps a row on later days, and the same same-day
# selection then lets it compete for that day's capacity. Everything downstream of the
# queue is untouched, so a policy change cannot alter the cost arithmetic by accident.

DAY_3 = pd.Timestamp("2022-09-11")
DAY_4 = pd.Timestamp("2022-09-12")


def _alerted_under(
    policy: str, rows: list[tuple[int, pd.Timestamp, float, int, float]], k: int
) -> list[int]:
    """Which accounts get worked under one overflow policy. Accounts, not rows."""
    queue = apply_overflow_policy(_frame(*rows), policy)
    return sorted({int(a) for a in queue.loc[select_alerts(queue, k), "account"]})


#: Account 2 misses day one's single slot and outscores the only account day two brings.
CARRY_ROWS = [
    (1, DAY_1, 0.91, 1, 2000.0),
    (2, DAY_1, 0.80, 0, 0.0),
    (3, DAY_2, 0.50, 0, 0.0),
]


def test_the_horizon_of_each_policy_is_declared_once() -> None:
    """``same_day`` keeps a candidate for the day it arrived. ``rollover_max_3d`` for three."""
    assert queue_policy_horizon("same_day") == 1
    assert queue_policy_horizon("rollover_max_3d") == 3


def test_an_unknown_overflow_policy_is_refused() -> None:
    """A misspelt policy must stop the build rather than fall back to same-day capacity."""
    with pytest.raises(SchemaError, match="overflow policy"):
        queue_policy_horizon("rollover_forever")

    with pytest.raises(SchemaError, match="overflow policy"):
        apply_overflow_policy(_frame(*CARRY_ROWS), "rollover_forever")


def test_same_day_capacity_leaves_the_queue_exactly_as_it_arrived() -> None:
    """The published build runs this policy, so it has to be the identity on the queue."""
    queue = _frame(*DAY_1_ROWS, *DAY_2_ROWS)

    pd.testing.assert_frame_equal(apply_overflow_policy(queue, "same_day"), queue)


def test_rollover_lets_yesterdays_leftover_take_todays_capacity() -> None:
    """The whole point of the policy, and the specific different answer it gives.

    Capacity is one a day. Under same-day capacity account 2 is discarded at the end of
    day one and day two spends its slot on account 3 at 0.50. Under rollover account 2
    carries its 0.80 into day two and takes the slot instead.
    """
    assert _alerted_under("same_day", CARRY_ROWS, k=1) == [1, 3]
    assert _alerted_under("rollover_max_3d", CARRY_ROWS, k=1) == [1, 2]


def test_rollover_expires_an_entry_after_three_days() -> None:
    """Four days, capacity one. Account 2 arrives on day one and must not survive to day four.

    Its 0.50 would outrank day four's only candidate at 0.10, so an unbounded backlog
    alerts account 2 and never reaches account 5. That is the specific wrong answer, and
    the expiry is what stops it.
    """
    ageing = [
        (1, DAY_1, 0.99, 0, 0.0),
        (2, DAY_1, 0.50, 0, 0.0),
        (3, DAY_2, 0.98, 0, 0.0),
        (4, DAY_3, 0.97, 0, 0.0),
        (5, DAY_4, 0.10, 0, 0.0),
    ]

    assert _alerted_under("rollover_max_3d", ageing, k=1) == [1, 3, 4, 5]


def test_rollover_still_spends_at_most_capacity_a_day() -> None:
    """A backlog changes who is worked, never how many. Ten arrive on day one, capacity 3."""
    busy = [(i, DAY_1, 1.0 - i / 100, 0, 0.0) for i in range(10)]
    quiet = [(99, DAY_2, 0.005, 0, 0.0)]

    rolled = apply_overflow_policy(_frame(*busy, *quiet), "rollover_max_3d")
    outcome = evaluate(rolled, k=3, costs=COSTS)

    assert outcome.n_alerts == 6
    assert outcome.n_days == 2


def test_rollover_changes_neither_the_population_nor_the_number_of_days() -> None:
    """Recall, the base rate and net per day all divide by these, so neither may move.

    The transformation adds account-days to the queue and no accounts and no days. If it
    added either, every rate reported beside a rollover figure would be measured against a
    different denominator from the same-day figure it is set against.
    """
    queue = _frame(*DAY_1_ROWS, *DAY_2_ROWS)
    rolled = apply_overflow_policy(queue, "rollover_max_3d")

    assert rolled["account"].nunique() == queue["account"].nunique()
    assert rolled["day"].nunique() == queue["day"].nunique()
    assert distinct_mules(rolled) == distinct_mules(queue)
    assert len(rolled) > len(queue)


def test_rollover_is_the_identity_when_every_account_is_eligible_every_day() -> None:
    """Nothing can roll over into a day that already held the same accounts."""
    everywhere = _frame(
        (1, DAY_1, 0.90, 1, 2000.0),
        (2, DAY_1, 0.10, 0, 0.0),
        (1, DAY_2, 0.90, 1, 2000.0),
        (2, DAY_2, 0.10, 0, 0.0),
    )

    rolled = apply_overflow_policy(everywhere, "rollover_max_3d")

    assert len(rolled) == len(everywhere)
    assert list(select_alerts(rolled, k=1)) == list(select_alerts(everywhere, k=1))


def test_the_footer_names_the_active_policy_alongside_the_four_numbers() -> None:
    """Five checkpoint assumptions, and the categorical one is the reason for this test.

    Without it a rollover figure and a same-day figure carry identical footers, and the
    policy is the assumption that decides which accounts were alerted at all.
    """
    footer = COSTS.as_footer()

    assert "overflow=same_day" in footer
    assert "capacity=3" in footer
    assert (
        "overflow=rollover_max_3d"
        in replace_cost(
            Costs(1150.0, 25.0, 75.0, 3, "rollover_max_3d"), "cost_investigation_eur", 50.0
        ).as_footer()
    )


def test_the_policy_cannot_be_swept_as_if_it_were_a_number() -> None:
    """A categorical assumption has no multiplier, so asking for one must fail loudly."""
    with pytest.raises(SchemaError, match="categorical"):
        replace_cost(COSTS, "queue_overflow_policy", 2.0)


def test_sweeping_a_cost_carries_the_policy_through_untouched() -> None:
    """A cost sweep must not quietly reset the policy to whatever the default happens to be."""
    rolled = Costs(1150.0, 25.0, 75.0, 3, "rollover_max_3d")

    moved = replace_cost(rolled, "cost_missed_mule_eur", 4600.0)

    assert moved.queue_overflow_policy == "rollover_max_3d"
    assert moved.cost_missed_mule_eur == pytest.approx(4600.0)


def test_a_two_day_window_cannot_reach_the_three_day_expiry() -> None:
    """Why the 3 in ``rollover_max_3d`` is inert on this project's windows, pinned as a test.

    ``label_window_days`` is 2, so a candidate is at most one day old before the window
    ends. The expiry can only bind from the third day onwards, which no window here has.
    """
    two_days = _frame(*CARRY_ROWS)

    horizon_3 = apply_overflow_policy(two_days, "rollover_max_3d")
    unbounded = pd.concat(
        [two_days, two_days.assign(day=two_days["day"] + pd.Timedelta(days=1))],
        ignore_index=True,
    )
    unbounded = unbounded.loc[unbounded["day"].isin(two_days["day"].unique())]

    assert len(horizon_3) == len(unbounded.drop_duplicates(["account", "day"]))


# --------------------------------------------------------------------------------------
# The two optimisation problems of F6 edge case (b), and the ratio between them.
#
# One day, 100 accounts scored 1.00 down to 0.01, mules at ranks 1, 5 and 20 carrying no
# exposure. Stripping exposure out leaves the cost parameters as the only thing moving the
# answer, which is what these tests are about. Reaching each mule costs, in wrong alerts:
#   rank 1  -> 25            rank 5  -> 25 + 3x100      rank 20 -> 25 + 14x100
# so the cost-optimal volume steps 0 -> 1 -> 5 -> 20 as a missed mule gets dearer.
# --------------------------------------------------------------------------------------

MULE_RANKS = (1, 5, 20)
VOLUMES = [0, 1, 5, 20, 100]
LADDER = _frame(
    *[(rank, DAY_1, 1.0 - rank / 100.0, int(rank in MULE_RANKS), 0.0) for rank in range(1, 101)]
)


@pytest.mark.parametrize(
    ("cost_missed", "expected"),
    [(10.0, 0), (100.0, 1), (500.0, 5), (2000.0, 20)],
)
def test_the_cost_optimal_volume_rises_as_a_missed_mule_gets_dearer(
    cost_missed: float, expected: int
) -> None:
    """The unconstrained problem. Nothing else changes, so the step is the cost alone."""
    costs = replace_cost(COSTS, "cost_missed_mule_eur", cost_missed)

    assert optimal_volume(LADDER, costs, VOLUMES) == expected


def test_the_budget_constrained_threshold_does_not_move_with_the_euro_costs() -> None:
    """The finding Checkpoint 4 exists to surface, asserted rather than asserted about.

    Under a fixed budget the threshold is a rank. Three of the four assumptions cannot
    reach it, so a sensitivity table that showed them moving it would be reporting a bug.
    """
    base = evaluate(LADDER, COSTS.analyst_capacity_per_day, COSTS)

    for name in ("cost_missed_mule_eur", "cost_investigation_eur", "cost_false_freeze_eur"):
        moved = replace_cost(COSTS, name, 1000.0 * getattr(COSTS, name))
        outcome = evaluate(LADDER, moved.analyst_capacity_per_day, moved)
        assert outcome.threshold == base.threshold
        assert outcome.n_alerts == base.n_alerts


def test_only_capacity_moves_the_budget_constrained_threshold() -> None:
    """The fourth assumption is the one that reaches it, and it moves it downward."""
    tight = evaluate(LADDER, 3, COSTS)
    loose = evaluate(LADDER, 30, COSTS)

    assert loose.threshold < tight.threshold
    assert loose.n_alerts == 30


def test_the_break_even_ratio_is_where_the_budget_starts_to_bind() -> None:
    """R = cost of a missed mule over cost of a wrong alert, which here is 25 + 75 = 100.

    Below the crossing the firm would choose to alert less than capacity, so the budget is
    not the binding constraint and the recommendation is a threshold. Above it the budget
    binds and the recommendation is capacity.
    """
    curve = break_even_curve(LADDER, COSTS, ratios=[0.1, 1.0, 5.0, 20.0], volumes=VOLUMES)

    assert list(curve["optimal_k"]) == [0.0, 1.0, 5.0, 20.0]
    assert list(curve["cost_missed_mule_eur"]) == [10.0, 100.0, 500.0, 2000.0]
    assert crossing_ratio(curve, capacity=3) == pytest.approx(5.0)


def test_a_sweep_that_never_crosses_says_so_rather_than_guessing() -> None:
    """NaN is the honest answer when the budget binds nowhere in the range examined."""
    curve = break_even_curve(LADDER, COSTS, ratios=[0.1, 1.0], volumes=VOLUMES)

    assert pd.isna(crossing_ratio(curve, capacity=3))


def test_the_sensitivity_table_moves_each_assumption_alone() -> None:
    """Four parameters against every multiplier, and the rest of the row held at config."""
    table = sensitivity(LADDER, COSTS, multipliers=[0.5, 1.0, 2.0])

    assert len(table) == 12
    assert set(table["parameter"]) == {
        "cost_missed_mule_eur",
        "cost_investigation_eur",
        "cost_false_freeze_eur",
        "analyst_capacity_per_day",
    }
    capacity = table.loc[table["parameter"] == "analyst_capacity_per_day"]
    assert list(capacity["alerts"]) == [1.0, 3.0, 6.0]


def test_replacing_a_parameter_that_does_not_exist_stops_the_build() -> None:
    """A typo in a sweep would otherwise report the base case four more times."""
    with pytest.raises(SchemaError, match="not a cost parameter"):
        replace_cost(COSTS, "cost_of_being_wrong", 1.0)


def test_a_queue_holding_an_unscored_account_stops_the_build() -> None:
    """The population is pinned by D9, so an account with no score is a defect upstream."""
    population = pd.Index([1, 2], name="account")
    queue = pd.DataFrame({"account": [1, 2, 3], "day": [DAY_1, DAY_1, DAY_1]})

    with pytest.raises(SchemaError, match="not in the scored population"):
        build_candidates(
            pd.Series([0.5, 0.5], index=population),
            pd.Series([0, 1], index=population),
            pd.Series([0.0, 10.0], index=population),
            queue,
        )


def test_the_break_even_precision_is_what_an_alert_must_reach_to_pay_for_itself() -> None:
    """With nothing recovered, an alert has to be right once in twelve to be worth raising.

    100 EUR to raise a wrong one, against 1150 avoided plus the 75 not spent freezing a
    clean account: 100 / 1225.
    """
    assert break_even_precision(COSTS, recovered_per_mule=0.0) == pytest.approx(0.081633, abs=1e-6)


def test_recovering_money_from_a_caught_mule_lowers_the_bar_sharply() -> None:
    """The term that swamps the other three. At 15,613 EUR recovered, 100 / 16,838."""
    assert break_even_precision(COSTS, recovered_per_mule=15_613.0) == pytest.approx(
        0.005939, abs=1e-6
    )


def test_an_optimum_at_the_edge_of_the_swept_grid_is_reported_as_such() -> None:
    """A clipped optimum is not an optimum, and presenting one as found would be a lie.

    The grid stops at 5 while the third mule sits at rank 20, so a high enough cost of
    missing one pins the answer against the edge with nowhere further to look.
    """
    curve = break_even_curve(LADDER, COSTS, ratios=[0.1, 1000.0], volumes=[0, 1, 5])

    assert list(curve["optimal_k"]) == [0.0, 5.0]
    assert list(curve["optimal_at_grid_edge"]) == [0.0, 1.0]


def test_a_crossing_at_the_floor_of_the_sweep_reports_the_floor_not_a_location() -> None:
    """The caller has to be able to tell 'crossed at 0.001' from 'crossed below 0.001'.

    A sweep that already binds at its smallest ratio has not found the crossing, it has run
    out of range. Returning the floor is correct and it is the caller's job to say so.
    """
    curve = break_even_curve(LADDER, COSTS, ratios=[5.0, 20.0], volumes=VOLUMES)

    assert crossing_ratio(curve, capacity=3) == pytest.approx(5.0)
    assert curve["ratio"].min() == pytest.approx(5.0)


def test_the_funnel_is_drawn_from_the_counts_the_stage_measured(tmp_path: Path) -> None:
    """The figure and the operating-point table have to come from one set of counts.

    A funnel drawn from anything else is a second source of truth for the two ceilings,
    and the whole point of publishing them is that they are arithmetic nobody chose.
    """
    candidates = _frame(*DAY_1_ROWS)
    outcome = evaluate(candidates, k=K, costs=COSTS)
    out = tmp_path / "figures" / "reachability_funnel.png"

    _write_ceiling_figures(
        candidates,
        {
            "exposure": {"median": 900.0, "mean": 1100.0},
            "break_even_precision": {
                "nothing_recovered": 0.08,
                "median_exposure": 0.006,
                "mean_exposure": 0.0001,
            },
            "base_rate": 0.0025,
        },
        unreachable=2,
        costs=COSTS,
        figures_dir=out.parent,
    )

    assert out.exists()
    assert (out.parent / "break_even.png").exists()
    # Three mules in the queue plus two the feature window cannot reach is the denominator.
    assert distinct_mules(candidates) == 3
    assert outcome.true_positives == 2


def _overflow_table(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """(window, scorer, policy, caught) into the shape ``overflow_verdict`` reads."""
    return pd.DataFrame(
        [
            {"window": w, "scorer": s, "policy": p, "caught": c, "precision_at_k": c / 400.0}
            for w, s, p, c in rows
        ]
    )


def test_the_overflow_verdict_counts_the_direction_rather_than_asserting_it() -> None:
    """The report used to state this in prose, and LI-Small falsified it. D37.

    One scorer improves under the alternative, one gets worse, one does not move. A verdict
    that collapses those into "rollover did not improve the operating point" is wrong on this
    table, which is exactly the shape the LI-Small run produced.
    """
    table = _overflow_table(
        [
            ("test", "logistic", "same_day", 2.0),
            ("test", "logistic", "rollover_max_3d", 3.0),
            ("test", "xgboost", "same_day", 5.0),
            ("test", "xgboost", "rollover_max_3d", 4.0),
            ("test", "rules", "same_day", 1.0),
            ("test", "rules", "rollover_max_3d", 1.0),
        ]
    )

    verdict = overflow_verdict(table, "same_day")

    assert verdict["improved"] == 1
    assert verdict["worsened"] == 1
    assert verdict["unchanged"] == 1
    assert verdict["largest_gain"] == 1.0
    assert verdict["largest_loss"] == 1.0


def test_a_verdict_where_nothing_moves_says_so() -> None:
    """The HI-Small case, which is what the hardcoded sentence was written from."""
    table = _overflow_table(
        [
            ("test", "xgboost", "same_day", 14.0),
            ("test", "xgboost", "rollover_max_3d", 14.0),
            ("val", "xgboost", "same_day", 9.0),
            ("val", "xgboost", "rollover_max_3d", 9.0),
        ]
    )

    verdict = overflow_verdict(table, "same_day")

    assert verdict["improved"] == 0
    assert verdict["worsened"] == 0
    assert verdict["unchanged"] == 2
    assert verdict["largest_gain"] == 0.0


def test_the_verdict_compares_against_the_configured_policy_and_not_a_fixed_one() -> None:
    """Configuring the other policy inverts every comparison, so the sign has to follow it."""
    table = _overflow_table(
        [
            ("test", "logistic", "same_day", 2.0),
            ("test", "logistic", "rollover_max_3d", 3.0),
        ]
    )

    assert overflow_verdict(table, "same_day")["improved"] == 1
    assert overflow_verdict(table, "rollover_max_3d")["worsened"] == 1
