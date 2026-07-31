"""Alert budget, cost matrix, sensitivity, break-even ratio, queue overflow. Implements F6.

The deliverable of this project is an operating point, and this module is where it is
computed. Three things are kept apart on purpose, because collapsing them is how an
economics section stops meaning anything:

* The **budget-constrained** threshold, which is a rank. It is set by
  ``analyst_capacity_per_day`` and by nothing else, so the three euro costs cannot move it.
* The **unconstrained** cost-minimising threshold, which is the volume a firm would choose
  if it could hire freely. The euro costs move this one.
* The **break-even ratio** between them, which is the number a reader substitutes their own
  costs into.

F6 edge case (b) says these are different optimisation problems and both get reported. The
constrained one is what the business faces.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src import charts
from src.definitions import (
    QUEUE_OVERFLOW_POLICIES,
    Costs,
    SchemaError,
    Window,
    build_windows,
    count_unscoreable_mules,
    daily_candidates,
    exposure,
    load_params,
    queue_policy_horizon,
    scoring_population,
    write_metrics,
)
from src.models import SCORERS

LOGGER = logging.getLogger(__name__)

#: The queue an alert budget is spent against. One row per account per day it was eligible.
CANDIDATE_COLUMNS: Final[tuple[str, ...]] = ("account", "day", "score", "is_mule", "exposure")


@dataclass(frozen=True)
class Outcome:
    """One budget-constrained run over a candidate queue. Every field of F6, kept apart.

    ``ev_recovered`` and ``ev_missed`` are reported as the PRD writes them, so the two
    directions of being wrong stay visible instead of netting off inside one number.
    """

    k: int
    n_days: int
    n_alerts: int
    true_positives: int
    false_positives: int
    missed: int
    precision_at_k: float
    recall: float
    threshold: float
    exposure_recovered: float
    investigation_cost: float
    freeze_cost: float
    ev_recovered: float
    ev_missed: float
    net: float

    @property
    def net_per_day(self) -> float:
        """The daily figure, which is the one a capacity decision is actually made on."""
        return self.net / self.n_days


def _validate(candidates: pd.DataFrame, k: int) -> None:
    """Refuse a queue that is missing a column or a capacity that cannot be spent."""
    missing = [column for column in CANDIDATE_COLUMNS if column not in candidates.columns]
    if missing:
        raise SchemaError(f"Candidate queue is missing columns: {missing}")
    if k < 0:
        raise SchemaError(f"Alert capacity must not be negative, got {k}.")


def select_alerts(candidates: pd.DataFrame, k: int) -> pd.Series:
    """Which rows of the queue get investigated. Boolean, aligned to ``candidates``.

    Top ``k`` by score on each day, and an account already investigated on an earlier day
    does not consume capacity again. Ties break to the lowest account id, which is
    arbitrary and declared, because the alternative is letting parquet row order decide who
    an analyst opens.

    Capacity is spent one day at a time and this function knows nothing about overflow. What
    happens to a candidate the day's capacity did not reach is decided before it gets here, by
    :func:`apply_overflow_policy`, which is the only thing that reads ``queue_overflow_policy``.
    A row present on a day is eligible on that day, whichever policy put it there.

    The loop runs once per day in the label window, which is two iterations here. It is not
    a loop over accounts, and it cannot become one: the number of iterations is set by
    ``label_window_days``.
    """
    _validate(candidates, k)

    ordered = candidates.sort_values(
        ["day", "score", "account"], ascending=[True, False, True], kind="stable"
    )
    alerted = pd.Series(False, index=candidates.index, name="alerted")

    worked: set[int] = set()
    for day in ordered["day"].drop_duplicates():
        todays = ordered.loc[ordered["day"] == day]
        picked = todays.loc[~todays["account"].isin(worked)].head(k)
        alerted.loc[picked.index] = True
        worked.update(int(account) for account in picked["account"])

    return alerted


def apply_overflow_policy(candidates: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Rewrite the queue so that F6 edge case (c) is a property of the queue, not of the pick.

    Under ``same_day`` this is the identity. Under ``rollover_max_3d`` a candidate the day's
    capacity did not reach keeps a row on each of the following days inside its horizon, and
    :func:`select_alerts` then lets it compete for that day's capacity against the accounts
    that arrived on it. Yesterday's spillover consumes today's budget, which is the effect
    the PRD says must be simulated rather than assumed away.

    Doing it here rather than inside the selection is what keeps the policy from reaching
    the cost arithmetic. Nothing downstream of the queue is aware a policy was applied.

    Three invariants hold by construction and are asserted in ``tests/test_economics.py``:

    * The set of accounts does not change, so every base rate and recall divides by the
      same denominator it did under same-day capacity.
    * The set of days does not change, because a carried row is kept only if that day
      already held candidates. ``net_per_day`` therefore divides by the same count.
    * At most ``k`` new accounts are worked per day, since capacity is still spent one day
      at a time and an account already worked is never picked again.

    A carried row's ``day`` is the day the capacity would be spent rather than the day the
    account arrived, so an alerted row reads as the day an analyst opened the case.
    """
    horizon = queue_policy_horizon(policy)
    _validate(candidates, 0)
    if horizon <= 1:
        return candidates

    days = candidates["day"].unique()
    carried = [candidates]
    for offset in range(1, horizon):
        shifted = candidates.assign(day=candidates["day"] + pd.Timedelta(days=offset))
        carried.append(shifted.loc[shifted["day"].isin(days)])

    return (
        pd.concat(carried, ignore_index=True)
        .drop_duplicates(["account", "day"])
        .sort_values(["day", "account"], kind="stable")
        .reset_index(drop=True)
    )


def distinct_mules(candidates: pd.DataFrame) -> int:
    """Mule accounts in the queue. Accounts, not rows: a mule seen twice is still one mule."""
    return int(candidates.loc[candidates["is_mule"] == 1, "account"].nunique())


def _coverage_row(candidates: pd.DataFrame, k: int, mules: int) -> dict[str, float]:
    """What a daily volume of ``k`` reaches. Counted once here, priced separately."""
    worked = candidates.loc[select_alerts(candidates, k)]
    caught_here = worked["is_mule"] == 1
    n_alerts = len(worked)
    caught = int(caught_here.sum())
    return {
        "k": float(k),
        "alerts": float(n_alerts),
        "caught": float(caught),
        "false_positives": float(n_alerts - caught),
        "missed": float(mules - caught),
        "exposure_recovered": float(worked.loc[caught_here, "exposure"].sum()),
        "threshold": float(worked["score"].min()) if n_alerts else float("nan"),
    }


def evaluate(candidates: pd.DataFrame, k: int, costs: Costs) -> Outcome:
    """Spend a daily budget of ``k`` against the queue and price both kinds of error.

    Implements the F6 expected-value formula. One run of the same counting and the same
    cost arithmetic a whole sweep uses, so a single operating point and a curve through it
    can never disagree.
    """
    n_days = int(candidates["day"].nunique())
    row = _coverage_row(candidates, k, distinct_mules(candidates))
    priced = price(pd.DataFrame([row]), costs, n_days).iloc[0]

    return Outcome(
        k=k,
        n_days=n_days,
        n_alerts=int(priced["alerts"]),
        true_positives=int(priced["caught"]),
        false_positives=int(priced["false_positives"]),
        missed=int(priced["missed"]),
        precision_at_k=float(priced["precision_at_k"]),
        recall=float(priced["recall"]),
        threshold=float(priced["threshold"]),
        exposure_recovered=float(priced["exposure_recovered"]),
        investigation_cost=float(priced["investigation_cost"]),
        freeze_cost=float(priced["freeze_cost"]),
        ev_recovered=float(priced["ev_recovered"]),
        ev_missed=float(priced["ev_missed"]),
        net=float(priced["net"]),
    )


#: Everything a volume reaches, before any cost parameter is applied to it.
COVERAGE_COLUMNS: Final[tuple[str, ...]] = (
    "k",
    "alerts",
    "caught",
    "false_positives",
    "missed",
    "exposure_recovered",
    "threshold",
)


def coverage_curve(candidates: pd.DataFrame, volumes: Sequence[int]) -> pd.DataFrame:
    """What each daily alert volume reaches. Nothing here reads a cost parameter.

    Kept apart from the pricing on purpose. The set of accounts a given volume alerts on
    does not depend on what an alert is worth, so a sweep over costs re-prices one ranking
    rather than re-ranking a population per cost. That is what makes a sweep over the whole
    population affordable, and an affordable sweep is what stops the optimum being reported
    from the edge of a grid chosen for speed.
    """
    if not volumes:
        raise SchemaError("A volume curve needs at least one candidate volume.")

    mules = distinct_mules(candidates)
    return pd.DataFrame([_coverage_row(candidates, k, mules) for k in volumes])


def price(coverage: pd.DataFrame, costs: Costs, n_days: int) -> pd.DataFrame:
    """Apply the cost matrix to a coverage curve. Vectorised across the whole curve."""
    priced = coverage.copy()
    priced["investigation_cost"] = coverage["alerts"] * costs.cost_investigation_eur
    priced["freeze_cost"] = coverage["false_positives"] * costs.cost_false_freeze_eur
    priced["ev_recovered"] = (
        coverage["exposure_recovered"] - priced["investigation_cost"] - priced["freeze_cost"]
    )
    priced["ev_missed"] = coverage["missed"] * costs.cost_missed_mule_eur
    priced["net"] = priced["ev_recovered"] - priced["ev_missed"]
    priced["net_per_day"] = priced["net"] / n_days
    priced["precision_at_k"] = (coverage["caught"] / coverage["alerts"]).fillna(0.0)
    reachable = coverage["caught"] + coverage["missed"]
    priced["recall"] = (coverage["caught"] / reachable).fillna(0.0)
    return priced


def volume_curve(candidates: pd.DataFrame, costs: Costs, volumes: Sequence[int]) -> pd.DataFrame:
    """Net expected value at each daily alert volume. The unconstrained problem, F6 (b).

    Sweeping the volume rather than the raw threshold keeps the axis in the units a
    capacity decision is made in, and it is the same axis the analyst budget sits on.
    """
    n_days = int(candidates["day"].nunique())
    return price(coverage_curve(candidates, volumes), costs, n_days)


def optimal_volume(candidates: pd.DataFrame, costs: Costs, volumes: Sequence[int]) -> int:
    """The daily volume maximising net expected value, ignoring the analyst budget.

    Ties go to the smaller volume, so a flat optimum reports the cheaper way to reach it.
    """
    return volume_optimum(volume_curve(candidates, costs, volumes))


def volume_optimum(priced: pd.DataFrame) -> int:
    """The argmax volume of an already-priced curve, so a sweep does not re-rank."""
    return int(priced["k"].iloc[int(priced["net"].argmax())])


def break_even_precision(costs: Costs, recovered_per_mule: float) -> float:
    """The precision one more alert must reach before it pays for itself.

    An extra alert costs ``cost_investigation_eur``, and a further ``cost_false_freeze_eur``
    if the account turns out clean. It returns ``recovered_per_mule`` plus the avoided
    ``cost_missed_mule_eur`` if the account turns out to be a mule. Setting the two equal
    and solving for the probability p that it is a mule:

        p (recovered + cost_missed + cost_freeze) = cost_investigation + cost_freeze

    Comparing this against the measured base rate is the whole economics argument in one
    line, and it is the form a reader can substitute their own costs into.
    """
    denominator = recovered_per_mule + costs.cost_missed_mule_eur + costs.cost_false_freeze_eur
    if denominator <= 0:
        raise SchemaError("Catching a mule must be worth something for a break-even to exist.")
    return (costs.cost_investigation_eur + costs.cost_false_freeze_eur) / denominator


def replace_cost(costs: Costs, name: str, value: float) -> Costs:
    """One numeric assumption changed, the rest held. Keeps the sweep from restating Costs.

    ``queue_overflow_policy`` is carried through untouched rather than being sweepable. It is
    categorical, so there is no value to multiply, and it is refused by name so that a sweep
    over it fails loudly instead of coercing a policy name to a float. D29.
    """
    if name == "queue_overflow_policy":
        raise SchemaError("queue_overflow_policy is categorical and has no multiplier. See D29.")
    if not hasattr(costs, name):
        raise SchemaError(f"{name!r} is not a cost parameter.")
    fields: dict[str, float] = {
        "cost_missed_mule_eur": costs.cost_missed_mule_eur,
        "cost_investigation_eur": costs.cost_investigation_eur,
        "cost_false_freeze_eur": costs.cost_false_freeze_eur,
        "analyst_capacity_per_day": costs.analyst_capacity_per_day,
    }
    fields[name] = value
    return Costs(
        cost_missed_mule_eur=fields["cost_missed_mule_eur"],
        cost_investigation_eur=fields["cost_investigation_eur"],
        cost_false_freeze_eur=fields["cost_false_freeze_eur"],
        analyst_capacity_per_day=int(fields["analyst_capacity_per_day"]),
        queue_overflow_policy=costs.queue_overflow_policy,
    )


COST_PARAMETERS: Final[tuple[str, ...]] = (
    "cost_missed_mule_eur",
    "cost_investigation_eur",
    "cost_false_freeze_eur",
    "analyst_capacity_per_day",
)


def sensitivity(
    candidates: pd.DataFrame, costs: Costs, multipliers: Sequence[float]
) -> pd.DataFrame:
    """Checkpoint 4. Each of the four assumptions moved one at a time, the rest held.

    One at a time rather than jointly, because the question the checkpoint asks is which
    single assumption the recommendation depends on.
    """
    rows: list[dict[str, float | str]] = []
    for name in COST_PARAMETERS:
        base = float(getattr(costs, name))
        for multiplier in multipliers:
            moved = replace_cost(costs, name, base * multiplier)
            outcome = evaluate(candidates, moved.analyst_capacity_per_day, moved)
            rows.append(
                {
                    "parameter": name,
                    "multiplier": multiplier,
                    "value": base * multiplier,
                    "k": float(outcome.k),
                    "alerts": float(outcome.n_alerts),
                    "caught": float(outcome.true_positives),
                    "missed": float(outcome.missed),
                    "precision_at_k": outcome.precision_at_k,
                    "recall": outcome.recall,
                    "threshold": outcome.threshold,
                    "ev_recovered": outcome.ev_recovered,
                    "ev_missed": outcome.ev_missed,
                    "net_per_day": outcome.net_per_day,
                }
            )
    return pd.DataFrame(rows)


def break_even_curve(
    candidates: pd.DataFrame,
    costs: Costs,
    ratios: Sequence[float],
    volumes: Sequence[int],
) -> pd.DataFrame:
    """Where the recommendation flips, expressed so a reader can substitute their costs.

    The ratio is the cost of missing one mule over the cost of one wrong alert:

        R = cost_missed_mule_eur / (cost_investigation_eur + cost_false_freeze_eur)

    At each R, the cost-optimal daily volume is recomputed. The number that matters is the
    R at which that optimum crosses ``analyst_capacity_per_day``. Below the crossing the
    budget does not bind, so the recommendation is a threshold. Above it the budget binds,
    and the recommendation is capacity rather than a threshold.
    """
    wrong_alert = costs.cost_investigation_eur + costs.cost_false_freeze_eur
    if wrong_alert <= 0:
        raise SchemaError("A wrong alert must cost something for the ratio to be defined.")

    n_days = int(candidates["day"].nunique())
    coverage = coverage_curve(candidates, volumes)
    at_budget = coverage_curve(candidates, [costs.analyst_capacity_per_day])
    largest = float(max(volumes))

    rows: list[dict[str, float]] = []
    for ratio in ratios:
        moved = replace_cost(costs, "cost_missed_mule_eur", ratio * wrong_alert)
        priced = price(coverage, moved, n_days)
        best = volume_optimum(priced)
        rows.append(
            {
                "ratio": ratio,
                "cost_missed_mule_eur": ratio * wrong_alert,
                "optimal_k": float(best),
                "net_at_optimum": float(priced["net"].max()),
                "net_at_budget": float(price(at_budget, moved, n_days)["net"].iloc[0]),
                "budget_binds": float(best > costs.analyst_capacity_per_day),
                "optimal_at_grid_edge": float(best == largest),
            }
        )
    return pd.DataFrame(rows)


def crossing_ratio(curve: pd.DataFrame, capacity: int) -> float:
    """The smallest swept ratio at which the cost-optimal volume exceeds capacity.

    Returns NaN when the sweep never crosses, which is a real answer and says the budget
    binds over the whole range examined, or over none of it.
    """
    above = curve.loc[curve["optimal_k"] > capacity]
    if above.empty:
        return float("nan")
    return float(above["ratio"].min())


def build_candidates(
    scores: pd.Series, labels: pd.Series, exposures: pd.Series, queue: pd.DataFrame
) -> pd.DataFrame:
    """Join a scorer's output onto the daily queue. One row per account per eligible day.

    Everything is indexed by the same pinned population (D9), so a missing score is a
    defect rather than something to fill in.
    """
    per_account = pd.DataFrame(
        {"score": scores, "is_mule": labels, "exposure": exposures}
    ).rename_axis("account")
    if per_account.isna().to_numpy().any():
        raise SchemaError("Scores, labels, and exposure must all be defined on the population.")

    merged = queue.merge(per_account, left_on="account", right_index=True, how="left")
    if merged[["score", "is_mule", "exposure"]].isna().to_numpy().any():
        raise SchemaError("The queue holds an account that is not in the scored population.")
    return merged[list(CANDIDATE_COLUMNS)]


def log_sensitivity(table: pd.DataFrame, costs: Costs) -> None:
    """Print the Checkpoint 4 table. Reporting only, it computes nothing."""
    shown = table.assign(
        value=table["value"].map(lambda v: f"{v:,.6g}"),
        precision_at_k=table["precision_at_k"].map(lambda v: f"{v:.4%}"),
        recall=table["recall"].map(lambda v: f"{v:.2%}"),
        threshold=table["threshold"].map(lambda v: f"{v:.6f}"),
        net_per_day=table["net_per_day"].map(lambda v: f"{v:,.0f}"),
    )[
        [
            "parameter",
            "multiplier",
            "value",
            "alerts",
            "caught",
            "precision_at_k",
            "recall",
            "threshold",
            "net_per_day",
        ]
    ]
    LOGGER.info("Checkpoint 4 sensitivity, %s\n%s", costs.as_footer(), shown.to_string(index=False))


def queue_for(
    txns: pd.DataFrame, window: Window, scores: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Assemble one scorer's candidate queue for one window, from the pinned population."""
    population = scoring_population(txns, window)
    return build_candidates(
        scores[column],
        scores["is_mule"],
        exposure(txns, window, population),
        daily_candidates(txns, window, population),
    )


def overflow_comparison(
    raw_queues: dict[tuple[str, str], pd.DataFrame], costs: Costs
) -> pd.DataFrame:
    """Every scorer's operating point under each overflow policy, on both windows. G3, D29.

    The configured policy's rows reproduce the published operating point exactly, because they
    are the same queue and the same arithmetic. That is the regression check: a comparison that
    cannot reproduce the figure it is set beside is measuring something else.

    Takes the raw queues rather than the ones the published path uses, so each policy is
    applied to the same starting point and neither is measured on the other's output.
    """
    rows: list[dict[str, float | str]] = []
    for (window, scorer), queue in sorted(raw_queues.items()):
        arrived = queue[["account", "day"]].assign(arrived_today=True)
        for policy in sorted(QUEUE_OVERFLOW_POLICIES):
            under = apply_overflow_policy(queue, policy)
            outcome = evaluate(under, costs.analyst_capacity_per_day, costs)
            worked = under.loc[select_alerts(under, costs.analyst_capacity_per_day)]
            tagged = worked.merge(arrived, on=["account", "day"], how="left")
            rows.append(
                {
                    "window": window,
                    "scorer": scorer,
                    "policy": policy,
                    "queue_rows": float(len(under)),
                    "largest_day": float(under.groupby("day").size().max()),
                    "alerts": float(outcome.n_alerts),
                    # Alerts spent on an account that did not arrive that day, so the whole of
                    # the policy's effect on who gets worked is one number.
                    "carried_alerts": float(tagged["arrived_today"].isna().sum()),
                    "caught": float(outcome.true_positives),
                    "precision_at_k": outcome.precision_at_k,
                    "recall": outcome.recall,
                    "threshold": outcome.threshold,
                    "exposure_recovered": outcome.exposure_recovered,
                    "net_per_day": outcome.net_per_day,
                }
            )
    return pd.DataFrame(rows)


def overflow_verdict(table: pd.DataFrame, configured: str) -> dict[str, float]:
    """Which way the alternative policy moved the operating point, counted per scorer. D37.

    The report stated this direction in prose. It was true of the file it was written on and
    false on the next one: LI-Small improves under rollover for two scorers and gets worse for
    the third. Counting it here keeps ``report.py`` to assembly and ties the claim to the run
    that produced it, which is the same rule every other number in the report already follows.

    Counted on catches rather than on net euro, because a single high-exposure account moves
    the euro figure by more than the policy does and would report the account rather than the
    policy.
    """
    configured_rows = table.loc[table["policy"] == configured].set_index(["window", "scorer"])
    alternatives = table.loc[table["policy"] != configured]
    if configured_rows.empty:
        raise SchemaError(f"No rows for the configured policy {configured!r} to compare against.")

    baseline = configured_rows["caught"].reindex(
        pd.MultiIndex.from_frame(alternatives[["window", "scorer"]])
    )
    delta = alternatives["caught"].to_numpy(dtype=float) - baseline.to_numpy(dtype=float)

    gains = delta[delta > 0]
    losses = delta[delta < 0]
    return {
        "improved": float((delta > 0).sum()),
        "worsened": float((delta < 0).sum()),
        "unchanged": float((delta == 0).sum()),
        "largest_gain": float(gains.max()) if gains.size else 0.0,
        "largest_loss": float(-losses.min()) if losses.size else 0.0,
    }


def log_overflow_comparison(table: pd.DataFrame, configured: str) -> None:
    """Print the two-policy comparison. Reporting only, it computes nothing."""
    shown = table.assign(
        precision_at_k=table["precision_at_k"].map(lambda v: f"{v:.4%}"),
        recall=table["recall"].map(lambda v: f"{v:.2%}"),
        threshold=table["threshold"].map(lambda v: f"{v:.6f}"),
        net_per_day=table["net_per_day"].map(lambda v: f"{v:,.0f}"),
        queue_rows=table["queue_rows"].map(lambda v: f"{v:,.0f}"),
        largest_day=table["largest_day"].map(lambda v: f"{v:,.0f}"),
        exposure_recovered=table["exposure_recovered"].map(lambda v: f"{v:,.0f}"),
    )
    LOGGER.info(
        "F6 (c) queue overflow, configured policy is %s\n%s",
        configured,
        shown.to_string(index=False),
    )


def _outcome_record(outcome: Outcome) -> dict[str, float]:
    """One operating point, flattened for the metrics file the report reads."""
    return {
        "k": float(outcome.k),
        "alerts": float(outcome.n_alerts),
        "caught": float(outcome.true_positives),
        "missed": float(outcome.missed),
        "precision_at_k": outcome.precision_at_k,
        "recall": outcome.recall,
        "threshold": outcome.threshold,
        "exposure_recovered": outcome.exposure_recovered,
        "ev_recovered": outcome.ev_recovered,
        "ev_missed": outcome.ev_missed,
        "net_per_day": outcome.net_per_day,
    }


def _log_operating_point(
    name: str, outcome: Outcome, unreachable: int, costs: Costs, base_rate: float
) -> None:
    """The operating point, with the reachability ceiling stated beside it rather than after."""
    LOGGER.info(
        "%-5s alerting %d accounts at random from a base rate of %.4f%% would be expected "
        "to catch %.1f mules. This scorer caught %d.",
        name,
        outcome.n_alerts,
        100.0 * base_rate,
        outcome.n_alerts * base_rate,
        outcome.true_positives,
    )
    LOGGER.info(
        "%-5s at k=%d/day over %d days: %d alerts, %d of %d reachable mules caught, "
        "precision@k %.4f%%, recall %.2f%%, threshold %.6f",
        name,
        outcome.k,
        outcome.n_days,
        outcome.n_alerts,
        outcome.true_positives,
        outcome.true_positives + outcome.missed,
        100.0 * outcome.precision_at_k,
        100.0 * outcome.recall,
        outcome.threshold,
    )
    window_mules = outcome.true_positives + outcome.missed + unreachable
    LOGGER.info(
        "%-5s recall over every mule in the label window: %d of %d = %.2f%% "
        "(%d unreachable, no feature-window inflow to score)",
        name,
        outcome.true_positives,
        window_mules,
        100.0 * outcome.true_positives / window_mules,
        unreachable,
    )
    LOGGER.info(
        "%-5s EV recovered %.0f (exposure %.0f - investigations %.0f - freezes %.0f), "
        "EV missed %.0f, net %.0f EUR over the window, %.0f EUR/day | %s",
        name,
        outcome.ev_recovered,
        outcome.exposure_recovered,
        outcome.investigation_cost,
        outcome.freeze_cost,
        outcome.ev_missed,
        outcome.net,
        outcome.net_per_day,
        costs.as_footer(),
    )


def _log_break_even(candidates: pd.DataFrame, costs: Costs) -> None:
    """The precision an alert has to reach, against the precision the population offers.

    This is the whole economics argument in two numbers. The recovered value per catch is
    measured from this window rather than assumed, and it is the term that decides the
    answer, so it is reported at the median and the mean and the two are far apart.
    """
    mules = candidates.drop_duplicates("account")
    caught_value = mules.loc[mules["is_mule"] == 1, "exposure"]
    base_rate = distinct_mules(candidates) / len(mules)

    LOGGER.info(
        "test  exposure of a mule: median %s EUR, mean %s EUR, total %s EUR over %d mules",
        f"{caught_value.median():,.0f}",
        f"{caught_value.mean():,.0f}",
        f"{caught_value.sum():,.0f}",
        len(caught_value),
    )
    for label, recovered in (
        ("nothing recovered", 0.0),
        ("median exposure", float(caught_value.median())),
        ("mean exposure", float(caught_value.mean())),
    ):
        needed = break_even_precision(costs, recovered)
        LOGGER.info(
            "test  break-even precision at %-18s (%12.0f EUR recovered per catch): "
            "%.4f%% against a base rate of %.4f%%, so a random alert %s",
            label,
            recovered,
            100.0 * needed,
            100.0 * base_rate,
            "pays for itself" if base_rate > needed else "destroys value",
        )


def _economics_metrics(
    queues: dict[tuple[str, str], pd.DataFrame], candidates: pd.DataFrame, costs: Costs
) -> dict[str, Any]:
    """Everything the report needs from this stage, measured once and handed forward.

    The exposure block lives here rather than in the report because the concentration is
    the largest single caveat on any money figure the project prints, and a caveat computed
    in the reporting layer is a caveat nobody can trace back to a module.
    """
    mules = candidates.drop_duplicates("account")
    exposures = mules.loc[mules["is_mule"] == 1, "exposure"].sort_values(ascending=False)
    total = float(exposures.sum())

    return {
        "operating_point": {
            f"{window}/{scorer}": _outcome_record(
                evaluate(queues[window, scorer], costs.analyst_capacity_per_day, costs)
            )
            for window in ("val", "test")
            for scorer in SCORERS
        },
        "exposure": {
            "total": total,
            "median": float(exposures.median()),
            "mean": float(exposures.mean()),
            "count": len(exposures),
            "top_1_share": float(exposures.iloc[0] / total),
            "top_5_share": float(exposures.head(5).sum() / total),
        },
        "break_even_precision": {
            "nothing_recovered": break_even_precision(costs, 0.0),
            "median_exposure": break_even_precision(costs, float(exposures.median())),
            "mean_exposure": break_even_precision(costs, float(exposures.mean())),
        },
        "base_rate": distinct_mules(candidates) / candidates["account"].nunique(),
    }


def _window_queues(
    txns: pd.DataFrame, window: Window, interim_dir: Path, costs: Costs
) -> tuple[dict[str, pd.DataFrame], int]:
    """One candidate queue per scorer for a window, with its operating points logged.

    The queues returned are raw, before any overflow policy is applied, because the two-policy
    comparison has to start both policies from the same queue. The operating points logged here
    are under the configured policy, which is what the published figures report.

    Returns the queues alongside the count of mules the feature window cannot reach, which
    is the reachability ceiling's numerator and is needed again when the funnel is drawn.
    """
    scores = pd.read_parquet(interim_dir / f"scores_{window.name}.parquet")
    population = scoring_population(txns, window)
    unreachable = count_unscoreable_mules(txns, window, population)
    scored = {scorer: queue_for(txns, window, scores, scorer) for scorer in SCORERS}

    first = scored[SCORERS[0]]
    LOGGER.info(
        "%-5s queue: %d account-days over %d days, from a population of %d",
        window.name,
        len(first),
        first["day"].nunique(),
        len(population),
    )
    LOGGER.info(
        "%-5s account-days per day: %s",
        window.name,
        ", ".join(
            f"{day:%m-%d} {count:,}"
            for day, count in first.groupby("day").size().sort_index().items()
        ),
    )
    base_rate = distinct_mules(first) / first["account"].nunique()
    for scorer, queue in scored.items():
        _log_operating_point(
            f"{window.name}/{scorer}",
            evaluate(
                apply_overflow_policy(queue, costs.queue_overflow_policy),
                costs.analyst_capacity_per_day,
                costs,
            ),
            unreachable,
            costs,
            base_rate=base_rate,
        )
    return scored, unreachable


def _write_ceiling_figures(
    candidates: pd.DataFrame,
    measured: dict[str, Any],
    *,
    unreachable: int,
    costs: Costs,
    figures_dir: Path,
) -> None:
    """The two figures that carry the findings rather than the detection numbers.

    Both are drawn from quantities this module has already measured. The funnel takes the
    reachable count from the queue and the unscoreable count from the population check, and
    the break-even figure takes its three thresholds and the base rate straight out of the
    metrics dictionary, so neither figure can drift from the tables printed beside it.
    """
    reachable = distinct_mules(candidates)
    in_window = reachable + unreachable
    outcome = evaluate(candidates, costs.analyst_capacity_per_day, costs)
    funnel = pd.DataFrame(
        [
            ("mules active in the label window", in_window),
            ("reachable by the feature window", reachable),
            ("alerts the budget allows", outcome.n_alerts),
            ("caught by xgboost", outcome.true_positives),
        ],
        columns=["stage", "count"],
    )
    funnel["share"] = funnel["count"] / in_window
    charts.reachability_funnel(
        funnel, window="test", costs=costs, out=figures_dir / "reachability_funnel.png"
    )
    LOGGER.info("wrote %s", figures_dir / "reachability_funnel.png")

    exposures = measured["exposure"]
    break_even_at = measured["break_even_precision"]
    thresholds = pd.DataFrame(
        [
            ("a freeze recovers nothing", break_even_at["nothing_recovered"]),
            (f"median exposure, {exposures['median']:,.0f} EUR", break_even_at["median_exposure"]),
            (f"mean exposure, {exposures['mean']:,.0f} EUR", break_even_at["mean_exposure"]),
        ],
        columns=["basis", "precision"],
    )
    charts.break_even(
        thresholds,
        base_rate=measured["base_rate"],
        window="test",
        costs=costs,
        out=figures_dir / "break_even.png",
    )
    LOGGER.info("wrote %s", figures_dir / "break_even.png")


def main() -> None:
    """Run the rules baseline through the budget and emit the Checkpoint 4 tables."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    costs = params.costs
    txns = pd.read_parquet(params.paths.interim_dir / "canonical_txns.parquet")
    windows = {
        window.name: window
        for window in build_windows(params.split, params.windows.label_window_days)
    }

    raw_queues: dict[tuple[str, str], pd.DataFrame] = {}
    unreachable_by_window: dict[str, int] = {}
    for name in ("val", "test"):
        scored, unreachable_by_window[name] = _window_queues(
            txns, windows[name], params.paths.interim_dir, costs
        )
        for scorer, queue in scored.items():
            raw_queues[name, scorer] = queue

    # Everything below this line runs under the configured policy. The comparison against the
    # other policy starts from the raw queues, so neither policy is measured on the other's
    # output. D29.
    queues = {
        key: apply_overflow_policy(queue, costs.queue_overflow_policy)
        for key, queue in raw_queues.items()
    }

    # The sweeps run on the test window, which is the clean comparison. Validation was the
    # early-stopping set for XGBoost, so its numbers carry the optimism that buys. D16.
    candidates = queues["test", "xgboost"]

    # One coverage curve per scorer, priced once, so the chart and the tables agree.
    volumes = [int(v) for v in np.unique(np.round(np.geomspace(1, 20_000, 50)).astype(int))]
    budget_curves = {
        scorer: price(
            coverage_curve(queues["test", scorer], volumes),
            costs,
            int(queues["test", scorer]["day"].nunique()),
        )
        for scorer in SCORERS
    }
    charts.alert_budget(
        budget_curves,
        capacity=costs.analyst_capacity_per_day,
        window="test",
        costs=costs,
        out=params.paths.figures_dir / "alert_budget.png",
    )
    LOGGER.info("wrote %s", params.paths.figures_dir / "alert_budget.png")

    measured = _economics_metrics(queues, candidates, costs)
    _write_ceiling_figures(
        candidates,
        measured,
        unreachable=unreachable_by_window["test"],
        costs=costs,
        figures_dir=params.paths.figures_dir,
    )

    log_sensitivity(sensitivity(candidates, costs, params.sensitivity_multipliers), costs)
    _log_break_even(candidates, costs)

    overflow = overflow_comparison(raw_queues, costs)
    log_overflow_comparison(overflow, costs.queue_overflow_policy)
    measured.update(
        overflow_comparison=overflow.to_dict(orient="records"),
        overflow_verdict=overflow_verdict(overflow, costs.queue_overflow_policy),
    )
    measured["queue_overflow_policy"] = costs.queue_overflow_policy
    # Every policy's horizon, not only the configured one, because the report describes the
    # alternative as well and a single number there would be the wrong policy's.
    measured["queue_policy_horizon_days"] = {
        policy: queue_policy_horizon(policy) for policy in sorted(QUEUE_OVERFLOW_POLICIES)
    }

    # The grid runs to the whole daily queue, so the unconstrained optimum is found rather
    # than clipped. Anything short of that reports the edge of the grid as an answer.
    ceiling = int(candidates.groupby("day").size().max())
    volumes = [int(v) for v in np.unique(np.round(np.geomspace(1, ceiling, 60)).astype(int))] + [
        costs.analyst_capacity_per_day
    ]
    volumes = sorted(set(volumes))

    best = optimal_volume(candidates, costs, volumes)
    LOGGER.info(
        "unconstrained cost-minimising volume at the configured costs: %d alerts/day "
        "against a budget of %d, over a grid running to the full daily queue of %d",
        best,
        costs.analyst_capacity_per_day,
        ceiling,
    )

    measured["sensitivity"] = sensitivity(
        candidates, costs, params.sensitivity_multipliers
    ).to_dict(orient="records")

    ratios = [float(r) for r in np.geomspace(0.001, 1000.0, 19)]
    curve = break_even_curve(candidates, costs, ratios, volumes)
    LOGGER.info(
        "break-even sweep, R = cost_missed / (cost_inv + cost_freeze), configured R = %.2f\n%s",
        costs.cost_missed_mule_eur / (costs.cost_investigation_eur + costs.cost_false_freeze_eur),
        curve.to_string(index=False),
    )
    measured["break_even_curve"] = curve.to_dict(orient="records")
    measured["optimal_volume"] = best
    measured["daily_queue_ceiling"] = ceiling

    crossing = crossing_ratio(curve, costs.analyst_capacity_per_day)
    floor = float(min(ratios))
    if np.isnan(crossing):
        LOGGER.info(
            "the budget of %d/day never binds anywhere in R = %.4g to %.4g",
            costs.analyst_capacity_per_day,
            floor,
            float(max(ratios)),
        )
    elif crossing <= floor:
        # Reporting the floor of a sweep as the crossing would be reading a number off the
        # edge of the grid, which is the same error the volume sweep was widened to avoid.
        LOGGER.info(
            "the budget of %d/day already binds at the smallest ratio swept, so the "
            "crossing is at or below R = %.4g and this sweep does not locate it",
            costs.analyst_capacity_per_day,
            floor,
        )
    else:
        LOGGER.info(
            "the budget of %d/day starts binding at R = %.4g",
            costs.analyst_capacity_per_day,
            crossing,
        )

    edge = float(curve["optimal_at_grid_edge"].mean())
    if edge > 0:
        LOGGER.info(
            "%.0f%% of the swept ratios put the optimum at the largest volume on the grid, "
            "so the unconstrained optimum is 'alert the whole queue' rather than a number "
            "this sweep found",
            100.0 * edge,
        )

    write_metrics(params.paths.reports_dir / "metrics_economics.json", measured)
    LOGGER.info("wrote %s", params.paths.reports_dir / "metrics_economics.json")


if __name__ == "__main__":
    main()
