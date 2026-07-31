"""The analyst brief generator, reduced form. G9, D36.

These pin the three judgement calls the brief makes: a feature is called out only above the
percentile floor, an account is briefed once however many days it was eligible, and a null
feature is never described. None of this is a claim about the model's reasoning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.brief import (
    next_step,
    population_percentiles,
    rank_alerts,
    standout_phrases,
)


def test_only_features_above_the_floor_are_called_out() -> None:
    """A feature at the 96th percentile is a standout at a 0.95 floor, one at the 40th is not."""
    values = pd.Series({"in_degree": 47.0, "out_degree": 3.0, "pass_through_ratio": 0.94})
    percentiles = pd.Series({"in_degree": 0.998, "out_degree": 0.40, "pass_through_ratio": 0.97})

    phrases = standout_phrases(values, percentiles, threshold=0.95)

    assert any("47 different accounts" in p for p in phrases)
    assert any("94% of the money" in p for p in phrases)
    assert not any("out to 3" in p for p in phrases)


def test_a_null_feature_is_never_a_standout() -> None:
    """pass_through_ratio is undefined for some accounts, and a null value describes nothing."""
    values = pd.Series({"in_degree": 5.0, "pass_through_ratio": np.nan})
    percentiles = pd.Series({"in_degree": 0.10, "pass_through_ratio": np.nan})

    assert standout_phrases(values, percentiles, threshold=0.95) == ()


def test_the_standout_list_is_capped() -> None:
    """A brief lists the few most unusual features, not every one above the floor."""
    values = pd.Series(
        {
            "in_degree": 9.0,
            "unique_counterparties_in": 9.0,
            "pass_through_ratio": 0.9,
            "total_inflow": 9.0,
            "out_degree": 9.0,
        }
    )
    percentiles = pd.Series(
        {
            "in_degree": 0.99,
            "unique_counterparties_in": 0.99,
            "pass_through_ratio": 0.99,
            "total_inflow": 0.99,
            "out_degree": 0.99,
        }
    )

    assert len(standout_phrases(values, percentiles, threshold=0.95)) == 3


def test_an_account_eligible_on_two_days_is_briefed_once() -> None:
    """The queue has a row per account per eligible day. The brief is per account."""
    alerted = pd.DataFrame(
        {
            "account": [7, 7, 3],
            "day": [
                pd.Timestamp("2022-09-09"),
                pd.Timestamp("2022-09-10"),
                pd.Timestamp("2022-09-09"),
            ],
            "score": [0.9, 0.9, 0.8],
            "is_mule": [True, True, False],
            "exposure": [100.0, 100.0, 50.0],
        }
    )

    ranked = rank_alerts(alerted)

    assert list(ranked["account"]) == [7, 3]
    assert list(ranked["rank"]) == [1, 2]


def test_ties_break_to_the_lowest_account_id() -> None:
    """Two accounts on the same score rank in id order, the order the queue spends capacity in."""
    alerted = pd.DataFrame(
        {
            "account": [9, 2],
            "day": [pd.Timestamp("2022-09-09"), pd.Timestamp("2022-09-09")],
            "score": [0.5, 0.5],
            "is_mule": [False, False],
            "exposure": [1.0, 1.0],
        }
    )

    ranked = rank_alerts(alerted)

    assert list(ranked["account"]) == [2, 9]


def test_the_percentile_of_a_null_stays_null() -> None:
    """So a feature the account has no value for cannot be ranked into a standout."""
    features = pd.DataFrame(
        {"pass_through_ratio": [0.1, 0.9, np.nan], "in_degree": [1.0, 2.0, 3.0]},
        index=pd.Index([1, 2, 3], name="account"),
    )

    pct = population_percentiles(features, ("pass_through_ratio", "in_degree"))

    assert bool(pct["pass_through_ratio"].isna()[3])
    assert float(pct["in_degree"][3]) == 1.0


def test_the_next_step_follows_the_strongest_signal() -> None:
    """A pass-through account and a collection account get different suggested actions."""
    assert "onward transfers" in next_step("pass_through_ratio")
    assert "inbound senders" in next_step("in_degree")
    assert next_step(None)  # a non-empty generic action when nothing stands out
