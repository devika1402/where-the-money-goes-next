"""The rules baseline: the rule as a fraud team would write it, and the same rule ranked.

The comparison against this baseline is the single most important thing the project
reports, so the baseline gets the same test discipline as the formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.definitions import LogisticSpec, RulesBaseline, SchemaError
from src.models import (
    LABEL,
    bootstrap_draws,
    fast_average_precision,
    flag_summary,
    log1p_features,
    matched_rules_flag,
    pr_auc,
    rules_flag,
    rules_score,
    scale_pos_weight,
    summarise_draws,
)

#: The configured rule, from config/params.yaml.
RULES = RulesBaseline(in_degree_percentile=0.99, pass_through_ratio_min=0.8)

#: A loosened rule for the small frames below, so the percentile cut has something to bite.
LOOSE = RulesBaseline(in_degree_percentile=0.5, pass_through_ratio_min=1.0)


def _features(in_degree: list[float], ratio: list[float]) -> pd.DataFrame:
    """A feature matrix carrying only the two columns the rules baseline reads."""
    return pd.DataFrame(
        {"in_degree": in_degree, "pass_through_ratio": ratio},
        index=pd.Index(range(len(in_degree)), name="account"),
    )


def test_the_rule_needs_both_conditions_not_either() -> None:
    """A conjunction. High in-degree alone is a busy account, not a mule."""
    features = _features(
        in_degree=[1.0, 100.0, 1.0, 100.0],
        ratio=[0.1, 0.1, 5.0, 5.0],
    )

    flagged = rules_flag(features, LOOSE)

    assert list(flagged) == [False, False, False, True]


def test_an_undefined_pass_through_ratio_is_never_flagged() -> None:
    """F2 returns NaN below the inflow floor. NaN is not evidence of passing money through."""
    features = _features(in_degree=[1.0, 100.0], ratio=[0.1, np.nan])

    flagged = rules_flag(features, LOOSE)

    assert list(flagged) == [False, False]


def test_the_in_degree_cut_is_a_percentile_of_the_population_not_a_constant() -> None:
    """The same account is flagged in one population and not in another. That is the point.

    A percentile cut moves with the population it is computed over, so the rule cannot be
    carried between windows as a hardcoded number.
    """
    busy = _features(in_degree=[10.0, 20.0, 30.0, 40.0], ratio=[5.0, 5.0, 5.0, 5.0])
    busier = _features(in_degree=[30.0, 500.0, 600.0, 700.0], ratio=[5.0, 5.0, 5.0, 5.0])

    # An in-degree of 30 clears the median of the first population and not of the second.
    assert bool(rules_flag(busy, LOOSE).iloc[2]) is True
    assert bool(rules_flag(busier, LOOSE).iloc[0]) is False


def test_the_score_thresholded_at_the_same_level_recovers_the_rule() -> None:
    """Why the score is allowed to be called the rules baseline rather than a new model.

    With both cuts at the same percentile q, the set scoring above q is exactly the set the
    conjunction flags. The score is the rule as a ranking, not a different rule.
    """
    values = [float(v) for v in range(1, 11)]
    features = pd.DataFrame(
        {"in_degree": values, "pass_through_ratio": values},
        index=pd.Index(range(10), name="account"),
    )
    q = 0.7
    rules = RulesBaseline(
        in_degree_percentile=q,
        pass_through_ratio_min=float(pd.Series(values).quantile(q)),
    )

    flagged = rules_flag(features, rules)
    above = rules_score(features, rules) > q

    assert list(flagged) == list(above)
    assert int(flagged.sum()) == 3


def test_the_score_is_defined_everywhere_so_a_budget_can_always_be_filled() -> None:
    """The rule flags fewer accounts than the two-day budget, so the tail must be ordered."""
    features = _features(in_degree=[1.0, 2.0, 3.0], ratio=[np.nan, 0.1, 900.0])

    scored = rules_score(features, RULES)

    assert not scored.isna().any()


def test_an_undefined_ratio_scores_below_a_defined_one() -> None:
    """An account that cannot be shown to pass money through ranks below one that can."""
    features = _features(in_degree=[5.0, 5.0], ratio=[np.nan, 0.001])

    scored = rules_score(features, RULES)

    assert scored.iloc[0] < scored.iloc[1]


def test_the_score_rises_with_both_quantities() -> None:
    """Monotone in each input, which is what makes it the same rule at every cut."""
    features = _features(
        in_degree=[1.0, 2.0, 3.0, 4.0],
        ratio=[1.0, 2.0, 3.0, 4.0],
    )

    scored = rules_score(features, RULES)

    assert scored.is_monotonic_increasing


def test_a_missing_feature_column_stops_the_build() -> None:
    """The baseline reads two columns by name. A rename should not silently score zeros."""
    features = _features(in_degree=[1.0], ratio=[1.0]).rename(
        columns={"pass_through_ratio": "ratio"}
    )

    with pytest.raises(KeyError):
        rules_flag(features, RULES)


# --------------------------------------------------------------------------------------
# Logistic regression and XGBoost. The two models see the same feature matrix and treat it
# very differently, which is the reason both are reported rather than one.
# --------------------------------------------------------------------------------------

LOGISTIC = LogisticSpec(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    log1p_features=("total_inflow",),
)


def test_log1p_compresses_only_the_declared_columns() -> None:
    """A declared list, so a feature cannot quietly acquire a transform later."""
    frame = pd.DataFrame(
        {"total_inflow": [0.0, np.e - 1.0], "reciprocity": [0.0, np.e - 1.0]},
        index=pd.Index([1, 2], name="account"),
    )

    transformed = log1p_features(frame, LOGISTIC.log1p_features)

    assert list(transformed["total_inflow"]) == pytest.approx([0.0, 1.0])
    assert list(transformed["reciprocity"]) == pytest.approx([0.0, np.e - 1.0])


def test_log1p_leaves_a_null_null_rather_than_inventing_a_zero() -> None:
    """Imputation is a separate, declared step. Silently filling here would hide it."""
    frame = pd.DataFrame({"total_inflow": [np.nan, 0.0]}, index=pd.Index([1, 2], name="account"))

    transformed = log1p_features(frame, LOGISTIC.log1p_features)

    assert pd.isna(transformed["total_inflow"].iloc[0])


def test_log1p_on_a_column_that_does_not_exist_stops_the_build() -> None:
    """A renamed feature must not silently drop out of the declared transform list."""
    frame = pd.DataFrame({"total_inflow": [1.0]}, index=pd.Index([1], name="account"))

    with pytest.raises(SchemaError, match="not in the feature matrix"):
        log1p_features(frame, ("total_inflow", "a_feature_that_was_renamed"))


def test_log1p_refuses_a_negative_column_rather_than_returning_minus_infinity() -> None:
    """degree_asymmetry reaches -0.99. log1p of anything at -1 or below is not a number."""
    frame = pd.DataFrame({"degree_asymmetry": [-0.5, 1.0]}, index=pd.Index([1, 2], name="account"))

    with pytest.raises(SchemaError, match="negative"):
        log1p_features(frame, ("degree_asymmetry",))


def test_scale_pos_weight_is_measured_from_the_labels_not_assumed() -> None:
    """A knob is not its outcome. 8 clean against 2 mules is a weight of 4."""
    labels = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=pd.RangeIndex(10))

    assert scale_pos_weight(labels) == pytest.approx(4.0)


def test_scale_pos_weight_refuses_a_window_with_no_positives() -> None:
    """Dividing by zero here would train a model against nothing and report a number."""
    with pytest.raises(SchemaError, match="no mule"):
        scale_pos_weight(pd.Series([0, 0, 0], index=pd.RangeIndex(3)))


def test_pr_auc_of_a_perfect_ranking_is_one() -> None:
    """Every mule above every clean account."""
    labels = pd.Series([0, 0, 1, 1], index=pd.RangeIndex(4))
    scores = pd.Series([0.1, 0.2, 0.3, 0.4], index=pd.RangeIndex(4))

    assert pr_auc(labels, scores) == pytest.approx(1.0)


def test_pr_auc_of_an_interleaved_ranking_is_the_average_of_its_precisions() -> None:
    """Mules at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333, computed by hand and asserted."""
    labels = pd.Series([1, 0, 1, 0], index=pd.RangeIndex(4))
    scores = pd.Series([0.4, 0.3, 0.2, 0.1], index=pd.RangeIndex(4))

    assert pr_auc(labels, scores) == pytest.approx(0.8333, abs=0.0001)


def test_pr_auc_of_a_scorer_that_learned_nothing_lands_near_the_base_rate() -> None:
    """The floor any model has to beat. One mule in five is a base rate of 0.2."""
    labels = pd.Series([0, 1, 0, 0, 0], index=pd.RangeIndex(5))
    flat = pd.Series([0.5] * 5, index=pd.RangeIndex(5))

    assert pr_auc(labels, flat) == pytest.approx(0.2)


# --------------------------------------------------------------------------------------
# Bootstrap intervals. 452 test positives is few enough that a point estimate on its own
# supports almost any story, so the comparison the project reports needs an interval.
# --------------------------------------------------------------------------------------


def test_the_fast_average_precision_matches_sklearn_on_continuous_scores() -> None:
    """It runs a thousand times per window, so it is written out. It must still agree."""
    rng = np.random.default_rng(0)
    labels = pd.Series(rng.integers(0, 2, 500), index=pd.RangeIndex(500))
    scores = pd.Series(rng.random(500), index=pd.RangeIndex(500))

    assert fast_average_precision(
        labels.to_numpy().astype(float), scores.to_numpy()
    ) == pytest.approx(pr_auc(labels, scores), abs=1e-12)


def test_the_fast_average_precision_collapses_ties_the_way_sklearn_does() -> None:
    """The bug this test exists for. A percentile-rank score is massively tied.

    Ranking tied accounts as though ordered credits the scorer with an ordering it never
    produced. On the real rules baseline that read 0.002451 against a true 0.002387.
    """
    labels = pd.Series([1, 0, 0, 1, 0, 0, 1, 0], index=pd.RangeIndex(8))
    tied = pd.Series([0.5] * 4 + [0.2] * 4, index=pd.RangeIndex(8))

    assert fast_average_precision(
        labels.to_numpy().astype(float), tied.to_numpy()
    ) == pytest.approx(pr_auc(labels, tied), abs=1e-12)


def test_a_scorer_compared_against_itself_has_no_difference_in_any_resample() -> None:
    """The paired bootstrap resamples the same accounts for both sides, so this is exact."""
    rng = np.random.default_rng(1)
    labels = pd.Series(rng.integers(0, 2, 300), index=pd.RangeIndex(300))
    scores = pd.DataFrame({"a": rng.random(300)}, index=pd.RangeIndex(300)).assign(
        b=lambda f: f["a"]
    )

    draws = bootstrap_draws(labels, scores, resamples=25, seed=7)

    assert ((draws["a"] - draws["b"]).abs() < 1e-12).all()


def test_the_bootstrap_interval_brackets_the_point_estimate() -> None:
    """A 95% percentile interval that excluded its own point estimate would be a defect."""
    rng = np.random.default_rng(2)
    labels = pd.Series(rng.integers(0, 2, 400), index=pd.RangeIndex(400))
    scores = pd.DataFrame({"only": rng.random(400)}, index=pd.RangeIndex(400))

    table = summarise_draws(bootstrap_draws(labels, scores, resamples=200, seed=3), labels, scores)
    row = table.iloc[0]

    assert row["low"] <= row["pr_auc"] <= row["high"]


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    """Two runs of the same seed must give the same interval, or nothing is citable."""
    rng = np.random.default_rng(4)
    labels = pd.Series(rng.integers(0, 2, 200), index=pd.RangeIndex(200))
    scores = pd.DataFrame({"only": rng.random(200)}, index=pd.RangeIndex(200))

    first = bootstrap_draws(labels, scores, resamples=50, seed=11)
    second = bootstrap_draws(labels, scores, resamples=50, seed=11)

    assert first.equals(second)


def test_the_ranking_never_reads_the_configured_ratio_cut() -> None:
    """D27. The published PR-AUC and the budget threshold come from this score.

    If the score moved with ``pass_through_ratio_min`` then the absolute 0.8 cut would be
    inside every reported comparison. It does not, and this pins that, because the claim
    that the headline comparison was unaffected by a misspecified cut rests on it.
    """
    features = _features(
        in_degree=[1.0, 2.0, 3.0, 4.0, 5.0],
        ratio=[0.1, 0.5, 0.9, 4000.0, 8_000_000.0],
    )

    at_configured = rules_score(features, RULES)
    at_absurd = rules_score(
        features, RulesBaseline(in_degree_percentile=0.01, pass_through_ratio_min=1e9)
    )

    pd.testing.assert_series_equal(at_configured, at_absurd)


def test_the_matched_flag_cuts_both_conditions_by_rank() -> None:
    """Both at the 50th percentile of five values keeps the accounts above both medians."""
    features = _features(
        in_degree=[1.0, 2.0, 3.0, 4.0, 5.0],
        ratio=[5.0, 4.0, 3.0, 2.0, 1.0],
    )

    flagged = matched_rules_flag(features, 0.5)

    assert list(flagged) == [False, False, False, False, False]


def test_the_matched_flag_is_stricter_than_a_cut_below_the_percentile() -> None:
    """The configured 0.8 sits below the ratio's 99th percentile, so it flags more."""
    rng = np.random.default_rng(0)
    features = _features(
        in_degree=list(rng.integers(1, 50, size=1000).astype(float)),
        ratio=list(rng.gamma(0.2, 500.0, size=1000)),
    )

    configured = rules_flag(features, RULES)
    matched = matched_rules_flag(features, RULES.in_degree_percentile)

    assert features["pass_through_ratio"].quantile(0.99) > RULES.pass_through_ratio_min
    assert int(matched.sum()) <= int(configured.sum())


def test_the_matched_flag_at_q_is_the_score_thresholded_at_q() -> None:
    """The two forms of the baseline describe one rule once both cuts are ranks."""
    features = _features(
        in_degree=[float(v) for v in range(1, 11)],
        ratio=[float(v) for v in range(1, 11)],
    )
    q = 0.7

    flagged = matched_rules_flag(features, q)
    above = rules_score(features, RULES) > q

    assert list(flagged) == list(above)


def test_an_undefined_ratio_is_never_flagged_by_the_matched_form() -> None:
    """F2 returns NaN below the inflow floor, and NaN > cut is False in both flag forms."""
    features = _features(in_degree=[1.0, 100.0], ratio=[float("nan"), float("nan")])

    assert int(matched_rules_flag(features, 0.5).sum()) == 0


def test_flag_summary_reports_zero_precision_rather_than_dividing_by_zero() -> None:
    """A rule that flags nothing has no precision, and the summary must not raise."""
    features = _features(in_degree=[1.0, 2.0], ratio=[0.1, 0.2])
    features[LABEL] = [0, 1]

    summary = flag_summary(features, pd.Series([False, False], index=features.index))

    assert summary["flagged"] == 0
    assert summary["precision"] == 0.0
    assert summary["mules"] == 1
