"""The analyst brief generator, reduced form. Section 11 Step 11, G9, D36.

The specified brief was one per flagged component, carrying the matched archetype. There are
no matched archetypes on this data and the component is not the right unit, both settled in
D36, so the specified form went with G8. What survives is a brief per alerted account carrying
its exposure, its rank in the day's queue, and the features that stand out about it in words.

Without SHAP the standout is descriptive, not causal. A feature is called out when the account
sits in the top few per cent of the scored population on it, so a brief lists what is unusual
about the account rather than what the model weighted. The brief says so, because claiming these
are the model's reasoning would be an attribution the project does not compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.definitions import (
    BriefSpec,
    SchemaError,
    Window,
    build_windows,
    daily_candidates,
    exposure,
    load_params,
    scoring_population,
    write_metrics,
)
from src.economics import apply_overflow_policy, build_candidates, select_alerts

LOGGER = logging.getLogger(__name__)

#: The interpretable features, with a plain-language template and how the value is formatted.
#: Every entry reads high as notable, so a top-percentile value is a suspicious value. Direction
#: matters: a feature where low is the signal would be misread by a top-percentile rule, so those
#: are left out rather than described the wrong way round.
STANDOUTS: tuple[tuple[str, str, str], ...] = (
    ("in_degree", "received payments from {v:,.0f} different accounts", "count"),
    ("unique_counterparties_in", "took money from {v:,.0f} distinct senders", "count"),
    ("pass_through_ratio", "sent on {v:.0%} of the money that came in", "ratio"),
    ("total_inflow", "took in {v:,.0f} EUR of inflow over the history window", "euro"),
    ("out_degree", "paid out to {v:,.0f} different accounts", "count"),
    (
        "sender_diversity",
        "its senders were unusually varied, {v:.2f} distinct per payment",
        "ratio",
    ),
    ("inflow_concentration", "one sender was {v:.0%} of everything it received", "ratio"),
    ("burstiness", "its activity was concentrated into a short burst", "ratio"),
)


@dataclass(frozen=True)
class Brief:
    """One alerted account, everything a brief says about it."""

    account: int
    rank: int
    alerts: int
    score: float
    exposure: float
    is_mule: bool
    standouts: tuple[str, ...]
    next_step: str


def population_percentiles(features: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Each account's percentile within the scored population, per feature.

    Nulls stay null, so a feature the account does not have a value for is never called a
    standout. The rank is over the whole population, which is what makes a top-percentile value
    mean unusual for this window rather than unusual in the abstract.
    """
    present = [c for c in columns if c in features.columns]
    return features[present].rank(pct=True)


def standout_phrases(
    values: pd.Series, percentiles: pd.Series, threshold: float, limit: int = 3
) -> tuple[str, ...]:
    """The plain-language lines for the features this account sits highest on, above the floor."""
    ranked = percentiles.dropna().sort_values(ascending=False)
    phrases: list[str] = []
    for feature, _, _ in STANDOUTS:
        if feature not in ranked.index or ranked[feature] < threshold:
            continue
        template = next(t for f, t, _ in STANDOUTS if f == feature)
        phrases.append(template.format(v=values[feature]))
        if len(phrases) == limit:
            break
    return tuple(phrases)


def next_step(top_feature: str | None) -> str:
    """A suggested next action, keyed on the account's strongest standout."""
    if top_feature == "pass_through_ratio":
        return "Review the onward transfers, and freeze pending source-of-funds checks."
    if top_feature in {"in_degree", "unique_counterparties_in", "sender_diversity"}:
        return "Review the spread of inbound senders for a collection pattern."
    return "Review this account's inbound sources and onward transfers."


def rank_alerts(alerted: pd.DataFrame) -> pd.DataFrame:
    """One row per alerted account, ranked by score with the same tie-break the queue uses.

    An account is worked once however many days it was eligible, so the brief is per account.
    The rank orders those accounts by score, breaking ties to the lowest account id, which is
    the order :func:`select_alerts` already spends capacity in.
    """
    per_account = (
        alerted.sort_values(["score", "account"], ascending=[False, True], kind="stable")
        .drop_duplicates("account")
        .reset_index(drop=True)
    )
    per_account["rank"] = per_account.index + 1
    return per_account


def build_briefs(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    exposures: pd.Series,
    queue: pd.DataFrame,
    *,
    spec: BriefSpec,
    capacity: int,
    policy: str,
) -> list[Brief]:
    """Every alerted account as a brief, highest rank first, using the configured scorer."""
    if spec.scorer not in scores.columns:
        raise SchemaError(
            f"Brief scorer {spec.scorer!r} is not a scored column: {list(scores.columns)}."
        )

    candidates = apply_overflow_policy(
        build_candidates(scores[spec.scorer], scores["is_mule"], exposures, queue), policy
    )
    alerted = candidates.loc[select_alerts(candidates, capacity)]
    ranked = rank_alerts(alerted)

    columns = tuple(f for f, _, _ in STANDOUTS)
    percentiles = population_percentiles(features, columns)

    accounts = [int(a) for a in ranked["account"]]
    ranks = [int(r) for r in ranked["rank"]]
    account_scores = [float(s) for s in ranked["score"]]
    account_exposure = [float(e) for e in ranked["exposure"]]
    account_mule = [bool(m) for m in ranked["is_mule"]]

    briefs: list[Brief] = []
    for i, account in enumerate(accounts):
        pct = cast("pd.Series", percentiles.loc[account])
        values = cast("pd.Series", features.loc[account])
        phrases = standout_phrases(values, pct, spec.standout_percentile)
        top = pct.dropna().sort_values(ascending=False)
        top_feature = (
            str(top.index[0])
            if not top.empty and float(top.iloc[0]) >= spec.standout_percentile
            else None
        )
        briefs.append(
            Brief(
                account=account,
                rank=ranks[i],
                alerts=len(accounts),
                score=account_scores[i],
                exposure=account_exposure[i],
                is_mule=account_mule[i],
                standouts=phrases,
                next_step=next_step(top_feature),
            )
        )
    return briefs


def render(brief: Brief, scorer: str) -> str:
    """One brief as plain markdown, readable by someone who has not seen the code."""
    lines = [
        f"# Account {brief.account}",
        "",
        f"Alert {brief.rank} of {brief.alerts}, ranked by the {scorer} score.",
        "",
        f"- Score: {brief.score:.6f}",
        f"- Exposure at stake this window: {brief.exposure:,.0f} EUR",
        "",
        "## Standout features of this account",
        "",
        "These are the features on which the account is in the top few per cent of the scored",
        "population. They describe the account, not the model's reasoning. The project does not",
        "attribute the model's reasoning per account.",
        "",
    ]
    if brief.standouts:
        lines += [f"- It {phrase}." for phrase in brief.standouts]
    else:
        lines.append("- Nothing above the standout threshold. It was alerted on the scorer alone.")
    lines += ["", f"**Suggested next step.** {brief.next_step}", ""]
    return "\n".join(lines)


def main() -> None:
    """Write the sample briefs and a one-line summary of the alerted set."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    interim, reports = params.paths.interim_dir, params.paths.reports_dir
    txns = pd.read_parquet(interim / "canonical_txns.parquet")
    scores = pd.read_parquet(interim / "scores_test.parquet")
    features = pd.read_parquet(interim / "features_test.parquet")

    windows = build_windows(params.split, params.windows.label_window_days)
    test_window: Window = windows[-1]
    population = scoring_population(txns, test_window)
    exposures = exposure(txns, test_window, population)
    queue = daily_candidates(txns, test_window, population)

    briefs = build_briefs(
        scores,
        features,
        exposures,
        queue,
        spec=params.brief,
        capacity=params.costs.analyst_capacity_per_day,
        policy=params.costs.queue_overflow_policy,
    )
    sample = sorted(briefs, key=lambda b: b.exposure, reverse=True)[: params.brief.sample]

    briefs_dir = reports / "briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for brief in sample:
        path = briefs_dir / f"account_{brief.account}.md"
        path.write_text(render(brief, params.brief.scorer), encoding="utf-8")
        written.append(path.name)
        LOGGER.info(
            "brief %s: rank %d/%d, exposure %s EUR, %d standouts",
            brief.account,
            brief.rank,
            brief.alerts,
            f"{brief.exposure:,.0f}",
            len(brief.standouts),
        )

    index = [
        "# Sample analyst briefs",
        "",
        f"Scorer: {params.brief.scorer}. {len(briefs)} accounts",
        "",
    ]
    index += [f"- [{name}]({name})" for name in written]
    (briefs_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    write_metrics(
        reports / "metrics_brief.json",
        {
            "scorer": params.brief.scorer,
            "alerts": len(briefs),
            "sample": len(sample),
            "mules_in_sample": int(sum(b.is_mule for b in sample)),
            "standout_percentile": params.brief.standout_percentile,
        },
    )


if __name__ == "__main__":
    main()
