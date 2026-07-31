"""The pass-through window sweep. Section 14, reported beside the published split. D40.

The published feature uses a 24 hour inflow sub-window because a 72 hour one swung the
feature's null rate across splits at Checkpoint 1 (D10). That decision settled the null rate,
not the operating point, and the operating point did not exist yet. This module varies
``pass_through_window_hours`` and reports what the operating point does, which is what section
14 asks for and what the earlier sweep could not answer.

It rebuilds the feature matrices in memory for each value, refits both models, scores the test
window, and reads off the operating point. It writes only ``reports/metrics_sweep.json`` and
touches no published parquet or figure, the same discipline as the G1 history scan. Only two of
the twenty features move with the window, ``pass_through_ratio`` and ``median_hours_to_outflow``,
so the sweep is a clean test of what those two are worth to the ranking.

The 24 hour row must reproduce the published operating point exactly. A drift there is a defect
in the sweep rather than a finding, and :func:`assert_reproduces_published` enforces it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.definitions import (
    Params,
    SchemaError,
    Window,
    build_windows,
    daily_candidates,
    exposure,
    label_accounts,
    load_params,
    scoring_population,
    write_metrics,
)
from src.economics import apply_overflow_policy, build_candidates, evaluate
from src.features import assert_no_unexpected_nulls, build_features, ratio_windows
from src.models import (
    SCORERS,
    fit_logistic,
    fit_xgboost,
    logistic_score,
    rules_score,
    xgboost_score,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OperatingPoint:
    """One scorer's budget-constrained outcome at one window length."""

    hours: int
    scorer: str
    inflow_hours: int
    caught: int
    alerts: int
    precision_at_k: float
    threshold: float
    net_per_day: float
    trees_kept: int


def feasible_hours(window: Window, requested: tuple[int, ...]) -> tuple[int, ...]:
    """The requested values that leave an inflow window inside the feature window.

    ``ratio_windows`` raises when the extension is at least the feature window, so a value like
    120 on a 96 hour window has no inflow sub-window and cannot be swept. The infeasible values
    are dropped here rather than allowed to raise mid-run, and the caller reports which and why.
    """
    kept = []
    for hours in requested:
        try:
            ratio_windows(window, hours)
        except SchemaError:
            continue
        kept.append(hours)
    return tuple(kept)


def _inflow_hours(window: Window, hours: int) -> int:
    """The length of the inflow sub-window that ``hours`` of extension leaves, in hours."""
    inflow_start, inflow_end, _ = ratio_windows(window, hours)
    return int((inflow_end - inflow_start) / pd.Timedelta(hours=1))


def _feature_window_hours(window: Window) -> int:
    """The length of the feature window in hours, for reporting what 120 exceeded."""
    return int((window.feature_end - window.feature_start) / pd.Timedelta(hours=1))


def operating_points_at(
    txns: pd.DataFrame, windows: tuple[Window, ...], hours: int, params: Params
) -> list[OperatingPoint]:
    """Fit and score at one window length, and read off every scorer's operating point.

    Features are rebuilt for all three windows at ``hours``, both models are refit on the
    training window, and the test queue is evaluated at the configured capacity under the
    configured overflow policy. Nothing is written, so the published run is never in this path.
    """
    matrices = {}
    for window in windows:
        features = build_features(txns, window, params.features, hours)
        assert_no_unexpected_nulls(features)
        labels = label_accounts(
            txns, window.label_start, window.label_end, population=scoring_population(txns, window)
        )
        matrices[window.name] = features.join(labels)

    pipeline = fit_logistic(matrices["train"], params.logistic, params.seed)
    classifier = fit_xgboost(matrices["train"], matrices["val"], params.model, params.seed)
    trees_kept = int(classifier.best_iteration) + 1

    test_window = windows[-1]
    test = matrices[test_window.name]
    population = scoring_population(txns, test_window)
    scores = {
        "rules": rules_score(test, params.rules_baseline),
        "logistic": logistic_score(pipeline, test, params.logistic),
        "xgboost": xgboost_score(classifier, test),
    }
    exposures = exposure(txns, test_window, population)
    queue = daily_candidates(txns, test_window, population)
    test_labels = test["is_mule"]

    points = []
    for scorer in SCORERS:
        candidates = apply_overflow_policy(
            build_candidates(scores[scorer], test_labels, exposures, queue),
            params.costs.queue_overflow_policy,
        )
        outcome = evaluate(candidates, params.costs.analyst_capacity_per_day, params.costs)
        points.append(
            OperatingPoint(
                hours=hours,
                scorer=scorer,
                inflow_hours=_inflow_hours(test_window, hours),
                caught=outcome.true_positives,
                alerts=outcome.n_alerts,
                precision_at_k=outcome.precision_at_k,
                threshold=outcome.threshold,
                net_per_day=outcome.net_per_day,
                trees_kept=trees_kept,
            )
        )
    return points


def read_published_operating_point(reports_dir: Path) -> dict[str, dict[str, Any]] | None:
    """The published economics operating point, or None if the pipeline has not been run."""
    path = reports_dir / "metrics_economics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = payload["operating_point"]
    return result


def assert_reproduces_published(
    points: list[OperatingPoint], published: dict[str, dict[str, Any]]
) -> None:
    """The configured-value row must match the published operating point for every scorer.

    The sweep refits from the same features with the same seed, so the configured window's
    numbers have to come out identical to what the economics stage wrote. A mismatch means the
    sweep is measuring something the published run did not, and it stops here.
    """
    for point in points:
        want = published[f"test/{point.scorer}"]
        got_thr, want_thr = round(point.threshold, 9), round(float(want["threshold"]), 9)
        if point.caught != int(want["caught"]) or got_thr != want_thr:
            raise SchemaError(
                f"Sweep does not reproduce the published {point.scorer} operating point: "
                f"caught {point.caught} vs {int(want['caught'])}, "
                f"threshold {got_thr} vs {want_thr}."
            )


def main() -> None:
    """Run the sweep on the test window and write the table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    txns = pd.read_parquet(params.paths.interim_dir / "canonical_txns.parquet")
    windows = tuple(build_windows(params.split, params.windows.label_window_days))
    test_window = windows[-1]

    requested = params.experiments.pass_through_sweep_hours
    feasible = feasible_hours(test_window, requested)
    dropped = [h for h in requested if h not in feasible]
    if dropped:
        LOGGER.info(
            "infeasible on a %dh feature window and left out: %s",
            _feature_window_hours(test_window),
            dropped,
        )

    configured = params.windows.pass_through_window_hours
    if configured not in feasible:
        raise SchemaError(
            f"The configured pass_through_window_hours {configured} is not in the swept set "
            f"{feasible}, so the sweep cannot check itself against the published run."
        )

    published = read_published_operating_point(params.paths.reports_dir)
    rows: list[OperatingPoint] = []
    for hours in feasible:
        points = operating_points_at(txns, windows, hours, params)
        if hours == configured and published is not None:
            assert_reproduces_published(points, published)
            LOGGER.info("%dh reproduces the published operating point", hours)
        for point in points:
            LOGGER.info(
                "%3dh (inflow %3dh) %-8s caught %2d  precision@k %.4f%%  threshold %.6f  "
                "net %s  trees %d",
                point.hours,
                point.inflow_hours,
                point.scorer,
                point.caught,
                100.0 * point.precision_at_k,
                point.threshold,
                f"{point.net_per_day:+,.0f}",
                point.trees_kept,
            )
        rows.extend(points)

    write_metrics(
        params.paths.reports_dir / "metrics_sweep.json",
        {
            "requested_hours": list(requested),
            "feasible_hours": list(feasible),
            "infeasible_hours": dropped,
            "configured_hours": configured,
            "feature_window_hours": _feature_window_hours(test_window),
            "rows": [asdict(row) for row in rows],
        },
    )


if __name__ == "__main__":
    main()
