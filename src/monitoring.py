"""Drift and decay monitoring. Implements F5, the Population Stability Index.

This is the check that would run in production and say whether the operating point chosen on
one window still means anything on the next. It is reported beside the model figures rather
than inside them, because a drift number is a statement about the data and not about the
scorer.

Two things it deliberately keeps apart:

* **PSI** answers where a distribution moved, bin by bin, and needs a reference period to
  define the bins. It is not symmetric in its two periods for that reason.
* **KS** answers how far apart two distributions are at their widest point, needs no binning
  and no reference, and is symmetric. Cheaper, and it cannot say which part moved.

F5's third edge case says PSI is not symmetric and asks for a test asserting that on two
proportion vectors. That is false as written and the correction is in D32: on proportions the
formula is symmetric to the last bit, and the asymmetry lives in where the bin edges come
from. Both facts are pinned by tests.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from src.definitions import SPLIT_NAMES, SchemaError, load_params, write_metrics
from src.models import LABEL, SCORERS

LOGGER = logging.getLogger(__name__)

#: Conventional readings for a PSI value, from PRD section 8 F5. Convention, not law, and the
#: report says so wherever it prints one.
#: PSI is undefined on fewer than two bins, because a single bin holds all the mass on both
#: sides by construction and reports no drift whatever the truth is.
MIN_BINS: Final[int] = 2

#: Distinct edges at or below this and the reference is effectively one bin, so PSI reports 0
#: however far the comparison moved. Three because the two infinite outer edges plus a single
#: interior value is the degenerate case: every quantile of a near-constant feature collapses
#: onto that value. Measured here on ``reciprocity``, zero for 99.52% of the training window.
DEGENERATE_EDGE_COUNT: Final[int] = 3

PSI_READINGS: tuple[tuple[float, str], ...] = (
    (0.10, "stable"),
    (0.25, "moderate shift"),
    (float("inf"), "significant shift"),
)


def psi_reading(value: float) -> str:
    """The conventional label for a PSI value. Reporting only, it decides nothing."""
    for bound, label in PSI_READINGS:
        if value < bound:
            return label
    return PSI_READINGS[-1][1]


def stabilise(proportions_in: Sequence[float] | np.ndarray, *, epsilon: float) -> np.ndarray:
    """Replace zero proportions with ``epsilon`` and renormalise. F5 edge case (b).

    An empty bin makes the log infinite on one side and undefined on the other, so the zero
    has to go before the log rather than after it. Renormalising afterwards matters more than
    it looks: without it the vector sums to 1 plus epsilon per empty bin, which is a silent
    error that grows with the number of empty bins.
    """
    values = np.asarray(proportions_in, dtype=float).copy()
    if values.size == 0:
        raise SchemaError("A period with no observations has no distribution to compare.")
    if np.any(values < 0.0):
        raise SchemaError("Proportions must not be negative.")
    values[values == 0.0] = epsilon
    total = values.sum()
    if total <= 0.0:
        raise SchemaError("Proportions must sum to something positive.")
    return np.asarray(values / total, dtype=float)


def psi(
    reference: Sequence[float] | np.ndarray,
    comparison: Sequence[float] | np.ndarray,
    *,
    epsilon: float,
) -> float:
    """F5. Population Stability Index between two proportion vectors over the same bins.

    ``PSI = sum over bins of (p_comp - p_ref) * ln(p_comp / p_ref)``

    **This function is symmetric and that is not a defect.** Swapping the two arguments
    negates both factors of every term, so each product is unchanged. The specification's
    instruction to assert inequality here cannot be satisfied by any correct implementation
    of the formula it also specifies. Pass the reference first anyway, because
    :func:`psi_from_values` is asymmetric and shares this argument order. D32.
    """
    ref = np.asarray(reference, dtype=float)
    comp = np.asarray(comparison, dtype=float)
    if ref.shape != comp.shape:
        raise SchemaError(
            f"Both periods must be binned into the same number of bins, got "
            f"{ref.shape} and {comp.shape}."
        )
    ref_p = stabilise(ref, epsilon=epsilon)
    comp_p = stabilise(comp, epsilon=epsilon)
    return float(np.sum((comp_p - ref_p) * np.log(comp_p / ref_p)))


def quantile_edges(values: np.ndarray | pd.Series, *, bins: int) -> np.ndarray:
    """Bin edges from the reference period only, at equally spaced quantiles. F5 edge case (a).

    The outer edges are opened to infinity, because a comparison value outside the reference
    range still has to land in a bin. Clipping it to the reference range would hide exactly
    the drift the measurement exists to find.

    Computed once from the reference and applied to both periods. Re-deriving them from the
    comparison is the error edge case (a) names, and it fails in the direction that reports
    no drift.
    """
    if bins < MIN_BINS:
        raise SchemaError(f"PSI needs at least {MIN_BINS} bins, got {bins}.")
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise SchemaError("A period with no observations cannot define bin edges.")
    edges = np.quantile(clean, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.asarray(edges, dtype=float)


def proportions(values: np.ndarray | pd.Series, edges: np.ndarray) -> np.ndarray:
    """Share of ``values`` falling in each bin of ``edges``. Sums to 1."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise SchemaError("A period with no observations has nothing to bin.")
    counts, _ = np.histogram(clean, bins=edges)
    return np.asarray(counts / counts.sum(), dtype=float)


def empty_bin_count(
    reference: Sequence[float] | np.ndarray, comparison: Sequence[float] | np.ndarray
) -> int:
    """Bins holding no mass on at least one side, counted before ``stabilise`` fills them.

    **This is what says whether a PSI value is a magnitude or a flag.** With no empty bin the
    value is a property of the data. With one, the emptied term is
    ``(epsilon - p_ref) * ln(epsilon / p_ref)``, whose size is set by the epsilon someone
    chose, and the honest reading of a large PSI becomes "a bin emptied" rather than a
    distance. Measured on this project's own data: the rules score reads 5.83 at epsilon 1e-4
    and 11.71 at 1e-8, on identical inputs. D34.
    """
    ref = np.asarray(reference, dtype=float)
    comp = np.asarray(comparison, dtype=float)
    return int(np.sum((ref == 0.0) | (comp == 0.0)))


def drift_row(
    reference: np.ndarray | pd.Series,
    comparison: np.ndarray | pd.Series,
    *,
    bins: int,
    epsilon: float,
) -> dict[str, float | str | int]:
    """One drift measurement with everything needed to read it. PSI alone is not enough."""
    edges = quantile_edges(reference, bins=bins)
    ref_p = proportions(reference, edges)
    comp_p = proportions(comparison, edges)
    value = psi(ref_p, comp_p, epsilon=epsilon)
    empty = empty_bin_count(ref_p, comp_p)
    return {
        "psi": value,
        "reading": psi_reading(value),
        "empty_bins": empty,
        # A reference that is constant enough for its quantiles to collapse cannot resolve
        # drift at all, and reports zero however far the comparison has moved.
        "distinct_reference_edges": len(np.unique(edges)),
        "epsilon_dependent": bool(empty > 0),
    }


def psi_from_values(
    reference: np.ndarray | pd.Series,
    comparison: np.ndarray | pd.Series,
    *,
    bins: int,
    epsilon: float,
) -> float:
    """PSI end to end from two raw samples, with the edges taken from the reference.

    **This is the asymmetric one.** ``psi_from_values(a, b)`` and ``psi_from_values(b, a)``
    read the same data and can differ by an order of magnitude, because the first bins by
    ``a``'s quantiles and the second by ``b``'s. Argument order is a silent bug here, which
    is what F5 edge case (c) is reaching for.
    """
    edges = quantile_edges(reference, bins=bins)
    return psi(
        proportions(reference, edges),
        proportions(comparison, edges),
        epsilon=epsilon,
    )


def ks_statistic(reference: np.ndarray | pd.Series, comparison: np.ndarray | pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the widest gap between the two ECDFs.

    Needs no bins and no reference period, so it is symmetric and cheap. It also cannot say
    which part of the distribution moved, which is why PSI is reported beside it rather than
    replaced by it.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    comp = np.sort(np.asarray(comparison, dtype=float))
    ref = ref[np.isfinite(ref)]
    comp = comp[np.isfinite(comp)]
    if ref.size == 0 or comp.size == 0:
        raise SchemaError("A period with no observations has no distribution to compare.")

    grid = np.concatenate([ref, comp])
    ref_cdf = np.searchsorted(ref, grid, side="right") / ref.size
    comp_cdf = np.searchsorted(comp, grid, side="right") / comp.size
    return float(np.max(np.abs(ref_cdf - comp_cdf)))


def feature_drift(
    reference: pd.DataFrame, comparison: pd.DataFrame, *, bins: int, epsilon: float
) -> pd.DataFrame:
    """PSI per feature between two windows, each feature on its own reference edges.

    One row per feature, in the frame's column order rather than sorted, so the table reads
    the same way the feature matrix does. Columns present in one frame and not the other are
    a defect rather than something to skip, because a feature that vanished between windows
    is the largest drift there is.
    """
    missing = [column for column in reference.columns if column not in comparison.columns]
    if missing:
        raise SchemaError(f"The comparison window is missing features: {missing}")

    rows: list[dict[str, object]] = []
    for column in reference.columns:
        ref_values = reference[column].to_numpy(dtype=float)
        comp_values = comparison[column].to_numpy(dtype=float)
        finite_ref = np.isfinite(ref_values).sum()
        finite_comp = np.isfinite(comp_values).sum()
        if finite_ref == 0 or finite_comp == 0:
            rows.append(
                {
                    "feature": column,
                    "psi": float("nan"),
                    "reading": "undefined, a window has no values",
                    "reference_null_share": 1.0 - finite_ref / len(ref_values),
                    "comparison_null_share": 1.0 - finite_comp / len(comp_values),
                }
            )
            continue
        row = drift_row(ref_values, comp_values, bins=bins, epsilon=epsilon)
        rows.append(
            {
                "feature": column,
                **row,
                "reference_null_share": 1.0 - finite_ref / len(ref_values),
                "comparison_null_share": 1.0 - finite_comp / len(comp_values),
            }
        )
    return pd.DataFrame(rows)


def score_drift(
    reference: pd.Series, comparison: pd.Series, *, bins: int, epsilon: float
) -> dict[str, float | str | int]:
    """Both drift measures on one scorer's output, so the pair can be read together."""
    row = drift_row(reference, comparison, bins=bins, epsilon=epsilon)
    row["ks"] = ks_statistic(reference, comparison)
    return row


def main() -> None:
    """Measure drift from the reference window to each later one and write the metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    spec = params.monitoring
    interim = params.paths.interim_dir

    reference_name = spec.reference_window
    if reference_name not in SPLIT_NAMES:
        raise SchemaError(f"Unknown reference window {reference_name!r}.")
    later = [name for name in SPLIT_NAMES if name != reference_name]

    features = {
        name: pd.read_parquet(interim / f"features_{name}.parquet").drop(
            columns=[LABEL], errors="ignore"
        )
        for name in SPLIT_NAMES
    }
    scores = {name: pd.read_parquet(interim / f"scores_{name}.parquet") for name in SPLIT_NAMES}

    measured: dict[str, Any] = {
        "reference_window": reference_name,
        "bins": spec.bins,
        "epsilon": spec.epsilon,
        "readings": {label: bound for bound, label in PSI_READINGS if bound != float("inf")},
        "feature_drift": {},
        "score_drift": {},
    }

    for name in later:
        table = feature_drift(
            features[reference_name], features[name], bins=spec.bins, epsilon=spec.epsilon
        )
        table = table.sort_values("psi", ascending=False, kind="stable").reset_index(drop=True)
        measured["feature_drift"][name] = table.to_dict(orient="records")
        shown = table.assign(psi=table["psi"].map(lambda v: f"{v:.4f}"))
        LOGGER.info(
            "feature drift %s -> %s, %d bins from the reference\n%s",
            reference_name,
            name,
            spec.bins,
            shown.to_string(index=False),
        )
        LOGGER.info(
            "%s -> %s: median feature PSI %.4f, %d of %d features read as a significant "
            "shift, %d have an emptied bin so their value moves with epsilon, %d have a "
            "reference too constant to resolve any drift",
            reference_name,
            name,
            float(table["psi"].median()),
            int((table["reading"] == "significant shift").sum()),
            len(table),
            int(table["epsilon_dependent"].sum()),
            int((table["distinct_reference_edges"] <= DEGENERATE_EDGE_COUNT).sum()),
        )

        measured["score_drift"][name] = {
            scorer: score_drift(
                scores[reference_name][scorer],
                scores[name][scorer],
                bins=spec.bins,
                epsilon=spec.epsilon,
            )
            for scorer in SCORERS
        }
        for scorer, pair in measured["score_drift"][name].items():
            LOGGER.info(
                "score drift %s -> %s, %-8s PSI %.4f (%s), KS %.4f, %d emptied bins%s",
                reference_name,
                name,
                scorer,
                pair["psi"],
                pair["reading"],
                pair["ks"],
                pair["empty_bins"],
                ", so the PSI magnitude is set by epsilon" if pair["epsilon_dependent"] else "",
            )

    # PRD section 9 Step 10 also asks for a performance-by-week decay curve. Ten usable days
    # do not contain weeks, and the three label windows are two days each, so there is no
    # sequence of weekly points to draw. Recorded as unmeasurable rather than approximated
    # with a curve over something that is not a week.
    measured["decay_by_week"] = None
    measured["decay_note"] = (
        "Not measurable. The usable span is ten days, so there are no weeks to decay over."
    )
    LOGGER.info("decay by week: not measurable, the usable span is ten days and holds no weeks")

    write_metrics(params.paths.reports_dir / "metrics_monitoring.json", measured)
    LOGGER.info("wrote %s", params.paths.reports_dir / "metrics_monitoring.json")


if __name__ == "__main__":
    main()
