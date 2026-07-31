"""Vectorised structural features per account per window.

Every feature here is a ``groupby.agg``, a sparse matrix operation, or a ``merge_asof``.
Nothing loops over accounts in Python. That constraint is what lets the prototype keep the
full graph at real scale rather than sampling down, and any feature that cannot be written
this way belongs in the cut list.

Implements F2 (pass-through ratio) from PRD section 8. F3 (FIFO pass-through latency) is
built here too and reported beside the published feature matrix rather than added to it,
which decision D31 fixed before F3 was measured.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.definitions import (
    FeatureSpec,
    SchemaError,
    Window,
    build_windows,
    label_accounts,
    load_params,
    scoring_population,
)

LOGGER = logging.getLogger(__name__)

#: Features allowed to be null, with the reason. Anything else null is a defect.
NULLABLE: tuple[str, ...] = (
    "pass_through_ratio",  # F2: undefined below the inflow floor
    "sender_diversity",  # undefined with no inbound transactions
    "mean_amount_ratio",  # undefined with no inbound or no outbound
    "median_hours_to_outflow",  # undefined with no matched inflow-outflow pair
)


def ratio_windows(
    window: Window, pass_through_window_hours: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Inflow window and outflow end for the pass-through ratio. F2 edge case (b) plus F4.

    Money arriving near the end of a window has not had time to leave, so every account
    looks like a hoarder at the edge. The fix is to count inflow over a shorter sub-window
    and allow outflow the extra hours to catch up.

    The extension stops at the leakage cutoff, never past it. With a 96 hour feature window
    and a 72 hour extension that leaves 24 hours of inflow, which is the trade-off the
    sweep over ``pass_through_window_hours`` exists to expose.
    """
    extension = pd.Timedelta(hours=pass_through_window_hours)
    inflow_end = window.feature_end - extension
    if inflow_end <= window.feature_start:
        raise SchemaError(
            f"A {pass_through_window_hours}h extension leaves no inflow window inside "
            f"[{window.feature_start}, {window.feature_end}). Shorten the extension or "
            f"lengthen the feature window."
        )
    return window.feature_start, inflow_end, window.feature_end


def pass_through_ratio(
    txns: pd.DataFrame,
    population: pd.Index,
    *,
    inflow_start: pd.Timestamp,
    inflow_end: pd.Timestamp,
    outflow_end: pd.Timestamp,
    min_flow_eur: float,
) -> pd.Series:
    """Outflow over inflow in base currency, indexed by the population. Implements F2.

    Returns NaN where inflow falls below ``min_flow_eur``, which covers the zero-inflow
    case. Zero would mean the account received money and kept all of it, which is the
    opposite of what an account with no inflow is doing.
    """
    inflows = txns.loc[(txns["timestamp"] >= inflow_start) & (txns["timestamp"] < inflow_end)]
    outflows = txns.loc[(txns["timestamp"] >= inflow_start) & (txns["timestamp"] < outflow_end)]

    inflow = (
        inflows.groupby("account_to")["amount_in_base_received"]
        .sum()
        .reindex(population, fill_value=0.0)
    )
    outflow = (
        outflows.groupby("account_from")["amount_in_base_paid"]
        .sum()
        .reindex(population, fill_value=0.0)
    )

    ratio = outflow / inflow.where(inflow >= min_flow_eur)
    return ratio.rename("pass_through_ratio")


def _directional_aggregates(txns: pd.DataFrame, population: pd.Index, *, side: str) -> pd.DataFrame:
    """Count, unique counterparties, total, max, and concentration for one direction."""
    own, other, amount = (
        ("account_to", "account_from", "amount_in_base_received")
        if side == "in"
        else ("account_from", "account_to", "amount_in_base_paid")
    )
    grouped = txns.groupby(own)
    frame = pd.DataFrame(
        {
            f"{side}_degree": grouped.size(),
            f"unique_counterparties_{side}": grouped[other].nunique(),
            f"total_{side}flow": grouped[amount].sum(),
            f"max_{side}flow": grouped[amount].max(),
        }
    )

    # Concentration: the largest single counterparty's share of the total.
    by_pair = txns.groupby([own, other])[amount].sum()
    largest = by_pair.groupby(level=0).max()
    frame[f"{side}flow_concentration"] = largest / frame[f"total_{side}flow"]

    return frame.reindex(population).fillna(
        {
            f"{side}_degree": 0,
            f"unique_counterparties_{side}": 0,
            f"total_{side}flow": 0.0,
            f"max_{side}flow": 0.0,
            f"{side}flow_concentration": 0.0,
        }
    )


def _pagerank_and_reciprocity(
    txns: pd.DataFrame, population: pd.Index, spec: FeatureSpec
) -> pd.DataFrame:
    """PageRank by sparse power iteration, and reciprocity from A elementwise A transpose.

    Both run on the whole feature-window graph rather than on the population alone, because
    an account's structural position depends on counterparties that may sit outside it.
    Betweenness is absent deliberately: it is O(nm) and would not finish at this scale.
    """
    nodes = pd.Index(
        np.union1d(txns["account_from"].unique(), txns["account_to"].unique()), name="account"
    )
    rows = nodes.get_indexer(pd.Index(txns["account_from"]))
    cols = nodes.get_indexer(pd.Index(txns["account_to"]))
    n = len(nodes)

    adjacency = sp.csr_matrix((np.ones(len(txns), dtype=np.float64), (rows, cols)), shape=(n, n))
    binary = adjacency.copy()
    binary.data[:] = 1.0
    binary.sum_duplicates()
    binary.data[:] = 1.0

    out_degree = np.asarray(binary.sum(axis=1)).ravel()
    inverse = np.divide(1.0, out_degree, out=np.zeros(n), where=out_degree > 0)
    transition = sp.diags(inverse) @ binary
    dangling = out_degree == 0

    damping = spec.pagerank_damping
    rank = np.full(n, 1.0 / n)
    for _ in range(spec.pagerank_max_iter):
        leaked = damping * rank[dangling].sum() / n
        updated = damping * (transition.T @ rank) + leaked + (1.0 - damping) / n
        if np.abs(updated - rank).sum() < spec.pagerank_tol:
            rank = updated
            break
        rank = updated

    mutual = binary.multiply(binary.T)
    reciprocated = np.asarray(mutual.sum(axis=1)).ravel()
    reciprocity = np.divide(reciprocated, out_degree, out=np.zeros(n), where=out_degree > 0)

    frame = pd.DataFrame({"pagerank": rank, "reciprocity": reciprocity}, index=nodes)
    return frame.reindex(population).fillna({"pagerank": 0.0, "reciprocity": 0.0})


def _activity(txns: pd.DataFrame, population: pd.Index) -> pd.DataFrame:
    """Active days and burstiness, over transactions on either side of the account."""
    both = pd.concat(
        [
            txns[["account_to", "timestamp"]].rename(columns={"account_to": "account"}),
            txns[["account_from", "timestamp"]].rename(columns={"account_from": "account"}),
        ],
        ignore_index=True,
    )
    both["day"] = both["timestamp"].dt.floor("D")

    per_day = both.groupby(["account", "day"]).size()
    totals = per_day.groupby(level=0).sum()
    frame = pd.DataFrame(
        {
            "active_days": per_day.groupby(level=0).size(),
            "burstiness": per_day.groupby(level=0).max() / totals,
        }
    )
    return frame.reindex(population).fillna({"active_days": 0, "burstiness": 0.0})


def median_hours_to_outflow(
    txns: pd.DataFrame,
    population: pd.Index,
    *,
    inflow_start: pd.Timestamp,
    inflow_end: pd.Timestamp,
    outflow_end: pd.Timestamp,
) -> pd.Series:
    """Median hours from an inflow to that account's next outflow, via ``merge_asof``.

    This was built as the vectorised stand-in for F3's FIFO attribution, ~~which is inherently
    sequential per account and sits in the cut list~~. **That reason was wrong.** F3 is
    :func:`fifo_pass_through_latency` and it vectorises, so the two now sit side by side and
    this one is kept because it is what every published figure was measured under. D31.

    They answer different questions and the difference is large. This one asks how soon after
    money arrives something leaves. F3 asks how long the money that arrived actually stayed.
    On the test window the medians are 9.25 and 19.80 hours.

    Inflows are counted over the sub-window and outflows are searched to the leakage
    cutoff, for the reason F2 edge case (b) gives and for a second one specific to a
    latency. Near the window edge the only forwards observable are the fast ones, because
    a slow forward has no outflow inside the window to match against. Counting the
    survivors measures how quickly the edge lets money leave rather than how quickly the
    account moves it. See D18.
    """
    windowed = txns.loc[(txns["timestamp"] >= inflow_start) & (txns["timestamp"] < outflow_end)]
    inflows = (
        windowed.loc[windowed["timestamp"] < inflow_end, ["account_to", "timestamp"]]
        .rename(columns={"account_to": "account"})
        .sort_values("timestamp")
    )
    outflows = (
        windowed[["account_from", "timestamp"]]
        .rename(columns={"account_from": "account", "timestamp": "outflow_time"})
        .sort_values("outflow_time")
    )
    if inflows.empty or outflows.empty:
        return pd.Series(np.nan, index=population, name="median_hours_to_outflow")

    matched = pd.merge_asof(
        inflows,
        outflows,
        left_on="timestamp",
        right_on="outflow_time",
        by="account",
        direction="forward",
        allow_exact_matches=False,
    )
    elapsed = matched["outflow_time"] - matched["timestamp"]
    matched["hours"] = elapsed.dt.total_seconds() / 3600.0

    median = matched.groupby("account")["hours"].median()
    return median.reindex(population).rename("median_hours_to_outflow")


def _directed_flows(
    txns: pd.DataFrame,
    *,
    inflow_start: pd.Timestamp,
    inflow_end: pd.Timestamp,
    outflow_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inflows and outflows on one time convention, each carrying a per-account cumsum.

    The window rules are the ones :func:`median_hours_to_outflow` already uses, so the two
    latencies answer their different questions over identical evidence. Anything else and a
    difference between them would be partly a difference of window.
    """
    windowed = txns.loc[(txns["timestamp"] >= inflow_start) & (txns["timestamp"] < outflow_end)]
    inflows = (
        windowed.loc[
            windowed["timestamp"] < inflow_end,
            ["account_to", "timestamp", "amount_in_base_received"],
        ]
        .rename(columns={"account_to": "account", "amount_in_base_received": "amount"})
        .sort_values(["account", "timestamp"], kind="stable")
    )
    outflows = (
        windowed[["account_from", "timestamp", "amount_in_base_paid"]]
        .rename(
            columns={
                "account_from": "account",
                "timestamp": "outflow_time",
                "amount_in_base_paid": "amount",
            }
        )
        .sort_values(["account", "outflow_time"], kind="stable")
    )
    inflows["in_cum"] = inflows.groupby("account")["amount"].cumsum()
    outflows["out_cum"] = outflows.groupby("account")["amount"].cumsum()
    return inflows, outflows


def fifo_latency_pairs(
    txns: pd.DataFrame,
    population: pd.Index,
    *,
    inflow_start: pd.Timestamp,
    inflow_end: pd.Timestamp,
    outflow_end: pd.Timestamp,
) -> pd.DataFrame:
    """F3's matched inflow-outflow pairs under FIFO, for every account at once.

    **The PRD cut F3 because FIFO attribution is "inherently sequential per account". It is
    not.** Put the money on a cumulative-amount axis and the sequencing disappears. Inflow
    ``k`` occupies the interval ``(A[k-1], A[k]]`` where ``A`` is the running total of
    inflow, outflow ``j`` occupies ``(B[j-1], B[j]]`` on the same axis, and FIFO says outflow
    ``j`` consumes exactly the inflow lying under it. So a matched pair is an overlap between
    two step functions, its weight is the length of that overlap, and the pairs come out of a
    ``merge_asof``. See D31.

    The construction, all of it vectorised across accounts:

    1. Cut the axis at every breakpoint, which is the union of both cumulative sums per
       account, keeping only breakpoints at or below ``min(total_in, total_out)``.
    2. Each resulting segment lies inside exactly one inflow interval and one outflow
       interval, so a forward ``merge_asof`` on each side names them.
    3. The segment's length is the amount attributed, and the latency is the gap between
       the two timestamps it names.

    **Unmatched money is dropped, which the PRD requires.** Truncating at
    ``min(total_in, total_out)`` is what does it: inflow an account is still holding has no
    outflow under it and produces no pair. Counting it as infinite latency would make every
    account still holding money look slow, which measures the window edge rather than the
    account.

    Returns one row per matched pair with ``account``, ``latency_hours`` and ``weight``.
    """
    inflows, outflows = _directed_flows(
        txns, inflow_start=inflow_start, inflow_end=inflow_end, outflow_end=outflow_end
    )
    empty = pd.DataFrame(
        {
            "account": pd.Series(dtype=population.dtype),
            "latency_hours": pd.Series(dtype="float64"),
            "weight": pd.Series(dtype="float64"),
        }
    )
    if inflows.empty or outflows.empty:
        return empty

    # How far up the axis both sides reach. Beyond this one side has run out, so there is
    # nothing to match against and the money is dropped rather than attributed.
    matched_total = pd.concat(
        [
            inflows.groupby("account")["in_cum"].max().rename("total"),
            outflows.groupby("account")["out_cum"].max().rename("total"),
        ],
        axis=1,
        keys=["inflow", "outflow"],
    ).min(axis=1)
    matched_total = matched_total.loc[matched_total > 0]
    if matched_total.empty:
        return empty

    breaks = pd.concat(
        [
            inflows[["account", "in_cum"]].rename(columns={"in_cum": "edge"}),
            outflows[["account", "out_cum"]].rename(columns={"out_cum": "edge"}),
            matched_total.rename("edge").reset_index(),
        ],
        ignore_index=True,
    )
    breaks = breaks.loc[breaks["edge"] <= breaks["account"].map(matched_total)]
    breaks = (
        breaks.drop_duplicates(["account", "edge"])
        .sort_values(["account", "edge"], kind="stable")
        .reset_index(drop=True)
    )
    breaks["weight"] = breaks["edge"] - breaks.groupby("account")["edge"].shift(1).fillna(0.0)
    segments = breaks.loc[breaks["weight"] > 0].reset_index(drop=True)
    if segments.empty:
        return empty

    # The first inflow whose running total reaches this segment is the one holding it, and
    # the same on the outflow side. Forward, because the interval is closed on the right.
    matched = pd.merge_asof(
        segments.sort_values("edge", kind="stable"),
        inflows.sort_values("in_cum", kind="stable")[["account", "in_cum", "timestamp"]],
        left_on="edge",
        right_on="in_cum",
        by="account",
        direction="forward",
    )
    matched = pd.merge_asof(
        matched.sort_values("edge", kind="stable"),
        outflows.sort_values("out_cum", kind="stable")[["account", "out_cum", "outflow_time"]],
        left_on="edge",
        right_on="out_cum",
        by="account",
        direction="forward",
    )
    matched = matched.dropna(subset=["timestamp", "outflow_time"])
    # FIFO consumes a running balance, so an outflow can only be attributed to inflow that
    # had already arrived. An account carrying a balance from before the window sends money
    # this window never saw arrive, and on the money axis that outflow still lands under the
    # first in-window inflow, which would report a negative latency. Same rule as the far
    # edge: the window cannot see the money, so the window does not price it. D31.
    matched = matched.loc[matched["outflow_time"] > matched["timestamp"]]

    elapsed = matched["outflow_time"] - matched["timestamp"]
    return pd.DataFrame(
        {
            "account": matched["account"].to_numpy(),
            "latency_hours": elapsed.dt.total_seconds().to_numpy() / 3600.0,
            "weight": matched["weight"].to_numpy(),
        }
    )


def weighted_median(pairs: pd.DataFrame, population: pd.Index, name: str) -> pd.Series:
    """Per-account weighted median of ``latency_hours`` by ``weight``. Vectorised.

    The convention is the one the F3 worked example uses: sort by latency, accumulate the
    weights, and take the first latency whose cumulative weight reaches half the total. On
    the worked example the halfway point is 175 of 350 and the answer is 12 hours.

    Declared rather than inherited from a library, because the alternatives (interpolating
    between the two straddling values, or averaging them) give a different number on the
    same input and the PRD fixes which one this project means.
    """
    if pairs.empty:
        return pd.Series(np.nan, index=population, name=name, dtype="float64")

    ordered = pairs.sort_values(["account", "latency_hours"], kind="stable")
    cumulative = ordered.groupby("account")["weight"].cumsum()
    half = ordered.groupby("account")["weight"].transform("sum") / 2.0
    reached = ordered.loc[cumulative >= half]
    median = reached.groupby("account")["latency_hours"].first()
    return median.reindex(population).rename(name)


def fifo_pass_through_latency(
    txns: pd.DataFrame,
    population: pd.Index,
    *,
    inflow_start: pd.Timestamp,
    inflow_end: pd.Timestamp,
    outflow_end: pd.Timestamp,
) -> pd.Series:
    """F3. Amount-weighted median latency over FIFO-matched pairs, per account.

    Null where an account produced no matched pair, which is any account that received
    nothing or sent nothing inside the window. That is the same population the naive proxy
    leaves null, so the two are comparable where both are defined.
    """
    pairs = fifo_latency_pairs(
        txns,
        population,
        inflow_start=inflow_start,
        inflow_end=inflow_end,
        outflow_end=outflow_end,
    )
    return weighted_median(pairs, population, "fifo_pass_through_latency")


def build_features(
    txns: pd.DataFrame, window: Window, spec: FeatureSpec, pass_through_window_hours: int
) -> pd.DataFrame:
    """Every feature for one window, indexed by that window's scoring population.

    Reads only transactions strictly before the leakage cutoff, which ``feature_rows``
    guarantees by taking the cutoff as an argument rather than deriving one.
    """
    population = scoring_population(txns, window)
    visible = txns.loc[
        (txns["timestamp"] >= window.feature_start) & (txns["timestamp"] < window.feature_end)
    ]

    inbound = _directional_aggregates(visible, population, side="in")
    outbound = _directional_aggregates(visible, population, side="out")
    graph = _pagerank_and_reciprocity(visible, population, spec)
    activity = _activity(visible, population)

    inflow_start, inflow_end, outflow_end = ratio_windows(window, pass_through_window_hours)
    ratio = pass_through_ratio(
        visible,
        population,
        inflow_start=inflow_start,
        inflow_end=inflow_end,
        outflow_end=outflow_end,
        min_flow_eur=spec.min_flow_eur,
    )

    features = pd.concat([inbound, outbound, graph, activity, ratio], axis=1)
    features["median_hours_to_outflow"] = median_hours_to_outflow(
        visible,
        population,
        inflow_start=inflow_start,
        inflow_end=inflow_end,
        outflow_end=outflow_end,
    )

    degree_total = features["in_degree"] + features["out_degree"]
    features["degree_asymmetry"] = np.divide(
        features["in_degree"] - features["out_degree"],
        degree_total,
        out=np.zeros(len(features)),
        where=degree_total > 0,
    )
    counterparty_total = (
        features["unique_counterparties_in"] + features["unique_counterparties_out"]
    )
    features["counterparty_asymmetry"] = np.divide(
        features["unique_counterparties_in"] - features["unique_counterparties_out"],
        counterparty_total,
        out=np.zeros(len(features)),
        where=counterparty_total > 0,
    )
    features["sender_diversity"] = features["unique_counterparties_in"] / features[
        "in_degree"
    ].where(features["in_degree"] > 0)

    mean_in = features["total_inflow"] / features["in_degree"].where(features["in_degree"] > 0)
    mean_out = features["total_outflow"] / features["out_degree"].where(features["out_degree"] > 0)
    features["mean_amount_ratio"] = mean_out / mean_in

    return features


def assert_no_unexpected_nulls(features: pd.DataFrame) -> None:
    """Every null must be one the feature definition allows. Anything else is a defect."""
    offenders = {
        column: int(features[column].isna().sum())
        for column in features.columns
        if column not in NULLABLE and features[column].isna().any()
    }
    if offenders:
        raise SchemaError(f"Unexpected nulls in feature columns: {offenders}")


def main() -> None:
    """Build and write one feature matrix per window."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    txns = pd.read_parquet(params.paths.interim_dir / "canonical_txns.parquet")
    windows = build_windows(params.split, params.windows.label_window_days)

    hours = params.windows.pass_through_window_hours
    for window in windows:
        inflow_start, inflow_end, _ = ratio_windows(window, hours)
        LOGGER.info(
            "%-5s ratio inflow window [%s, %s), outflow to the cutoff at %s",
            window.name,
            inflow_start,
            inflow_end,
            window.feature_end,
        )

        features = build_features(txns, window, params.features, hours)
        assert_no_unexpected_nulls(features)

        population = scoring_population(txns, window)
        labels = label_accounts(txns, window.label_start, window.label_end, population=population)
        matrix = features.join(labels)

        destination = params.paths.interim_dir / f"features_{window.name}.parquet"
        matrix.to_parquet(destination)
        LOGGER.info(
            "%-5s %d accounts x %d features, %d mules, wrote %s",
            window.name,
            len(matrix),
            len(features.columns),
            int(matrix["is_mule"].sum()),
            destination.name,
        )


if __name__ == "__main__":
    main()
