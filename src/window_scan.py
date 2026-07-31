"""What a longer history window would have bought. Session G1, alongside the published split.

The published feature window is four days because ten usable days had to carry three two-day
label windows, not because four days was measured against anything. Two of the project's
binding numbers come out of that leftover: 57.8% of label-window mules are reachable, and the
feature-window graph has a mean degree of 2.39.

This module holds one label window fixed and varies only the length of the history before it,
so the same mule accounts are the target at every length. It reads no new data and writes
nothing the published pipeline reads, which is what makes it an experiment reported beside the
split rather than a change to it.

Nothing here redefines a shared quantity. The scoring population, the label rule, the
unscoreable count and the leakage invariant all come from :mod:`src.definitions`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.definitions import (
    SchemaError,
    Window,
    build_windows,
    count_unscoreable_mules,
    label_accounts,
    load_params,
    scoring_population,
    write_metrics,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphStats:
    """Structure of the undirected history-window graph, on the whole graph not the population.

    Senders are nodes here even when they are never scored, because an account's structural
    position depends on counterparties that sit outside the scoring population.
    """

    nodes: int
    directed_edges: int
    undirected_edges: int
    mean_degree: float
    max_degree: int
    triangles: int
    accounts_in_triangle: int
    zero_triangle_share: float


@dataclass(frozen=True)
class ScanRow:
    """One history length measured against the fixed label window."""

    history_days: int
    feature_start: str
    population: int
    mules_in_label_window: int
    reachable_mules: int
    unscoreable_mules: int
    reachability: float
    base_rate: float
    graph: GraphStats


def rehistory(window: Window, history_days: int, data_start: pd.Timestamp) -> Window:
    """The same label window, with ``history_days`` of history before it.

    ``Window.__post_init__`` refuses to exist with ``feature_end != label_start``, so the
    invariant that makes the leakage cutoff unambiguous is enforced by the constructor rather
    than restated here.

    Raises rather than clamping when the history would start before the usable span, because a
    silently shortened window would be reported under the length that was asked for.
    """
    feature_start = window.label_start - pd.Timedelta(days=history_days)
    if feature_start < data_start:
        raise SchemaError(
            f"A {history_days} day history before {window.label_start} starts at "
            f"{feature_start}, which is outside the usable span opening at {data_start}."
        )
    return Window(
        name=f"{window.name}_h{history_days}",
        feature_start=feature_start,
        feature_end=window.label_start,
        label_start=window.label_start,
        label_end=window.label_end,
    )


def structure(rows: np.ndarray, cols: np.ndarray, nodes: int, block_rows: int) -> GraphStats:
    """Degree and triangle structure from an ordered edge list over ``nodes`` accounts.

    Every sparse object stays inside this function, because ``disallow_any_unimported`` means
    a scipy type in a signature fails the type gate.

    Triangles come from ``U.multiply(U @ U)``. Appendix A cut the clustering features for
    needing a Python loop over accounts, and that stated reason was measured to be wrong: this
    is a sparse matrix operation. The blocking is over row ranges rather than accounts, and it
    exists because the full ``U @ U`` materialises every path of length two, which a single hub
    of degree 9,220 alone pushes into tens of millions of nonzeros.
    """
    directed = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(nodes, nodes)
    )
    directed.sum_duplicates()
    directed.data[:] = 1.0

    undirected = (directed + directed.T).tocsr()
    undirected.setdiag(0.0)
    undirected.eliminate_zeros()
    undirected.data[:] = 1.0

    degree = np.asarray(undirected.sum(axis=1)).ravel()

    # Exact integers, so counted as integers. A float total would be a rounded quantity, and
    # comparing counts across history lengths is the whole point of this scan.
    triangles = np.zeros(nodes, dtype=np.int64)
    for start in range(0, nodes, block_rows):
        stop = min(start + block_rows, nodes)
        block = undirected[start:stop]
        closed = block.multiply(block @ undirected)
        paths = np.asarray(closed.sum(axis=1)).ravel()
        triangles[start:stop] = paths.round().astype(np.int64) // 2
    in_triangle = int((triangles > 0).sum())

    return GraphStats(
        nodes=nodes,
        directed_edges=int(directed.nnz),
        undirected_edges=int(undirected.nnz // 2),
        mean_degree=float(degree.mean()),
        max_degree=int(degree.max()),
        triangles=int(triangles.sum() // 3),
        accounts_in_triangle=in_triangle,
        zero_triangle_share=1.0 - in_triangle / nodes,
    )


def graph_stats(txns: pd.DataFrame, window: Window, block_rows: int) -> GraphStats:
    """Structure of the history-window graph, over every account in it rather than the population.

    Senders are nodes here even when they are never scored, because an account's structural
    position depends on counterparties that sit outside the scoring population.
    """
    edges = txns.loc[
        (txns["timestamp"] >= window.feature_start) & (txns["timestamp"] < window.feature_end)
    ]
    nodes = pd.Index(
        np.union1d(edges["account_from"].unique(), edges["account_to"].unique()), name="account"
    )
    return structure(
        nodes.get_indexer(pd.Index(edges["account_from"])),
        nodes.get_indexer(pd.Index(edges["account_to"])),
        len(nodes),
        block_rows,
    )


def scan_row(
    txns: pd.DataFrame,
    published: Window,
    history_days: int,
    data_start: pd.Timestamp,
    block_rows: int,
) -> ScanRow:
    """Measure one history length against the published window's label window."""
    window = rehistory(published, history_days, data_start)
    population = scoring_population(txns, window)
    labels = label_accounts(txns, window.label_start, window.label_end, population=population)

    reachable = int(labels["is_mule"].sum())
    unscoreable = count_unscoreable_mules(txns, window, population)
    in_label_window = reachable + unscoreable

    return ScanRow(
        history_days=history_days,
        feature_start=str(window.feature_start.date()),
        population=len(population),
        mules_in_label_window=in_label_window,
        reachable_mules=reachable,
        unscoreable_mules=unscoreable,
        reachability=reachable / in_label_window,
        base_rate=reachable / len(population),
        graph=graph_stats(txns, window, block_rows),
    )


def scan(
    txns: pd.DataFrame,
    published: Window,
    history_lengths: tuple[int, ...],
    data_start: pd.Timestamp,
    block_rows: int,
) -> tuple[ScanRow, ...]:
    """Every configured history length, against one fixed label window.

    The mule count in the label window depends on the label window alone, so it has to come out
    the same at every history length. A drift there would mean the label rule is reading the
    history window, which is the leak this project's boundary test exists to catch.
    """
    rows = tuple(
        scan_row(txns, published, length, data_start, block_rows) for length in history_lengths
    )
    counted = {row.mules_in_label_window for row in rows}
    if len(counted) > 1:
        raise SchemaError(
            f"Mules in the fixed label window changed with the history length: {sorted(counted)}. "
            "The label rule must not read the history window."
        )
    return rows


def main() -> None:
    """Run the scan on the test label window and write the table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    txns = pd.read_parquet(params.paths.interim_dir / "canonical_txns.parquet")
    windows = build_windows(params.split, params.windows.label_window_days)
    published = windows[-1]

    LOGGER.info(
        "label window held at [%s, %s), varying history over %s days",
        published.label_start,
        published.label_end,
        list(params.experiments.history_window_days),
    )
    rows = scan(
        txns,
        published,
        params.experiments.history_window_days,
        params.split.data_start,
        params.experiments.triangle_block_rows,
    )

    for row in rows:
        LOGGER.info(
            "%2dd from %s | pop %7d | reachable %3d of %3d (%.1f%%) | base rate %.4f%% | "
            "mean degree %.2f | zero-triangle %.1f%%",
            row.history_days,
            row.feature_start,
            row.population,
            row.reachable_mules,
            row.mules_in_label_window,
            100.0 * row.reachability,
            100.0 * row.base_rate,
            row.graph.mean_degree,
            100.0 * row.graph.zero_triangle_share,
        )

    write_metrics(
        params.paths.reports_dir / "metrics_window_scan.json",
        {
            "label_window": [str(published.label_start), str(published.label_end)],
            "published_history_days": params.split.feature_window_days,
            "rows": [asdict(row) for row in rows],
        },
    )


if __name__ == "__main__":
    main()
