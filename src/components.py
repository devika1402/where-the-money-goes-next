"""Component structure around flagged accounts. The measurement that gates G8, decision D35.

PRD section 9 Step 8 would extract the component around each flagged account and match it
against structural definitions for fan-in, fan-out, cycle, scatter-gather and gather-scatter.
That operation has to be well defined before it is worth two days, and it has two opposite
ways of not being. A two node fragment is compatible with every archetype and distinguishes
none. A component holding most of the graph is not a motif at all, and matching it against a
star says nothing without a local extraction rule the specification does not give.

This module measures which of the two the graph produces, and it decides nothing on its own.
The decision is recorded after the numbers exist, against a prediction written before them.

Every quantity comes from :mod:`src.definitions` or is a sparse matrix operation here. Nothing
loops over accounts, and nothing published reads what this writes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from src.definitions import (
    Window,
    build_windows,
    load_params,
    scoring_population,
    write_metrics,
)
from src.economics import apply_overflow_policy, queue_for, select_alerts
from src.models import LABEL

LOGGER = logging.getLogger(__name__)

#: A component with fewer nodes than this cannot express any archetype. Fan-in and fan-out
#: need a centre and at least two counterparties, so three is the smallest shape that is a
#: star rather than an edge, and even three is a shape every archetype contains a copy of.
MIN_ARCHETYPE_NODES: int = 4

#: Above this a component stops being a motif somebody could read off a page. The bound is
#: declared rather than derived, and it is generous: an analyst brief describing a 50 account
#: component is already not describing a shape.
MAX_ARCHETYPE_NODES: int = 50

#: A component of one node is an account with no counterparty in the window. Two is a single
#: edge. Three is a path or a triangle. None of them tells one archetype from another, so they
#: are banded together and named as the floor rather than left as bare numbers in a comparison.
ISOLATED_NODES: int = 1
FRAGMENT_MIN_NODES: int = 2
FRAGMENT_MAX_NODES: int = 3


@dataclass(frozen=True)
class ComponentProfile:
    """What the connected components of one feature-window graph look like."""

    nodes: int
    edges: int
    components: int
    largest_component: int
    largest_share: float
    median_component: float
    singletons: int
    components_with_a_cycle: int
    cycle_share_outside_largest: float


def undirected_edges(txns: pd.DataFrame, window: Window) -> tuple[np.ndarray, np.ndarray, int]:
    """Edge list of the feature-window graph, as integer node ids over its own accounts.

    Senders are nodes even when they are never scored, because the shape around a flagged
    account is made of counterparties whether or not those counterparties are scoreable.
    """
    edges = txns.loc[
        (txns["timestamp"] >= window.feature_start) & (txns["timestamp"] < window.feature_end)
    ]
    nodes = pd.Index(
        np.union1d(edges["account_from"].unique(), edges["account_to"].unique()), name="account"
    )
    rows = nodes.get_indexer(pd.Index(edges["account_from"]))
    cols = nodes.get_indexer(pd.Index(edges["account_to"]))
    return rows, cols, len(nodes)


def component_labels(
    rows: np.ndarray, cols: np.ndarray, nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Connected component id per node, and the edge count of each component.

    Weak connectivity, because an archetype is a shape rather than a direction. A fan-in and a
    fan-out are the same undirected star, and the direction is what tells them apart once the
    shape is known.
    """
    undirected = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(nodes, nodes)
    )
    undirected = ((undirected + undirected.T) > 0).astype(np.int8).tocsr()
    undirected.setdiag(0)
    undirected.eliminate_zeros()

    count, labels = connected_components(undirected, directed=False)

    # Each undirected edge appears twice in the symmetric matrix, and both endpoints carry the
    # same component label, so summing degree by component and halving gives edges per
    # component with no loop over components.
    degree = np.asarray(undirected.sum(axis=1)).ravel()
    edges_per_component = np.bincount(labels, weights=degree, minlength=count) / 2.0
    return labels, edges_per_component


def profile(labels: np.ndarray, edges_per_component: np.ndarray) -> ComponentProfile:
    """Summarise the component structure. A component is a tree exactly when m == n - 1.

    So a component contains at least one cycle exactly when its edge count reaches its node
    count, which is countable directly and needs no cycle enumeration.
    """
    sizes = np.bincount(labels)
    largest = int(sizes.max())
    has_cycle = edges_per_component >= sizes

    outside = sizes < largest
    cycles_outside = int(has_cycle[outside].sum())
    return ComponentProfile(
        nodes=int(sizes.sum()),
        edges=int(edges_per_component.sum()),
        components=len(sizes),
        largest_component=largest,
        largest_share=float(largest / sizes.sum()),
        median_component=float(np.median(sizes)),
        singletons=int((sizes == 1).sum()),
        components_with_a_cycle=int(has_cycle.sum()),
        cycle_share_outside_largest=float(cycles_outside / max(int(outside.sum()), 1)),
    )


def account_component_sizes(
    accounts: pd.Index, nodes_index: pd.Index, labels: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Size of the component each account sits in. Accounts off the graph give 0."""
    positions = nodes_index.get_indexer(accounts)
    found = positions >= 0
    out = np.zeros(len(accounts), dtype=np.int64)
    out[found] = sizes[labels[positions[found]]]
    return out


def band_table(component_sizes: np.ndarray, label: str) -> pd.DataFrame:
    """How a set of accounts is spread across component sizes. The row G8 turns on is the
    middle band: big enough to carry a shape and small enough to be one."""
    bands = [
        ("not in the graph", component_sizes == 0),
        (f"{ISOLATED_NODES}, isolated", component_sizes == ISOLATED_NODES),
        (
            f"{FRAGMENT_MIN_NODES} to {FRAGMENT_MAX_NODES}, an edge or a path",
            (component_sizes >= FRAGMENT_MIN_NODES) & (component_sizes <= FRAGMENT_MAX_NODES),
        ),
        (
            f"{MIN_ARCHETYPE_NODES} to {MAX_ARCHETYPE_NODES}, could be a motif",
            (component_sizes >= MIN_ARCHETYPE_NODES) & (component_sizes <= MAX_ARCHETYPE_NODES),
        ),
        (f"over {MAX_ARCHETYPE_NODES}", component_sizes > MAX_ARCHETYPE_NODES),
    ]
    return pd.DataFrame(
        [
            {"set": label, "band": name, "accounts": int(mask.sum()), "share": float(mask.mean())}
            for name, mask in bands
        ]
    )


def main() -> None:
    """Measure the component structure the test window offers, and the shape G8 would read."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    txns = pd.read_parquet(params.paths.interim_dir / "canonical_txns.parquet")
    window = {w.name: w for w in build_windows(params.split, params.windows.label_window_days)}[
        "test"
    ]

    rows, cols, node_count = undirected_edges(txns, window)
    edge_frame = txns.loc[
        (txns["timestamp"] >= window.feature_start) & (txns["timestamp"] < window.feature_end)
    ]
    nodes_index = pd.Index(
        np.union1d(edge_frame["account_from"].unique(), edge_frame["account_to"].unique()),
        name="account",
    )
    labels, edges_per_component = component_labels(rows, cols, node_count)
    sizes = np.bincount(labels)
    shape = profile(labels, edges_per_component)

    LOGGER.info("test feature-window graph: %s", asdict(shape))

    scores = pd.read_parquet(params.paths.interim_dir / "scores_test.parquet")
    queue = apply_overflow_policy(
        queue_for(txns, window, scores, "xgboost"), params.costs.queue_overflow_policy
    )
    alerted = queue.loc[select_alerts(queue, params.costs.analyst_capacity_per_day)]
    alerted_accounts = pd.Index(sorted(set(alerted["account"])), name="account")

    population = scoring_population(txns, window)
    mules = pd.Index(
        sorted(set(scores.loc[scores[LABEL] == 1].index.to_numpy())), name="account"
    ).intersection(population)

    sets = {
        "alerted by xgboost": alerted_accounts,
        "reachable mules": mules,
        "whole scoring population": population,
    }
    tables = []
    for name, accounts in sets.items():
        account_sizes = account_component_sizes(accounts, nodes_index, labels, sizes)
        tables.append(band_table(account_sizes, name))
        LOGGER.info(
            "%-24s n=%d, median component %d, largest %d, share in the giant component %.2f%%",
            name,
            len(accounts),
            int(np.median(account_sizes)),
            int(account_sizes.max()),
            100.0 * float((account_sizes == shape.largest_component).mean()),
        )
    bands = pd.concat(tables, ignore_index=True)
    shown = bands.assign(share=bands["share"].map(lambda v: f"{v:.2%}"))
    LOGGER.info("component size bands\n%s", shown.to_string(index=False))

    measured = {
        "window": window.name,
        "graph": asdict(shape),
        "bands": bands.to_dict(orient="records"),
        "min_archetype_nodes": MIN_ARCHETYPE_NODES,
        "max_archetype_nodes": MAX_ARCHETYPE_NODES,
    }
    write_metrics(params.paths.reports_dir / "metrics_components.json", measured)
    LOGGER.info("wrote %s", params.paths.reports_dir / "metrics_components.json")


if __name__ == "__main__":
    main()
