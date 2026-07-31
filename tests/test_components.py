"""Component structure, the measurement that gates G8. Decision D35.

The load-bearing rule here is that a connected component is a tree exactly when its edge count
is one below its node count, so it contains a cycle exactly when the edge count reaches the
node count. That is what makes the cycle archetype countable without enumerating cycles, and
it is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.components import (
    FRAGMENT_MAX_NODES,
    MAX_ARCHETYPE_NODES,
    account_component_sizes,
    band_table,
    component_labels,
    profile,
    undirected_edges,
)
from src.definitions import Window

WINDOW = Window(
    name="fixture",
    feature_start=pd.Timestamp("2022-09-01"),
    feature_end=pd.Timestamp("2022-09-05"),
    label_start=pd.Timestamp("2022-09-05"),
    label_end=pd.Timestamp("2022-09-07"),
)


def _txns(edges: list[tuple[int, int]], when: str = "2022-09-02") -> pd.DataFrame:
    """A transaction frame carrying exactly the given directed edges inside the window."""
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(when)] * len(edges),
            "account_from": [a for a, _ in edges],
            "account_to": [b for _, b in edges],
        }
    )


def _shape(edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols, nodes = undirected_edges(_txns(edges), WINDOW)
    return component_labels(rows, cols, nodes)


def test_a_path_is_a_tree_and_has_no_cycle() -> None:
    """Three nodes, two edges. m == n - 1, so no cycle."""
    labels, edges_per = _shape([(1, 2), (2, 3)])

    shape = profile(labels, edges_per)

    assert shape.nodes == 3
    assert shape.edges == 2
    assert shape.components == 1
    assert shape.components_with_a_cycle == 0


def test_a_triangle_has_a_cycle() -> None:
    """Three nodes, three edges. m == n, so the component closes."""
    labels, edges_per = _shape([(1, 2), (2, 3), (3, 1)])

    shape = profile(labels, edges_per)

    assert shape.nodes == 3
    assert shape.edges == 3
    assert shape.components_with_a_cycle == 1


def test_a_star_is_a_tree_however_many_arms_it_has() -> None:
    """Fan-in is a star, and a star is a tree. Direction is what names it, not shape."""
    labels, edges_per = _shape([(2, 1), (3, 1), (4, 1), (5, 1)])

    shape = profile(labels, edges_per)

    assert shape.nodes == 5
    assert shape.edges == 4
    assert shape.components_with_a_cycle == 0
    assert shape.largest_component == 5


def test_a_two_way_pair_is_one_edge_rather_than_a_cycle() -> None:
    """A pays B and B pays A. Undirected that is a single edge, so it does not close a loop.

    Reciprocity is a directed property and this measurement is deliberately undirected, so a
    mutual pair must not be counted as a cycle. Counting it would put a cycle in every
    reciprocal pair in the graph and make the cycle archetype look common.
    """
    labels, edges_per = _shape([(1, 2), (2, 1)])

    shape = profile(labels, edges_per)

    assert shape.nodes == 2
    assert shape.edges == 1
    assert shape.components_with_a_cycle == 0


def test_disconnected_components_are_counted_separately() -> None:
    """A triangle and a separate edge: two components, one of which closes."""
    labels, edges_per = _shape([(1, 2), (2, 3), (3, 1), (10, 11)])

    shape = profile(labels, edges_per)

    assert shape.components == 2
    assert shape.largest_component == 3
    assert shape.median_component == 2.5
    assert shape.components_with_a_cycle == 1
    assert shape.cycle_share_outside_largest == 0.0


def test_the_giant_component_share_is_measured_against_every_node() -> None:
    """The number G8 turns on, on a fixture where it is checkable by hand."""
    labels, edges_per = _shape([(1, 2), (2, 3), (3, 4), (10, 11)])

    shape = profile(labels, edges_per)

    assert shape.largest_component == 4
    assert shape.largest_share == 4 / 6


def test_an_account_off_the_graph_reports_a_component_size_of_zero() -> None:
    """Never scored and never a counterparty, so it sits in no component at all."""
    frame = _txns([(1, 2), (2, 3)])
    rows, cols, nodes = undirected_edges(frame, WINDOW)
    labels, _ = component_labels(rows, cols, nodes)
    nodes_index = pd.Index(np.union1d(frame["account_from"], frame["account_to"]), name="account")
    sizes = np.bincount(labels)

    got = account_component_sizes(pd.Index([1, 999]), nodes_index, labels, sizes)

    assert list(got) == [3, 0]


def test_the_bands_split_on_the_declared_bounds() -> None:
    """Every account lands in exactly one band, and the shares sum to one."""
    sizes = np.array([0, 1, 2, FRAGMENT_MAX_NODES, 4, MAX_ARCHETYPE_NODES, MAX_ARCHETYPE_NODES + 1])

    table = band_table(sizes, "fixture")

    assert int(table["accounts"].sum()) == len(sizes)
    assert float(table["share"].sum()) == 1.0
    assert int(table.loc[table["band"].str.startswith("4 to"), "accounts"].iloc[0]) == 2
    assert int(table.loc[table["band"].str.startswith("over"), "accounts"].iloc[0]) == 1
