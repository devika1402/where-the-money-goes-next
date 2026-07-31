"""Worked-example test for F5 (Population Stability Index), plus the two rules it depends on.

The numbers come from PRD section 8 F5 and are used exactly as written there. The binning
block asserts the specific wrong value the edge case produces: re-binning the comparison
period by its own quantiles collapses PSI to exactly zero on a pair of samples that share no
values at all, so the fixed-edge rule is enforced by the test rather than by a comment.

**One assertion the PRD asks for is not made, because it cannot be true.** F5 edge case (c)
says PSI is not symmetric and instructs a test that ``psi(a, b) != psi(b, a)`` on the worked
example. On two proportion vectors the formula is symmetric by construction, so the assertion
here is equality. The asymmetry the PRD is reaching for is real and lives one level up, in
where the bin edges come from, and it has its own test below. See D32.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.definitions import SchemaError
from src.monitoring import (
    DEGENERATE_EDGE_COUNT,
    drift_row,
    empty_bin_count,
    feature_drift,
    ks_statistic,
    proportions,
    psi,
    psi_from_values,
    quantile_edges,
    stabilise,
)

EPSILON = 1e-6
BINS = 3

# PRD section 8 F5, three quantile bins from the reference period.
P_REF = [0.40, 0.40, 0.20]
P_COMP = [0.60, 0.40, 0.00]


def test_f5_matches_the_worked_example() -> None:
    """PSI = 0.081092 + 0.000000 + 2.441204 = 2.5223. PRD section 8 F5."""
    assert psi(P_REF, P_COMP, epsilon=EPSILON) == pytest.approx(2.5223, abs=0.001)


def test_f5_epsilon_replaces_the_zero_bin_and_renormalises() -> None:
    """Edge case (b). The zero bin becomes 1e-6 and the vector still sums to 1.

    Without the renormalisation the proportions sum to 1.000001, which is a silent error
    that grows with the number of empty bins.
    """
    stabilised = stabilise(P_COMP, epsilon=EPSILON)

    assert stabilised.sum() == pytest.approx(1.0)
    assert stabilised[2] == pytest.approx(1e-6, rel=1e-3)
    assert stabilised[0] == pytest.approx(0.5999994, abs=1e-7)


def test_f5_of_a_distribution_against_itself_is_exactly_zero() -> None:
    """Not approximately zero. Every term is (p - p) times log(p/p), so the sum is 0.0."""
    assert psi(P_REF, P_REF, epsilon=EPSILON) == 0.0
    assert psi(P_COMP, P_COMP, epsilon=EPSILON) == 0.0


def test_f5_is_symmetric_on_proportions_which_contradicts_the_specification() -> None:
    """The PRD says to assert inequality here. The formula makes that impossible.

    PSI is the sum over bins of (q - p) ln(q / p). Swapping p and q negates both factors,
    so the product is unchanged and the sum is identical to the last bit. The specification's
    edge case (c) is right that argument order matters and wrong about where. See D32.
    """
    forward = psi(P_REF, P_COMP, epsilon=EPSILON)
    backward = psi(P_COMP, P_REF, epsilon=EPSILON)

    assert forward == backward
    assert forward - backward == 0.0


# --------------------------------------------------------------------------------------
# Where argument order actually bites: the bin edges come from the reference period.

REF_VALUES = np.arange(1, 13, dtype=float)
#: Disjoint from the reference and free of ties, so its own quantile bins split it exactly
#: into thirds and the re-binning bug produces exactly zero rather than merely a small number.
DISJOINT_VALUES = np.arange(20, 32, dtype=float)
#: Overlapping and tied, which is what makes the two directions differ.
SKEWED_VALUES = np.array([1, 1, 1, 2, 2, 2, 3, 3, 20, 21, 22, 23], dtype=float)
#: Spread across all three of the reference's bins, so nothing empties and PSI is a magnitude
#: rather than a flag.
FILLED_VALUES = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 12, 12], dtype=float)


def test_bin_edges_come_from_the_reference_and_are_unbounded_at_both_ends() -> None:
    """A comparison value outside the reference range has to land somewhere. Edge case (a)."""
    edges = quantile_edges(REF_VALUES, bins=BINS)

    assert len(edges) == BINS + 1
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert proportions(DISJOINT_VALUES, edges).sum() == pytest.approx(1.0)


def test_applying_the_reference_edges_to_the_comparison_measures_the_drift() -> None:
    """Every comparison value sits above the reference's top bin edge, so it is [0, 0, 1]."""
    edges = quantile_edges(REF_VALUES, bins=BINS)

    assert list(proportions(REF_VALUES, edges)) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert list(proportions(DISJOINT_VALUES, edges)) == pytest.approx([0.0, 0.0, 1.0])
    assert psi_from_values(
        REF_VALUES, DISJOINT_VALUES, bins=BINS, epsilon=EPSILON
    ) == pytest.approx(9.210313, abs=1e-5)


def test_rebinning_the_comparison_independently_collapses_psi_to_zero() -> None:
    """Edge case (a), and the specific wrong value it produces.

    The two samples share no values whatsoever, so the honest answer is 9.21. Binning each
    period by its own quantiles gives both of them a third of their mass per bin, and PSI
    reports exactly 0.0. This is the bug the fixed-edge rule exists to prevent, and it fails
    silently in the direction that says nothing has drifted.
    """
    correct = psi_from_values(REF_VALUES, DISJOINT_VALUES, bins=BINS, epsilon=EPSILON)

    own_edges = quantile_edges(DISJOINT_VALUES, bins=BINS)
    rebinned = psi(
        proportions(REF_VALUES, quantile_edges(REF_VALUES, bins=BINS)),
        proportions(DISJOINT_VALUES, own_edges),
        epsilon=EPSILON,
    )

    assert correct == pytest.approx(9.210313, abs=1e-5)
    assert rebinned == 0.0


def test_psi_from_values_is_not_symmetric_because_the_edges_move() -> None:
    """The asymmetry F5 edge case (c) is reaching for, in the place it actually lives.

    Same two samples, opposite roles. Forward, the reference's quantiles put two thirds of
    the comparison into one bin. Reversed, the comparison's own quantiles spread the
    reference across all three. 4.47 against 0.24, from identical data.
    """
    forward = psi_from_values(REF_VALUES, SKEWED_VALUES, bins=BINS, epsilon=EPSILON)
    reversed_ = psi_from_values(SKEWED_VALUES, REF_VALUES, bins=BINS, epsilon=EPSILON)

    assert forward == pytest.approx(4.470002, abs=1e-5)
    assert reversed_ == pytest.approx(0.239181, abs=1e-5)
    assert forward != reversed_


def test_an_empty_period_is_refused_rather_than_scored() -> None:
    """A window with no observations has no distribution, so PSI has nothing to compare."""
    with pytest.raises(SchemaError, match="no observations"):
        quantile_edges(np.array([], dtype=float), bins=BINS)

    with pytest.raises(SchemaError, match="no observations"):
        proportions(np.array([], dtype=float), quantile_edges(REF_VALUES, bins=BINS))


def test_proportion_vectors_of_different_lengths_are_refused() -> None:
    """Comparing three bins against four is a binning error, not a drift measurement."""
    with pytest.raises(SchemaError, match="same number of bins"):
        psi([0.5, 0.5], [0.3, 0.3, 0.4], epsilon=EPSILON)


# --------------------------------------------------------------------------------------
# KS on the score distribution, the second half of PRD section 9 Step 10.


def test_ks_of_a_distribution_against_itself_is_zero() -> None:
    """The same sample twice has no maximum gap between its two step functions."""
    assert ks_statistic(REF_VALUES, REF_VALUES) == pytest.approx(0.0)


def test_ks_of_two_disjoint_samples_is_one() -> None:
    """Every comparison value exceeds every reference value, so the gap reaches its maximum."""
    assert ks_statistic(REF_VALUES, DISJOINT_VALUES) == pytest.approx(1.0)


def test_ks_is_symmetric_where_psi_from_values_is_not() -> None:
    """The contrast worth stating: KS needs no binning, so it needs no reference period.

    That makes it the cheaper check and the less informative one. It cannot say which part
    of the distribution moved, which is the whole reason PSI is reported per bin beside it.
    """
    forward = ks_statistic(REF_VALUES, SKEWED_VALUES)
    backward = ks_statistic(SKEWED_VALUES, REF_VALUES)

    assert forward == backward


def test_a_drift_table_reports_one_row_per_feature() -> None:
    """The shape the report reads. One PSI per feature, computed on that feature's own edges."""
    reference = pd.DataFrame({"a": REF_VALUES, "b": REF_VALUES})
    comparison = pd.DataFrame({"a": REF_VALUES, "b": DISJOINT_VALUES})

    table = feature_drift(reference, comparison, bins=BINS, epsilon=EPSILON)

    assert list(table["feature"]) == ["a", "b"]
    assert float(table.loc[table["feature"] == "a", "psi"].iloc[0]) == 0.0
    assert float(table.loc[table["feature"] == "b", "psi"].iloc[0]) == pytest.approx(
        9.210313, abs=1e-5
    )


# --------------------------------------------------------------------------------------
# Whether a PSI value is a magnitude or a flag. D34.


def test_an_emptied_bin_makes_the_psi_magnitude_a_function_of_epsilon() -> None:
    """The number every large PSI in this project has to be read against.

    Two disjoint samples empty two of the reference's three bins. The emptied term is
    ``(epsilon - p_ref) * ln(epsilon / p_ref)``, so the reported value moves with a parameter
    nobody measured: 4.6 at 1e-3 against 13.8 at 1e-9 on identical data. Without an empty bin
    the value does not move at all, which the second half asserts.
    """
    loose = psi_from_values(REF_VALUES, DISJOINT_VALUES, bins=BINS, epsilon=1e-3)
    tight = psi_from_values(REF_VALUES, DISJOINT_VALUES, bins=BINS, epsilon=1e-9)

    assert loose == pytest.approx(4.591382, abs=1e-5)
    assert tight == pytest.approx(13.815511, abs=1e-5)
    assert tight > 2 * loose

    reference_edges = quantile_edges(REF_VALUES, bins=BINS)
    assert (
        empty_bin_count(
            proportions(REF_VALUES, reference_edges),
            proportions(FILLED_VALUES, reference_edges),
        )
        == 0
    )
    settled_loose = psi_from_values(REF_VALUES, FILLED_VALUES, bins=BINS, epsilon=1e-3)
    settled_tight = psi_from_values(REF_VALUES, FILLED_VALUES, bins=BINS, epsilon=1e-9)

    assert settled_loose == pytest.approx(settled_tight, abs=1e-12)
    assert settled_loose == pytest.approx(0.115525, abs=1e-5)


def test_a_drift_row_says_how_many_bins_emptied() -> None:
    """PSI on its own cannot be read. The row carries what is needed to read it."""
    row = drift_row(REF_VALUES, DISJOINT_VALUES, bins=BINS, epsilon=EPSILON)

    assert row["empty_bins"] == 2
    assert row["epsilon_dependent"] is True
    assert row["psi"] == pytest.approx(9.210313, abs=1e-5)


def test_a_reference_too_constant_to_bin_reports_no_drift_however_far_it_moved() -> None:
    """The blind spot of quantile binning, and the reason the row counts distinct edges.

    A feature that is one value for almost every account in the reference collapses every
    quantile onto that value, so both periods land wholly in one bin and PSI is exactly 0.
    That zero means the measurement has no resolution, not that nothing moved. It happens on
    real data here: `reciprocity` is 0 for 99.52% of the training window.
    """
    constant = np.zeros(100)
    constant[-1] = 5.0
    moved = np.full(100, 900.0)

    row = drift_row(constant, moved, bins=BINS, epsilon=EPSILON)

    assert row["psi"] == 0.0
    assert int(row["distinct_reference_edges"]) <= DEGENERATE_EDGE_COUNT
