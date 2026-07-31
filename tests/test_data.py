"""Worked-example tests for F1 (currency normalisation) and F4 (leakage boundary).

Both examples come from PRD section 8 and the numbers are used exactly as written there.
Every test asserts the correct value and the specific wrong value that its edge case
produces, so a regression back to the wrong form is caught rather than passing quietly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import account_flows, feature_rows, load_transactions, normalise_amounts
from src.definitions import (
    SchemaError,
    UnknownCurrencyError,
    Window,
    count_unscoreable_mules,
    label_accounts,
    load_params,
    scoring_population,
)

# F1 fixes USD to EUR at 0.90. The rate is read from the shared table rather than restated
# here, so the test fails if the table and the worked example ever disagree.
RATES = load_params().fx_to_base

ACCOUNT_A = 1
LABEL_START = pd.Timestamp("2024-03-08 00:00:00")
LABEL_END = pd.Timestamp("2024-03-15 00:00:00")
ONLY_ACCOUNT_A = pd.Index([ACCOUNT_A], name="account")


def test_the_shared_fx_table_holds_the_rate_the_worked_example_assumes() -> None:
    """F1 is only meaningful if the shared table still says USD to EUR is 0.90."""
    assert RATES["US Dollar"] == 0.90
    assert RATES["Euro"] == 1.0


# --------------------------------------------------------------------------------------
# F1. Currency-normalised flow
# --------------------------------------------------------------------------------------


def _f1_frame() -> pd.DataFrame:
    """The F1 worked example as a transaction frame.

    Account A receives 100.00 USD and 50.00 EUR, and pays out 120.00 USD. Rows 1 and 3
    cross currencies, which is the condition under which reading the wrong amount column
    produces a different number instead of an identical one.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2022-09-01 00:00", "2022-09-01 01:00", "2022-09-01 02:00"]
            ),
            "account_from": [2, 3, ACCOUNT_A],
            "account_to": [ACCOUNT_A, ACCOUNT_A, 4],
            "amount_received": [100.00, 50.00, 100.00],
            "receiving_currency": ["US Dollar", "Euro", "Euro"],
            "amount_paid": [95.00, 50.00, 120.00],
            "payment_currency": ["Euro", "Euro", "US Dollar"],
        }
    )


def test_f1_inflow_and_outflow_match_the_worked_example() -> None:
    """inflow = (100.00 x 0.90) + 50.00 = 140.00. outflow = 120.00 x 0.90 = 108.00."""
    flows = account_flows(normalise_amounts(_f1_frame(), RATES))

    assert flows.loc[ACCOUNT_A, "inflow"] == pytest.approx(140.00, abs=0.005)
    assert flows.loc[ACCOUNT_A, "outflow"] == pytest.approx(108.00, abs=0.005)


def test_f1_reading_the_wrong_amount_column_produces_the_wrong_numbers() -> None:
    """The receiver's inflow uses amount_received. The sender's outflow uses amount_paid.

    Exchanging the two columns gives inflow 145.00 and outflow 100.00. Both differ from
    the correct values, so the swap cannot pass unnoticed on cross-currency rows.
    """
    frame = _f1_frame().rename(
        columns={
            "amount_received": "amount_paid",
            "amount_paid": "amount_received",
            "receiving_currency": "payment_currency",
            "payment_currency": "receiving_currency",
        }
    )
    flows = account_flows(normalise_amounts(frame, RATES))

    assert flows.loc[ACCOUNT_A, "inflow"] == pytest.approx(145.00, abs=0.005)
    assert flows.loc[ACCOUNT_A, "outflow"] == pytest.approx(100.00, abs=0.005)


def test_f1_swapping_the_account_columns_inverts_direction() -> None:
    """Gotcha 1. The source CSV names both account columns the same, so a positional read
    silently reverses every directional feature.

    Reversing direction does not simply exchange the two totals, because inflow reads
    amount_received and outflow reads amount_paid whichever way the edge points. Account A
    becomes the sender on rows 1 and 2 and the receiver on row 3, giving inflow 100.00 and
    outflow 145.00. Both differ from the correct 140.00 and 108.00.
    """
    frame = _f1_frame().rename(columns={"account_from": "account_to", "account_to": "account_from"})
    flows = account_flows(normalise_amounts(frame, RATES))

    assert flows.loc[ACCOUNT_A, "inflow"] == pytest.approx(100.00, abs=0.005)
    assert flows.loc[ACCOUNT_A, "outflow"] == pytest.approx(145.00, abs=0.005)


def test_f1_summing_without_conversion_gives_the_uncorrected_total() -> None:
    """Adding raw amounts across currencies gives 150.00 rather than 140.00."""
    frame = _f1_frame()
    uncorrected = frame.loc[frame["account_to"] == ACCOUNT_A, "amount_received"].sum()

    assert uncorrected == pytest.approx(150.00, abs=0.005)


def test_f1_an_unknown_currency_raises_rather_than_defaulting() -> None:
    """A missing rate is a data problem and stops the build. No silent fallback to 1.0."""
    frame = _f1_frame()
    frame.loc[0, "receiving_currency"] = "Klingon Darsek"

    with pytest.raises(UnknownCurrencyError, match="Klingon Darsek"):
        normalise_amounts(frame, RATES)


# --------------------------------------------------------------------------------------
# F4. Leakage boundary
# --------------------------------------------------------------------------------------


def _f4_frame() -> pd.DataFrame:
    """The F4 worked example: three timestamps either side of a cutoff at 2024-03-08.

    The laundering flag sits on the 00:00:00 row, which belongs to the label window.
    Admitting that row into the features leaks the label.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-03-07 23:59:59",
                    "2024-03-08 00:00:00",
                    "2024-03-08 00:00:01",
                ]
            ),
            "account_from": [2, 2, 2],
            "account_to": [ACCOUNT_A, ACCOUNT_A, ACCOUNT_A],
            "amount_received": [10.0, 20.0, 30.0],
            "receiving_currency": ["Euro", "Euro", "Euro"],
            "amount_paid": [10.0, 20.0, 30.0],
            "payment_currency": ["Euro", "Euro", "Euro"],
            "is_laundering": [0, 1, 0],
        }
    )


def test_f4_features_use_only_transactions_strictly_before_the_cutoff() -> None:
    """Strict inequality. The 23:59:59 row is the only one a feature may see."""
    rows = feature_rows(_f4_frame(), cutoff=LABEL_START)

    assert len(rows) == 1
    assert rows["timestamp"].iloc[0] == pd.Timestamp("2024-03-07 23:59:59")


def test_f4_an_inclusive_boundary_would_admit_a_second_row() -> None:
    """The specific wrong answer. An off-by-one to <= gives 2 rows and leaks the label."""
    frame = _f4_frame()
    inclusive = frame[frame["timestamp"] <= LABEL_START]

    assert len(inclusive) == 2
    assert len(feature_rows(frame, cutoff=LABEL_START)) == 1


def test_f4_the_label_window_carries_the_laundering_flag() -> None:
    """The receiving account is labelled positive from the 00:00:00 row."""
    labels = label_accounts(
        _f4_frame(), start=LABEL_START, end=LABEL_END, population=ONLY_ACCOUNT_A
    )

    assert labels.loc[ACCOUNT_A, "is_mule"] == 1


def test_f4_the_pre_cutoff_rows_alone_label_negative() -> None:
    """Everything a feature is allowed to see carries no laundering flag.

    This is what makes the boundary matter. If the feature rows could label the account
    positive on their own, the cutoff would not be separating anything.
    """
    visible = feature_rows(_f4_frame(), cutoff=LABEL_START)
    labels = label_accounts(
        visible,
        start=pd.Timestamp("2024-03-01"),
        end=LABEL_START,
        population=ONLY_ACCOUNT_A,
    )

    assert labels.loc[ACCOUNT_A, "is_mule"] == 0


# --------------------------------------------------------------------------------------
# D9. The scoring population, pinned.
#
# Every base rate and every precision@k divides by this choice. Three defensible
# populations existed for the test window and they differed by 49%, so it is now an
# explicit argument rather than something derived from whatever frame a caller passes.
# --------------------------------------------------------------------------------------


def _population_frame() -> pd.DataFrame:
    """Account 1 receives in the feature window. Account 7 first appears in the labels.

    Account 7 is a mule with no inbound history, so it has no features and cannot be
    scored. Counting it would put an undetectable positive into the denominator.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-03-05 12:00:00", "2024-03-09 12:00:00", "2024-03-09 13:00:00"]
            ),
            "account_from": [2, 2, 3],
            "account_to": [ACCOUNT_A, ACCOUNT_A, 7],
            "is_laundering": [0, 1, 1],
        }
    )


def test_the_scoring_population_is_receivers_active_in_the_feature_window() -> None:
    """D9. Account 7 never receives before the cutoff, so it is not scoreable."""
    window = Window(
        name="test",
        feature_start=pd.Timestamp("2024-03-04"),
        feature_end=LABEL_START,
        label_start=LABEL_START,
        label_end=LABEL_END,
    )

    assert list(scoring_population(_population_frame(), window)) == [ACCOUNT_A]


def test_the_label_index_is_exactly_the_population_it_was_given() -> None:
    """The denominator is the population, never the accounts that happen to be in the frame.

    Account 99 never appears in the data at all and must still be present as a negative.
    Account 7 is a mule in the window and must be absent, because it is not scoreable.
    """
    population = pd.Index([ACCOUNT_A, 99], name="account")
    labels = label_accounts(
        _population_frame(), start=LABEL_START, end=LABEL_END, population=population
    )

    assert list(labels.index) == [ACCOUNT_A, 99]
    assert labels.loc[ACCOUNT_A, "is_mule"] == 1
    assert labels.loc[99, "is_mule"] == 0
    assert 7 not in labels.index


def test_unscoreable_mules_are_counted_rather_than_silently_dropped() -> None:
    """Excluding a positive is defensible. Not knowing how many were excluded is not."""
    window = Window(
        name="test",
        feature_start=pd.Timestamp("2024-03-04"),
        feature_end=LABEL_START,
        label_start=LABEL_START,
        label_end=LABEL_END,
    )
    frame = _population_frame()
    population = scoring_population(frame, window)

    assert count_unscoreable_mules(frame, window, population) == 1


# --------------------------------------------------------------------------------------
# Load-time gotchas. Both cost an hour each if hit cold, and neither is visible later.
# --------------------------------------------------------------------------------------

RAW_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
A_TO_B = "2022/09/01 00:20,010,AAA,020,BBB,100.00,Euro,100.00,Euro,ACH,0"


def _write_csv(tmp_path: Path, rows: list[str], header: str = RAW_HEADER) -> Path:
    """Write a source-shaped CSV, duplicate account header included."""
    path = tmp_path / "trans.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_ingest_maps_the_duplicate_account_headers_to_the_right_direction(
    tmp_path: Path,
) -> None:
    """Gotcha 1. Both account columns share a header, so pandas yields Account.1.

    Getting this backwards inverts every directional feature while the model trains
    happily, which is why the mapping is asserted against the original strings rather
    than trusted.
    """
    result = load_transactions(_write_csv(tmp_path, [A_TO_B]))
    row = result.txns.iloc[0]

    assert result.account_keys[row["account_from"]] == "010|AAA"
    assert result.account_keys[row["account_to"]] == "020|BBB"


def test_ingest_rejects_a_header_that_does_not_match_the_data_contract(
    tmp_path: Path,
) -> None:
    """A renamed or reordered column stops the build instead of shifting every field."""
    bad_header = RAW_HEADER.replace("Amount Received", "Amount Recieved")

    with pytest.raises(SchemaError, match="header"):
        load_transactions(_write_csv(tmp_path, [A_TO_B], header=bad_header))


def test_ingest_drops_self_transfers_and_reports_the_count(tmp_path: Path) -> None:
    """11.6% of the source moves money from an account to itself.

    Left in, each one raises inflow and outflow by the same amount, which pulls the
    pass-through ratio toward 1.0 for accounts doing nothing suspicious.
    """
    rows = [
        "2022/09/01 00:20,010,AAA,010,AAA,100.00,Euro,100.00,Euro,Reinvestment,0",
        A_TO_B,
    ]
    result = load_transactions(_write_csv(tmp_path, rows))

    assert result.n_self_transfers_dropped == 1
    assert len(result.txns) == 1


def test_ingest_keys_accounts_on_bank_and_account_together(tmp_path: Path) -> None:
    """D6. The same account id under two banks is two nodes.

    Four account ids in the source appear under more than one bank. Keying on the id
    alone merges them and gives the merged node the combined flow of two real accounts.
    """
    rows = [
        A_TO_B,
        "2022/09/01 00:21,030,AAA,020,BBB,100.00,Euro,100.00,Euro,ACH,0",
    ]
    result = load_transactions(_write_csv(tmp_path, rows))

    assert result.txns["account_from"].nunique() == 2
    assert result.txns["account_to"].nunique() == 1


def test_ingest_factorises_accounts_to_int32(tmp_path: Path) -> None:
    """Gotcha 2. 5M rows of string ids sit near 2GB and slow every groupby."""
    result = load_transactions(_write_csv(tmp_path, [A_TO_B]))

    assert result.txns["account_from"].dtype == "int32"
    assert result.txns["account_to"].dtype == "int32"


def test_ingest_parses_the_timestamp_once(tmp_path: Path) -> None:
    """The source format is %Y/%m/%d %H:%M and is parsed at load, never in a groupby."""
    result = load_transactions(_write_csv(tmp_path, [A_TO_B]))

    assert result.txns["timestamp"].dtype == "datetime64[ns]"
    assert result.txns["timestamp"].iloc[0] == pd.Timestamp("2022-09-01 00:20:00")
