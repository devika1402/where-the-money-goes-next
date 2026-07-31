"""Ingest, currency normalisation, temporal windowing, and label derivation.

Implements F1 (currency-normalised flow) and F4 (leakage boundary) from PRD section 8.
Every shared quantity used here is imported from :mod:`src.definitions` rather than
restated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src import charts
from src.definitions import (
    DataSpec,
    SchemaError,
    build_windows,
    count_unscoreable_mules,
    fx_to_base,
    label_accounts,
    load_params,
    scoring_population,
    write_metrics,
)

LOGGER = logging.getLogger(__name__)

#: The source header, verbatim. ``Account`` appears twice, which is why pandas produces
#: ``Account.1`` for the receiving side and why nothing here is read by position.
RAW_HEADER: Final[str] = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)

#: Explicit rename of the auto-suffixed duplicate. Getting this pair backwards inverts
#: every directional feature and the model still trains.
_COLUMN_RENAMES: Final[dict[str, str]] = {
    "From Bank": "bank_from",
    "Account": "account_from_key",
    "To Bank": "bank_to",
    "Account.1": "account_to_key",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

_SOURCE_TIMESTAMP_FORMAT: Final[str] = "%Y/%m/%d %H:%M"


@dataclass(frozen=True)
class Ingested:
    """The loaded transaction table plus what was dropped getting there."""

    txns: pd.DataFrame
    account_keys: pd.Index
    n_self_transfers_dropped: int
    n_laundering_dropped: int
    n_rows_read: int


def _assert_header(path: Path) -> None:
    """Stop the build if the source header is not the one the data contract expects."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            found = handle.readline().strip()
    except OSError as exc:
        raise SchemaError(f"Could not read header from {path}") from exc

    if found != RAW_HEADER:
        raise SchemaError(
            f"Unexpected header in {path.name}.\n  expected: {RAW_HEADER}\n  found:    {found}"
        )


def load_transactions(path: Path, *, drop_self_transfers: bool = True) -> Ingested:
    """Read the source CSV into a canonical frame. Handles both load-time gotchas.

    Accounts are keyed on the bank and the account id together, because four account ids
    in the source appear under more than one bank. The pair is factorised to int32 across
    both directions at once, so the sending and receiving columns share one codebook and
    a node has the same id whichever end of an edge it sits on.
    """
    _assert_header(path)

    frame = pd.read_csv(
        path,
        dtype={
            "From Bank": "string",
            "Account": "string",
            "To Bank": "string",
            "Account.1": "string",
            "Amount Received": "float64",
            "Receiving Currency": "category",
            "Amount Paid": "float64",
            "Payment Currency": "category",
            "Payment Format": "category",
            "Is Laundering": "int8",
        },
    ).rename(columns=_COLUMN_RENAMES)

    n_rows_read = len(frame)

    # Parsed once, here. Never inside a groupby.
    frame["timestamp"] = pd.to_datetime(frame["Timestamp"], format=_SOURCE_TIMESTAMP_FORMAT)

    from_key = frame["bank_from"] + "|" + frame["account_from_key"]
    to_key = frame["bank_to"] + "|" + frame["account_to_key"]
    codes, uniques = pd.factorize(pd.concat([from_key, to_key], ignore_index=True))
    frame["account_from"] = codes[:n_rows_read].astype("int32")
    frame["account_to"] = codes[n_rows_read:].astype("int32")

    self_mask = frame["account_from"] == frame["account_to"]
    n_self = int(self_mask.sum())
    n_self_laundering = int(frame.loc[self_mask, "is_laundering"].sum())
    if drop_self_transfers:
        frame = frame.loc[~self_mask]
    else:
        n_self_laundering = 0

    keep = [
        "timestamp",
        "account_from",
        "account_to",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]
    return Ingested(
        txns=frame[keep].reset_index(drop=True),
        account_keys=pd.Index(uniques, name="account_key"),
        n_self_transfers_dropped=n_self,
        n_laundering_dropped=n_self_laundering,
        n_rows_read=n_rows_read,
    )


def validate_against_contract(loaded: Ingested, spec: DataSpec) -> None:
    """Assert the measured counts match the data contract. A mismatch stops the build."""
    checks = (
        ("rows", loaded.n_rows_read, spec.expected_rows),
        ("accounts", len(loaded.account_keys), spec.expected_accounts),
        ("self transfers", loaded.n_self_transfers_dropped, spec.expected_self_transfers),
        (
            "laundering rows",
            int(loaded.txns["is_laundering"].sum()) + loaded.n_laundering_dropped,
            spec.expected_laundering_rows,
        ),
    )
    mismatches = [
        f"{name}: expected {expected:,}, measured {measured:,}"
        for name, measured, expected in checks
        if measured != expected
    ]
    if mismatches:
        raise SchemaError("Data contract mismatch.\n  " + "\n  ".join(mismatches))


def normalise_amounts(txns: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Add ``amount_in_base_received`` and ``amount_in_base_paid``. Implements F1.

    The two sides are converted separately because a transaction may cross currencies,
    in which case the amount the receiver gets and the amount the sender pays are
    different numbers in different units.
    """
    out = txns.copy()
    out["amount_in_base_received"] = fx_to_base(
        out["amount_received"], out["receiving_currency"], rates
    )
    out["amount_in_base_paid"] = fx_to_base(out["amount_paid"], out["payment_currency"], rates)
    return out


def account_flows(txns: pd.DataFrame) -> pd.DataFrame:
    """Total inflow and outflow per account, in base currency. Implements F1.

    The receiving account's inflow uses the received amount. The sending account's
    outflow uses the paid amount. Reading one column for both sides is silent on
    same-currency rows and wrong on the 1.4% that cross currencies.
    """
    inflow = txns.groupby("account_to")["amount_in_base_received"].sum()
    outflow = txns.groupby("account_from")["amount_in_base_paid"].sum()

    accounts = inflow.index.union(outflow.index)
    accounts.name = "account"
    return pd.DataFrame(
        {
            "inflow": inflow.reindex(accounts, fill_value=0.0),
            "outflow": outflow.reindex(accounts, fill_value=0.0),
        }
    )


def feature_rows(txns: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Transactions a feature is allowed to see. Implements F4.

    Strictly before the cutoff. A transaction at exactly the cutoff belongs to the label
    window, so admitting it leaks the label into the features. The inequality being
    strict is the entire content of this function.
    """
    return txns.loc[txns["timestamp"] < cutoff]


def daily_volume_table(txns: pd.DataFrame) -> pd.DataFrame:
    """Transactions and laundering share per calendar day, over the whole file.

    This is the evidence behind D8 and it is computed over every day the file holds rather
    than the usable span, because the point of the table is what happens outside it.
    """
    day = txns["timestamp"].dt.floor("D")
    daily = txns.groupby(day).agg(
        transactions=("is_laundering", "size"),
        laundering=("is_laundering", "sum"),
    )
    daily["laundering_share"] = daily["laundering"] / daily["transactions"]
    daily.index.name = "day"
    return daily


def _daily_for_metrics(daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Serialise the daily table with an explicit date key, so the report can read it back."""
    return {
        str(day)[:10]: {
            "transactions": int(values["transactions"]),
            "laundering": int(values["laundering"]),
            "laundering_share": float(values["laundering_share"]),
        }
        for day, values in daily.iterrows()
    }


def _log_observed_span(txns: pd.DataFrame) -> None:
    """Report the span the PRD could not verify before the download."""
    observed_min = txns["timestamp"].min()
    observed_max = txns["timestamp"].max()
    span_days = (observed_max - observed_min) / pd.Timedelta(days=1)
    LOGGER.info("observed span: %s to %s (%.2f days)", observed_min, observed_max, span_days)


def main() -> None:
    """Load the source, assert the data contract, and write the canonical table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    source = params.paths.raw_dir / params.data.trans_file

    LOGGER.info("reading %s", source)
    loaded = load_transactions(source, drop_self_transfers=params.data.drop_self_transfers)
    LOGGER.info("rows read: %d", loaded.n_rows_read)
    LOGGER.info("accounts (bank and id together): %d", len(loaded.account_keys))
    LOGGER.info(
        "self transfers dropped: %d (%.1f%% of rows), carrying %d laundering rows",
        loaded.n_self_transfers_dropped,
        100.0 * loaded.n_self_transfers_dropped / loaded.n_rows_read,
        loaded.n_laundering_dropped,
    )

    validate_against_contract(loaded, params.data)
    LOGGER.info("data contract: all counts match")

    _log_observed_span(loaded.txns)

    txns = normalise_amounts(loaded.txns, params.fx_to_base)
    usable = txns.loc[
        (txns["timestamp"] >= params.split.data_start) & (txns["timestamp"] < params.split.data_end)
    ].reset_index(drop=True)
    LOGGER.info(
        "usable span %s to %s holds %d rows (%.1f%% of the file)",
        params.split.data_start,
        params.split.data_end,
        len(usable),
        100.0 * len(usable) / len(txns),
    )

    daily = daily_volume_table(txns)
    charts.daily_volume(
        daily,
        cutoff=params.split.data_end,
        costs=params.costs,
        out=params.paths.figures_dir / "daily_volume.png",
    )
    LOGGER.info("wrote %s", params.paths.figures_dir / "daily_volume.png")

    measured: dict[str, Any] = {
        "rows_read": loaded.n_rows_read,
        "accounts": len(loaded.account_keys),
        "self_transfers_dropped": loaded.n_self_transfers_dropped,
        "laundering_in_self_transfers": loaded.n_laundering_dropped,
        "observed_start": str(loaded.txns["timestamp"].min()),
        "observed_end": str(loaded.txns["timestamp"].max()),
        "usable_rows": len(usable),
        "usable_start": str(params.split.data_start),
        "usable_end": str(params.split.data_end),
        "daily": _daily_for_metrics(daily),
        "windows": {},
    }

    windows = build_windows(params.split, params.windows.label_window_days)
    for window in windows:
        population = scoring_population(usable, window)
        labels = label_accounts(usable, window.label_start, window.label_end, population=population)
        positives = int(labels["is_mule"].sum())
        unscoreable = count_unscoreable_mules(usable, window, population)
        measured["windows"][window.name] = {
            "feature_start": str(window.feature_start),
            "feature_end": str(window.feature_end),
            "label_start": str(window.label_start),
            "label_end": str(window.label_end),
            "population": len(population),
            "mules": positives,
            "base_rate": positives / len(population),
            "unscoreable_mules": unscoreable,
            "mules_in_window": positives + unscoreable,
            "reachability_ceiling": positives / (positives + unscoreable),
        }
        LOGGER.info(
            "%-5s features [%s, %s) | labels [%s, %s) | population %7d | mules %4d "
            "(%.4f%%) | unscoreable mules %3d",
            window.name,
            window.feature_start.date(),
            window.feature_end.date(),
            window.label_start.date(),
            window.label_end.date(),
            len(population),
            positives,
            100.0 * positives / len(population),
            unscoreable,
        )

    params.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    destination = params.paths.interim_dir / "canonical_txns.parquet"
    usable.to_parquet(destination, index=False)
    LOGGER.info("wrote %s", destination)

    write_metrics(params.paths.reports_dir / "metrics_data.json", measured)
    LOGGER.info("wrote %s", params.paths.reports_dir / "metrics_data.json")


if __name__ == "__main__":
    main()
