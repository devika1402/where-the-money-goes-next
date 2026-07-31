"""Figures. Presentation only, and it computes nothing that a stage module has not already.

Every stage that owns a number also owns the chart of that number, and hands the finished
frame here to be drawn. That keeps the layering the PRD asks for: the module that measured
a quantity is the module that can be traced back to for it.

The palette is the validated categorical default, slots 1 to 3 in fixed order, checked with
the accompanying validator on both the adjacent and the all-pairs lists. Worst all-pairs
separation is 9.2 under deutan simulation and 24.0 for normal vision, both clear of their
floors. Slot 3 sits at 2.74:1 against the surface, which is below the 3:1 contrast bar, so
every chart here carries the values in its legend and the report carries the same numbers
as a table. That is the documented relief for a sub-3:1 slot and it is not optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.definitions import Costs

matplotlib.use("Agg")

#: Categorical slots 1 to 3, in the fixed order the validator was run against. Never cycled.
SERIES: Final[dict[str, str]] = {
    "rules": "#2a78d6",
    "logistic": "#eb6834",
    "xgboost": "#1baf7a",
}

#: Chart chrome. Text never wears a series colour.
SURFACE: Final[str] = "#fcfcfb"
INK_PRIMARY: Final[str] = "#0b0b0b"
INK_SECONDARY: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#898781"
GRIDLINE: Final[str] = "#e1e0d9"
BASELINE: Final[str] = "#c3c2b7"

#: matplotlib ships DejaVu Sans, so this resolves everywhere without a warning.
#: Substitute a brand UI sans here if the figures are ever re-skinned.
FONT: Final[list[str]] = ["DejaVu Sans", "sans-serif"]

#: Below this a share tick needs two decimals to say anything.
ONE_PERCENT: Final[float] = 0.01
THOUSAND: Final[float] = 1e3
MILLION: Final[float] = 1e6


def _money(value: float) -> str:
    """Axis tick for a euro amount that spans thousands to tens of millions."""
    if value == 0:
        return "0"
    if abs(value) >= MILLION:
        return f"{value / MILLION:+.0f}M"
    return f"{value / THOUSAND:+.0f}k"


def _style(axes: Any) -> None:
    """Recessive grid and axes, so the marks carry the chart rather than the furniture."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    axes.xaxis.label.set_color(INK_SECONDARY)
    axes.yaxis.label.set_color(INK_SECONDARY)


def _finish(figure: Any, costs: Costs, destination: Path, note: str = "") -> None:
    """Write the figure with the five active assumptions in its footer. D2 requires this."""
    footer = costs.as_footer()
    if note:
        footer = f"{footer}  |  {note}"
    figure.text(0.008, 0.012, footer, fontsize=6.5, color=INK_MUTED)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def daily_volume(daily: pd.DataFrame, *, cutoff: pd.Timestamp, costs: Costs, out: Path) -> None:
    """The evidence behind D8, in two panels sharing one x axis.

    Total volume and laundering share are different measures on different scales, so they
    get a panel each rather than a second y axis. A dual-axis chart invites the reader to
    infer a relationship from where two lines happen to cross.
    """
    plt.rcParams["font.family"] = FONT
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 5.4), sharex=True, height_ratios=[1.4, 1.0]
    )

    kept = daily.index < cutoff
    for axes, column, label in (
        (top, "transactions", "transactions per day"),
        (bottom, "laundering_share", "laundering share of the day"),
    ):
        _style(axes)
        axes.bar(
            daily.index[kept],
            daily.loc[kept, column],
            color=SERIES["rules"],
            width=0.72,
            zorder=2,
        )
        axes.bar(
            daily.index[~kept],
            daily.loc[~kept, column],
            color=BASELINE,
            width=0.72,
            zorder=2,
        )
        axes.set_ylabel(label)
        axes.axvline(cutoff, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=3)

    # Both panels are log. The share panel spans 0.03% to 100%, so on a linear axis the
    # included days are sub-pixel and read as missing data rather than as nearly zero.
    top.set_yscale("log")
    bottom.set_yscale("log")
    bottom.yaxis.set_major_formatter(lambda v, _: f"{v:.2%}" if v < ONE_PERCENT else f"{v:.0%}")
    top.set_title(
        "Background traffic stops on 09-11 while injected laundering keeps completing",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    top.annotate(
        "excluded from the usable span (D8)",
        xy=(cutoff, daily["transactions"].max() * 0.35),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
        va="center",
    )
    bottom.set_xlabel("date")
    figure.autofmt_xdate(rotation=45, ha="right")
    _finish(figure, costs, out, "grey bars are excluded days")


def pr_curve(
    curves: dict[str, tuple[Any, Any]],
    summary: pd.DataFrame,
    *,
    base_rate: float,
    window: str,
    costs: Costs,
    out: Path,
) -> None:
    """All three scorers on one axis, with the base rate drawn as the floor.

    Without that floor line a precision-recall curve at a 0.28% base rate reads as though
    a scorer has found something when it has found nothing.
    """
    plt.rcParams["font.family"] = FONT
    figure, axes = plt.subplots(figsize=(7.4, 5.0))
    _style(axes)

    lookup = summary.set_index("scorer")
    for name, (recall, precision) in curves.items():
        row = lookup.loc[name]
        axes.plot(
            recall,
            precision,
            color=SERIES[name],
            linewidth=2.0,
            zorder=3,
            label=f"{name}  PR-AUC {row['pr_auc']:.4f}  [{row['low']:.4f}, {row['high']:.4f}]",
        )
    axes.axhline(
        base_rate,
        color=INK_MUTED,
        linewidth=1.2,
        linestyle="--",
        zorder=2,
        label=f"base rate {base_rate:.4%}, what finding nothing looks like",
    )

    axes.set_yscale("log")
    axes.set_xlabel("recall")
    axes.set_ylabel("precision")
    axes.set_title(
        f"Precision against recall, {window} window, with 95% bootstrap intervals",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    legend = axes.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
    legend.get_frame().set_edgecolor(GRIDLINE)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _finish(figure, costs, out)


def reachability_funnel(stages: pd.DataFrame, *, window: str, costs: Costs, out: Path) -> None:
    """The two arithmetic ceilings and what is left after them, on one linear scale.

    Every bar is a share of the same denominator, the mule accounts active in the label
    window, so the last bar is a sliver. A funnel drawn stage-relative would make each step
    look survivable and hide that the compounding is what matters.

    ``share`` arrives measured rather than divided here, so this stays a drawing.
    """
    plt.rcParams["font.family"] = FONT
    figure, axes = plt.subplots(figsize=(8.0, 3.4))
    _style(axes)
    axes.grid(False)

    # Grey for what the window contains, ink for the budget, and the xgboost slot for what
    # the model recovers, so a colour means the same thing here as on the other figures.
    colours = [BASELINE, BASELINE, INK_SECONDARY, SERIES["xgboost"]]
    positions = range(len(stages))
    axes.barh(
        list(positions),
        stages["share"],
        color=colours[: len(stages)],
        height=0.62,
        zorder=3,
    )
    for position, count, share in zip(positions, stages["count"], stages["share"], strict=True):
        axes.annotate(
            f"{int(count):,}   {float(share):.1%}",
            xy=(float(share), position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
        )

    axes.set_yticks(list(positions), list(stages["stage"]), fontsize=9)
    axes.tick_params(axis="y", labelcolor=INK_SECONDARY)
    axes.invert_yaxis()
    axes.set_xlim(0.0, 1.18)  # Headroom for the label that sits outside the longest bar.
    axes.set_xticks([])
    axes.spines["bottom"].set_visible(False)
    axes.set_title(
        f"What is reachable before any model runs, {window} window",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    _finish(figure, costs, out, "every bar is a share of the mules active in the label window")


def break_even(
    thresholds: pd.DataFrame, *, base_rate: float, window: str, costs: Costs, out: Path
) -> None:
    """Break-even precision against what the population offers, on one log axis.

    The finding is positional. The base rate falls between the threshold implied by the
    median mule exposure and the one implied by the mean, so the same cost matrix over the
    same distribution recommends alerting nobody or alerting everybody depending on which
    summary statistic is picked. Drawing it on one axis is the whole point of the figure.
    """
    plt.rcParams["font.family"] = FONT
    figure, axes = plt.subplots(figsize=(8.0, 3.6))
    _style(axes)

    positions = range(len(thresholds))
    axes.axvspan(0.0, base_rate, color=SERIES["xgboost"], alpha=0.10, zorder=1)
    for position, value in zip(positions, thresholds["precision"], strict=True):
        precision = float(value)
        met = precision <= base_rate
        axes.hlines(position, base_rate, precision, color=GRIDLINE, linewidth=1.4, zorder=2)
        axes.plot(
            precision,
            position,
            marker="o",
            markersize=9,
            color=SERIES["xgboost"] if met else SERIES["logistic"],
            zorder=4,
        )
        axes.annotate(
            f"{precision:.4%}   {'pays for itself' if met else 'destroys value'}",
            xy=(precision, position),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    axes.axvline(base_rate, color=INK_PRIMARY, linewidth=1.4, zorder=5)
    axes.annotate(
        f"what the population offers, {base_rate:.4%}",
        xy=(base_rate, -0.55),
        xytext=(7, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_PRIMARY,
    )

    axes.set_xscale("log")
    axes.set_yticks(list(positions), list(thresholds["basis"]), fontsize=9)
    axes.tick_params(axis="y", labelcolor=INK_SECONDARY)
    axes.set_ylim(-0.7, len(thresholds) - 0.3)
    # Descending threshold down the page, so the figure reads in the same order as the
    # break-even table in the report rather than in the reverse of it.
    axes.invert_yaxis()
    axes.xaxis.set_major_formatter(lambda v, _: f"{v:.2%}" if v >= ONE_PERCENT else f"{v:.3%}")
    axes.set_xlabel("precision an alert has to reach to pay for itself (log scale)")
    axes.set_title(
        f"What a catch has to be worth, {window} window",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    _finish(figure, costs, out, "shaded region is precision the population already offers")


def alert_budget(
    curves: dict[str, pd.DataFrame], *, capacity: int, window: str, costs: Costs, out: Path
) -> None:
    """The deliverable: what a daily alert budget buys, and what it costs to be wrong.

    Precision and money are different measures, so they get a panel each on a shared x
    axis rather than a second y scale.
    """
    plt.rcParams["font.family"] = FONT
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)

    for axes, column, label in (
        (top, "precision_at_k", "precision at k"),
        (bottom, "net_per_day", "net expected value, EUR per day"),
    ):
        _style(axes)
        for name, curve in curves.items():
            axes.plot(
                curve["k"], curve[column], color=SERIES[name], linewidth=2.0, zorder=3, label=name
            )
        axes.set_xscale("log")
        axes.set_ylabel(label)
        axes.axvline(capacity, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=4)

    # Both measures span regimes that a linear axis cannot show together. Precision runs
    # from 0 at the rules baseline to 75% in the top two accounts, and the net runs from
    # -280k a day to +66M once a single enormous account enters the alerted set. symlog
    # keeps zero representable, which a log axis would not, and keeps the decision region
    # around the configured capacity legible instead of flattened against the axis.
    top.set_yscale("symlog", linthresh=0.01)
    bottom.set_yscale("symlog", linthresh=1e5)
    top.set_ylim(bottom=0.0)  # symlog is symmetric by default, and precision is not.
    top.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= ONE_PERCENT else f"{v:.1%}")
    bottom.yaxis.set_major_formatter(lambda v, _: _money(v))
    bottom.axhline(0.0, color=BASELINE, linewidth=1.2, zorder=2)
    bottom.set_xlabel("alerts worked per day (log scale)")
    top.set_title(
        f"What the analyst budget buys, {window} window",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    top.annotate(
        f"configured capacity, {capacity}/day",
        xy=(capacity, top.get_ylim()[1]),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )
    legend = top.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
    legend.get_frame().set_edgecolor(GRIDLINE)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _finish(figure, costs, out, "every point is a re-run of F6 at that volume")
