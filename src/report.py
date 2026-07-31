"""Assembly. This module computes nothing. PRD invariant 6.

Every number below was measured by a named upstream stage and written to a metrics file:
``metrics_data.json`` by :mod:`src.data`, ``metrics_models.json`` by :mod:`src.models`,
``metrics_economics.json`` by :mod:`src.economics`, and ``metrics_monitoring.json`` by
:mod:`src.monitoring`. If a number is not in one of those files it does not appear here,
which is the same rule the run log enforces by hand.

The only arithmetic performed is formatting: percentages, thousands separators, and
rounding for display. Nothing is derived, combined, or recomputed.
"""

from __future__ import annotations

import logging
from typing import Any

from src.charts import SERIES
from src.definitions import SPLIT_NAMES, Costs, load_params, read_metrics
from src.models import SCORERS

LOGGER = logging.getLogger(__name__)


def _percent(value: float, places: int = 4) -> str:
    """Format a fraction as a percentage. Display only."""
    return f"{100.0 * value:.{places}f}%"


def _money(value: float) -> str:
    """Format a euro amount with thousands separators. Display only."""
    return f"{value:,.0f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    """A markdown table. The table view every chart is required to have a twin of."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _windows_section(data: dict[str, Any]) -> str:
    """Population, positives, and the reachability ceiling per split."""
    # SPLIT_NAMES, not the JSON key order, because the splits are temporal and reading
    # them alphabetically puts test before train.
    rows = [
        [
            name,
            f"{data['windows'][name]['population']:,}",
            f"{data['windows'][name]['mules']:,}",
            _percent(data["windows"][name]["base_rate"]),
            f"{data['windows'][name]['unscoreable_mules']:,}",
            _percent(data["windows"][name]["reachability_ceiling"], 1),
        ]
        for name in SPLIT_NAMES
    ]
    header = ["split", "population", "mule accounts", "base rate"]
    header += ["unreachable", "reachability ceiling"]
    return _table(header, rows)


def _pr_section(models: dict[str, Any], window: str) -> str:
    """PR-AUC per scorer with its bootstrap interval, and the paired comparisons."""
    block = models["windows"][window]
    marginal = _table(
        ["scorer", "PR-AUC", "2.5%", "97.5%"],
        [
            [row["scorer"], f"{row['pr_auc']:.6f}", f"{row['low']:.6f}", f"{row['high']:.6f}"]
            for row in block["pr_auc"]
        ],
    )
    paired = _table(
        ["comparison", "difference", "2.5%", "97.5%", "crosses zero"],
        [
            [
                row["comparison"],
                f"{row['difference']:.6f}",
                f"{row['low']:.6f}",
                f"{row['high']:.6f}",
                "yes" if row["crosses_zero"] else "no",
            ]
            for row in block["paired"]
        ],
    )
    note = (
        "Paired differences. Both scorers see the same re-drawn accounts, so the variation they "
        "share cancels. `crosses zero` means the interval includes zero, so the sign of the "
        "difference is not established."
    )
    return f"{marginal}\n\n{note}\n\n{paired}"


def _operating_section(economics: dict[str, Any], window: str) -> str:
    """What the configured budget bought, per scorer."""
    # SCORERS order, so the baseline the comparison turns on is read first every time.
    rows = []
    for scorer in SCORERS:
        point = economics["operating_point"][f"{window}/{scorer}"]
        rows.append(
            [
                scorer,
                f"{point['alerts']:,.0f}",
                f"{point['caught']:,.0f}",
                _percent(point["precision_at_k"]),
                _percent(point["recall"], 2),
                f"{point['threshold']:.6f}",
                _money(point["net_per_day"]),
            ]
        )
    return _table(
        ["scorer", "alerts", "caught", "precision@k", "recall", "threshold", "net EUR/day"], rows
    )


def _overflow_section(economics: dict[str, Any]) -> str:
    """F6 edge case (c), both overflow policies side by side. G3, D29.

    Rows for the configured policy reproduce the operating-point table above, because they are
    the same queue and the same arithmetic. Reading the two tables against each other is how a
    reader checks that the comparison is set beside the published figure rather than replacing it.
    """
    rows = []
    for record in economics["overflow_comparison"]:
        rows.append(
            [
                record["window"],
                record["scorer"],
                record["policy"],
                f"{record['alerts']:,.0f}",
                f"{record['carried_alerts']:,.0f}",
                f"{record['caught']:,.0f}",
                _percent(record["precision_at_k"]),
                f"{record['threshold']:.6f}",
                _money(record["net_per_day"]),
            ]
        )
    return _table(
        [
            "window",
            "scorer",
            "policy",
            "alerts",
            "carried",
            "caught",
            "precision@k",
            "threshold",
            "net EUR/day",
        ],
        rows,
    )


def _overflow_verdict_sentence(economics: dict[str, Any]) -> str:
    """Render the counted direction of the policy comparison. Assembly only, D37.

    This sentence used to be written by hand. It said rollover improved nothing, which was
    true of HI-Small and false of LI-Small, where two of the three scorers improve under it.
    The counts come from :func:`src.economics.overflow_verdict` and nothing is decided here.
    """
    verdict = economics["overflow_verdict"]
    improved = int(verdict["improved"])
    worsened = int(verdict["worsened"])
    unchanged = int(verdict["unchanged"])

    if improved == 0 and worsened == 0:
        return (
            f"**The alternative policy left the operating point unchanged in all {unchanged} "
            "scorer and window comparisons.**"
        )

    counted = [
        (improved, f"improves it in {improved}"),
        (worsened, f"makes it worse in {worsened}"),
        (unchanged, f"leaves it unchanged in {unchanged}"),
    ]
    parts = [phrase for count, phrase in counted if count]
    total = improved + worsened + unchanged
    gain = int(verdict["largest_gain"])
    loss = int(verdict["largest_loss"])
    return (
        f"**Across the {total} scorer and window comparisons the alternative policy "
        f"{', '.join(parts[:-1])} and {parts[-1]}.** The largest movement either way is "
        f"{max(gain, loss)} caught account{'' if max(gain, loss) == 1 else 's'}."
    )


def _overflow_alternative(economics: dict[str, Any]) -> str:
    """Name the policy that is not configured, with its own horizon rather than the active one.

    The horizons are read per policy because a single figure here would print the configured
    policy's horizon while describing the other one, which is how the first draft was wrong.
    """
    horizons: dict[str, int] = economics["queue_policy_horizon_days"]
    active = economics["queue_overflow_policy"]
    others = [name for name in sorted(horizons) if name != active]
    if not others:
        return ""
    described = [
        f"`{name}`, which carries a candidate for {horizons[name]} "
        f"{'day' if horizons[name] == 1 else 'days'}"
        for name in others
    ]
    lead = "The alternative is" if len(others) == 1 else "The alternatives are"
    return (
        f"{lead} {', '.join(described)}, so the accounts left over from yesterday compete for "
        "today's capacity."
    )


def _drift_section(monitoring: dict[str, Any], window: str) -> str:
    """Feature drift from the reference window, with what each PSI value can be read as.

    ``empty`` is the number of bins holding no mass on one side. A PSI computed with any of
    them is a flag saying a bin emptied rather than a distance, because the size of an emptied
    bin's term is set by the epsilon that replaced the zero. D34.
    """
    rows = []
    for record in monitoring["feature_drift"][window]:
        rows.append(
            [
                str(record["feature"]),
                f"{record['psi']:.4f}",
                str(record["reading"]),
                f"{record['empty_bins']:.0f}",
                "flag" if record["epsilon_dependent"] else "magnitude",
            ]
        )
    return _table(["feature", "PSI", "reading", "empty bins", "reads as"], rows)


def _score_drift_section(monitoring: dict[str, Any], window: str) -> str:
    """PSI and KS on each scorer's output between the reference window and this one."""
    rows = []
    for scorer in SCORERS:
        record = monitoring["score_drift"][window][scorer]
        rows.append(
            [
                scorer,
                f"{record['psi']:.4f}",
                str(record["reading"]),
                f"{record['ks']:.4f}",
                f"{record['empty_bins']:.0f}",
                "flag" if record["epsilon_dependent"] else "magnitude",
            ]
        )
    return _table(["scorer", "PSI", "reading", "KS", "empty bins", "reads as"], rows)


def _assumptions_line(costs: Costs) -> str:
    """The five active assumptions, printed as the final line of a successful run."""
    return costs.as_footer()


def build(
    data: dict[str, Any],
    models: dict[str, Any],
    economics: dict[str, Any],
    monitoring: dict[str, Any],
    costs: Costs,
) -> str:
    """Assemble the report. Reads measured values and formats them, nothing else."""
    exposure = economics["exposure"]
    breakeven = economics["break_even_precision"]

    return f"""# Where the Money Goes Next: results

`make report` generates this document. Every figure comes from an upstream stage that measured
it and wrote it to a metrics file. This module computes nothing.

**A simulator generated the labels.** Nothing below is a detection claim about money
laundering. Three of the four cost parameters are assumptions.

The terms used here are defined in `docs/glossary.md`.

Active assumptions: `{_assumptions_line(costs)}`

## The data, after two exclusions

Rows read: {data["rows_read"]:,}, over {data["accounts"]:,} accounts keyed as (bank, account).
One account is one (bank, account) pair, so the same account number at two banks counts twice.
Self-transfers removed at load: {data["self_transfers_dropped"]:,}, carrying
{data["laundering_in_self_transfers"]} laundering rows. A self-transfer has the same account on
both sides. The observed span is {data["observed_start"]} to {data["observed_end"]}.
{data["usable_rows"]:,} rows fall inside the usable span {data["usable_start"]} to
{data["usable_end"]}.

![Daily volume](figures/daily_volume.png)

The cliff in that chart is the evidence for excluding everything from 09-11 onward. Ordinary
traffic falls by a factor of about a thousand, and the laundering share rises from about 0.1% to
near 100%. Stating that rule needs no label at all, which is the test every exclusion in this
project has to pass.

Each split below has its own feature window, and the label window that follows it. The population
is the accounts that received money during the feature window.

{_windows_section(data)}

**The reachability ceiling is a limit no model can pass.** It is the share of the label window's
mule accounts that fall inside the population. An account with no incoming payment before the
cutoff has no features, so the pipeline cannot score it. The `unreachable` column counts the
mule accounts left outside.

![Reachability and budget](figures/reachability_funnel.png)

Both ceilings use one denominator: the mule accounts active in the test label window.

## The model against the rules, validation window

PR-AUC is the area under the precision-recall curve. It rises when mule accounts rank nearer the
top. A scorer that has learned nothing scores the base rate, which is the share of the population
that are mule accounts. The 2.5% and 97.5% columns are the range the middle 95% of
{models["bootstrap_resamples"]:,} re-draws fall in.

![PR curve, validation](figures/pr_curve_val.png)

{_pr_section(models, "val")}

## The same comparison on the test window

XGBoost uses the validation window to early-stop, so the validation scores are optimistic and
the test window is the clean comparison.

![PR curve, test](figures/pr_curve_test.png)

{_pr_section(models, "test")}

Trees kept by XGBoost: {models["trees_kept"]}, of an allowed {models["trees_allowed"]}. The
pipeline measured `scale_pos_weight` from the training labels at {models["scale_pos_weight"]:.2f}.
That setting tells XGBoost how much more one mule account counts than one clean account. The
training window holds {models["train_clean"]:,} clean accounts and {models["train_mules"]:,} mule
accounts.

## The operating point under a fixed analyst budget

An alert is an account the analyst team opens and works. The budget allows
{costs.analyst_capacity_per_day} alerts a day, and the `alerts` column is the total over the
label window. `precision@k` is the share of those alerts that are mule accounts. The threshold is
the score of the last account inside the budget.

![Alert budget](figures/alert_budget.png)

Validation window:

{_operating_section(economics, "val")}

Test window:

{_operating_section(economics, "test")}

### Accounts the capacity did not reach

`queue_overflow_policy` is configured as `{economics["queue_overflow_policy"]}`. It discards, at
the end of the day, everything above the threshold that the budget did not reach.
{_overflow_alternative(economics)} Both policies are measured below, and every other figure in this
report was produced under the configured one.

{_overflow_section(economics)}

The alert count is the same in every row, so the policy changes which accounts are worked and
never how many. `carried` is how many of those alerts were spent on an account that did not
arrive that day. That is the whole of the policy's effect on who gets worked. Under same-day
capacity it is zero by definition.

A backlog needs a quiet day after a busy one to have anywhere to go. The mechanism has room on the
test window and almost none on validation. The daily volume chart at the top of this report shows
the same asymmetry from the other side.

{_overflow_verdict_sentence(economics)} No confidence interval was computed for any of it, so the
direction is an observation and not a result. A three-day expiry cannot bind on a label window
this short, so what is measured here is a single carry.

## What the money rests on

Exposure is the money that arrived in an account during its label window. It prices a catch,
because a freeze is assumed to recover what arrived. Exposure over the test window totals
{_money(exposure["total"])} EUR across {exposure["count"]} accounts, with a median of
{_money(exposure["median"])} and a mean of {_money(exposure["mean"])}. **The largest single
account carries {_percent(exposure["top_1_share"], 1)} of it, and the top five carry
{_percent(exposure["top_5_share"], 1)}.**

![Break-even precision](figures/break_even.png)

The precision one more alert must reach to cover its cost, against a measured base rate of
{_percent(economics["base_rate"])}:

{
        _table(
            ["value recovered per catch", "break-even precision", "verdict for a random alert"],
            [
                ["nothing", _percent(breakeven["nothing_recovered"]), "destroys value"],
                ["median exposure", _percent(breakeven["median_exposure"]), "destroys value"],
                ["mean exposure", _percent(breakeven["mean_exposure"]), "covers its cost"],
            ],
        )
    }

**The sign of the recommendation flips between the median and the mean of the same measured
distribution.** None of the four cost parameters does anything comparable.

## Data drift between windows

This section measures drift from the `{monitoring["reference_window"]}` window, the one the models
were fitted on, to the test window. Drift here means the Population Stability Index, PSI. It cuts
the reference window into {monitoring["bins"]} bins of equal size. It then applies those same bin
edges to the later window and compares how the mass is spread. The conventional readings are below
0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant. They are a convention and not a law.

**A PSI value cannot be read without knowing whether a bin emptied.** An empty bin has its zero
replaced by a very small number before the logarithm. That replacement then sets the size of the
bin's term. Rows marked `flag` mean a bin emptied. Rows marked `magnitude` can be read as a
distance.

{_drift_section(monitoring, "test")}

A PSI of exactly 0.0000 is the other failure of the same kind. A feature that takes one value for
almost every account in the reference collapses all its quantile edges onto that value. Both
windows then fall into a single bin, and no drift can be resolved whatever happened. That is a
measurement without resolution, and not a stable feature.

The table below reads the scorers' own output the same way. KS is the largest gap between two
cumulative distributions.

{_score_drift_section(monitoring, "test")}

KS needs no bins and no reference period, so it is symmetric and cheap. It cannot say which part of
the distribution moved. That is why both are here.

**None of this supports a claim about model decay.** The training feature window contains the two
busiest days in the file and the test feature window does not. What is measured is the simulator's
calendar, and not behaviour changing over time. {monitoring["decay_note"]}

## Colour

Series colours are the validated categorical slots 1 to 3 in fixed order:
{", ".join(f"{name} `{hex_}`" for name, hex_ in SERIES.items())}. Every chart above has a table
beside it. That table is the relief required for the third slot, which falls below 3:1 contrast
against the chart surface.
"""


def main() -> None:
    """Read every stage's metrics and write the report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    reports = params.paths.reports_dir

    document = build(
        read_metrics(reports / "metrics_data.json"),
        read_metrics(reports / "metrics_models.json"),
        read_metrics(reports / "metrics_economics.json"),
        read_metrics(reports / "metrics_monitoring.json"),
        params.costs,
    )

    destination = reports / "report.md"
    destination.write_text(document, encoding="utf-8")
    LOGGER.info("wrote %s", destination)
    LOGGER.info("active assumptions: %s", params.costs.as_footer())


if __name__ == "__main__":
    main()
