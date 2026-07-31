"""Single source of truth for every quantity shared across modules.

PRD invariants 1 to 3 live here. The FX table, the ``is_mule`` label rule, the window
boundaries, and the cost parameters are defined once and imported. Restating any of them
in another module is a build failure, because two modules disagreeing about a definition
produce two results that look comparable and are not.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH: Final[Path] = PROJECT_ROOT / "config" / "params.yaml"

#: Names which parameter file the pipeline reads. It carries no parameter of its own, so
#: PRD invariant 1 is intact: every quantity still comes from a parameter file. See D37.
PARAMS_PATH_ENV: Final[str] = "PARAMS_PATH"


class UnknownCurrencyError(ValueError):
    """A currency appeared in the data with no rate in the fixed FX table."""


class SchemaError(ValueError):
    """A source file did not match the schema the data contract expects."""


@dataclass(frozen=True)
class Paths:
    """Where the pipeline reads and writes. All relative to the project root."""

    raw_dir: Path
    interim_dir: Path
    reports_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class DataSpec:
    """The data contract. Every count here is asserted at load and stops the build."""

    variant: str
    trans_file: str
    patterns_file: str
    expected_rows: int
    expected_accounts: int
    expected_laundering_rows: int
    expected_self_transfers: int
    drop_self_transfers: bool


@dataclass(frozen=True)
class SplitSpec:
    """The temporal split. Reshaped from the PRD default after the Step 1 span scan."""

    data_start: pd.Timestamp
    data_end: pd.Timestamp
    feature_window_days: int


@dataclass(frozen=True)
class Window:
    """One split: a feature window, then the label window that follows it.

    ``feature_end`` and ``label_start`` are the same instant by construction. That instant
    is the leakage cutoff of F4, computed once here and passed everywhere else as an
    argument. No other module derives its own.
    """

    name: str
    feature_start: pd.Timestamp
    feature_end: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp

    def __post_init__(self) -> None:
        """Refuse to exist in a state where a feature could see its own label."""
        if self.feature_end != self.label_start:
            raise SchemaError(
                f"Window {self.name!r}: feature_end {self.feature_end} must equal "
                f"label_start {self.label_start}, otherwise the cutoff is ambiguous."
            )
        if self.feature_start >= self.feature_end:
            raise SchemaError(f"Window {self.name!r}: empty feature window.")
        if self.label_start >= self.label_end:
            raise SchemaError(f"Window {self.name!r}: empty label window.")


@dataclass(frozen=True)
class WindowSpec:
    """Label window length and the outflow extension used by the pass-through ratio."""

    label_window_days: int
    pass_through_window_hours: int


@dataclass(frozen=True)
class FeatureSpec:
    """Thresholds and solver settings for the feature stage."""

    min_flow_eur: float
    pagerank_damping: float
    pagerank_max_iter: int
    pagerank_tol: float


@dataclass(frozen=True)
class RulesBaseline:
    """The rules a fraud team could write in an afternoon. The comparison that matters."""

    in_degree_percentile: float
    pass_through_ratio_min: float


@dataclass(frozen=True)
class LogisticSpec:
    """Logistic regression settings, plus the features it log1p transforms first.

    XGBoost is scale-invariant and consumes NaN natively, so none of this applies to it.
    The asymmetry is deliberate and it is why both are reported.
    """

    max_iter: int
    C: float  # sklearn's name for the inverse regularisation strength.
    class_weight: str
    solver: str
    log1p_features: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationSpec:
    """How the headline metric is reported. An interval is part of the number, not an extra."""

    bootstrap_resamples: int


#: What each queue overflow policy does with a candidate the day's capacity did not reach,
#: expressed as the number of days it stays eligible counting the day it arrived. F6 edge
#: case (c) and the PRD assumption register name both policies; this is the one place the
#: horizon of each is written down. The 3 in ``rollover_max_3d`` is part of the policy's
#: definition rather than a free parameter, so it is not separately configurable. D29.
QUEUE_OVERFLOW_POLICIES: Final[dict[str, int]] = {"same_day": 1, "rollover_max_3d": 3}


def _checked_policy(policy: str) -> str:
    """The policy name, having proved it names a policy. Returns it so it reads at the call."""
    queue_policy_horizon(policy)
    return policy


def queue_policy_horizon(policy: str) -> int:
    """How many days an unreached candidate stays in the queue, including the day it arrived.

    Raises rather than defaulting, because a misspelt policy that silently fell back to
    same-day capacity would report a rollover figure that never rolled anything over.
    """
    if policy not in QUEUE_OVERFLOW_POLICIES:
        known = ", ".join(sorted(QUEUE_OVERFLOW_POLICIES))
        raise SchemaError(f"Unknown queue overflow policy {policy!r}. Known policies: {known}.")
    return QUEUE_OVERFLOW_POLICIES[policy]


@dataclass(frozen=True)
class Costs:
    """The five checkpoint assumptions. Printed in every figure footer and the run log.

    Only ``cost_missed_mule_eur`` is anchored in a published figure. The other four are
    stated, not measured, and the README says so.

    Four of the five are numbers an alert is priced with. ``queue_overflow_policy`` is
    categorical and it decides which accounts get alerted rather than what an alert is worth,
    so it carries no multiplier and the Checkpoint 4 sweep leaves it alone. Its sensitivity is
    the two-policy comparison recorded beside the published figures.
    """

    cost_missed_mule_eur: float
    cost_investigation_eur: float
    cost_false_freeze_eur: float
    analyst_capacity_per_day: int
    queue_overflow_policy: str

    def as_footer(self) -> str:
        """One-line rendering of the active assumptions, for chart footers and logs."""
        return (
            f"assumptions: cost_missed={self.cost_missed_mule_eur:g} "
            f"cost_inv={self.cost_investigation_eur:g} "
            f"cost_freeze={self.cost_false_freeze_eur:g} "
            f"capacity={self.analyst_capacity_per_day:g} "
            f"overflow={self.queue_overflow_policy}"
        )


@dataclass(frozen=True)
class MonitoringSpec:
    """F5 settings. The reference window is named once here and read, never assumed. G6."""

    bins: int
    epsilon: float
    reference_window: str


@dataclass(frozen=True)
class ExperimentSpec:
    """Settings for work reported beside the published split rather than inside it. G1, G4."""

    history_window_days: tuple[int, ...]
    triangle_block_rows: int
    pass_through_sweep_hours: tuple[int, ...]


@dataclass(frozen=True)
class BriefSpec:
    """The reduced analyst brief generator. One brief per alerted account. G9, D36."""

    scorer: str
    sample: int
    standout_percentile: float


@dataclass(frozen=True)
class Params:
    """Everything the pipeline reads from config/params.yaml."""

    seed: int
    paths: Paths
    data: DataSpec
    base_currency: str
    fx_to_base: dict[str, float]
    split: SplitSpec
    windows: WindowSpec
    features: FeatureSpec
    model: dict[str, Any]
    logistic: LogisticSpec
    evaluation: EvaluationSpec
    rules_baseline: RulesBaseline
    costs: Costs
    sensitivity_multipliers: tuple[float, ...]
    monitoring: MonitoringSpec
    experiments: ExperimentSpec
    brief: BriefSpec


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, raising a specific error rather than returning something odd."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise SchemaError(f"Parameter file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SchemaError(f"Parameter file is not valid YAML: {path}") from exc

    if not isinstance(loaded, dict):
        raise SchemaError(f"Parameter file must contain a mapping: {path}")
    return loaded


def default_params_path() -> Path:
    """The parameter file to read, which a second dataset variant can redirect. D37.

    A relative override resolves against the project root, so the variant can be named the
    way the Makefile names it. An override pointing at a file that is not there raises rather
    than falling back, because a silent fallback would publish a full set of figures measured
    on one variant under the heading of another.
    """
    override = os.environ.get(PARAMS_PATH_ENV)
    if override is None:
        return DEFAULT_PARAMS_PATH

    path = Path(override)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise SchemaError(f"{PARAMS_PATH_ENV} names a parameter file that does not exist: {path}")
    return path


@functools.lru_cache(maxsize=4)
def load_params(path: Path | None = None) -> Params:
    """Load and validate the parameter file. Cached, so the file is read once per run."""
    if path is None:
        path = default_params_path()
    raw = _read_yaml(path)

    paths_raw = raw["paths"]
    data_raw = raw["data"]
    currency_raw = raw["currency"]
    split_raw = raw["split"]
    costs_raw = raw["costs"]

    fx = {str(name): float(rate) for name, rate in currency_raw["fx_to_base"].items()}
    base = str(currency_raw["base_currency"])
    if fx.get(base) != 1.0:
        raise SchemaError(f"Base currency {base!r} must have an FX rate of exactly 1.0")

    return Params(
        seed=int(raw["seed"]),
        paths=Paths(
            raw_dir=PROJECT_ROOT / str(paths_raw["raw_dir"]),
            interim_dir=PROJECT_ROOT / str(paths_raw["interim_dir"]),
            reports_dir=PROJECT_ROOT / str(paths_raw["reports_dir"]),
            figures_dir=PROJECT_ROOT / str(paths_raw["figures_dir"]),
        ),
        data=DataSpec(
            variant=str(data_raw["variant"]),
            trans_file=str(data_raw["trans_file"]),
            patterns_file=str(data_raw["patterns_file"]),
            expected_rows=int(data_raw["expected_rows"]),
            expected_accounts=int(data_raw["expected_accounts"]),
            expected_laundering_rows=int(data_raw["expected_laundering_rows"]),
            expected_self_transfers=int(data_raw["expected_self_transfers"]),
            drop_self_transfers=bool(data_raw["drop_self_transfers"]),
        ),
        base_currency=base,
        fx_to_base=fx,
        split=SplitSpec(
            data_start=pd.Timestamp(str(split_raw["data_start"])),
            data_end=pd.Timestamp(str(split_raw["data_end"])),
            feature_window_days=int(split_raw["feature_window_days"]),
        ),
        windows=WindowSpec(
            label_window_days=int(raw["windows"]["label_window_days"]),
            pass_through_window_hours=int(raw["windows"]["pass_through_window_hours"]),
        ),
        features=FeatureSpec(
            min_flow_eur=float(raw["features"]["min_flow_eur"]),
            pagerank_damping=float(raw["features"]["pagerank_damping"]),
            pagerank_max_iter=int(raw["features"]["pagerank_max_iter"]),
            pagerank_tol=float(raw["features"]["pagerank_tol"]),
        ),
        model=dict(raw["model"]),
        logistic=LogisticSpec(
            max_iter=int(raw["logistic"]["max_iter"]),
            C=float(raw["logistic"]["C"]),
            class_weight=str(raw["logistic"]["class_weight"]),
            solver=str(raw["logistic"]["solver"]),
            log1p_features=tuple(str(name) for name in raw["logistic"]["log1p_features"]),
        ),
        evaluation=EvaluationSpec(
            bootstrap_resamples=int(raw["evaluation"]["bootstrap_resamples"]),
        ),
        rules_baseline=RulesBaseline(
            in_degree_percentile=float(raw["rules_baseline"]["in_degree_percentile"]),
            pass_through_ratio_min=float(raw["rules_baseline"]["pass_through_ratio_min"]),
        ),
        costs=Costs(
            cost_missed_mule_eur=float(costs_raw["cost_missed_mule_eur"]),
            cost_investigation_eur=float(costs_raw["cost_investigation_eur"]),
            cost_false_freeze_eur=float(costs_raw["cost_false_freeze_eur"]),
            analyst_capacity_per_day=int(costs_raw["analyst_capacity_per_day"]),
            # Validated at load rather than at first use, so a misspelt policy stops the run
            # before any figure is drawn with a footer naming a policy that does not exist.
            queue_overflow_policy=_checked_policy(str(costs_raw["queue_overflow_policy"])),
        ),
        sensitivity_multipliers=tuple(float(m) for m in raw["sensitivity"]["multipliers"]),
        monitoring=MonitoringSpec(
            bins=int(raw["monitoring"]["bins"]),
            epsilon=float(raw["monitoring"]["epsilon"]),
            reference_window=str(raw["monitoring"]["reference_window"]),
        ),
        experiments=ExperimentSpec(
            history_window_days=tuple(
                int(days) for days in raw["experiments"]["history_window_days"]
            ),
            triangle_block_rows=int(raw["experiments"]["triangle_block_rows"]),
            pass_through_sweep_hours=tuple(
                int(hours) for hours in raw["experiments"]["pass_through_sweep_hours"]
            ),
        ),
        brief=BriefSpec(
            scorer=str(raw["brief"]["scorer"]),
            sample=int(raw["brief"]["sample"]),
            standout_percentile=float(raw["brief"]["standout_percentile"]),
        ),
    )


#: The three splits, in temporal order. Named once so nothing downstream invents a name.
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    """Write one stage's measured numbers where ``report.py`` can read them.

    PRD invariant 6 says the report assembles and computes nothing, which only holds if
    every stage hands its numbers forward. A number that is not in one of these files
    cannot appear in the report, which is the same rule the run log enforces by hand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def read_metrics(path: Path) -> dict[str, Any]:
    """Read a stage's metrics, raising rather than returning an empty dict if it is absent."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded: dict[str, Any] = json.load(handle)
    except FileNotFoundError as exc:
        raise SchemaError(
            f"{path.name} is missing. Run the stage that writes it before the report."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path.name} is not valid JSON: {exc}") from exc
    return loaded


def build_windows(split: SplitSpec, label_window_days: int) -> tuple[Window, ...]:
    """Lay out the three splits over the usable span. Computes the leakage cutoff once.

    Each split gets a feature window of fixed length followed immediately by its label
    window. The label windows are contiguous, disjoint, and in time order, and together
    they fill the span from the end of the first feature window to ``data_end``.
    """
    label_delta = pd.Timedelta(days=label_window_days)
    feature_delta = pd.Timedelta(days=split.feature_window_days)
    first_label_start = split.data_start + feature_delta

    required = feature_delta + label_delta * len(SPLIT_NAMES)
    available = split.data_end - split.data_start
    if available < required:
        raise SchemaError(
            f"Usable span of {available} is too short for {len(SPLIT_NAMES)} label windows "
            f"of {label_delta} plus a {feature_delta} feature window ({required} needed)."
        )

    windows = []
    for index, name in enumerate(SPLIT_NAMES):
        label_start = first_label_start + label_delta * index
        windows.append(
            Window(
                name=name,
                feature_start=label_start - feature_delta,
                feature_end=label_start,
                label_start=label_start,
                label_end=label_start + label_delta,
            )
        )
    return tuple(windows)


def fx_to_base(amounts: pd.Series, currencies: pd.Series, rates: dict[str, float]) -> pd.Series:
    """Convert amounts to the base currency using the fixed table. Implements F1's conversion.

    Raises rather than defaulting, because a missing rate is a data problem and a silent
    fallback to 1.0 would understate every non-base amount without changing anything visible.
    """
    unknown = sorted(set(currencies.dropna().unique()) - set(rates))
    if unknown:
        raise UnknownCurrencyError(f"No FX rate for: {', '.join(unknown)}")
    return amounts.astype("float64") * currencies.map(rates).astype("float64")


def scoring_population(txns: pd.DataFrame, window: Window) -> pd.Index:
    """The accounts eligible to be scored in this window. Defined once here. D9.

    Receivers active in the feature window, and nothing else. An account with no inbound
    transaction before the cutoff has no features to compute, so scoring it is impossible
    and counting it pads the denominator with an automatic negative.

    This choice sets every base rate and every precision@k in the project. Three
    defensible populations existed and they differed by 49% on the test window, so it is
    pinned here rather than derived from whatever frame a caller happens to pass.
    """
    in_features = txns.loc[
        (txns["timestamp"] >= window.feature_start) & (txns["timestamp"] < window.feature_end)
    ]
    return pd.Index(in_features["account_to"].unique(), name="account").sort_values()


def label_accounts(
    txns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    population: pd.Index,
) -> pd.DataFrame:
    """Assign ``is_mule`` per account over the label window ``[start, end)``.

    The label goes to the receiving account, which is the whole argument of the project.
    An account is a mule in this window if it received at least one transaction carrying
    the laundering flag inside it. Defined once here. PRD invariant 2.

    ``population`` is required rather than derived. The returned frame is indexed by
    exactly that population, so the denominator cannot drift with the caller's frame.
    """
    in_window = txns.loc[(txns["timestamp"] >= start) & (txns["timestamp"] < end)]
    flagged = in_window.groupby("account_to")["is_laundering"].max()

    labels = flagged.reindex(population, fill_value=0).astype("int8")
    return labels.rename("is_mule").to_frame()


def exposure(txns: pd.DataFrame, window: Window, population: pd.Index) -> pd.Series:
    """Money arriving into each account over its label window, in base currency. D3.

    F6 sums this over caught mules to value a catch. The PRD left it undefined, and D3
    settles it as label-window inflow, on the grounds that this is the money that passed
    through the account and was therefore at stake.

    This reads transactions after the leakage cutoff on purpose. It values an outcome
    rather than describing an account, so it is never a feature and no scorer sees it.
    """
    in_window = txns.loc[
        (txns["timestamp"] >= window.label_start) & (txns["timestamp"] < window.label_end)
    ]
    arriving = in_window.groupby("account_to")["amount_in_base_received"].sum()
    return arriving.reindex(population, fill_value=0.0).rename("exposure")


def daily_candidates(txns: pd.DataFrame, window: Window, population: pd.Index) -> pd.DataFrame:
    """Which accounts an analyst could alert on, on which day. Defined once here. D11.

    An account is a candidate on every day of its label window that it receives money on,
    because that is when a transaction monitoring queue would surface it. Accounts appear
    on more than one day, and :func:`src.economics.select_alerts` works each one at most
    once.

    The rule reads timestamps and account ids only. It never reads ``is_laundering``, so
    it passes the same target-independence test as the two data exclusions.
    """
    in_window = txns.loc[
        (txns["timestamp"] >= window.label_start)
        & (txns["timestamp"] < window.label_end)
        & (txns["account_to"].isin(population))
    ]
    pairs = pd.DataFrame(
        {
            "account": in_window["account_to"].to_numpy(),
            "day": in_window["timestamp"].dt.floor("D").to_numpy(),
        }
    )
    return pairs.drop_duplicates().sort_values(["day", "account"]).reset_index(drop=True)


def count_unscoreable_mules(txns: pd.DataFrame, window: Window, population: pd.Index) -> int:
    """Mules in the label window that the population excludes, because they have no history.

    Excluding them is defensible. Not knowing how many were excluded is not, because they
    are undetectable by construction and belong in the report as a stated ceiling.
    """
    in_window = txns.loc[
        (txns["timestamp"] >= window.label_start) & (txns["timestamp"] < window.label_end)
    ]
    mules = in_window.loc[in_window["is_laundering"] == 1, "account_to"].unique()
    return len(pd.Index(mules).difference(population))
