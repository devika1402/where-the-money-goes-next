"""Scorers. The rules baseline, plus the two fitted models it has to be compared against.

The rules baseline is the comparison the whole project turns on. Without it there is no
evidence the graph features bought anything over what a fraud team could write in an
afternoon, which is why it is on the never-cut list and why it gets tested like a formula.

It exists in two forms and they answer different questions:

* :func:`rules_flag` is the rule exactly as configured, a conjunction of two conditions.
  It returns a set, and a set is what a fraud team would actually write down.
* :func:`rules_score` is the same two quantities made rankable. A daily alert budget has to
  choose an order, and a set does not have one.

Thresholding the score at a percentile recovers the flag at that percentile, so the second
is the first as a ranking rather than a different rule. ``tests/test_models.py`` asserts it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import charts
from src.definitions import (
    LogisticSpec,
    RulesBaseline,
    SchemaError,
    build_windows,
    load_params,
    write_metrics,
)

LOGGER = logging.getLogger(__name__)

#: scikit-learn ships no type stubs, so a fitted estimator is Any to mypy. Named once here
#: rather than left as a bare Any drifting through several signatures.
Fitted = Any

#: The two columns the baseline reads. Named once so a rename fails loudly.
IN_DEGREE: str = "in_degree"
PASS_THROUGH_RATIO: str = "pass_through_ratio"

#: The label column, which is never a feature.
LABEL: Final[str] = "is_mule"

#: The comparisons the report makes. Each gets a paired interval.
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("xgboost", "rules"),
    ("xgboost", "logistic"),
    ("logistic", "rules"),
)

#: The three scorers, in the order they are reported. Named once.
SCORERS: Final[tuple[str, str, str]] = ("rules", "logistic", "xgboost")


def rules_flag(features: pd.DataFrame, rules: RulesBaseline) -> pd.Series:
    """In-degree above a percentile of the population, and pass-through ratio above a floor.

    The in-degree cut is a percentile rather than a count, so the rule travels between
    windows whose activity differs. The ratio cut is an absolute value, because the ratio
    already has a meaning: above 1.0 an account paid out more than it took in.

    An undefined ratio compares False, which is the intended reading. F2 returns NaN when
    inflow sits below the floor, and an account too small to measure is not evidence.
    """
    cut = features[IN_DEGREE].quantile(rules.in_degree_percentile)
    busy = features[IN_DEGREE] > cut
    passing = features[PASS_THROUGH_RATIO] > rules.pass_through_ratio_min
    return (busy & passing).rename("rules_flag")


def matched_rules_flag(features: pd.DataFrame, percentile: float) -> pd.Series:
    """The conjunction with both conditions cut at the same percentile. G2, D27.

    :func:`rules_flag` is the rule as configured and it is left alone, because what was
    specified before any model ran is part of the evidence. This is the form that matches
    :func:`rules_score`, which cuts both quantities by rank, so the two forms of the baseline
    describe one rule rather than two.

    An undefined ratio compares False here as it does in the configured form. A quantile over
    a column with nulls is taken over the values that exist, so the cut is a percentile of the
    measurable population rather than of the population padded with unmeasurable accounts.
    """
    busy = features[IN_DEGREE] > features[IN_DEGREE].quantile(percentile)
    passing = features[PASS_THROUGH_RATIO] > features[PASS_THROUGH_RATIO].quantile(percentile)
    return (busy & passing).rename("matched_rules_flag")


def rules_score(features: pd.DataFrame, rules: RulesBaseline) -> pd.Series:
    """The same rule as a ranking: the lower of the two percentile ranks.

    An account clears a conjunction of two percentile cuts at level q exactly when the
    lower of its two ranks exceeds q, so this is the family of rules the configured one
    belongs to, indexed by where the cut is put. Sweeping the score sweeps the rule.

    ``rules`` is taken as an argument although the score does not read it. The signature
    matches :func:`rules_flag` deliberately, because the two must stay the same rule, and a
    scorer that quietly stopped depending on the configured quantities would be a different
    baseline wearing the name.

    An undefined ratio ranks lowest rather than being dropped. The budget has to be filled
    from somewhere and these accounts are the weakest evidence available, not the strongest.
    """
    del rules
    ranks = features[[IN_DEGREE, PASS_THROUGH_RATIO]].rank(pct=True, method="max", na_option="top")
    return ranks.min(axis=1).rename("rules")


def _null_rates(matrix: pd.DataFrame) -> dict[str, float]:
    """Null rate per feature, keeping only the columns that have any."""
    rates = matrix[feature_columns(matrix)].isna().mean()
    return {str(column): float(value) for column, value in rates.items() if value > 0}


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Every column a model is allowed to see. The label is not one of them."""
    return [column for column in matrix.columns if column != LABEL]


def log1p_features(features: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Compress the declared heavy-tailed columns before standardising. D15.

    Logistic regression fits a coefficient per feature on the raw scale, so a column whose
    99th percentile is near 3,900 and whose maximum is 8.25 million dominates the fit for
    reasons of units rather than evidence. XGBoost is scale-invariant and never sees this.

    Nulls stay null. Imputation is a separate declared step and filling here would hide it.
    """
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise SchemaError(f"Declared for log1p but not in the feature matrix: {missing}")

    negative = [column for column in columns if (features[column] < 0).any()]
    if negative:
        raise SchemaError(
            f"log1p is declared for columns that hold negative values: {negative}. "
            "At -1 it returns negative infinity, so the declaration is wrong."
        )

    transformed = features.copy()
    transformed[list(columns)] = np.log1p(transformed[list(columns)])
    return transformed


def scale_pos_weight(labels: pd.Series) -> float:
    """Clean accounts per mule, measured from the labels handed in.

    XGBoost's ``scale_pos_weight`` is a knob and a knob is not its outcome, so this reads
    the realised class balance of the training window rather than trusting a base rate
    written down somewhere earlier.
    """
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        raise SchemaError("The training window holds no mule accounts, so there is nothing to fit.")
    return negatives / positives


def fit_logistic(train: pd.DataFrame, spec: LogisticSpec, seed: int) -> Fitted:
    """Median imputation, standardisation, then logistic regression, all fitted on train.

    The imputer and the scaler are fitted on the training window alone and applied forward
    unchanged. Refitting them per window would mean a feature meant something different in
    each split, which is the same objection D7 raised against expanding feature windows.
    """
    columns = feature_columns(train)
    pipeline = Pipeline(
        [
            # No add_indicator, and the reason is measured rather than assumed. The nulls
            # here are exactly the accounts with out_degree == 0: 131,604 in the training
            # window on both counts, the same set. out_degree is already a feature, so the
            # model is not missing the fact that an account forwarded nothing, and an
            # indicator column would restate a column it already has. D19.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=spec.max_iter,
                    C=spec.C,
                    class_weight=spec.class_weight,
                    solver=spec.solver,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(log1p_features(train[columns], spec.log1p_features), train[LABEL])
    return pipeline


def logistic_score(pipeline: Fitted, features: pd.DataFrame, spec: LogisticSpec) -> pd.Series:
    """Probability of being a mule, indexed by the population it was asked about."""
    columns = feature_columns(features)
    prepared = log1p_features(features[columns], spec.log1p_features)
    return pd.Series(pipeline.predict_proba(prepared)[:, 1], index=features.index, name="logistic")


def fit_xgboost(
    train: pd.DataFrame, validation: pd.DataFrame, model: dict[str, Any], seed: int
) -> XGBClassifier:
    """One fixed configuration, one fit, no hyperparameter search.

    ``early_stopping_rounds`` is a constructor argument in XGBoost 2.0 and later, and
    passing it to ``.fit()`` raises. ``device='cpu'`` with ``tree_method='hist'`` replaces
    the removed ``gpu_hist``. Both per PRD section 5.

    The validation window is the early-stopping set, so validation numbers for this model
    carry the optimism that buys, and the test window is the clean comparison. D16.
    """
    columns = feature_columns(train)
    weight = scale_pos_weight(train[LABEL])
    LOGGER.info(
        "xgboost scale_pos_weight measured from the training window: %.2f "
        "(%d clean against %d mules)",
        weight,
        int((train[LABEL] == 0).sum()),
        int((train[LABEL] == 1).sum()),
    )

    classifier = XGBClassifier(
        n_estimators=int(model["n_estimators"]),
        max_depth=int(model["max_depth"]),
        learning_rate=float(model["learning_rate"]),
        early_stopping_rounds=int(model["early_stopping_rounds"]),
        subsample=float(model["subsample"]),
        colsample_bytree=float(model["colsample_bytree"]),
        device=str(model["device"]),
        tree_method=str(model["tree_method"]),
        scale_pos_weight=weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    classifier.fit(
        train[columns],
        train[LABEL],
        eval_set=[(validation[columns], validation[LABEL])],
        verbose=False,
    )
    LOGGER.info(
        "xgboost stopped at %d of %d trees, best validation aucpr %.6f",
        classifier.best_iteration + 1,
        int(model["n_estimators"]),
        classifier.best_score,
    )
    return classifier


def xgboost_score(classifier: XGBClassifier, features: pd.DataFrame) -> pd.Series:
    """Probability of being a mule. NaN is consumed natively, so nothing is imputed."""
    columns = feature_columns(features)
    return pd.Series(
        classifier.predict_proba(features[columns])[:, 1], index=features.index, name="xgboost"
    )


def pr_auc(labels: pd.Series, scores: pd.Series) -> float:
    """Average precision, which is the area under the precision-recall curve.

    Precision-recall rather than ROC, because at a base rate near 0.25% the ROC curve is
    dominated by true negatives and looks respectable for a scorer that finds nothing.
    A scorer that has learned nothing scores about the base rate here.
    """
    if not labels.index.equals(scores.index):
        raise SchemaError("Labels and scores must be indexed by the same population.")
    return float(average_precision_score(labels, scores))


def fast_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, written out because the bootstrap calls it thousands of times.

    Tied scores collapse into one threshold, which is not a detail here. The rules baseline
    is a percentile rank and is heavily tied, and ranking tied accounts as though ordered
    credits the scorer with an ordering it never produced. Checked against
    :func:`pr_auc` in the test suite on both continuous and tied inputs.
    """
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    positives = hits.sum()
    if positives == 0:
        return float("nan")

    ranked = scores[order]
    boundaries = np.flatnonzero(np.diff(ranked)) if len(ranked) > 1 else np.empty(0, dtype=int)
    ends = np.append(boundaries, len(ranked) - 1)

    caught = np.cumsum(hits)[ends]
    precision = caught / (ends + 1)
    recall = caught / positives
    gained = np.diff(np.concatenate(([0.0], recall)))
    return float((precision * gained).sum())


def bootstrap_draws(
    labels: pd.Series, scores: pd.DataFrame, resamples: int, seed: int
) -> pd.DataFrame:
    """Resample accounts with replacement and recompute average precision each time.

    Every scorer sees the same resampled accounts on each draw. That is what makes the
    differences between columns paired, and the paired difference is the comparison worth
    reporting: the shared sampling variation cancels, and at 452 positives it is large.
    """
    if resamples < 1:
        raise SchemaError(f"Need at least one resample, got {resamples}.")

    truth = labels.to_numpy().astype(np.float64)
    columns = {name: scores[name].to_numpy() for name in scores.columns}
    rng = np.random.default_rng(seed)
    n = len(truth)

    drawn = {name: np.empty(resamples) for name in columns}
    for draw in range(resamples):
        picked = rng.integers(0, n, n)
        resampled_truth = truth[picked]
        for name, vector in columns.items():
            drawn[name][draw] = fast_average_precision(resampled_truth, vector[picked])
    return pd.DataFrame(drawn)


def summarise_draws(draws: pd.DataFrame, labels: pd.Series, scores: pd.DataFrame) -> pd.DataFrame:
    """Point estimate and 95% percentile interval per scorer."""
    truth = labels.to_numpy().astype(np.float64)
    return pd.DataFrame(
        [
            {
                "scorer": name,
                "pr_auc": fast_average_precision(truth, scores[name].to_numpy()),
                "low": float(np.nanpercentile(draws[name], 2.5)),
                "high": float(np.nanpercentile(draws[name], 97.5)),
            }
            for name in draws.columns
        ]
    )


def paired_differences(
    draws: pd.DataFrame, labels: pd.Series, scores: pd.DataFrame, pairs: Sequence[tuple[str, str]]
) -> pd.DataFrame:
    """Difference between two scorers on the same resampled accounts, with its interval.

    An interval that crosses zero means the ordering between those two scorers is not
    established by this window, whatever the point estimates say.
    """
    truth = labels.to_numpy().astype(np.float64)
    rows = []
    for left, right in pairs:
        delta = draws[left] - draws[right]
        low = float(np.nanpercentile(delta, 2.5))
        high = float(np.nanpercentile(delta, 97.5))
        rows.append(
            {
                "comparison": f"{left} - {right}",
                "difference": fast_average_precision(truth, scores[left].to_numpy())
                - fast_average_precision(truth, scores[right].to_numpy()),
                "low": low,
                "high": high,
                "crosses_zero": bool(low <= 0.0 <= high),
            }
        )
    return pd.DataFrame(rows)


def pr_table(scores: pd.DataFrame) -> pd.DataFrame:
    """PR-AUC per scorer against the base rate it has to beat. Reported for one window."""
    labels = scores[LABEL]
    base_rate = float((labels == 1).mean())
    return pd.DataFrame(
        [
            {
                "scorer": name,
                "pr_auc": pr_auc(labels, scores[name]),
                "base_rate": base_rate,
                "lift_over_base_rate": pr_auc(labels, scores[name]) / base_rate,
            }
            for name in SCORERS
        ]
    )


def flag_summary(features: pd.DataFrame, flagged: pd.Series) -> dict[str, float]:
    """What one flag form catches, so two forms of the same rule can be compared. G2."""
    mule = features[LABEL] == 1
    count = int(flagged.sum())
    caught = int((flagged & mule).sum())
    return {
        "flagged": count,
        "share": count / len(features),
        "mules_caught": caught,
        "mules": int(mule.sum()),
        "precision": caught / count if count else 0.0,
    }


def _log_rules_diagnostics(name: str, features: pd.DataFrame, rules: RulesBaseline) -> None:
    """What the rule as configured flags, and what each of its two conditions flags alone.

    A conjunction that flags almost nothing can be either condition's doing, and the two
    are fixed by different parameters, so reporting only the conjunction hides which.
    """
    flagged = rules_flag(features, rules)
    mule = features[LABEL] == 1
    n_flagged = int(flagged.sum())

    cut = features[IN_DEGREE].quantile(rules.in_degree_percentile)
    busy = features[IN_DEGREE] > cut
    passing = features[PASS_THROUGH_RATIO] > rules.pass_through_ratio_min
    LOGGER.info(
        "%-5s in_degree p%.0f cut = %.4g, flagging %d (%.2f%%); ratio > %.3g flags "
        "%d (%.2f%%); ratio p99 = %.4g",
        name,
        100.0 * rules.in_degree_percentile,
        cut,
        int(busy.sum()),
        100.0 * busy.mean(),
        rules.pass_through_ratio_min,
        int(passing.sum()),
        100.0 * passing.mean(),
        features[PASS_THROUGH_RATIO].quantile(0.99),
    )
    LOGGER.info(
        "%-5s rule as configured: flags %d of %d (%.2f%%), catching %d of %d mules, "
        "precision %.4f%%",
        name,
        n_flagged,
        len(features),
        100.0 * n_flagged / len(features),
        int((flagged & mule).sum()),
        int(mule.sum()),
        100.0 * int((flagged & mule).sum()) / n_flagged if n_flagged else 0.0,
    )


def main() -> None:
    """Fit all three scorers and write one scores parquet per window.

    Economics reads those files rather than recomputing a score, so the operating point and
    anything else downstream describe the same scorer. PRD invariant 5.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()
    rules = params.rules_baseline
    windows = build_windows(params.split, params.windows.label_window_days)
    matrices = {
        window.name: pd.read_parquet(params.paths.interim_dir / f"features_{window.name}.parquet")
        for window in windows
    }

    for name, features in matrices.items():
        nulls = features[feature_columns(features)].isna().mean()
        LOGGER.info(
            "%-5s null rate by feature, the columns logistic regression has to impute: %s",
            name,
            ", ".join(f"{c} {v:.1%}" for c, v in nulls[nulls > 0].sort_values().items()),
        )

    pipeline = fit_logistic(matrices["train"], params.logistic, params.seed)
    classifier = fit_xgboost(matrices["train"], matrices["val"], params.model, params.seed)

    measured: dict[str, Any] = {
        "scale_pos_weight": scale_pos_weight(matrices["train"][LABEL]),
        "train_clean": int((matrices["train"][LABEL] == 0).sum()),
        "train_mules": int((matrices["train"][LABEL] == 1).sum()),
        "trees_kept": int(classifier.best_iteration) + 1,
        "trees_allowed": int(params.model["n_estimators"]),
        "bootstrap_resamples": params.evaluation.bootstrap_resamples,
        "null_rates": {name: _null_rates(frame) for name, frame in matrices.items()},
        "windows": {},
    }

    importance = pd.Series(
        classifier.feature_importances_, index=feature_columns(matrices["train"])
    ).sort_values(ascending=False)
    measured["importance"] = {k: float(v) for k, v in importance.items()}
    LOGGER.info(
        "xgboost gain-weighted importance, top 8:\n%s",
        importance.head(8).to_string(float_format=lambda v: f"{v:.4f}"),
    )

    for window in windows:
        features = matrices[window.name]
        _log_rules_diagnostics(window.name, features, rules)

        # Two forms of one rule, reported side by side. D27. The configured form is the one
        # specified before any model ran and it is not touched.
        configured = flag_summary(features, rules_flag(features, rules))
        matched = flag_summary(features, matched_rules_flag(features, rules.in_degree_percentile))
        LOGGER.info(
            "%-5s flag as configured: %d flagged (%.3f%%), %d mules, precision %.4f%% | "
            "both cuts at p%.0f: %d flagged (%.3f%%), %d mules, precision %.4f%%",
            window.name,
            configured["flagged"],
            100.0 * configured["share"],
            configured["mules_caught"],
            100.0 * configured["precision"],
            100.0 * rules.in_degree_percentile,
            matched["flagged"],
            100.0 * matched["share"],
            matched["mules_caught"],
            100.0 * matched["precision"],
        )

        scores = pd.DataFrame(
            {
                "rules": rules_score(features, rules),
                "rules_flag": rules_flag(features, rules),
                "logistic": logistic_score(pipeline, features, params.logistic),
                "xgboost": xgboost_score(classifier, features),
                LABEL: features[LABEL],
            }
        )
        if scores[list(SCORERS)].isna().to_numpy().any():
            raise SchemaError(f"{window.name}: a scorer returned NaN on the pinned population.")

        destination = params.paths.interim_dir / f"scores_{window.name}.parquet"
        scores.to_parquet(destination)
        LOGGER.info("%-5s wrote %s", window.name, destination.name)

        table = pr_table(scores)
        base_rate = float(table["base_rate"].iloc[0])
        LOGGER.info(
            "%-5s PR-AUC against a base rate of %.4f%%\n%s",
            window.name,
            100.0 * base_rate,
            table.to_string(index=False, float_format=lambda v: f"{v:.6f}"),
        )
        measured["windows"][window.name] = {
            "base_rate": base_rate,
            "flag_configured": configured,
            "flag_matched": matched,
        }

        # Train performance is never a reported claim, so it gets neither an interval nor
        # a figure. The bootstrap is the slowest thing in the build and this halves it.
        if window.name == "train":
            continue

        labels = scores[LABEL]
        ranked = scores[list(SCORERS)]
        draws = bootstrap_draws(labels, ranked, params.evaluation.bootstrap_resamples, params.seed)
        summary = summarise_draws(draws, labels, ranked)
        paired = paired_differences(draws, labels, ranked, PAIRS)
        LOGGER.info(
            "%-5s PR-AUC with %d-resample 95%% intervals\n%s\n%s",
            window.name,
            params.evaluation.bootstrap_resamples,
            summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"),
            paired.to_string(index=False, float_format=lambda v: f"{v:.6f}"),
        )
        measured["windows"][window.name]["pr_auc"] = summary.to_dict(orient="records")
        measured["windows"][window.name]["paired"] = paired.to_dict(orient="records")

        curves = {}
        for name in SCORERS:
            precision, recall, _ = precision_recall_curve(labels, scores[name])
            curves[name] = (recall, precision)
        figure_path = params.paths.figures_dir / f"pr_curve_{window.name}.png"
        charts.pr_curve(
            curves,
            summary,
            base_rate=base_rate,
            window=window.name,
            costs=params.costs,
            out=figure_path,
        )
        LOGGER.info("%-5s wrote %s", window.name, figure_path)

    write_metrics(params.paths.reports_dir / "metrics_models.json", measured)
    LOGGER.info("wrote %s", params.paths.reports_dir / "metrics_models.json")


if __name__ == "__main__":
    main()
