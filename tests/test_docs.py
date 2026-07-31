"""The build's only check on prose. Everything else here enforces numbers and behaviour.

Three defects in one session came from the same gap: ``make lint`` reads types, ``make test``
reads behaviour, and nothing read a sentence. Four module docstrings had drifted from their code,
``src/report.py`` shipped a qualitative claim that was true of one dataset variant and false of the
next, and the test count in ``README.md`` went stale in three places across four sessions with
nothing to catch it.

These tests do not check that the prose is good. They check the three kinds of claim a machine can
verify: a count of something the repo contains, a list of files a module reads, and a measured
number that also appears in a document. That is a narrow slice of what can go wrong and it is the
slice that actually went wrong.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.definitions import PROJECT_ROOT, load_params

TESTS_DIR = PROJECT_ROOT / "tests"
README = PROJECT_ROOT / "README.md"
RESULTS_PAGE = PROJECT_ROOT / "docs" / "results.md"
GLOSSARY = PROJECT_ROOT / "docs" / "glossary.md"
PIPELINE_DIAGRAM = PROJECT_ROOT / "docs" / "diagrams" / "pipeline.svg"
REPORT_MODULE = PROJECT_ROOT / "src" / "report.py"

#: The generated report, read from the configured variant's own directory so that a run under
#: ``PARAMS_PATH`` checks its own output. Public, unlike the pages that carried these guards
#: before: ``docs/method.md`` and ``docs/model_card.md`` are local only, so a published snapshot
#: would ship this test without the files it read.
REPORT_PAGE = load_params().paths.reports_dir / "report.md"

#: Every prose document a reader of the published repository gets.
PUBLIC_DOCUMENTS = (README, RESULTS_PAGE, GLOSSARY, REPORT_PAGE)

#: The two banned characters, as escapes. Writing them literally would make this file fail the
#: rule it enforces, which ``ruff`` points out with RUF001.
EM_DASH = "\u2014"
EN_DASH = "\u2013"

#: Documents that state the test count, and have each been wrong about it at least once.
COUNT_DOCUMENTS = (README, PIPELINE_DIAGRAM)


def _collected_whole_suite(request: pytest.FixtureRequest) -> bool:
    """Whether this run collected every test module, so a total means anything.

    Running one file or filtering with ``-k`` collects a subset, and asserting a total against
    that would fail for a reason that has nothing to do with the documentation.
    """
    collected = {Path(str(item.path)).stem for item in request.session.items}
    on_disk = {path.stem for path in TESTS_DIR.glob("test_*.py")}
    return collected == on_disk


def test_the_documented_test_count_is_the_count_the_suite_actually_has(
    request: pytest.FixtureRequest,
) -> None:
    """The count is read from pytest's own collection, never from counting ``def test_``.

    Counting definitions currently gives 156 where pytest collects 159, because one test is
    parametrised. A guard that disagrees with the thing it guards is worse than none, and that
    gap is why this reads ``request.session`` rather than the source.
    """
    if not _collected_whole_suite(request):
        pytest.skip("partial run, so the collected total is not the suite's total")

    total = len(request.session.items)
    wrong = [
        document.relative_to(PROJECT_ROOT)
        for document in COUNT_DOCUMENTS
        if f"{total} tests" not in document.read_text(encoding="utf-8")
    ]

    assert not wrong, (
        f"The suite has {total} tests and these documents do not say so: {wrong}. "
        f"Update them rather than this test."
    )


def test_no_document_states_a_test_count_that_is_not_the_current_one(
    request: pytest.FixtureRequest,
) -> None:
    """The other half. Adding the right number somewhere does not remove a wrong one elsewhere.

    This is how the count survived being updated twice in one session: the README was corrected
    and the number baked into the diagram was not.
    """
    if not _collected_whole_suite(request):
        pytest.skip("partial run, so the collected total is not the suite's total")

    total = len(request.session.items)
    stale: list[str] = []
    for document in COUNT_DOCUMENTS:
        for found in re.findall(r"(\d[\d,]*) tests", document.read_text(encoding="utf-8")):
            if int(found.replace(",", "")) != total:
                stale.append(f"{document.relative_to(PROJECT_ROOT)} says {found} tests")

    assert not stale, f"The suite has {total} tests. Stale claims: {stale}"


def _metrics_files_read_by(module: Path) -> set[str]:
    """Every ``metrics_*.json`` the module loads, taken from its source rather than a list."""
    return set(re.findall(r'read_metrics\(\s*reports / "([^"]+)"', module.read_text("utf-8")))


def _module_docstring(module: Path) -> str:
    parsed = ast.parse(module.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(parsed)
    assert docstring is not None, f"{module.name} has no module docstring"
    return docstring


def test_the_report_docstring_names_every_metrics_file_it_reads() -> None:
    """The exact defect this file exists for, pinned.

    ``src/report.py`` states as an exhaustive rule which metrics files its numbers come from.
    G6 added a fourth file and the docstring kept naming three, so the rule was false while
    looking like every other invariant in the project.
    """
    named = _module_docstring(REPORT_MODULE)
    unnamed = sorted(name for name in _metrics_files_read_by(REPORT_MODULE) if name not in named)

    assert not unnamed, (
        f"src/report.py reads {unnamed} and its docstring does not name them. The docstring "
        f"claims to list every source of every number in the report."
    )


def test_the_report_docstring_does_not_name_a_metrics_file_it_stopped_reading() -> None:
    """The other direction. A docstring promising a source that is gone is the same defect."""
    named = _module_docstring(REPORT_MODULE)
    actually_read = _metrics_files_read_by(REPORT_MODULE)
    claimed = set(re.findall(r"``(metrics_\w+\.json)``", named))

    assert not claimed - actually_read, (
        f"src/report.py names {sorted(claimed - actually_read)} in its docstring and does not "
        f"read them."
    )


def _published_metrics() -> dict[str, Any] | None:
    """The metrics of the configured variant, or None if the pipeline has not been run."""
    reports = load_params().paths.reports_dir
    loaded: dict[str, Any] = {}
    for stem in ("data", "models", "economics"):
        path = reports / f"metrics_{stem}.json"
        if not path.is_file():
            return None
        loaded[stem] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


#: Claims a machine can locate and check, each anchored on the words around it so the number is
#: read inside its own sentence. A bare value search is not enough: ``6,924,049`` appears several
#: times across these documents, so changing one occurrence and leaving the rest would pass a search
#: and still ship a set of documents that contradict each other. The list spans the README and the
#: pages the deep material moved to, so a number is checked wherever it now lives. Each entry is
#: (document, pattern, metric).
ANCHORED_CLAIMS: tuple[tuple[Path, str, str], ...] = (
    (README, r"A pipeline over ([\d,]+) synthetic transactions", "rows_read"),
    (README, r"raw data of ([\d,]+) payments is cleaned down to", "rows_read"),
    (README, r"payments is cleaned down to ([\d,]+) accounts", "accounts"),
    (README, r"Out of\n([\d,]+) accounts that took money in", "test_population"),
    (README, r"\| Caught of ([\d,]+) \|", "test_mules"),
    (RESULTS_PAGE, r"\| Test window, ([\d,]+) accounts \| Flagged", "test_population"),
    (RESULTS_PAGE, r"re-draws of the ([\d,]+) scored\s+accounts", "test_population"),
    (RESULTS_PAGE, r"Mule accounts caught of ([\d,]+) \| Precision", "test_mules"),
    (PIPELINE_DIAGRAM, r"Raw data of ([\d,]+) payments", "rows_read"),
    (PIPELINE_DIAGRAM, r"cleaned down to ([\d,]+) accounts", "accounts"),
    (PIPELINE_DIAGRAM, r">([\d,]+) payments</text>", "rows_read"),
    (PIPELINE_DIAGRAM, r">([\d,]+) accounts, ten usable days</text>", "accounts"),
    (REPORT_PAGE, r"Rows read: ([\d,]+), over", "rows_read"),
    (REPORT_PAGE, r"over ([\d,]+) accounts keyed as \(bank, account\)", "accounts"),
    (REPORT_PAGE, r"\| test \| ([\d,]+) \|", "test_population"),
    (REPORT_PAGE, r"\| test \| [\d,]+ \| ([\d,]+) \|", "test_mules"),
)


def _measured(metrics: dict[str, Any]) -> dict[str, int]:
    """The quantities the anchored claims are checked against."""
    data = metrics["data"]
    test_window = data["windows"]["test"]
    return {
        "rows_read": int(data["rows_read"]),
        "accounts": int(data["accounts"]),
        "test_population": int(test_window["population"]),
        "test_mules": int(test_window["mules"]),
    }


def test_every_anchored_claim_carries_the_number_the_pipeline_measured() -> None:
    """The real guard. Each number is read out of the sentence that makes the claim.

    A pattern that stops matching is a failure too. Prose gets reworded, and a claim whose anchor
    has moved is no longer being checked, which is the silent version of the defect.
    """
    metrics = _published_metrics()
    if metrics is None:
        pytest.skip("no metrics files, so there is nothing measured to check the documents against")

    measured = _measured(metrics)
    wrong: list[str] = []
    for document, pattern, metric in ANCHORED_CLAIMS:
        where = document.relative_to(PROJECT_ROOT).as_posix()
        found = re.findall(pattern, document.read_text(encoding="utf-8"))
        if not found:
            wrong.append(f"{where}: pattern {pattern!r} matched nothing, so {metric} is unchecked")
            continue
        for value in found:
            if int(value.replace(",", "")) != measured[metric]:
                wrong.append(f"{where}: says {value} for {metric}, measured {measured[metric]:,}")

    assert not wrong, "Documents disagree with the pipeline:\n  " + "\n  ".join(wrong)


def test_the_documents_carry_no_dash_the_voice_rules_ban() -> None:
    """Em dashes and en dashes, banned outright, and re-checked here rather than by grepping."""
    offenders = {
        document.relative_to(PROJECT_ROOT).as_posix(): document.read_text("utf-8").count(dash)
        for document in PUBLIC_DOCUMENTS
        if document.is_file()
        for dash in (EM_DASH, EN_DASH)
        if dash in document.read_text(encoding="utf-8")
    }

    assert not offenders, f"Banned dashes found: {offenders}"


#: One term per meaning, and the forms that mean the same thing and are therefore out. ASD-STE100
#: asks for one word for one meaning, and synonym cycling is what made these pages hard to follow:
#: the same window was a history window, a lookback, and a scoring window inside one page.
REJECTED_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("scoring window", "label window"),
    ("scored window", "label window"),
    ("history window", "feature window"),
    ("lookback", "feature window"),
    ("the generator", "the simulator"),
    ("mules", "mule accounts"),
)


def test_no_public_document_cycles_a_synonym_for_a_term_that_is_already_fixed() -> None:
    """A reader who is new to the domain cannot tell two names for one thing apart."""
    offenders: list[str] = []
    for document in PUBLIC_DOCUMENTS:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        where = document.relative_to(PROJECT_ROOT).as_posix()
        offenders.extend(
            f"{where} says {rejected!r}, which this project calls {preferred!r}"
            for rejected, preferred in REJECTED_SYNONYMS
            if re.search(rf"\b{re.escape(rejected)}\b", text, flags=re.IGNORECASE)
        )

    assert not offenders, "Synonyms for a fixed term:\n  " + "\n  ".join(offenders)


#: Terms that carry the argument, and the phrase that defines each one. The definition has to
#: reach the reader before the term does, so this checks position and not just presence. Each
#: entry is (document, term, defining phrase).
GLOSSED_ON_FIRST_USE: tuple[tuple[Path, str, str], ...] = (
    (README, "mule account", "That account is a mule account"),
    (README, "leakage", "leakage, which means a feature reading a payment"),
    (RESULTS_PAGE, "mule account", "A mule account is an account that received at least one"),
    (RESULTS_PAGE, "in-degree", "The measure is in-degree, the count of"),
    (RESULTS_PAGE, "pass-through ratio", "The measure is the pass-through ratio, the money"),
    (RESULTS_PAGE, "PR-AUC", "PR-AUC, the area under the precision-recall curve"),
    (RESULTS_PAGE, "base rate", "the base rate, which is the share of the"),
    (RESULTS_PAGE, "crosses zero", "An interval that crosses zero leaves the sign"),
    (RESULTS_PAGE, "threshold", "The threshold is the score of the last account"),
    (RESULTS_PAGE, "PSI", "the Population Stability Index, PSI"),
    (REPORT_PAGE, "PR-AUC", "PR-AUC is the area under the precision-recall curve"),
    (REPORT_PAGE, "alert", "An alert is an account the analyst team opens"),
    (REPORT_PAGE, "PSI", "the Population Stability Index, PSI"),
)


def test_every_load_bearing_term_is_defined_before_the_document_first_uses_it() -> None:
    """The defect this file was extended for: a page that assumes the reader knows the field.

    A definition that has drifted below its term stops doing the one job it has, and reads as
    correct to anyone who already knew the term.
    """
    wrong: list[str] = []
    for document, term, definition in GLOSSED_ON_FIRST_USE:
        if not document.is_file():
            pytest.skip(f"{document.name} is not present, so nothing can be read from it")
        # Line wrapping is not a defect, so both sides are read as one flowed line. Without this
        # the guard fails whenever a paragraph is re-wrapped, which trains a reader to edit the
        # test rather than the prose.
        text = re.sub(r"\s+", " ", document.read_text(encoding="utf-8"))
        flowed = re.sub(r"\s+", " ", definition)
        where = document.relative_to(PROJECT_ROOT).as_posix()

        defined_at = text.find(flowed)
        if defined_at == -1:
            wrong.append(f"{where}: no definition of {term!r} matching {definition!r}")
            continue
        used_at = text.lower().find(term.lower())
        if used_at < defined_at:
            wrong.append(f"{where}: uses {term!r} at {used_at} and defines it at {defined_at}")

    assert not wrong, "Terms used before they are defined:\n  " + "\n  ".join(wrong)


def test_the_glossary_defines_every_feature_the_drift_table_prints() -> None:
    """The drift table names all twenty measures, and a reader meets most of them only there."""
    reports = load_params().paths.reports_dir
    monitoring_path = reports / "metrics_monitoring.json"
    if not monitoring_path.is_file() or not GLOSSARY.is_file():
        pytest.skip("no monitoring metrics or no glossary, so there is nothing to compare")

    monitoring = json.loads(monitoring_path.read_text(encoding="utf-8"))
    glossary = GLOSSARY.read_text(encoding="utf-8")
    printed = {str(record["feature"]) for record in monitoring["feature_drift"]["test"]}
    undefined = sorted(name for name in printed if f"`{name}`" not in glossary)

    assert not undefined, (
        f"The drift table prints these and the glossary does not define them: {undefined}"
    )
