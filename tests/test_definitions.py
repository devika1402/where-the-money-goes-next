"""Which parameter file the pipeline reads. Decision D37.

G4 runs the whole pipeline on a second dataset variant beside the published one rather than
in place of it, so the variant needs its own parameter file and its own output directories.
The override names which file to read and nothing else, which keeps PRD invariant 1 intact:
every quantity still comes from a parameter file and none of them moves to a command line.

The failure these tests exist to prevent is silent. An override pointing at a file that is not
there must stop the run, because falling back to the published parameters would produce a full
set of HI-Small figures under an LI-Small heading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.definitions import (
    DEFAULT_PARAMS_PATH,
    PARAMS_PATH_ENV,
    PROJECT_ROOT,
    SchemaError,
    default_params_path,
)


def test_the_published_parameter_file_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing set, the pipeline reads the file every published figure came from."""
    monkeypatch.delenv(PARAMS_PATH_ENV, raising=False)

    assert default_params_path() == DEFAULT_PARAMS_PATH


def test_an_override_names_the_file_to_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other = tmp_path / "params_other.yaml"
    other.write_text("seed: 42\n", encoding="utf-8")
    monkeypatch.setenv(PARAMS_PATH_ENV, str(other))

    assert default_params_path() == other


def test_a_relative_override_resolves_against_the_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """So the variant can be named the way the Makefile names it, without an absolute path."""
    monkeypatch.setenv(PARAMS_PATH_ENV, "config/params.yaml")

    assert default_params_path() == PROJECT_ROOT / "config" / "params.yaml"


def test_an_override_pointing_at_nothing_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point. A typo must not silently republish the default parameters."""
    monkeypatch.setenv(PARAMS_PATH_ENV, str(tmp_path / "absent.yaml"))

    with pytest.raises(SchemaError, match="does not exist"):
        default_params_path()
