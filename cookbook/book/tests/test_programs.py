"""Hermetic unit tests for how a chapter finds the surfaces it runs.

A chapter that shells out to the `jammi` CLI measures the CLI built from the
code beside it. When that binary already exists — CI renders the book against
the binaries its run built — `JAMMI_CLI_BIN` names it, and the chapter must
run that binary rather than compile another.
"""

from __future__ import annotations

from jammi_cookbook import programs


def test_cli_runs_the_binary_jammi_cli_bin_names(monkeypatch, tmp_path):
    binary = tmp_path / "jammi"
    binary.write_text("")
    monkeypatch.setenv("JAMMI_CLI_BIN", str(binary))
    monkeypatch.setattr(programs, "_run", _refuse_to_build)
    assert programs.cli("0.0.0") == str(binary)


def _refuse_to_build(args, cwd=None, env=None):
    raise AssertionError(f"a named CLI must not be built: {' '.join(args)}")
