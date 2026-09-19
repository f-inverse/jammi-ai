"""`python -m jammi.session_journal` and `jammi.SessionWindow`.

The runner is driven as a real subprocess over throwaway scripts, one per
process shape a leak can hide in: the command itself, a grandchild whose exit
status the command never reads, an interpreter that dies without running an
exit hook, and a Jupyter kernel a notebook client starts and shuts down. Every
script is hermetic: a remote channel dials nothing until it is used, and the
embedded route runs over a fake native handle whose label is a real directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

import jammi
from jammi import EmbeddedBackend, SessionWindow
from jammi._session_journal import JOURNAL_ENV

TARGET = "grpc://127.0.0.1:8081"
# A remote session's label is its endpoint, without the scheme.
LABEL = "127.0.0.1:8081"

_FAKE_EMBEDDED = """
import jammi

class _FakeNative:
    def close(self, release=False):
        pass
"""


def _run_under_journal(tmp_path: Path, body: str, *extra: str) -> subprocess.CompletedProcess:
    script = tmp_path / "lane.py"
    script.write_text(textwrap.dedent(body))
    env = {k: v for k, v in os.environ.items() if k != JOURNAL_ENV}
    return subprocess.run(
        [sys.executable, "-m", "jammi.session_journal", "--", sys.executable, str(script), *extra],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_command_that_leaves_a_session_open_fails_by_label(tmp_path):
    result = _run_under_journal(
        tmp_path,
        f"""
        import jammi
        jammi.connect({TARGET!r})
        """,
    )
    assert result.returncode == 1, result.stderr
    assert "opened and never closed" in result.stderr
    assert repr(LABEL) in result.stderr


def test_a_command_that_closes_every_session_passes(tmp_path):
    result = _run_under_journal(
        tmp_path,
        f"""
        import jammi
        with jammi.connect({TARGET!r}):
            pass
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "1 process(es) imported jammi, 1 session(s) opened" in result.stderr


def test_a_leak_in_a_grandchild_whose_status_nobody_reads_still_fails(tmp_path):
    leaker = tmp_path / "leaker.py"
    leaker.write_text(f"import jammi\njammi.connect({TARGET!r})\n")
    result = _run_under_journal(
        tmp_path,
        f"""
        import subprocess, sys
        subprocess.run([sys.executable, {str(leaker)!r}], check=False)
        """,
    )
    assert result.returncode == 1, result.stderr
    assert repr(LABEL) in result.stderr


@pytest.mark.parametrize(
    "death",
    ["os._exit(0)", "os.kill(os.getpid(), signal.SIGKILL)"],
    ids=["os._exit", "SIGKILL"],
)
def test_a_leak_in_a_process_that_runs_no_exit_hook_is_still_reported(tmp_path, death):
    leaker = tmp_path / "leaker.py"
    leaker.write_text(f"import os, signal, jammi\njammi.connect({TARGET!r})\n{death}\n")
    result = _run_under_journal(
        tmp_path,
        f"""
        import subprocess, sys
        subprocess.run([sys.executable, {str(leaker)!r}], check=False)
        """,
    )
    assert result.returncode == 1, result.stderr
    assert repr(LABEL) in result.stderr


def test_a_failing_command_keeps_its_own_status_and_the_leak_is_a_warning(tmp_path):
    result = _run_under_journal(
        tmp_path,
        f"""
        import sys, jammi
        jammi.connect({TARGET!r})
        sys.exit(3)
        """,
    )
    assert result.returncode == 3, result.stderr
    assert "session journal: warning:" in result.stderr
    assert "session journal: error:" not in result.stderr
    assert repr(LABEL) in result.stderr


def test_a_session_closed_after_its_directory_was_removed_fails_by_label(tmp_path):
    result = _run_under_journal(
        tmp_path,
        _FAKE_EMBEDDED
        + """
import tempfile
with tempfile.TemporaryDirectory() as catalog:
    db = jammi.EmbeddedBackend(_FakeNative(), label=catalog)
    print(catalog)
db.close()
""",
    )
    assert result.returncode == 1, result.stderr
    assert "closed after their directory was removed" in result.stderr
    assert repr(result.stdout.strip()) in result.stderr


def test_a_session_closed_inside_its_directory_scope_passes(tmp_path):
    result = _run_under_journal(
        tmp_path,
        _FAKE_EMBEDDED
        + """
import tempfile
with tempfile.TemporaryDirectory() as catalog:
    with jammi.EmbeddedBackend(_FakeNative(), label=catalog):
        pass
""",
    )
    assert result.returncode == 0, result.stderr


def test_without_the_variable_nothing_is_journaled(tmp_path):
    script = tmp_path / "lane.py"
    script.write_text(f"import jammi\njammi.connect({TARGET!r})\n")
    env = {k: v for k, v in os.environ.items() if k != JOURNAL_ENV}
    before = set(Path(tempfile.gettempdir()).glob("jammi-session-journal-*"))
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert set(Path(tempfile.gettempdir()).glob("jammi-session-journal-*")) == before


def test_a_leak_inside_a_jupyter_kernel_fails_the_command_that_ran_the_notebook(tmp_path):
    """The shape a document renderer has: the command starts a kernel through a
    notebook client, the leak is in the kernel, and the command never reads the
    kernel's exit status."""
    result = _run_under_journal(
        tmp_path,
        f"""
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.v4.new_notebook()
        notebook.cells = [
            nbformat.v4.new_code_cell("import jammi"),
            nbformat.v4.new_code_cell("db = jammi.connect({TARGET!r})"),
        ]
        NotebookClient(notebook, kernel_name="python3").execute()
        """,
    )
    assert result.returncode == 1, result.stderr
    assert repr(LABEL) in result.stderr


def test_the_runner_refuses_a_command_line_without_the_separator():
    result = subprocess.run(
        [sys.executable, "-m", "jammi.session_journal", "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr


# --------------------------------------------------------------------------- #
# SessionWindow, in process
# --------------------------------------------------------------------------- #


class _FakeNative:
    def close(self, release: bool = False) -> None:
        pass


def test_a_window_reports_what_it_saw_open_and_not_close():
    with SessionWindow() as window:
        leaked = jammi.connect(TARGET)
        with jammi.connect("grpc://127.0.0.1:8082"):
            pass
    try:
        assert list(window.leaked().values()) == [LABEL]
    finally:
        leaked.close()


def test_a_window_does_not_answer_for_a_session_opened_before_it():
    earlier = jammi.connect(TARGET)
    with SessionWindow() as window:
        earlier.close()
    assert window.leaked() == {}
    assert window.closed_after_removal() == {}


def test_a_window_sees_a_session_dropped_without_a_reference():
    with SessionWindow() as window:
        EmbeddedBackend(_FakeNative(), label="/data/dropped")
    assert list(window.leaked().values()) == ["/data/dropped"]


def test_a_window_reports_a_directory_backed_session_closed_after_removal(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    with SessionWindow() as window:
        db = EmbeddedBackend(_FakeNative(), label=str(catalog))
        shutil.rmtree(catalog)
        db.close()
    assert window.leaked() == {}
    assert list(window.closed_after_removal().values()) == [str(catalog)]


def test_a_window_cannot_be_opened_twice():
    window = SessionWindow().open()
    try:
        with pytest.raises(RuntimeError):
            window.open()
    finally:
        window.close()
