"""A session journal across processes, and the runner that judges it.

A :class:`~jammi.SessionWindow` answers "what did this process leave open" from
inside the process. A command that spawns other interpreters — a script that
runs scripts, a document renderer that starts a Jupyter kernel — leaks in a
descendant whose exit status the command never reads, and a killed interpreter
runs no exit hook at all. So the evidence is written out of band, as it
happens: when ``JAMMI_SESSION_JOURNAL`` names a directory, every process that
imports `jammi` appends each session open and close to its own file there, one
``write`` per event, with no buffering and nothing deferred to exit.

Run a command under the journal and fail it for what it left open::

    python -m jammi.session_journal -- python examples/quickstart.py

The runner exits with the command's own status when the command fails (what it
left open is printed as a warning, never stacked on the real cause), and with
status 1 when the command succeeds but a process under it left a session open,
or closed a directory-backed session after its directory was removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ._sessions import describe_sessions, observe

JOURNAL_ENV = "JAMMI_SESSION_JOURNAL"


class _JournalWriter:
    """Appends this process's session events to its own file in `directory`.

    The file is keyed by pid and reopened after a fork: handles are per-process
    counters, so two processes must never share one file's handle space.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._lock = threading.Lock()
        self._pid: Optional[int] = None
        self._fd: Optional[int] = None
        self._directories: Dict[int, str] = {}

    def _descriptor(self) -> int:
        pid = os.getpid()
        if self._fd is None or self._pid != pid:
            path = self._directory / f"{pid}-{uuid.uuid4().hex}.jsonl"
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._pid = pid
            self._directories = {}
        return self._fd

    def _append(self, event: dict) -> None:
        line = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        with self._lock:
            os.write(self._descriptor(), line)

    def on_register(self, handle: int, label: str) -> None:
        self._append({"event": "open", "handle": handle, "label": label})
        if os.path.isdir(label):
            with self._lock:
                self._directories[handle] = label

    def on_unregister(self, handle: int, _label: Optional[str]) -> None:
        with self._lock:
            directory = self._directories.pop(handle, None)
        removed = directory is not None and not os.path.isdir(directory)
        self._append({"event": "close", "handle": handle, "after_removal": removed})


def activate_from_env() -> None:
    """Journal this process's sessions for its whole life, if
    ``JAMMI_SESSION_JOURNAL`` names a directory. Called once, when `jammi` is
    imported — before any session can exist."""
    directory = os.environ.get(JOURNAL_ENV)
    if not directory:
        return
    writer = _JournalWriter(Path(directory))
    observe(writer.on_register, writer.on_unregister)


@dataclass
class JournalVerdict:
    """What the processes under one journal directory did with their sessions.
    Keys are `(journal file name, handle)` — a handle is unique only within
    the process that assigned it."""

    processes: int = 0
    opened: int = 0
    leaked: Dict[Tuple[str, int], str] = field(default_factory=dict)
    closed_after_removal: Dict[Tuple[str, int], str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.leaked and not self.closed_after_removal

    def report(self) -> List[str]:
        lines = []
        if self.leaked:
            lines.append(
                f"{len(self.leaked)} jammi session(s) were opened and never closed: "
                + describe_sessions({k[1]: v for k, v in self.leaked.items()})
            )
        if self.closed_after_removal:
            lines.append(
                f"{len(self.closed_after_removal)} jammi session(s) were closed after "
                "their directory was removed (an embedded engine holds its catalog "
                "until close() returns): "
                + describe_sessions({k[1]: v for k, v in self.closed_after_removal.items()})
            )
        return lines


def read_journal(directory: Path) -> JournalVerdict:
    """Fold every process's journal under `directory` into one verdict."""
    verdict = JournalVerdict()
    for path in sorted(directory.glob("*.jsonl")):
        verdict.processes += 1
        open_labels: Dict[int, str] = {}
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                event = json.loads(line)
                handle = event["handle"]
                if event["event"] == "open":
                    open_labels[handle] = event["label"]
                    verdict.opened += 1
                elif event["event"] == "close":
                    # A close with no open in this file is a session a forked
                    # child inherited: its opener's journal answers for it.
                    label = open_labels.pop(handle, None)
                    if label is not None and event["after_removal"]:
                        verdict.closed_after_removal[(path.name, handle)] = label
                else:
                    raise ValueError(f"unknown event {event['event']!r}")
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{number}: unreadable journal line: {error}") from error
        for handle, label in open_labels.items():
            verdict.leaked[(path.name, handle)] = label
    return verdict


def run(command: Sequence[str]) -> int:
    """Run `command` under a fresh journal; return the exit status to use."""
    with tempfile.TemporaryDirectory(prefix="jammi-session-journal-") as directory:
        env = dict(os.environ, **{JOURNAL_ENV: directory})
        status = subprocess.run(list(command), env=env, check=False).returncode
        verdict = read_journal(Path(directory))

    print(
        f"session journal: {verdict.processes} process(es) imported jammi, "
        f"{verdict.opened} session(s) opened",
        file=sys.stderr,
    )
    if verdict.clean:
        return status
    severity = "warning" if status != 0 else "error"
    for line in verdict.report():
        print(f"session journal: {severity}: {line}", file=sys.stderr)
    return status if status != 0 else 1


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2 or argv[0] != "--":
        print("usage: python -m jammi.session_journal -- <command> [args…]", file=sys.stderr)
        return 2
    return run(argv[1:])
