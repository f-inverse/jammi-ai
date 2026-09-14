"""No test module in this suite binds `jammi.connect` at import time.

`conftest.py`'s `_no_leaked_sessions` guard (`closes_escape: esc-112`) catches a
leaked session by monkeypatching the `jammi.connect` MODULE ATTRIBUTE for the
duration of each test. That patch only reaches call sites that look the attribute
up at call time — `jammi.connect(...)`, however the target URL is built. A module
that instead binds a name to the function once, at IMPORT time —

    from jammi import connect          # holds a pre-patch reference
    connect = jammi.connect            # same shape, a bare alias assignment

— captures the pre-patch function object, so every later call through that name
runs unpatched and the guard never sees the session open. This file is the
enumerating gate for that one precondition: it does not re-implement any part of
the leak detector itself, and it does not reason about `with`/`finally`/tempdir
shapes (that closed-vs-leaked question is the runtime guard's job, proven by
`test_session_lifecycle_guard.py`). It only asks whether a module bound the name
before the guard ever got to run.

Scope: every `.py` file directly under this `tests/` directory — the set the
guard's autouse fixture actually applies to. A file that binds the alias in a
nested function body (not at module scope) does not pre-empt the patch, since the
lookup then happens after `_no_leaked_sessions` has already substituted the
attribute; this gate only flags a column-zero, module-level binding.
"""

from __future__ import annotations

import re
from pathlib import Path

# Positive shape 1: `from jammi import connect` (possibly alongside other names).
_IMPORT_ALIAS = re.compile(r"^from jammi import .*\bconnect\b")
# Positive shape 2: a bare alias assignment to the function itself — no trailing
# `(`, which is what distinguishes `connect = jammi.connect` from the ordinary,
# legal `db = jammi.connect(f"file://{d}")` call-through-the-attribute shape.
_ASSIGN_ALIAS = re.compile(r"^\w+\s*=\s*jammi\.connect\b(?!\()")


def _alias_offenses(text: str) -> list[str]:
    """Every line of `text` that binds `jammi.connect` at module (column-zero)
    scope, either shape. Returns the offending lines themselves so a caller can
    report them; an empty list means the text is clean."""
    return [
        line
        for line in text.splitlines()
        if _IMPORT_ALIAS.match(line) or _ASSIGN_ALIAS.match(line)
    ]


def test_alias_detector_controls():
    """The detector, on synthetic text, before it is ever pointed at the real
    suite — a detector that matched nothing at all would pass the real gate for
    the wrong reason."""
    # positive: import-alias shape, flagged regardless of what else is imported.
    assert _alias_offenses("from jammi import connect, Capability\n")
    assert _alias_offenses("from jammi import Capability, connect\n")
    # positive: bare assignment-alias shape.
    assert _alias_offenses("connect = jammi.connect\n")
    assert _alias_offenses("local_connect = jammi.connect\n")

    # negative: an import that does not name `connect` at all.
    assert not _alias_offenses("from jammi import Capability, Session\n")
    # negative: the legal module-attribute *call* shape (a trailing `(` makes it
    # a call, not an alias) — this is how every test in the suite is meant to
    # open a session.
    assert not _alias_offenses('db = jammi.connect(f"file://{d}")\n')
    assert not _alias_offenses('    db = jammi.connect("grpc://127.0.0.1:8081")\n')
    # negative: `jammi.connect` mentioned only in a docstring/comment about the
    # attribute, not bound to a name — no leading `word =` or `from jammi import`.
    assert not _alias_offenses("# patches jammi.connect for the duration of a test\n")
    # negative: the same two alias shapes, but indented (a nested function
    # scope, not module scope) — the Scope paragraph above says this gate only
    # flags a column-zero binding, so these must NOT be flagged.
    assert not _alias_offenses("    from jammi import connect\n")
    assert not _alias_offenses("    connect = jammi.connect\n")


def test_no_test_module_binds_connect_at_import_time():
    """The real gate: sweep every `.py` module directly under this directory."""
    offenders = [
        f"{p.name}:{n}: {line.strip()}"
        for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if _IMPORT_ALIAS.match(line) or _ASSIGN_ALIAS.match(line)
    ]
    assert not offenders, (
        "these lines bind `jammi.connect` at import time, which escapes the "
        f"conftest module-attribute leak guard: {offenders}"
    )
