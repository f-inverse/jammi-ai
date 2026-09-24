"""What every cache-building script shares: where the engine's own fixtures
are, and the checksum record written beside a chapter's committed artifacts.

A build script (``scripts/build_*_cache.py``) runs the engine once and commits
what it measured; these are the two steps each of them takes the same way.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def engine_fixtures_root(arg: str | None, *required: str) -> Path:
    """The engine checkout whose fixtures a build script runs against.

    ``arg`` is the script's ``--fixtures-root`` (else ``JAMMI_FIXTURES_ROOT``);
    every path in ``required`` (relative to the checkout) must exist, and the
    first one missing is named in the refusal.
    """
    root = arg or os.environ.get("JAMMI_FIXTURES_ROOT")
    if not root:
        raise SystemExit(
            "pass --fixtures-root (or set JAMMI_FIXTURES_ROOT) to the engine checkout "
            f"carrying {', '.join(required)}"
        )
    checkout = Path(root).resolve()
    for relative in required:
        if not (checkout / relative).exists():
            raise SystemExit(f"--fixtures-root {checkout} has no {relative}")
    return checkout


def write_checksums(artifacts: Path) -> None:
    """Record the sha256 of every committed artifact in ``artifacts`` as
    ``checksums.json`` beside them."""
    sums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(artifacts.glob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (artifacts / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))
