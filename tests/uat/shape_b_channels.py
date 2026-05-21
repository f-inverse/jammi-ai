"""UAT shape-B: drive the provenance-channel primitive from a Python client.

A single-tenant Jammi session (catalog on disk, no tenant binding)
registers a new provenance channel, appends columns to it, and verifies
the append-only invariant (retyping an existing column fails). Validates
that the PyO3 binding contract Item 1 shipped is wired end-to-end.

The "over Flight SQL + gRPC" wire variant is queued for a future
iteration — the substrate-side surface is exercised here through the
embedded `jammi.connect` path, which is what every adopter who builds
on the PyO3 surface will see.

Run with `python3 tests/uat/shape_b_channels.py`. Exits 0 on success.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import jammi


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = str(Path(tmp))
        db = jammi.connect(artifact_dir=artifact_dir)

        db.register_channel(
            "scored_by",
            priority=3,
            columns=[("ranker", "Utf8"), ("rank_score", "Float32")],
        )
        db.add_channel_columns(
            "scored_by",
            columns=[("model_version", "Utf8")],
        )

        # Append-only invariant: retyping an existing column must fail.
        try:
            db.add_channel_columns(
                "scored_by",
                columns=[("ranker", "Int32")],
            )
        except (RuntimeError, ValueError) as exc:
            msg = str(exc).lower()
            assert "ranker" in msg, f"error message lost column name: {exc}"
        else:
            raise AssertionError("retype must raise")

        # Duplicate-register must fail.
        try:
            db.register_channel(
                "scored_by",
                priority=4,
                columns=[("ranker", "Utf8")],
            )
        except (RuntimeError, ValueError) as exc:
            assert "scored_by" in str(exc), f"duplicate-register error lost id: {exc}"
        else:
            raise AssertionError("duplicate register must raise")

    print("shape_b_channels: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
