"""UAT shape-C: tenant-scope isolation across primitives from a Python client.

Single Python session rebinds between two tenants on the same artifact
directory and verifies that the tenant-scope analyzer filters rows
correctly across the mutable-table and topic-catalog surfaces:

  - Tenant A registers `notes` mutable table and `events.a` topic.
  - Tenant B (same session, rebound) sees zero `notes` rows and no
    `events.a` topic.
  - Rebinding back to tenant A restores the original visibility.

Validates that the cross-tenant isolation property cp9's Phase 3 plan
committed to holds at the PyO3-binding boundary.

Run with `python3 tests/uat/shape_c_isolation.py`. Exits 0 on success.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa

import jammi

# Generic UUIDs — not coupled to any flagship tenant identity.
TENANT_A = "00000000-0000-0000-0000-0000000000a1"
TENANT_B = "00000000-0000-0000-0000-0000000000b2"


def _notes_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("note_id", pa.int64(), nullable=False),
            pa.field("body", pa.string(), nullable=False),
        ]
    )


def _events_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_id", pa.int64(), nullable=False),
            pa.field("payload", pa.string(), nullable=False),
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = str(Path(tmp))
        db = jammi.connect(artifact_dir=artifact_dir)

        # Tenant A: register both primitives and insert one row.
        db.with_tenant(TENANT_A)
        db.create_mutable_table(
            "notes", schema=_notes_schema(), primary_key=["note_id"]
        )
        db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (1, 'a-row')")
        db.register_topic("events.a", schema=_events_schema())

        count_a = (
            db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
            .column("n")
            .to_pylist()[0]
        )
        assert count_a == 1, f"tenant A sees {count_a} rows, expected 1"
        assert "events.a" in db.list_topics(), "tenant A's topic missing"

        # Tenant B: same session, rebound. Row authored under A must be
        # invisible; topic registered under A must be excluded.
        db.with_tenant(TENANT_B)
        count_b = (
            db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
            .column("n")
            .to_pylist()[0]
        )
        assert count_b == 0, f"tenant B leaked {count_b} rows from tenant A"
        assert "events.a" not in db.list_topics(), "tenant B leaked tenant A's topic"

        # Rebinding back to tenant A restores visibility.
        db.with_tenant(TENANT_A)
        count_a_again = (
            db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
            .column("n")
            .to_pylist()[0]
        )
        assert count_a_again == 1, (
            f"tenant A lost row on rebind: {count_a_again}"
        )
        assert "events.a" in db.list_topics(), "tenant A lost topic on rebind"

        db.drop_topic("events.a")
        db.drop_mutable_table("notes")

    print("shape_c_isolation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
