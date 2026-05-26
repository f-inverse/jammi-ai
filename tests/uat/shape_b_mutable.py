"""UAT shape-B: drive the mutable-table primitive from a Python client.

Creates a mutable companion table through the PyO3 binding, writes rows
through DataFusion DML, reads them back, and drops the table. Validates
that the create/drop lifecycle Item 1 shipped round-trips end-to-end.

Run with `python3 tests/uat/shape_b_mutable.py`. Exits 0 on success.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa

import jammi_ai


def _notes_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("note_id", pa.int64(), nullable=False),
            pa.field("body", pa.string(), nullable=False),
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = str(Path(tmp))
        db = jammi_ai.connect(artifact_dir=artifact_dir)

        table_id = db.create_mutable_table(
            "notes",
            schema=_notes_schema(),
            primary_key=["note_id"],
        )
        assert table_id == "notes", f"expected 'notes', got {table_id}"

        db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (1, 'one')")
        db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (2, 'two')")
        db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (3, 'three')")

        count = (
            db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
            .column("n")
            .to_pylist()[0]
        )
        assert count == 3, f"expected 3 rows, got {count}"

        bodies = sorted(
            db.sql("SELECT body FROM mutable.public.notes ORDER BY note_id")
            .column("body")
            .to_pylist()
        )
        assert bodies == ["one", "three", "two"], f"unexpected rows {bodies}"

        db.drop_mutable_table("notes")

        # After drop the federated SQL surface no longer resolves the table.
        try:
            db.sql("SELECT COUNT(*) FROM mutable.public.notes")
        except RuntimeError:
            pass
        else:
            raise AssertionError("post-drop SELECT must raise")

        # Idempotent drop with if_exists.
        db.drop_mutable_table("notes", if_exists=True)

    print("shape_b_mutable: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
