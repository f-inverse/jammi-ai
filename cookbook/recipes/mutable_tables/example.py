"""Create / insert / select / drop on a Jammi mutable companion table.

Run with `python cookbook/recipes/mutable_tables/example.py`. Exits 0 on
success.
"""

from __future__ import annotations

import tempfile

import pyarrow as pa

import jammi_ai


def notes_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("note_id", pa.int64(), nullable=False),
            pa.field("body", pa.string(), nullable=False),
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db = jammi_ai.connect(artifact_dir=tmp, gpu_device=-1)

        # 1. Create the mutable table. The catalog now resolves
        #    `mutable.public.notes` for SQL DML and reads.
        table_id = db.create_mutable_table(
            "notes",
            schema=notes_schema(),
            primary_key=["note_id"],
        )
        assert table_id == "notes", f"expected 'notes', got {table_id}"

        # 2. Insert three rows through DataFusion's INSERT path.
        db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (1, 'one')")
        db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (2, 'two')")
        db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (3, 'three')")

        # 3. Count + ordered SELECT round-trip the rows we just wrote.
        count = (
            db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
            .column("n")
            .to_pylist()[0]
        )
        assert count == 3, f"expected 3 rows, got {count}"

        bodies = (
            db.sql("SELECT body FROM mutable.public.notes ORDER BY note_id")
            .column("body")
            .to_pylist()
        )
        assert bodies == ["one", "two", "three"], f"unexpected rows {bodies}"

        # 4. Drop — after which the federated SQL surface no longer resolves
        #    the table.
        db.drop_mutable_table("notes")
        try:
            db.sql("SELECT COUNT(*) FROM mutable.public.notes")
        except RuntimeError:
            pass
        else:
            raise AssertionError("post-drop SELECT must raise")

        # 5. Idempotent drop — `if_exists=True` does not raise when the
        #    table is already gone.
        db.drop_mutable_table("notes", if_exists=True)

    print("mutable_tables: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
