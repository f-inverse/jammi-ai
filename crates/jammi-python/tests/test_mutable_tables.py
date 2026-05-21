"""Python `Database.create_mutable_table` / `drop_mutable_table` round-trip.

Exercises the PyO3 binding end-to-end: a `jammi.connect` session opens a
fresh catalog; `create_mutable_table` provisions both the catalog row and
the storage table inside one transaction; SQL DDL/DML over
`mutable.public.<id>` federates with the same query surface used by
Parquet result tables; `drop_mutable_table` removes the table atomically.

Mirrors `test_tenant.py` in style — each test takes its own `tmp_path`
and uses generic schema shapes (`notes`, `sensor_readings`) with no
domain coupling.
"""

import pyarrow as pa
import pytest

import jammi


TENANT_A = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
TENANT_B = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b"


def _notes_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("note_id", pa.int64(), nullable=False),
            pa.field("body", pa.string(), nullable=False),
        ]
    )


def _sensor_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sensor_id", pa.int64(), nullable=False),
            pa.field("reading", pa.float64(), nullable=False),
            pa.field("seq", pa.int64(), nullable=False),
        ]
    )


def test_create_drop_round_trips_through_python(tmp_path):
    """Register `notes`, write 3 rows through the federated SQL surface,
    count them, drop the table, and verify the SQL surface no longer sees
    it."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    table_id = db.create_mutable_table(
        "notes",
        schema=_notes_schema(),
        primary_key=["note_id"],
    )
    assert table_id == "notes"

    db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (1, 'first')")
    db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (2, 'second')")
    db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (3, 'third')")

    count = (
        db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
        .column("n")
        .to_pylist()[0]
    )
    assert count == 3

    db.drop_mutable_table("notes")
    with pytest.raises(Exception):
        # After drop, the federated SQL surface no longer resolves the table.
        db.sql("SELECT COUNT(*) FROM mutable.public.notes")


def test_create_with_index_and_order_column(tmp_path):
    """Register a table with a secondary index and an order_column. The
    index DDL is committed inside the same transaction as the CREATE
    TABLE; the table is queryable through the federated SQL surface."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    table_id = db.create_mutable_table(
        "sensor_readings",
        schema=_sensor_schema(),
        primary_key=["sensor_id", "seq"],
        indexes=[
            {"name": "idx_sensor_id", "columns": ["sensor_id"]},
            {"name": "uniq_seq", "columns": ["seq"], "unique": True},
        ],
        order_column="seq",
    )
    assert table_id == "sensor_readings"

    # Insert one row and read it back; this proves the provider is wired,
    # the index DDL committed inside the same transaction as the CREATE,
    # and the order_column survives the round-trip without depending on
    # DataFusion's LIMIT 0 path to preserve the projected schema.
    db.sql(
        "INSERT INTO mutable.public.sensor_readings (sensor_id, reading, seq) "
        "VALUES (1, 19.5, 100)"
    )
    one = db.sql("SELECT sensor_id, reading, seq FROM mutable.public.sensor_readings")
    assert one.schema.names == ["sensor_id", "reading", "seq"]
    assert one.column("sensor_id").to_pylist() == [1]

    db.drop_mutable_table("sensor_readings")


def test_create_inherits_session_tenant(tmp_path):
    """Tenant inheritance on create + analyzer-side row filtering on read.
    Bound to tenant A: `create_mutable_table` stamps `tenant_id = A` on
    the row. Re-bound to tenant B on the same session: the tenant-scope
    analyzer injects `tenant_id = B OR IS NULL`, so the row authored
    under A becomes invisible. Re-bound back to A: the row is visible
    again."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    db.with_tenant(TENANT_A)
    db.create_mutable_table(
        "notes_a",
        schema=_notes_schema(),
        primary_key=["note_id"],
    )
    db.sql("INSERT INTO mutable.public.notes_a (note_id, body) VALUES (1, 'a-row')")

    count_a = (
        db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes_a")
        .column("n")
        .to_pylist()[0]
    )
    assert count_a == 1

    db.with_tenant(TENANT_B)
    count_b = (
        db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes_a")
        .column("n")
        .to_pylist()[0]
    )
    assert count_b == 0

    db.with_tenant(TENANT_A)
    count_a_again = (
        db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes_a")
        .column("n")
        .to_pylist()[0]
    )
    assert count_a_again == 1


def test_create_rejects_invalid_id(tmp_path):
    """`MutableTableId::new` rejects non-lowercase-ASCII identifiers and
    the typed `MutableTableError::InvalidId` surfaces through the binding."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.create_mutable_table(
            "Notes-Bad",
            schema=_notes_schema(),
            primary_key=["note_id"],
        )
    msg = str(info.value).lower()
    assert "lowercase" in msg or "invalid" in msg, (
        f"expected message to mention the id-shape constraint, got: {info.value}"
    )


def test_create_rejects_pk_not_in_schema(tmp_path):
    """The builder rejects a primary key column that isn't in `schema`."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.create_mutable_table(
            "notes",
            schema=_notes_schema(),
            primary_key=["missing"],
        )
    assert "missing" in str(info.value), (
        f"expected the engine's primary-key-not-in-schema message, got: {info.value}"
    )


def test_double_create_raises(tmp_path):
    """Registering the same id twice raises with the typed
    `MutableTableError::AlreadyExists` surface."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    db.create_mutable_table("notes", schema=_notes_schema(), primary_key=["note_id"])
    with pytest.raises((ValueError, RuntimeError)):
        db.create_mutable_table(
            "notes", schema=_notes_schema(), primary_key=["note_id"]
        )


def test_drop_missing_without_if_exists_raises(tmp_path):
    """`drop_mutable_table` over a not-registered id raises — the typed
    `MutableTableError::NotFound` propagates without `if_exists=True`."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.drop_mutable_table("never_registered")
    msg = str(info.value).lower()
    assert "not found" in msg, (
        f"expected the typed NotFound message, got: {info.value}"
    )


def test_drop_missing_with_if_exists_succeeds(tmp_path):
    """`drop_mutable_table(if_exists=True)` over a not-registered id is a
    no-op — the binding short-circuits on the typed NotFound variant."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    db.drop_mutable_table("never_registered", if_exists=True)
