"""SPEC-03 §12 #6 — Python `Database` tenant surface end-to-end.

Tests the PyO3 binding from the consumer's seat: a `jammi_ai.connect`
yields a Database; `set_tenant` mutates the underlying engine binding
in place, while `tenant_scope` binds for the duration of a `with` block and
restores the prior tenant on exit; subsequent `sql` / `list_sources` calls
observe the tenant-scoped predicate the engine's `TenantScopeAnalyzerRule`
injects. Mirrors the engine-side SPEC-03 §12 #2 federated split (Parquet local
source with a tenant_id column, 10 rows split 6/4) but reaches it through the
Python API.

The tests do not call `jammi_ai.connect` from a fixture because the engine's
catalog is path-scoped — each test takes its own `tmp_path` for a fresh
catalog.

These tests run via the `test-python` CI job, which builds the wheel
with `maturin develop --release` before invoking `pytest`.
"""

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jammi_ai


TENANT_A = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
TENANT_B = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b"


def _write_split_parquet(path: str, n_a: int = 6, n_b: int = 4) -> None:
    """Write `(note_id INT64, tenant_id UTF8)` with `n_a` + `n_b` rows."""
    n = n_a + n_b
    note_ids = pa.array(list(range(n)), type=pa.int64())
    tenants = pa.array(
        [TENANT_A] * n_a + [TENANT_B] * n_b,
        type=pa.string(),
    )
    table = pa.Table.from_arrays([note_ids, tenants], names=["note_id", "tenant_id"])
    pq.write_table(table, path)


def test_set_tenant_filters_federated_source(tmp_path):
    """SPEC-03 §12 #2 + #6 — Python session bound to tenant A reads 6 rows;
    tenant B reads 4; tenant binding is observable through the SQL surface."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    pq_path = tmp_path / "notes.parquet"
    _write_split_parquet(str(pq_path))

    # Register the source once with an unbound session. Both per-tenant
    # sessions reload it from the catalog on connect.
    registrar = jammi_ai.connect(f"file://{artifact_dir}")
    registrar.add_source("notes", url=str(pq_path), format="parquet")
    del registrar

    db_a = jammi_ai.connect(f"file://{artifact_dir}")
    db_a.set_tenant(TENANT_A)

    db_b = jammi_ai.connect(f"file://{artifact_dir}")
    db_b.set_tenant(TENANT_B)

    count_a = (
        db_a.sql("SELECT COUNT(*) AS n FROM notes.public.notes")
        .column("n")
        .to_pylist()[0]
    )
    count_b = (
        db_b.sql("SELECT COUNT(*) AS n FROM notes.public.notes")
        .column("n")
        .to_pylist()[0]
    )
    assert count_a == 6
    assert count_b == 4

    ids_a = set(
        db_a.sql("SELECT note_id FROM notes.public.notes")
        .column("note_id")
        .to_pylist()
    )
    ids_b = set(
        db_b.sql("SELECT note_id FROM notes.public.notes")
        .column("note_id")
        .to_pylist()
    )
    assert ids_a.isdisjoint(ids_b)


def test_set_tenant_rejects_invalid_uuid(tmp_path):
    """ADR-00 invariant: a non-UUID tenant id raises `ValueError`
    (PyO3 mapping of `JammiError::Tenant`)."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.set_tenant("not-a-uuid")
    msg = str(info.value).lower()
    assert "tenant" in msg or "uuid" in msg or "invalid" in msg, (
        f"expected the message to mention the tenant/uuid problem, got: {info.value}"
    )


def test_tenant_scope_rejects_invalid_uuid_at_call_site(tmp_path):
    """`tenant_scope` validates its id eagerly: a malformed id raises where the
    caller names it, not on `__enter__`."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.tenant_scope("not-a-uuid")
    msg = str(info.value).lower()
    assert "tenant" in msg or "uuid" in msg or "invalid" in msg, (
        f"expected the message to mention the tenant/uuid problem, got: {info.value}"
    )


def test_empty_tenant_clears_binding(tmp_path):
    """`set_tenant('')` clears the binding — the engine's API contract
    treats empty string as unbind."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    db.set_tenant(TENANT_A)
    assert db.tenant() == TENANT_A
    db.set_tenant("")
    assert db.tenant() is None


def test_tenant_scope_scopes_and_restores(tmp_path):
    """`with db.tenant_scope(t)` scopes catalog listings + discriminator-column
    row reads to `t` for the block, then restores the prior scope on exit. Uses a
    real `tenant_id`-column source so row-level isolation is genuinely exercised."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    pq_path = tmp_path / "notes.parquet"
    _write_split_parquet(str(pq_path))

    registrar = jammi_ai.connect(f"file://{artifact_dir}")
    registrar.add_source("notes", url=str(pq_path), format="parquet")
    del registrar

    db = jammi_ai.connect(f"file://{artifact_dir}")
    assert db.tenant() is None  # starts unscoped

    def count() -> int:
        return (
            db.sql("SELECT COUNT(*) AS n FROM notes.public.notes")
            .column("n")
            .to_pylist()[0]
        )

    with db.tenant_scope(TENANT_A):
        assert db.tenant() == TENANT_A
        # `notes` is a tenant_id-column source visible to A (its rows + globals).
        assert {"notes"}.issubset({s["source_id"] for s in db.list_sources()})
        assert count() == 6  # discriminator-column row isolation

    # Prior (unscoped) scope restored on exit.
    assert db.tenant() is None
    assert count() == 0  # unscoped sees only tenant_id IS NULL rows (none here)


def test_tenant_scope_nests_and_restores_prior(tmp_path):
    """Nested `tenant_scope` blocks restore the OUTER tenant on inner exit, not a
    blind clear to unscoped."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    pq_path = tmp_path / "notes.parquet"
    _write_split_parquet(str(pq_path))

    registrar = jammi_ai.connect(f"file://{artifact_dir}")
    registrar.add_source("notes", url=str(pq_path), format="parquet")
    del registrar

    db = jammi_ai.connect(f"file://{artifact_dir}")

    def count() -> int:
        return (
            db.sql("SELECT COUNT(*) AS n FROM notes.public.notes")
            .column("n")
            .to_pylist()[0]
        )

    with db.tenant_scope(TENANT_A):
        assert db.tenant() == TENANT_A
        assert count() == 6
        with db.tenant_scope(TENANT_B):
            assert db.tenant() == TENANT_B
            assert count() == 4
        # Inner block exit restores A, not unscoped.
        assert db.tenant() == TENANT_A
        assert count() == 6
    assert db.tenant() is None


def test_tenant_scope_restores_on_exception(tmp_path):
    """A `tenant_scope` block restores the prior tenant even when the body
    raises — the restore rides `__exit__`, which fires on the exceptional path."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    db.set_tenant(TENANT_A)
    with pytest.raises(RuntimeError):
        with db.tenant_scope(TENANT_B):
            assert db.tenant() == TENANT_B
            raise RuntimeError("boom")
    assert db.tenant() == TENANT_A


def test_register_channel_round_trips_through_python(tmp_path):
    """SPEC-01 §7 dual-language hook landed in this iteration: register
    a new evidence-provenance channel from Python; merging in the engine
    sees the declared columns."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    db.register_channel(
        "scored_by",
        priority=3,
        columns=[("ranker", "Utf8"), ("rank_score", "Float32")],
    )
    # Adding a second column with the same name + a different dtype is the
    # append-only invariant; reject with a typed error carrying the engine's
    # message verbatim.
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.add_channel_columns("scored_by", columns=[("ranker", "Int32")])
    msg = str(info.value)
    assert "cannot redeclare as Int32" in msg, (
        f"expected the engine's append-only message, got: {info.value}"
    )
