"""Unit tests for the three rails (provenance, tenancy, measurement).

The tenancy rail is exercised against a *real* embedded ``jammi_ai`` engine —
hermetic (no network, no GPU). It asserts the engine's two genuine isolation
layers and the honest caveat the Air Routes showcase teaches:

* catalog-listing isolation — tenant A's ``list_sources`` excludes B's source;
* row-level discriminator-column isolation — a ``tenant_id``-tagged source
  returns disjoint rows under A vs B;
* the caveat — a discriminator-less source is globally readable.

Each test genuinely exercises its property and fails if it broke.
"""

from __future__ import annotations

import os
import tempfile
import uuid

import jammi_ai
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jammi_cookbook import rails

TENANT_A = "11111111-1111-1111-1111-111111111111"
TENANT_B = "22222222-2222-2222-2222-222222222222"


def _register(db, name: str, table: pa.Table, *, root: str) -> None:
    # The embedded engine resolves a registered source's table by the parquet file
    # stem, so the file is named for the source (as the cookbook loaders do).
    path = os.path.join(root, f"{name}.parquet")
    pq.write_table(table, path)
    db.add_source(name, url=path, format="parquet")


def test_tenant_context_restores_prior_scope():
    """``rails.tenant`` binds the scope in place and restores it on exit."""
    db = jammi_ai.connect(f"file://{tempfile.mkdtemp()}")
    assert db.tenant() is None
    with rails.tenant(db, TENANT_A) as scoped:
        assert scoped is db  # binds in place, yields the same handle
        assert db.tenant() == TENANT_A
    assert db.tenant() is None  # restored


def test_catalog_listing_isolation():
    """Tenant A's ``list_sources`` excludes a source registered under tenant B.

    The first genuine layer: the registry is filtered to ``tenant_id = $cur OR
    IS NULL``. Would fail if listing leaked a foreign tenant's registration.
    """
    root = tempfile.mkdtemp()
    db = jammi_ai.connect(f"file://{tempfile.mkdtemp()}")
    with rails.tenant(db, TENANT_A):
        _register(db, "src_a", pa.table({"code": ["AAA", "BBB"]}), root=root)
    with rails.tenant(db, TENANT_B):
        _register(db, "src_b", pa.table({"code": ["XXX", "YYY"]}), root=root)

    with rails.tenant(db, TENANT_A):
        a_listed = sorted(s["source_id"] for s in db.list_sources())
    assert "src_b" not in a_listed, "A's listing must exclude B's source"
    rails.assert_listing_isolated(a_listed, {"src_b"}, tenant_id=TENANT_A)


def test_discriminator_column_row_isolation():
    """One ``tenant_id``-tagged source returns disjoint rows under A vs B.

    The second genuine layer: the analyzer injects ``tenant_id = $cur OR IS NULL``
    onto the ``TableScan`` because the schema carries the discriminator column.
    Would fail if the analyzer did not filter (both tenants would see all rows).
    """
    root = tempfile.mkdtemp()
    db = jammi_ai.connect(f"file://{tempfile.mkdtemp()}")
    with rails.tenant(db, ""):  # register the shared source globally
        _register(db, "tagged", pa.table({
            "code": ["AAA", "BBB", "XXX", "YYY"],
            "tenant_id": [TENANT_A, TENANT_A, TENANT_B, TENANT_B],
        }), root=root)

    with rails.tenant(db, TENANT_A):
        seen_a = sorted(r["code"] for r in db.sql(
            "SELECT code FROM tagged.public.tagged").to_pylist())
    with rails.tenant(db, TENANT_B):
        seen_b = sorted(r["code"] for r in db.sql(
            "SELECT code FROM tagged.public.tagged").to_pylist())
    assert seen_a == ["AAA", "BBB"], "A must see only its own tagged rows"
    assert seen_b == ["XXX", "YYY"], "B must see only its own tagged rows"
    rails.assert_rows_isolated(seen_a, set(seen_b), tenant_id=TENANT_A)


def test_discriminator_less_source_is_globally_readable():
    """A source with NO ``tenant_id`` column is globally readable (the caveat).

    The honest limit, asserted positively: with no discriminator column there is
    nothing for the analyzer to filter on and the engine does not authenticate, so
    tenant A reads ALL of a source registered under tenant B when it names it. The
    remedy is a discriminator column or an access gate above the engine.
    """
    root = tempfile.mkdtemp()
    db = jammi_ai.connect(f"file://{tempfile.mkdtemp()}")
    with rails.tenant(db, TENANT_B):
        _register(db, "b_open", pa.table({"code": ["XXX", "YYY"]}), root=root)

    with rails.tenant(db, TENANT_A):
        seen = sorted(r["code"] for r in db.sql(
            "SELECT code FROM b_open.public.b_open").to_pylist())
    assert seen == ["XXX", "YYY"], "a discriminator-less source is globally readable"


def test_assert_listing_isolated_flags_a_leak():
    with pytest.raises(AssertionError, match="Catalog-listing isolation is violated"):
        rails.assert_listing_isolated(["src_a", "src_b"], {"src_b"}, tenant_id=TENANT_A)


def test_assert_rows_isolated_flags_a_leak():
    with pytest.raises(AssertionError, match="Row-level isolation is violated"):
        rails.assert_rows_isolated(["AAA", "XXX"], {"XXX", "YYY"}, tenant_id=TENANT_A)


def test_provenance_extracts_audit_trail():
    trail = rails.provenance({"source": "edges", "context_ref": ["a", "b"]})
    assert trail == {"source": "edges", "context_ref": ["a", "b"]}


def test_provenance_without_trail_raises():
    with pytest.raises(ValueError, match="no provenance"):
        rails.provenance({"mean": 1.0})


def test_tenant_id_is_an_opaque_uuid():
    # the showcase uses fixed readable UUIDs; any valid UUID form is accepted.
    uuid.UUID(TENANT_A)
    uuid.UUID(TENANT_B)
