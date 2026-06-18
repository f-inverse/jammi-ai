"""The three rails as thin helpers: provenance, tenancy, measurement.

The rails are the columns of the book's 4-tier × 3-rail grid — woven through
every tier, not a chapter of their own. This module keeps them *thin*: it
composes ``jammi_ai`` and enforces the cookbook's contracts; it implements no
graph or ML logic. The dedicated rails chapter (K-rails) deepens the prose and
the worked examples, but every tier recipe calls these same helpers, so the
rails are real wiring rather than a gesture.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from . import contracts

# --------------------------------------------------------------------------- #
# Rail 1 — provenance
# --------------------------------------------------------------------------- #


def provenance(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the audit trail off a context-conditioned result.

    A prediction here is auditable: the ``source`` fact says how the context was
    assembled (``ann`` | ``edges`` | ``hybrid``) and ``context_ref`` records which
    rows informed it. This pulls those out of a ``predict_with_context_predictor``
    / ``assemble_context`` result for display — the queryable provenance that
    neither a monograph proof nor a throwaway PyG notebook carries.
    """
    trail: dict[str, Any] = {}
    if "source" in result:
        trail["source"] = result["source"]
    if "context_ref" in result:
        trail["context_ref"] = result["context_ref"]
    # The uncertainty channel carries context_ref nested when present.
    unc = result.get("uncertainty")
    if isinstance(unc, dict) and "context_ref" in unc:
        trail.setdefault("context_ref", unc["context_ref"])
    if not trail:
        raise ValueError(
            "result carries no provenance (no 'source' / 'context_ref'); a "
            "context-conditioned output must ride its audit trail."
        )
    return trail


# --------------------------------------------------------------------------- #
# Rail 2 — tenancy (the two-tenant test)
# --------------------------------------------------------------------------- #


@contextmanager
def tenant(db, tenant_id: str):
    """Bind a tenant scope for the duration of the block.

    A tenant UUID delegates to the engine's own ``tenant_scope`` context manager,
    which binds ``tenant_id`` for the block and restores the prior scope on exit.
    The empty string is the *global* scope (``tenant_id IS NULL``) used to register
    a shared source; ``tenant_scope`` only accepts a UUID, so the global case binds
    in place with ``set_tenant`` and restores the prior scope manually. Either way
    the same ``db`` is yielded so a block reads naturally. The bound scope drives
    the engine's two isolation layers:

    * **catalog-listing** — ``list_sources`` / ``list`` filter to
      ``tenant_id = $cur OR tenant_id IS NULL``, so the block sees only this
      tenant's registrations plus globals;
    * **discriminator-column row filtering** — a ``TableScan`` over a source whose
      schema carries a ``tenant_id`` column is rewritten with
      ``tenant_id = $cur OR IS NULL``, so the same source yields disjoint rows
      under different tenants.

    It does **not** make a source that lacks a ``tenant_id`` column invisible: a
    discriminator-less source named directly in SQL is globally readable (the
    engine does not authenticate — access-gating lives above it). ``tenant_id`` is
    an opaque UUID; the engine validates the form, never who the tenant is.
    """
    if tenant_id == "":
        prior = db.tenant()
        db.set_tenant("")
        try:
            yield db
        finally:
            db.set_tenant(prior or "")
    else:
        with db.tenant_scope(tenant_id):
            yield db


def assert_listing_isolated(listed: list[str], forbidden: set[str], *, tenant_id: str) -> None:
    """Assert a tenant's source listing excludes another tenant's registrations.

    The catalog-listing isolation property: ``listed`` (source ids ``list_sources``
    returned under ``tenant_id``) must be disjoint from ``forbidden`` (source ids
    registered under another tenant). The engine filters the registry to
    ``tenant_id = $cur OR IS NULL``, so a foreign tenant's sources do not appear —
    a real, regression-tested isolation layer, not a domain feature.
    """
    leaked = set(listed) & forbidden
    if leaked:
        raise AssertionError(
            f"tenant '{tenant_id}' listed {len(leaked)} source(s) registered under "
            f"another tenant: {sorted(leaked)[:5]}{'…' if len(leaked) > 5 else ''}. "
            f"Catalog-listing isolation is violated."
        )


def assert_rows_isolated(rows_seen: list[str], forbidden: set[str], *, tenant_id: str) -> None:
    """Assert a discriminator-column read surfaced none of another tenant's rows.

    The row-level data-isolation property: for a source whose schema carries a
    ``tenant_id`` column, the analyzer injects ``tenant_id = $cur OR IS NULL`` onto
    the ``TableScan``, so ``rows_seen`` (keys a query under ``tenant_id`` returned
    from such a source) must be disjoint from ``forbidden`` (rows tagged for another
    tenant). This holds only because the source carries the discriminator column —
    not for a discriminator-less source (see :func:`tenant`).
    """
    leaked = set(rows_seen) & forbidden
    if leaked:
        raise AssertionError(
            f"tenant '{tenant_id}' saw {len(leaked)} discriminator-tagged row(s) "
            f"belonging to another tenant: {sorted(leaked)[:5]}"
            f"{'…' if len(leaked) > 5 else ''}. Row-level isolation is violated."
        )


# --------------------------------------------------------------------------- #
# Rail 3 — measurement (R1/R2)
# --------------------------------------------------------------------------- #


def measure(metric: str, observed: float) -> float:
    """Assert a measured number against the frozen golden metric.

    The no-deferral policy applied to numbers: a recipe without a real number
    (computed from the committed artifacts) is not done. Delegates to
    :func:`contracts.assert_close`, returning ``observed`` so a cell asserts and
    displays at once.
    """
    return contracts.assert_close(metric, observed)
