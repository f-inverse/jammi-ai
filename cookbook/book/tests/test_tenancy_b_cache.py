"""Cache-backed checks on the committed tenant-isolation vertical (B2).

These run on CPU against the committed cache (no GPU) and assert the tenancy vertical's
load-bearing invariants — the engine's TRUE isolation model measured as properties:
catalog-listing isolation (hard zero), discriminator-column row isolation (hard zero),
the honest discriminator-less caveat (a positive visible count), and tenant-conditioned
metric parity (the same recipe under two tenants yields each its own scoped result over a
disjoint partition). It must NOT encode any false "a separate source hides data" claim.

If the emitted cache is absent the heavy artifacts are skipped, but the golden metrics,
once committed, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_TN = contracts._dataset_dir("tenancy_b")
_HAVE_CACHE = (_TN / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="tenancy_b cache not emitted")


@_needs_cache
def test_isolation_layers_are_hard_zeros():
    """Catalog-listing and discriminator-column isolation are each a HARD zero leak."""
    listing = contracts.golden("tenancy_b.listing_leak")
    discriminator = contracts.golden("tenancy_b.discriminator_leak")
    assert listing.value == 0.0 and listing.tol == 0.0, "listing isolation is a hard zero"
    assert discriminator.value == 0.0 and discriminator.tol == 0.0, "row isolation is a hard zero"

    record = contracts.load_artifact("tenancy_b.record")
    assert record["listing_leak"] == 0, "A's listing must exclude B's source"
    assert record["discriminator_leak"] == 0, "A must surface no B-tagged row"
    # the discriminator read returns exactly tenant A's tagged rows.
    assert record["discriminator_rows_seen"] == record["tenant_a_papers"]


@_needs_cache
def test_caveat_discriminatorless_source_is_globally_visible():
    """The honest caveat: a discriminator-LESS source is globally readable.

    A positive assertion — tenant A sees ALL of B's rows when it names B's
    discriminator-less source. This is the limit the KV-air audit corrected; it must not
    be hidden behind a false "separate source isolates" claim.
    """
    visible = contracts.golden("tenancy_b.caveat_visible")
    assert visible.value > 0.0, "a discriminator-less source IS globally visible"
    record = contracts.load_artifact("tenancy_b.record")
    assert record["caveat_visible"] == record["tenant_b_papers"], "A sees ALL of B's rows"


@_needs_cache
def test_tenant_conditioned_metric_parity_over_a_disjoint_partition():
    """The same recall recipe under two tenants yields each its own scoped result.

    The scopes are a disjoint partition that tiles the full cache; each tenant's recall
    is a real number in range, scoped to rows it alone can see.
    """
    record = contracts.load_artifact("tenancy_b.record")
    a = record["parity_a_recall_at_10"]
    b = record["parity_b_recall_at_10"]
    assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0
    assert contracts.golden("tenancy_b.parity_a_recall_at_10").contains(a)
    assert contracts.golden("tenancy_b.parity_b_recall_at_10").contains(b)
    # the two scopes tile the cache exactly (disjoint partition).
    assert record["parity_a_visible"] + record["parity_b_visible"] == (
        record["tenant_a_papers"] + record["tenant_b_papers"])
    # each tenant scopes to its own row count.
    assert record["parity_a_visible"] == record["tenant_a_papers"]
    assert record["parity_b_visible"] == record["tenant_b_papers"]
