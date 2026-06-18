"""Cache-backed checks on the committed KV-air on-ramp artifacts.

These run on CPU against the committed Air Routes cache (no GPU, no recompute) and
assert the on-ramp's load-bearing facts: the artifact files exist with the
contracted shape, the tier 01–02 golden numbers are real, and — the showcase —
the tenancy record captures the engine's two genuine isolation layers plus the
honest caveat.

Air Routes is the clean 01–02 on-ramp: embed + neighbor graph + continent
homophily (tier 01), route-graph propagation + raw→propagated recall (tier 02),
and the tenancy showcase. There is NO learn/predict tier here — the continent
label is near-solved by lat/lon, so a tier-03/04 would be manufactured; that
spine lives on ogbn-arxiv (the keystone).

If the emitted cache is absent the heavy artifacts are skipped (the unit suite
stays runnable without the GPU emit), but the committed golden metrics are always
asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_AIR = contracts._dataset_dir("air")
_HAVE_CACHE = (_AIR / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="air cache not emitted")


@_needs_cache
def test_declared_hierarchy_is_more_homophilous_than_route():
    """Tier 01 — the construct contrast: the declared containment hierarchy is
    near-perfectly continent-consistent (~0.99), the route topology highly but not
    fully intracontinental (~0.83). A declared hierarchy carries clean structure the
    route graph only approximates.
    """
    route = contracts.golden("air.tier01.route_homophily").value
    contains = contracts.golden("air.tier01.contains_homophily").value
    assert 0.5 < route < 1.0, "route graph is mostly-but-not-fully intracontinental"
    assert contains > route, "the declared hierarchy is more continent-consistent"
    assert contains > 0.95, "the continent→airport hierarchy is near-perfectly homophilous"


@_needs_cache
def test_route_propagation_helps_continent_recall():
    """Tier 02 — propagation as a low-pass filter lifts same-continent recall.

    Propagating the raw airport embeddings over the declared route graph (a node
    toward its flight-neighbours) improves the label-target retrieval — a real
    denoising gain over an embedding-independent relevance target. A teaching label
    (continent ≈ solved by lat/lon), measured honestly.
    """
    raw = contracts.golden("air.tier01.recall_at_10").value
    prop = contracts.golden("air.tier02.recall_at_10").value
    delta = contracts.golden("air.tier02.recall_delta").value
    assert 0.0 <= raw <= 1.0
    assert prop > raw, "route-graph propagation must improve continent recall"
    # raw/prop/delta are each committed rounded to 3dp, so recomputing the delta
    # from the rounded recalls agrees with the recorded delta within one rounding ULP.
    assert abs((prop - raw) - delta) <= 1e-3, "the recorded delta is internally consistent"


@_needs_cache
def test_tenancy_record_captures_the_two_layers_and_the_caveat():
    """The showcase — the engine's two genuine isolation layers + the honest caveat.

    Catalog-listing isolation (tenant A's listing excludes B's source) and
    row-level discriminator-column isolation (a tenant_id-tagged source returns
    disjoint rows under A vs B) are each a HARD zero leak — not a tolerance. The
    caveat is recorded as a positive value: a discriminator-less source is globally
    readable, so tenant A sees ALL of tenant B's rows.
    """
    listing = contracts.golden("air.tenancy.listing_leak")
    discriminator = contracts.golden("air.tenancy.discriminator_leak")
    visible = contracts.golden("air.tenancy.global_source_visible")
    assert listing.value == 0.0 and listing.tol == 0.0, "listing isolation is a hard zero"
    assert discriminator.value == 0.0 and discriminator.tol == 0.0, "row isolation is a hard zero"
    assert visible.value > 0.0, "the caveat: a discriminator-less source IS globally visible"

    record = contracts.load_artifact("air.tenancy")
    assert record["listing_leak"] == 0, "A's listing must exclude B's source"
    assert record["discriminator_leak"] == 0, "A must surface no B-tagged row"
    assert record["tenant_a_airports"] > 0 and record["tenant_b_airports"] > 0
    # The discriminator-column read returns exactly tenant A's tagged rows…
    assert record["discriminator_rows_seen"] == record["tenant_a_airports"]
    # …while the discriminator-less source exposes ALL of tenant B's rows (caveat).
    assert record["global_source_visible"] == record["tenant_b_airports"]


@_needs_cache
def test_committed_air_artifacts_match_contract():
    """Every parquet/edge_table air artifact loads and carries its contracted columns."""
    for name in ("air.airports", "air.embeddings", "air.neighbor_graph",
                 "air.route_edges", "air.contains_edges", "air.propagated"):
        art = contracts.artifact(name)
        table = contracts.load_artifact(name)
        assert table.num_rows > 0
        for col in art.columns:
            assert col in table.column_names, f"{name} missing column {col}"
