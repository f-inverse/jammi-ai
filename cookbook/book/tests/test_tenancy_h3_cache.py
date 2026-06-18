"""Cache-backed checks for the per-verb isolation matrix + BYO-auth seam (§3.5).

These run on CPU against the committed **embedded-canonical** matrix (no server,
no GPU, no recompute) and assert the chapter's load-bearing facts:

* **the matrix-to-golden oracle** — every committed verdict matches its frozen
  golden in ``artifacts/tenancy_h3/golden_metrics.json``: the golden-stability gate;
* **the per-verb hard zeros** — every tenant-scoped catalog read / describe / the
  discriminator-column sql row read leaks ZERO of tenant A's resource to tenant B;
* **the stated-positives** — B sees the engine's built-in global channels; A reads
  a discriminator-less source whole (the honest caveat, a real positive count);
* **the collisions** — a duplicate mutable-table name across tenants ERRORS on the
  global PK; duplicate topic/channel ids isolate per-tenant — never a clobber;
* **the destructive-verb survival** — A's mutable table / topic SURVIVES a
  foreign-tenant drop: no cross-tenant data destruction (the property the standing
  oracle guards; the headline no-leak finding);
* **the BYO-auth seam** — two authenticated tenants get isolated reads; a
  missing/invalid credential is rejected, not run unscoped.

The cross-transport ``remote == embedded`` parity is a ONE-TIME emit-side LIVE
check (recorded in ``tenancy_h3.json``); PR CI never re-diffs two static artifacts.

The library-level BYO-auth seam primitives (``mint_token`` / ``verify_token`` /
the gateway shape) are exercised here in pure Python with no engine — a forged
token must NOT verify regardless of the committed cache.

If the emitted cache is absent the matrix-backed checks skip; the committed golden
metrics, once present, are always asserted.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from jammi_cookbook import contracts

_TENANCY = contracts._dataset_dir("tenancy_h3")
_HAVE_CACHE = (_TENANCY / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="tenancy_h3 cache not emitted")

# The hard-zero observables — a leak here is an isolation failure.
_HARD_ZEROS = (
    "source_listing_leak",
    "describe_source_leak",
    "mutable_listing_leak",
    "topic_listing_leak",
    "channel_listing_leak",
    "model_listing_leak",
    "sql_row_leak",
)


def _matrix() -> dict:
    return contracts.load_artifact("tenancy_h3.matrix")


def _record() -> dict:
    return contracts.load_artifact("tenancy_h3.record")


# --------------------------------------------------------------------------- #
# the matrix-to-golden oracle
# --------------------------------------------------------------------------- #


@_needs_cache
def test_every_verdict_matches_golden():
    """Every committed matrix verdict matches its frozen golden — the golden the
    chapter renders against. A drift in any cell fails CI here."""
    for metric in (
        "hard_zero.source_listing_leak",
        "hard_zero.describe_source_leak",
        "hard_zero.mutable_listing_leak",
        "hard_zero.topic_listing_leak",
        "hard_zero.channel_listing_leak",
        "hard_zero.model_listing_leak",
        "hard_zero.sql_row_leak",
    ):
        # every hard zero is exactly 0
        contracts.assert_close(f"tenancy_h3.{metric}", 0.0)
    for metric in (
        "collision.mutable_dup_errored",
        "collision.topic_dup_isolated",
        "collision.channel_dup_isolated",
        "destructive.mt_survives_foreign_drop",
        "destructive.tp_survives_foreign_drop",
        "parity.cross_transport_equal",
        "byo_auth.a_isolated",
        "byo_auth.b_isolated",
        "byo_auth.missing_rejected",
        "byo_auth.invalid_rejected",
        "byo_auth.legit_after_forgery",
    ):
        contracts.assert_close(f"tenancy_h3.{metric}", 1.0)


# --------------------------------------------------------------------------- #
# the per-verb hard zeros — the measured core
# --------------------------------------------------------------------------- #


@_needs_cache
def test_every_tenant_scoped_verb_leaks_nothing():
    """For every tenant-scoped verb, tenant B sees/reaches ZERO of tenant A's
    resource — the standing oracle's property, measured per verb as a hard zero."""
    m = _matrix()
    for cell in _HARD_ZEROS:
        assert m[cell] == 0, f"{cell} leaked {m[cell]} (expected a hard zero)"


@_needs_cache
def test_no_leak_finding_recorded():
    """The headline: no leak was found. The record states it explicitly, and the
    matrix backs it — every hard zero is 0 and A survives every foreign destructive
    call. If this ever flips, the record carries the candidate engine finding."""
    m = _matrix()
    assert all(m[cell] == 0 for cell in _HARD_ZEROS)
    assert m["mt_survives_foreign_drop"] is True
    assert m["tp_survives_foreign_drop"] is True
    assert _record()["leak_finding"].startswith("NONE")


# --------------------------------------------------------------------------- #
# the stated-positives — documented visibility, asserted as real positives
# --------------------------------------------------------------------------- #


@_needs_cache
def test_stated_positives_are_real_positive_counts():
    """The honest caveats are POSITIVE counts, not hidden zeros: B sees the
    engine's built-in global channels, and A reads a discriminator-less source
    whole. Both asserted to golden as the visible counts they are."""
    m = _matrix()
    assert m["builtin_channels_visible"] > 0
    assert m["caveat_visible"] > 0
    contracts.assert_close(
        "tenancy_h3.stated_positive.builtin_channels_visible",
        float(m["builtin_channels_visible"]),
    )
    contracts.assert_close(
        "tenancy_h3.stated_positive.caveat_visible", float(m["caveat_visible"])
    )


# --------------------------------------------------------------------------- #
# collisions — error or isolated namespace, never a cross-tenant clobber
# --------------------------------------------------------------------------- #


@_needs_cache
def test_duplicate_ids_error_or_isolate_never_clobber():
    """A duplicate mutable-table name across tenants ERRORS on the global catalog
    PK (it does not clobber A's table); duplicate topic / channel ids isolate as
    B's own per-tenant resource. Either way, no cross-tenant overwrite."""
    m = _matrix()
    assert m["mutable_dup_errored"] is True
    assert m["topic_dup_isolated"] is True
    assert m["channel_dup_isolated"] is True


# --------------------------------------------------------------------------- #
# destructive verbs are tenant-scoped — A's resource survives
# --------------------------------------------------------------------------- #


@_needs_cache
def test_destructive_verbs_do_not_reach_across_tenants():
    """B names A's mutable table / topic in a destructive call; B's call resolves
    in B's OWN namespace and A's resource SURVIVES — the load-bearing guarantee
    against cross-tenant data destruction. (The mutable-table drop by B is a
    not-found in B's namespace; A's table is still listed afterwards.)"""
    m = _matrix()
    assert m["mt_survives_foreign_drop"] is True
    assert m["tp_survives_foreign_drop"] is True


# --------------------------------------------------------------------------- #
# the cross-transport parity verdict (recorded, asserted as a fact)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_remote_equals_embedded_for_cross_transport_verbs():
    """The recorded one-time live parity verdict: remote == embedded for every
    cross-transport observable (the catalog reads + the discriminator sql row
    read). Asserted as a committed fact + the golden."""
    rec = _record()
    parity = rec["parity"]
    assert parity, "the parity record must carry per-observable verdicts"
    assert all(p["equal"] for p in parity), "every cross-transport observable must agree"
    assert rec["parity_verdict"].startswith("remote == embedded")
    contracts.assert_close("tenancy_h3.parity.cross_transport_equal", 1.0)


# --------------------------------------------------------------------------- #
# the BYO-auth seam (committed verdict + the pure-Python primitives)
# --------------------------------------------------------------------------- #


@_needs_cache
def test_byo_auth_seam_verdict():
    """The committed BYO-auth verdict: two authenticated tenants get isolated
    reads; a missing credential is rejected (not run unscoped); an invalid
    credential is rejected, and a valid token for the same tenant still resolves
    (the rejection was the signature, not a tenant blocklist)."""
    rec = _record()
    byo = rec["byo_auth"]
    assert byo["a_isolated"] is True
    assert byo["b_isolated"] is True
    assert byo["missing_rejected"] is True
    assert byo["invalid_rejected"] is True
    assert byo["legit_after_forgery"] is True
    # the seam is documented as the consumer's, never the engine's, auth
    assert "engine ships the seam" in rec["byo_auth_note"].lower()


def test_byo_auth_primitives_reject_a_forged_token():
    """The library-level seam primitives are sound in pure Python (no engine, no
    cache): a token whose signature does not cover its claim must NOT verify, and a
    tampered tenant claim must change the required signature — the forgery the
    signed claim defends against."""
    from scripts.build_tenancy_h3_cache import mint_token, verify_token

    tenant_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    tenant_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    # a freshly minted token verifies to its tenant claim
    token = mint_token("subject-a", tenant_a)
    assert verify_token(token) == tenant_a

    # a missing token, a non-bearer token, and a malformed token all reject
    assert verify_token(None) is None
    assert verify_token("subject-a." + tenant_a + ".deadbeef") is None
    assert verify_token("Bearer no-dots-here") is None

    # a forged tenant claim (swap the tenant, keep the old signature) must NOT
    # verify — the signature covers the claim
    claim, _, sig = token[len("Bearer "):].rpartition(".")
    forged_claim = claim.replace(tenant_a, tenant_b)
    forged = f"Bearer {forged_claim}.{sig}"
    assert verify_token(forged) is None

    # and the forgery's correct signature is a DIFFERENT mac (the claim is signed)
    correct_for_forged = hmac.new(
        b"consumer-issuer-signing-key-not-jammi",
        forged_claim.encode(),
        hashlib.sha256,
    ).hexdigest()
    assert correct_for_forged != sig
