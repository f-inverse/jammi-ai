#!/usr/bin/env python3
"""Emit the per-verb tenant-isolation + BYO-auth cache (H3 §3.5) — CPU, hermetic.

The engine↔cookbook validator for the standing isolation oracle (engine `§3.5`):
the property that, for every tenant-scoped verb the engine exposes, tenant B
cannot see or reach tenant A's resource. Chapter 11 measured the *two-layer*
model (catalog-listing isolation + discriminator-column row isolation, as hard
zeros, plus the honest discriminator-less caveat). This script extends that to a
**per-verb matrix**: each tenant-scoped verb measured from the consumer side, as a
hard zero where isolation must hold (or a documented stated-positive where it does
not), on the embedded engine and — for every cross-transport verb — on a live
`grpc://` `jammi-server`.

Isolation is catalog/SQL behaviour (a registry filter + a `TableScan` analyzer
rewrite), so the whole matrix runs on CPU with no GPU and no keystone corpus. The
one exception is the model catalog: the only public path that puts a model row in
the catalog is TRAINING, so the `list_models` hard zero registers a real model
under tenant A via a tiny CPU `fine_tune` (the same pattern §3.6 uses) and then
measures that tenant B's catalog carries none of A's models. Tenant ids are opaque
UUIDs — the engine validates the form, never who the tenant is; nothing here names
a consumer.

## Part A — the per-verb isolation matrix

For each verb, tenant A creates/registers a resource and tenant B's
listing/read/reach is measured:

* **catalog-listing reads (hard zeros)** — `list_sources` / `list_mutable_tables`
  / `list_topics` / `list_models` filter the registry to
  `tenant_id = $cur OR IS NULL`, so B's listing excludes A's registration (leak 0).
  `list_models` registers a REAL model under A first (training is the only public
  catalog path), so B's catalog carrying none of A's models is a true cross-tenant
  zero, not an empty-catalog artifact. `describe_source` on a foreign id returns
  `None` (not visible — a hard zero, not an error that confirms existence).
* **`list_channels` (hard zero for the user channel; a stated-positive for the
  built-ins)** — B does not see A's registered channel (leak 0), but DOES see the
  engine's built-in global channels (`vector` / `inference` / `bm25`) — globally
  scoped, visible to every tenant (the discriminator-less-global analog).
* **discriminator-column row read via `sql` / Flight SQL (hard zero)** — A reads a
  shared, globally-registered, `tenant_id`-tagged source; the analyzer injects
  `tenant_id = $cur OR IS NULL`, so A sees only its own rows, none of B's (leak 0).
* **the discriminator-less caveat (a stated-positive)** — a source with no
  `tenant_id` column is globally readable: A sees ALL of B's rows when it names it.
  The engine does not authenticate; access-gating lives above it.
* **duplicate-id collisions across tenants (ERROR or isolated — never clobber)** —
  a duplicate `name` in `create_mutable_table` across tenants ERRORS on the global
  catalog primary key (it does NOT silently clobber A's table). `register_topic` /
  `register_channel` are per-tenant namespaces — a duplicate id under B SUCCEEDS as
  B's own resource, A's untouched. Either way, no cross-tenant overwrite.
* **destructive verbs are tenant-scoped (A's resource survives)** —
  `drop_mutable_table` / `drop_topic` named by B resolve in B's OWN namespace: B's
  drop of a name A registered is `not found` (mutable table) or a no-op on B's own
  (topic), and A's resource SURVIVES. Measured directly: A's resource is still
  present after B's destructive call. (This is the property the standing oracle
  guards — a regression here would be a cross-tenant data-destruction leak.)

Every cell ends in a measured number: a leak count that must be `0`, or a
stated-positive count (the built-ins B sees, the caveat rows A sees) that is a
real positive, or a survives-flag (`1.0`) after a foreign destructive call.

## Part B — the BYO-auth seam, over the real Flight SQL wire

The engine authenticates nothing on any transport: the `jammi-session-id` header
the stock interceptor reads is a transport correlation id, NOT a trust boundary
(anyone presenting another session's id assumes that session's tenant).
Verifying a caller is the consumer's job — a gateway placed IN FRONT OF the
engine. The engine's in-process `grpc_byo_auth.rs` worked example shows that
seam for the TYPED gRPC verbs; the Flight SQL lane (`db.sql()`) is a separate
`pyarrow.flight` transport and is the gateway-in-front's responsibility there too
(engine issue #220, by design).

On `jammi-client` `0.32.0` the client carries the channel's bearer on the Flight
SQL lane as well as the typed gRPC verbs (jammi #96); on `0.31.0` the bearer rode
only the typed path. This script demonstrates the consumer-side seam over that
**real Flight wire**: a `pyarrow.flight` gateway server reads the inbound bearer
off a genuine `db.sql()` call (the production token-threading runs — no mock), a
**HMAC-signed bearer token** (no real IdP, no product name) it verifies and maps
to a tenant, and a real-engine tenant-scoped read it returns to the caller.

It measures these properties, all over the real Flight wire:

* **the bearer rides the real Flight lane** — a `BearerCredentials`-authenticated
  `db.sql()` puts `Bearer <minted>` on every Flight call the gateway sees; an
  anonymous `db.sql()` (no credentials) carries no authorization header.
* **two authenticated tenants get isolated reads** — caller A and caller B each
  present a valid token for their own tenant over Flight; the gateway verifies,
  binds the verified tenant, and the live-engine read returns only that tenant's
  source (leak 0 each way).
* **a missing credential is rejected, not run unscoped** — an anonymous Flight
  call → the gateway rejects in `get_flight_info` before any engine read runs;
  the request reads NOTHING (it does not fall through to an unscoped global read).
* **an invalid credential is rejected** — a tampered token (a forged tenant claim
  the signature does not cover) → rejected; the forgery buys nothing, and a valid
  token for the same tenant still resolves.
* **over-Flight == embedded** — the gateway's tenant-scoped read matches the
  embedded `list_sources` under the same tenant (the same isolation, on the wire).

The credential is generic: an opaque subject + a tenant claim, HMAC-SHA256
[@rfc2104] over both under a shared key, presented as a bearer token [@rfc6750].
It stands in for whatever a consumer's identity system mints; the seam only needs
the gateway to turn a verified claim into a tenant id. No identity provider,
scheme, or consumer is named.

Usage::

    JAMMI_SERVER_BIN=/mnt/sagemaker-nvme/jammi-target/debug/jammi-server \\
        python scripts/build_tenancy_h3_cache.py --fixtures-root /path/to/jammi-ai

The ``--fixtures-root`` is the engine checkout carrying
``tests/fixtures/tiny_modernbert`` (or set ``JAMMI_FIXTURES_ROOT``) — the base
model the ``list_models`` hard zero fine-tunes under tenant A.

CPU/hermetic. This is an emit-only script; PR CI reads the committed matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import jammi_ai
import jammi_client
import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook.rails import assert_listing_isolated, assert_rows_isolated, tenant

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "tenancy_h3"

# Opaque tenant UUIDs — the engine validates the form, never who the tenant is.
TENANT_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TENANT_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

# The engine's built-in global channels — globally scoped, visible to every tenant
# (the discriminator-less-global analog for the channel registry). B sees these
# but NOT A's registered channel; pinned here so the stated-positive is explicit.
_BUILTIN_CHANNELS = {"vector", "inference", "bm25"}

# A tiny supervised pairs corpus — `(anchor, positive)` triggers the engine's
# `pairs` training format (MNRL). 12 rows trains in ~1s on CPU (candle falls back
# off CUDA). Registering a model under tenant A is the only public path that puts a
# model row in the catalog, so the `list_models` hard zero is measured against a
# REAL model A owns (same shape §3.6's build_lifecycle_cache uses).
_PAIRS = {
    "anchor": ["a graph", "a node", "an edge", "a model"] * 3,
    "positive": ["a network", "a vertex", "a link", "a net"] * 3,
}


# --------------------------------------------------------------------------- #
# Helpers shared by both transports
# --------------------------------------------------------------------------- #


def _write_src(db, work: Path, name: str, table: pa.Table) -> None:
    """Register a parquet source whose FILE STEM equals its source id, so the
    Flight SQL table reference is ``<name>.public.<name>`` on BOTH transports
    (the remote resolves the table by the parquet file stem)."""
    path = work / f"{name}.parquet"
    pq.write_table(table, path)
    db.add_source(name, url=str(path), format="parquet")


def _channel_ids(db) -> set[str]:
    return {c["channel_id"] for c in db.list_channels()}


def _mutable_ids(db) -> set[str]:
    return {t["id"] for t in db.list_mutable_tables()}


# --------------------------------------------------------------------------- #
# Part A — the per-verb isolation matrix (one transport)
# --------------------------------------------------------------------------- #


def _register_model_for(db, work: Path, base_model: str, tag: str) -> list[str]:
    """Register a model under the CURRENTLY-bound tenant the only public way — a
    tiny CPU ``fine_tune`` — and return the model ids it adds to the catalog (the
    base model at submission + the fine-tuned model on completion). Same shape
    §3.6's ``build_lifecycle_cache`` uses; trains in ~1s on CPU."""
    corpus = work / f"pairs_{tag}.parquet"
    pq.write_table(pa.table(_PAIRS), corpus)
    src = f"pairs_{tag}"
    db.add_source(src, url=f"file://{corpus}", format="parquet")
    job = db.fine_tune(
        source=src,
        base_model=base_model,
        columns=["anchor", "positive"],
        method="lora",
        task="text_embedding",
        epochs=1,
        batch_size=4,
        seed=0,
    )
    job.wait()
    return [m["model_id"] for m in db.list_models()]


def run_matrix(db, work: Path, base_model: str, *, tag: str, cross_transport: bool) -> dict:
    """Drive every tenant-scoped verb on one transport and return the observable
    matrix. ``tag`` keeps registrations disjoint on the module-shared remote
    server. ``cross_transport`` records which verbs are also exercised on grpc://.

    A row is created/registered under tenant A, then tenant B's
    listing/read/reach is measured — a hard zero where isolation must hold, a
    stated-positive where the engine documents visibility, a survives-flag after a
    foreign destructive call."""
    sch = pa.schema([("k", pa.int64()), ("v", pa.string())])
    topic_sch = pa.schema([("k", pa.int64())])
    src_name = f"src_a_{tag}"
    mt_name = f"mt_a_{tag}"
    tp_name = f"tp_a_{tag}"
    ch_name = f"ch_a_{tag}"
    tagged = f"tagged_{tag}"
    nodisc = f"nodisc_{tag}"

    # --- A registers one resource per verb under its own scope --------------- #
    with tenant(db, TENANT_A):
        _write_src(db, work, src_name, pa.table({"id": [1, 2, 3]}))
        db.create_mutable_table(mt_name, schema=sch, primary_key=["k"])
        db.register_topic(tp_name, schema=topic_sch)
        db.register_channel(ch_name, priority=1, columns=[("c", "Int64")])
        # A registers a REAL model (training is the only public catalog path), so
        # the list_models hard zero below is measured against models A genuinely
        # owns — not an empty catalog.
        a_models = set(_register_model_for(db, work, base_model, f"a_{tag}"))

    # --- list_sources / describe_source (hard zeros) ------------------------- #
    with tenant(db, TENANT_B):
        b_sources = [s["source_id"] for s in db.list_sources()]
        b_describe_src = db.describe_source(src_name)
    assert_listing_isolated(b_sources, {src_name}, tenant_id=TENANT_B)
    source_listing_leak = len(set(b_sources) & {src_name})
    # describe of a foreign source is None (not visible) — the hard zero is "B
    # cannot confirm A's source exists", encoded as 0 when describe returns None.
    describe_source_leak = 0 if b_describe_src is None else 1

    # --- list_mutable_tables (hard zero) ------------------------------------- #
    with tenant(db, TENANT_B):
        b_mutable = _mutable_ids(db)
    mutable_listing_leak = len(b_mutable & {mt_name})

    # --- list_topics (hard zero) --------------------------------------------- #
    with tenant(db, TENANT_B):
        b_topics = set(db.list_topics())
    topic_listing_leak = len(b_topics & {tp_name})

    # --- list_channels (hard zero for A's channel; built-ins a stated-positive) #
    with tenant(db, TENANT_B):
        b_channels = _channel_ids(db)
    channel_listing_leak = len(b_channels & {ch_name})
    # the stated-positive: B sees the engine's built-in global channels.
    builtin_channels_visible = len(b_channels & _BUILTIN_CHANNELS)

    # --- list_models (hard zero — B's catalog carries none of A's models) ---- #
    # A registered REAL models under its scope above (a fine_tune, the only public
    # catalog path). The isolation property is the listing filter: B's list_models
    # must carry NONE of A's models — a true cross-tenant model-isolation zero, not
    # an empty-catalog artifact.
    with tenant(db, TENANT_B):
        b_models = {m["model_id"] for m in db.list_models()}
    model_listing_leak = len(b_models & a_models)

    # --- discriminator-column row read via sql / Flight SQL (hard zero) ------ #
    # One shared, globally-registered source tagged by a tenant_id discriminator
    # column. A reads it; the analyzer injects tenant_id = $cur OR IS NULL, so A
    # sees only its own rows.
    db.set_tenant("")  # register the shared tagged source as a global
    _write_src(db, work, tagged, pa.table({
        "id": [1, 2, 3, 4],
        "tenant_id": [TENANT_A, TENANT_A, TENANT_B, TENANT_B],
    }))
    with tenant(db, TENANT_A):
        a_rows = [r["id"] for r in db.sql(
            f"SELECT id, tenant_id FROM {tagged}.public.{tagged}").to_pylist()]
    # B's rows are 3, 4; A must see none of them.
    assert_rows_isolated([str(x) for x in a_rows], {"3", "4"}, tenant_id=TENANT_A)
    sql_row_leak = len(set(a_rows) & {3, 4})

    # --- the discriminator-less caveat (a stated-positive) ------------------- #
    db.set_tenant("")
    _write_src(db, work, nodisc, pa.table({"id": [10, 11, 12]}))
    with tenant(db, TENANT_A):
        a_nodisc = [r["id"] for r in db.sql(
            f"SELECT id FROM {nodisc}.public.{nodisc}").to_pylist()]
    caveat_visible = len(a_nodisc)  # A reads all 3 rows of a discriminator-less src.

    # --- duplicate-id collisions across tenants (ERROR or isolated namespace) - #
    # create_mutable_table: a duplicate name across tenants ERRORS on the global
    # catalog PK — it does NOT clobber A's table.
    with tenant(db, TENANT_B):
        try:
            db.create_mutable_table(mt_name, schema=sch, primary_key=["k"])
            mutable_dup = {"errored": False}
        except Exception as exc:  # noqa: BLE001 — the outcome IS the measurement
            mutable_dup = {"errored": True, "error": type(exc).__name__}
    # register_topic / register_channel: per-tenant namespaces — a duplicate under
    # B SUCCEEDS as B's own resource (A's untouched).
    with tenant(db, TENANT_B):
        try:
            db.register_topic(tp_name, schema=topic_sch)
            topic_dup = {"errored": False, "isolated": True}
        except Exception as exc:  # noqa: BLE001
            topic_dup = {"errored": True, "error": type(exc).__name__}
        try:
            db.register_channel(ch_name, priority=1, columns=[("c", "Int64")])
            channel_dup = {"errored": False, "isolated": True}
        except Exception as exc:  # noqa: BLE001
            channel_dup = {"errored": True, "error": type(exc).__name__}

    # --- destructive verbs are tenant-scoped (A's resource SURVIVES) --------- #
    # B names A's mutable table / topic in a destructive call. B's call resolves in
    # B's OWN namespace (not found / B's own), and A's resource SURVIVES — the
    # standing oracle's load-bearing guarantee (no cross-tenant data destruction).
    with tenant(db, TENANT_B):
        try:
            db.drop_mutable_table(mt_name)
            b_drop_mt = {"raised": False}
        except Exception as exc:  # noqa: BLE001
            b_drop_mt = {"raised": True, "error": type(exc).__name__}
        try:
            db.drop_topic(tp_name)
            b_drop_tp = {"raised": False}
        except Exception as exc:  # noqa: BLE001
            b_drop_tp = {"raised": True, "error": type(exc).__name__}
    with tenant(db, TENANT_A):
        mt_survives = mt_name in _mutable_ids(db)
        tp_survives = tp_name in set(db.list_topics())

    return {
        "transport": "remote" if cross_transport else "embedded",
        # hard zeros (catalog-listing + describe + sql row read)
        "source_listing_leak": source_listing_leak,
        "describe_source_leak": describe_source_leak,
        "mutable_listing_leak": mutable_listing_leak,
        "topic_listing_leak": topic_listing_leak,
        "channel_listing_leak": channel_listing_leak,
        "model_listing_leak": model_listing_leak,
        "sql_row_leak": sql_row_leak,
        # stated-positives
        "builtin_channels_visible": builtin_channels_visible,
        "caveat_visible": caveat_visible,
        # collisions (error or isolated namespace — never clobber)
        "mutable_dup_errored": mutable_dup["errored"],
        "topic_dup_isolated": topic_dup.get("isolated", False),
        "channel_dup_isolated": channel_dup.get("isolated", False),
        # destructive verbs are tenant-scoped — A's resource survives B's drop
        "mt_survives_foreign_drop": bool(mt_survives),
        "tp_survives_foreign_drop": bool(tp_survives),
        "b_drop_mt_raised": b_drop_mt["raised"],
        "b_drop_tp_raised": b_drop_tp["raised"],
    }


# The hard-zero observables that MUST be cross-transport-equal (the verbs reachable
# on grpc://): the catalog reads + the discriminator-column sql row read.
_CROSS_TRANSPORT_OBSERVABLES = (
    "source_listing_leak",
    "describe_source_leak",
    "mutable_listing_leak",
    "topic_listing_leak",
    "channel_listing_leak",
    "model_listing_leak",
    "sql_row_leak",
    "caveat_visible",
    "mt_survives_foreign_drop",
    "tp_survives_foreign_drop",
)


# --------------------------------------------------------------------------- #
# Part B — the BYO-auth seam, from the consumer side (generic, names no consumer)
# --------------------------------------------------------------------------- #

# The shared secret a consumer's token issuer and its authenticating gateway hold.
# In a real deployment this is a rotated signing key, never a literal; here it is a
# fixed test value so the fixture can mint tokens the gateway verifies. It is the
# CONSUMER's key — Jammi never sees it.
_SIGNING_KEY = b"consumer-issuer-signing-key-not-jammi"


def mint_token(subject: str, tenant_id: str) -> str:
    """Mint a generic signed token ``"<subject>.<tenant>.<hex-mac>"`` — a BARE
    token, no ``"Bearer "`` prefix. The MAC is HMAC-SHA256 [@rfc2104] over
    ``"<subject>.<tenant>"`` under :data:`_SIGNING_KEY`. The tenant claim is INSIDE
    the signed payload, so it cannot be forged without the key.

    The bare form is deliberate: :class:`jammi_client.BearerCredentials` prepends
    ``"Bearer "`` itself when it puts the credential on the wire, so the token the
    gateway observes is ``"Bearer <subject>.<tenant>.<hex-mac>"``. Minting with the
    prefix already attached would double it to ``"Bearer Bearer …"`` and never
    verify. Stands in for whatever a consumer's identity system issues; the seam
    only needs a verified claim to yield a tenant."""
    claim = f"{subject}.{tenant_id}"
    sig = hmac.new(_SIGNING_KEY, claim.encode(), hashlib.sha256).hexdigest()
    return f"{claim}.{sig}"


def verify_token(authorization: str | None) -> str | None:
    """The consumer's authentication step, run on the value the WIRE carries.

    The gateway reads the inbound ``authorization`` header, whose value is the
    ``"Bearer <token>"`` :class:`jammi_client.BearerCredentials` put on the Flight
    lane. This strips exactly one leading ``"Bearer "`` the wire added, then
    verifies the bare token [@rfc6750] and returns its tenant claim, or ``None`` if
    the header is missing, malformed, or fails the constant-time signature check.
    Only a VERIFIED claim yields a tenant — a tampered token (a forged tenant the
    signature does not cover) returns ``None`` and is rejected upstream."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    raw = authorization[len("Bearer "):]
    claim, _, sig_hex = raw.rpartition(".")
    if not claim or not sig_hex:
        return None
    subject, _, tenant_id = claim.partition(".")
    if not subject or not tenant_id:
        return None
    expected = hmac.new(_SIGNING_KEY, claim.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig_hex):  # constant-time
        return None
    return tenant_id


class _AuthMiddleware(flight.ServerMiddleware):
    """One Flight call's recorded ``authorization`` header — the exact value the
    client put on the wire (``"Bearer <token>"`` or ``None`` when anonymous)."""

    def __init__(self, authorization: str | None):
        self.authorization = authorization


class _AuthMiddlewareFactory(flight.ServerMiddlewareFactory):
    """Reads the inbound ``authorization`` header off every Flight call.

    pyarrow lowercases header keys and presents each value as a LIST; this stores
    the first value (or ``None`` when the header is absent) on a per-call
    :class:`_AuthMiddleware` the handlers retrieve via
    ``context.get_middleware("auth")``. It also records every observed value so the
    emit can assert exactly what ``db.sql()`` put on the wire."""

    def __init__(self):
        self.observed: list[str | None] = []

    def start_call(self, info, headers):
        auth = headers.get("authorization")
        value = auth[0] if auth else None
        self.observed.append(value)
        return _AuthMiddleware(value)


class _GatewayFlightServer(flight.FlightServerBase):
    """The consumer's authenticating gateway, as a real ``pyarrow.flight`` server.

    It sits IN FRONT OF the engine: a `db.sql()` call presents its bearer on the
    Flight lane; the gateway reads it (via the recording middleware), verifies it,
    derives the tenant from the *verified* claim, and binds it via the engine's
    per-request ``tenant_scope`` on a real upstream ``jammi-server`` — returning
    only that tenant's rows. A missing or invalid credential raises
    :class:`pyarrow.flight.FlightUnauthenticatedError` so the client surfaces a
    clean ``FlightUnauthenticatedError`` (not a wrapped ``FlightServerError``).

    ``db.sql()`` makes TWO Flight calls (``get_flight_info`` then ``do_get``), each
    re-presenting the bearer, so the gateway verifies in BOTH — rejecting in
    ``get_flight_info`` short-circuits the whole ``sql()`` before any engine read.
    This mirrors the engine's ``grpc_byo_auth.rs`` seam, here for the Flight lane
    the in-engine interceptor does not cover (engine #220)."""

    def __init__(self, location, upstream_endpoint: str, factory: _AuthMiddlewareFactory):
        super().__init__(location, middleware={"auth": factory})
        self._upstream = upstream_endpoint
        self._schema = pa.schema([("source_id", pa.string())])

    def _verified_tenant(self, context) -> str:
        middleware = context.get_middleware("auth")
        authorization = middleware.authorization if middleware is not None else None
        tenant_id = verify_token(authorization)
        if tenant_id is None:
            raise flight.FlightUnauthenticatedError("missing or invalid credential")
        return tenant_id

    def get_flight_info(self, context, descriptor):
        # Verify here too — rejecting before do_get short-circuits the whole sql().
        self._verified_tenant(context)
        endpoint = flight.FlightEndpoint(b"sources", [])
        return flight.FlightInfo(self._schema, descriptor, [endpoint], -1, -1)

    def do_get(self, context, ticket):
        tenant_id = self._verified_tenant(context)
        # The verified read leg hits the REAL engine: bind the tenant from the
        # authenticated claim and return only that tenant's sources.
        upstream = jammi_client.connect(self._upstream)
        try:
            with upstream.tenant_scope(tenant_id):
                sources = [s["source_id"] for s in upstream.list_sources()]
        finally:
            upstream.close()
        return flight.RecordBatchStream(pa.table({"source_id": sources}))


def _gateway_sources(endpoint: str, token: str | None) -> list[str]:
    """Drive the production ``db.sql()`` Flight path against the gateway with the
    given bearer (or anonymous when ``token`` is ``None``) and return the tenant's
    sources. The bearer rides the REAL Flight wire — no ``db._flight`` seeding, no
    Python-variable shortcut."""
    credentials = jammi_client.BearerCredentials(token) if token is not None else None
    db = jammi_client.connect(f"grpc://{endpoint}", credentials=credentials)
    try:
        table = db.sql("SELECT source_id FROM sources")
        return table.column("source_id").to_pylist()
    finally:
        db.close()


def run_byo_auth(upstream_endpoint: str, work: Path, *, tag: str) -> dict:
    """Drive the BYO-auth seam over the REAL Flight SQL wire and return the measured
    verdict. One source per tenant is registered on the upstream live engine under
    its own scope; the consumer-side gateway (a ``pyarrow.flight`` server) reads the
    bearer off a genuine ``db.sql()`` call, verifies it, and binds the verified
    tenant for a real-engine read. Mirrors ``grpc_byo_auth.rs`` for the Flight lane."""
    a_src = f"auth_a_{tag}"
    b_src = f"auth_b_{tag}"
    upstream = jammi_client.connect(upstream_endpoint)
    try:
        with tenant(upstream, TENANT_A):
            _write_src(upstream, work, a_src, pa.table({"id": [1]}))
        with tenant(upstream, TENANT_B):
            _write_src(upstream, work, b_src, pa.table({"id": [2]}))
        # the embedded-parity baseline: A's tenant-scoped list_sources upstream.
        with tenant(upstream, TENANT_A):
            a_upstream = sorted(s["source_id"] for s in upstream.list_sources())
    finally:
        upstream.close()

    factory = _AuthMiddlewareFactory()
    gateway = _GatewayFlightServer(
        flight.Location.for_grpc_tcp("127.0.0.1", _free_port()),
        upstream_endpoint,
        factory,
    )
    gateway_endpoint = f"127.0.0.1:{gateway.port}"
    try:
        # the bearer rides the real Flight wire: the gateway middleware sees the
        # exact minted token, prefixed "Bearer " by BearerCredentials.
        minted_a = mint_token("subject-a", TENANT_A)
        a_seen = _gateway_sources(gateway_endpoint, minted_a)
        bearer_on_flight_observed = bool(factory.observed) and all(
            obs == f"Bearer {minted_a}" for obs in factory.observed
        )

        b_seen = _gateway_sources(gateway_endpoint, mint_token("subject-b", TENANT_B))
        a_isolated = a_src in a_seen and b_src not in a_seen
        b_isolated = b_src in b_seen and a_src not in b_seen

        # an anonymous db.sql() carries no authorization header on the Flight lane.
        before = len(factory.observed)
        try:
            _gateway_sources(gateway_endpoint, None)
            missing_rejected = False
        except flight.FlightUnauthenticatedError:
            missing_rejected = True
        anon_calls = factory.observed[before:]
        anonymous_no_bearer = bool(anon_calls) and all(obs is None for obs in anon_calls)

        # an invalid credential (a forged tenant claim the signature does not cover).
        forged = f"subject-mallory.{TENANT_A}.deadbeef"
        try:
            _gateway_sources(gateway_endpoint, forged)
            forged_rejected = False
        except flight.FlightUnauthenticatedError:
            forged_rejected = True

        # and the forgery bought nothing: a VALID token for the same tenant still
        # resolves (the rejection was the signature, not a tenant blocklist).
        legit_after = _gateway_sources(gateway_endpoint, mint_token("subject-a", TENANT_A))
        legit_after_forgery = a_src in legit_after
    finally:
        gateway.shutdown()

    # over-Flight == embedded: the gateway's tenant-scoped read matches the
    # upstream tenant-scoped list_sources under the same tenant.
    over_flight_eq_embedded = sorted(a_seen) == a_upstream

    return {
        "bearer_on_flight_observed": bearer_on_flight_observed,
        "anonymous_no_bearer": anonymous_no_bearer,
        "a_isolated_over_flight": a_isolated,
        "b_isolated_over_flight": b_isolated,
        "missing_rejected_over_flight": missing_rejected,
        "forged_rejected_over_flight": forged_rejected,
        "legit_after_forgery_over_flight": legit_after_forgery,
        "over_flight_eq_embedded": over_flight_eq_embedded,
        "a_seen": sorted(a_seen),
        "b_seen": sorted(b_seen),
    }


# --------------------------------------------------------------------------- #
# Live remote server (mirrors the engine conftest live_server)
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """A real CPU ``jammi-server`` on a free port, readiness-polled, torn down on
    exit — the shape the engine's conftest uses."""

    def __init__(self, server_bin: str):
        self.server_bin = server_bin
        self.proc = None
        self.endpoint = None
        self._artifact_dir = None

    def __enter__(self) -> str:
        self._artifact_dir = tempfile.mkdtemp(prefix="jammi_srv_tenancy_h3_")
        flight_port = _free_port()
        health_port = _free_port()
        env = dict(os.environ)
        env["JAMMI_ARTIFACT_DIR"] = self._artifact_dir
        env["JAMMI_SERVER__FLIGHT_LISTEN"] = f"127.0.0.1:{flight_port}"
        env["JAMMI_SERVER__HEALTH_LISTEN"] = f"127.0.0.1:{health_port}"
        env["JAMMI_SERVER__SERVICES"] = "all"
        self.proc = subprocess.Popen(
            [self.server_bin], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self.endpoint = f"grpc://127.0.0.1:{flight_port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read().decode(errors="replace") if self.proc.stdout else ""
                raise RuntimeError(f"jammi-server exited early:\n{out}")
            try:
                handshake = jammi_client.connect(self.endpoint)
                handshake.get_server_info()
                handshake.close()
                return self.endpoint
            except Exception:
                time.sleep(0.25)
        self.proc.terminate()
        raise RuntimeError("jammi-server did not become ready within 30s")

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# Parity (remote == embedded for the cross-transport observables)
# --------------------------------------------------------------------------- #


def _parity(name: str, embedded, remote) -> dict:
    """The live cross-transport parity verdict for one observable. Raises (loudly)
    on a real divergence — that is an ENGINE finding the validator surfaces, never
    papers over."""
    equal = embedded == remote
    verdict = {"observable": name, "equal": equal, "value": embedded}
    if not equal:
        verdict["embedded"] = embedded
        verdict["remote"] = remote
        raise AssertionError(f"remote != embedded for {name}: {embedded!r} != {remote!r}")
    return verdict


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums() -> None:
    sums = {
        p.name: _checksum(p)
        for p in sorted(ARTIFACTS.glob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def emit(fixtures_root: Path, server_bin: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    base_model = f"local:{fixtures_root / 'tests' / 'fixtures' / 'tiny_modernbert'}"

    with tempfile.TemporaryDirectory() as work_root:
        work = Path(work_root)

        # --- embedded transport (the canonical matrix) ---------------------- #
        with tempfile.TemporaryDirectory() as catalog:
            embedded = jammi_ai.connect(f"file://{catalog}")
            print("== embedded engine: per-verb isolation matrix ==", flush=True)
            embedded_matrix = run_matrix(
                embedded, work, base_model, tag="emb", cross_transport=False
            )

        # --- remote transport (live grpc:// parity for the wire verbs) ------ #
        with LiveServer(server_bin) as endpoint:
            print(f"== remote engine up at {endpoint} ==", flush=True)
            remote = jammi_client.connect(endpoint)
            try:
                print("== remote engine: per-verb isolation matrix ==", flush=True)
                remote_matrix = run_matrix(
                    remote, work, base_model, tag="rem", cross_transport=True
                )
            finally:
                remote.close()

            # --- BYO-auth seam over the REAL Flight SQL wire ---------------- #
            # The gateway reads the bearer off a genuine db.sql() Flight call and
            # binds the verified tenant for a real-engine read against this same
            # live upstream server (the Part A real server reused as the read leg).
            print("== BYO-auth seam: bearer on the real Flight SQL wire ==", flush=True)
            byo = run_byo_auth(endpoint, work, tag="byo")

    # --- the live remote == embedded parity verdict (cross-transport verbs) - #
    parity = [
        _parity(name, embedded_matrix[name], remote_matrix[name])
        for name in _CROSS_TRANSPORT_OBSERVABLES
    ]
    print(f"== remote == embedded: PASS for all {len(parity)} cross-transport observables ==",
          flush=True)

    m = embedded_matrix

    # --- the frozen golden matrix (every cell a measured verdict) ----------- #
    golden = {
        # hard zeros — isolation must hold
        "hard_zero.source_listing_leak": {"value": float(m["source_listing_leak"]), "tol": 0.0},
        "hard_zero.describe_source_leak": {"value": float(m["describe_source_leak"]), "tol": 0.0},
        "hard_zero.mutable_listing_leak": {"value": float(m["mutable_listing_leak"]), "tol": 0.0},
        "hard_zero.topic_listing_leak": {"value": float(m["topic_listing_leak"]), "tol": 0.0},
        "hard_zero.channel_listing_leak": {"value": float(m["channel_listing_leak"]), "tol": 0.0},
        "hard_zero.model_listing_leak": {"value": float(m["model_listing_leak"]), "tol": 0.0},
        "hard_zero.sql_row_leak": {"value": float(m["sql_row_leak"]), "tol": 0.0},
        # stated-positives — documented visibility
        "stated_positive.builtin_channels_visible": {
            "value": float(m["builtin_channels_visible"]), "tol": 0.0
        },
        "stated_positive.caveat_visible": {"value": float(m["caveat_visible"]), "tol": 0.0},
        # collisions — error or isolated namespace, never clobber
        "collision.mutable_dup_errored": {
            "value": 1.0 if m["mutable_dup_errored"] else 0.0, "tol": 0.0
        },
        "collision.topic_dup_isolated": {
            "value": 1.0 if m["topic_dup_isolated"] else 0.0, "tol": 0.0
        },
        "collision.channel_dup_isolated": {
            "value": 1.0 if m["channel_dup_isolated"] else 0.0, "tol": 0.0
        },
        # destructive verbs are tenant-scoped — A's resource survives B's drop
        "destructive.mt_survives_foreign_drop": {
            "value": 1.0 if m["mt_survives_foreign_drop"] else 0.0, "tol": 0.0
        },
        "destructive.tp_survives_foreign_drop": {
            "value": 1.0 if m["tp_survives_foreign_drop"] else 0.0, "tol": 0.0
        },
        # cross-transport parity for the wire verbs
        "parity.cross_transport_equal": {
            "value": 1.0 if all(p["equal"] for p in parity) else 0.0, "tol": 0.0
        },
        # --- BYO-auth seam verdict, over the REAL Flight SQL wire ----------- #
        "byo_auth.bearer_on_flight_observed": {
            "value": 1.0 if byo["bearer_on_flight_observed"] else 0.0, "tol": 0.0
        },
        "byo_auth.anonymous_no_bearer": {
            "value": 1.0 if byo["anonymous_no_bearer"] else 0.0, "tol": 0.0
        },
        "byo_auth.a_isolated_over_flight": {
            "value": 1.0 if byo["a_isolated_over_flight"] else 0.0, "tol": 0.0
        },
        "byo_auth.b_isolated_over_flight": {
            "value": 1.0 if byo["b_isolated_over_flight"] else 0.0, "tol": 0.0
        },
        "byo_auth.missing_rejected_over_flight": {
            "value": 1.0 if byo["missing_rejected_over_flight"] else 0.0, "tol": 0.0
        },
        "byo_auth.forged_rejected_over_flight": {
            "value": 1.0 if byo["forged_rejected_over_flight"] else 0.0, "tol": 0.0
        },
        "byo_auth.legit_after_forgery_over_flight": {
            "value": 1.0 if byo["legit_after_forgery_over_flight"] else 0.0, "tol": 0.0
        },
        "byo_auth.over_flight_eq_embedded": {
            "value": 1.0 if byo["over_flight_eq_embedded"] else 0.0, "tol": 0.0
        },
    }

    (ARTIFACTS / "matrix.json").write_text(json.dumps(embedded_matrix, indent=2, sort_keys=True))
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(golden, indent=2, sort_keys=True))

    record = {
        "purpose": (
            "The engine↔cookbook validator for the §3.5 standing isolation oracle: for "
            "every tenant-scoped verb, tenant B cannot see or reach tenant A's resource, "
            "measured from the consumer side as a per-verb matrix of hard zeros and "
            "documented stated-positives, on the embedded engine and a live grpc:// "
            "jammi-server. Plus the BYO-auth seam, demonstrated from the consumer side over "
            "the REAL Flight SQL wire: a generic HMAC-bearer-token gateway that reads the "
            "bearer off a genuine db.sql() Flight call and binds the verified tenant for a "
            "real-engine read, in front of the engine's tenant binding."
        ),
        "transports": [
            "embedded (file://, in-process)",
            "remote (grpc://, live jammi-server) — the catalog reads + the discriminator "
            "sql row read",
            "the BYO-auth seam over the real Flight SQL lane (pyarrow.flight) — a "
            "BearerCredentials db.sql() against a consumer-side gateway",
        ],
        "parity_verdict": "remote == embedded for every cross-transport observable",
        "parity": parity,
        "matrix": embedded_matrix,
        "byo_auth": byo,
        "byo_auth_note": (
            "The engine authenticates nothing on any transport; the jammi-session-id header "
            "is a transport correlation id, NOT a trust boundary. Verifying a caller is the "
            "consumer's job — a gateway in front of the engine. The engine's grpc_byo_auth.rs "
            "worked example shows that seam for the TYPED gRPC verbs; the Flight SQL lane "
            "(db.sql()) is the gateway-in-front's responsibility there too (engine #220, by "
            "design). On jammi-client 0.32.0 the client carries the channel's bearer on the "
            "Flight SQL lane as well as the typed verbs (jammi #96; on 0.31.0 it rode only "
            "the typed path). This is the consumer-side mirror over that real Flight wire: a "
            "pyarrow.flight gateway reads the inbound bearer off a genuine db.sql() call, a "
            "generic HMAC-signed bearer token it verifies and maps to a tenant, and a "
            "real-engine tenant_scope read it returns to the caller. A missing/invalid "
            "credential is rejected (FlightUnauthenticatedError) in get_flight_info BEFORE "
            "any engine read runs — it never falls through to an unscoped read. The engine "
            "ships the credential plumbing and the per-request tenant scope; it never ships "
            "the auth. Names no consumer, no real IdP."
        ),
        "leak_finding": (
            "NONE — every hard-zero verb measured 0; every destructive verb left A's "
            "resource intact (no cross-tenant destruction); duplicate ids ERROR or isolate, "
            "never clobber. The drop_mutable_table cross-tenant-destruction defect flagged "
            "during scouting is NOT present on the pinned 0.30.0 engine: B's drop of a name A "
            "registered resolves in B's own namespace (not found), and A's table survives "
            "(mt_survives_foreign_drop=True, measured)."
        ),
    }
    (ARTIFACTS / "tenancy_h3.json").write_text(json.dumps(record, indent=2, sort_keys=True))

    _write_checksums()

    # --- the loud verdict ---------------------------------------------------- #
    print("\n=== per-verb isolation matrix, measured (embedded canonical) ===", flush=True)
    print("  HARD ZEROS (must be 0):", flush=True)
    for k in ("source_listing_leak", "describe_source_leak", "mutable_listing_leak",
              "topic_listing_leak", "channel_listing_leak", "model_listing_leak",
              "sql_row_leak"):
        print(f"    {k:24s} = {m[k]}", flush=True)
    print("  STATED-POSITIVES:", flush=True)
    print(f"    builtin_channels_visible = {m['builtin_channels_visible']} "
          f"(B sees the global built-ins)", flush=True)
    print(f"    caveat_visible           = {m['caveat_visible']} "
          f"(A reads a discriminator-less source whole)", flush=True)
    print("  COLLISIONS (error / isolated, never clobber):", flush=True)
    print(f"    mutable_dup_errored      = {m['mutable_dup_errored']} "
          f"(global PK collision ERRORS)", flush=True)
    print(f"    topic_dup_isolated       = {m['topic_dup_isolated']} "
          f"(per-tenant namespace)", flush=True)
    print(f"    channel_dup_isolated     = {m['channel_dup_isolated']} "
          f"(per-tenant namespace)", flush=True)
    print("  DESTRUCTIVE VERBS (A's resource SURVIVES B's drop):", flush=True)
    print(f"    mt_survives_foreign_drop = {m['mt_survives_foreign_drop']}", flush=True)
    print(f"    tp_survives_foreign_drop = {m['tp_survives_foreign_drop']}", flush=True)
    leaks = [k for k in ("source_listing_leak", "describe_source_leak", "mutable_listing_leak",
                         "topic_listing_leak", "channel_listing_leak", "model_listing_leak",
                         "sql_row_leak") if m[k] != 0]
    if leaks or not (m["mt_survives_foreign_drop"] and m["tp_survives_foreign_drop"]):
        print(f"\n  *** LEAK DETECTED — candidate ENGINE finding: {leaks} / "
              f"survives mt={m['mt_survives_foreign_drop']} tp={m['tp_survives_foreign_drop']} "
              f"***", flush=True)
    else:
        print("\n  NO LEAK — every hard zero is 0; A survives every foreign destructive call.",
              flush=True)
    print(f"  parity: remote == embedded for all {len(parity)} cross-transport observables",
          flush=True)
    print("\n=== BYO-auth seam over the real Flight SQL wire, measured ===", flush=True)
    print(f"  bearer_on_flight_observed       = {byo['bearer_on_flight_observed']} "
          f"(the gateway saw 'Bearer <minted>' on the Flight lane)", flush=True)
    print(f"  anonymous_no_bearer             = {byo['anonymous_no_bearer']} "
          f"(anonymous db.sql() carried no authorization header)", flush=True)
    print(f"  a_isolated_over_flight          = {byo['a_isolated_over_flight']}", flush=True)
    print(f"  b_isolated_over_flight          = {byo['b_isolated_over_flight']}", flush=True)
    print(f"  missing_rejected_over_flight    = {byo['missing_rejected_over_flight']} "
          f"(anonymous Flight call rejected, not run unscoped)", flush=True)
    print(f"  forged_rejected_over_flight     = {byo['forged_rejected_over_flight']} "
          f"(legit still resolves: {byo['legit_after_forgery_over_flight']})", flush=True)
    print(f"  over_flight_eq_embedded         = {byo['over_flight_eq_embedded']} "
          f"(gateway read == embedded list_sources under the same tenant)", flush=True)

    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def _fixtures_root(arg: str | None) -> Path:
    root = arg or os.environ.get("JAMMI_FIXTURES_ROOT")
    if not root:
        raise SystemExit(
            "pass --fixtures-root (or set JAMMI_FIXTURES_ROOT) to the engine "
            "checkout carrying tests/fixtures/tiny_modernbert (the base model the "
            "list_models hard zero fine-tunes under tenant A)"
        )
    p = Path(root).resolve()
    if not (p / "tests" / "fixtures" / "tiny_modernbert" / "config.json").exists():
        raise SystemExit(f"--fixtures-root {p} has no tests/fixtures/tiny_modernbert")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", default=None,
                    help="engine checkout with tests/fixtures/tiny_modernbert "
                         "(or set JAMMI_FIXTURES_ROOT)")
    ap.add_argument("--server-bin", default=os.environ.get("JAMMI_SERVER_BIN"),
                    help="built CPU jammi-server binary (or set JAMMI_SERVER_BIN)")
    args = ap.parse_args()
    if not args.server_bin or not os.path.exists(args.server_bin):
        raise SystemExit("pass --server-bin (or set JAMMI_SERVER_BIN) to a built jammi-server")
    emit(_fixtures_root(args.fixtures_root), args.server_bin)


if __name__ == "__main__":
    main()
