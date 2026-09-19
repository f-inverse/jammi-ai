"""Conformance: the embed wheel's remote arm IS `jammi-ai`, by construction.

The whole point of the composition is that the remote surface is
DEFINED ONCE — in `jammi-ai` — and `jammi-ai`'s remote target delegates to
it. So these tests assert the construction holds rather than re-listing a
parallel surface that could drift:

  1. `jammi.connect(remote)` returns a `jammi.RemoteDatabase` — the
     embed wheel's remote-capable surface IS the client's, not a copy.
  2. `connect(target)` routes by scheme: `file://` → the compiled local engine
     (`jammi.EmbeddedBackend`); `https://`/`grpc://` → the client's
     `RemoteDatabase`.
  3. The pure client, asked for a `file://` local target, raises the truthful
     no-embedded-engine error (the runtime echo of the Rust `#[cfg]` gate).
  4. The remote verb surface the two wheels expose is the SAME set of method
     names with the SAME signatures — which is automatic here (it is one class),
     but pinned so a future hand-rolled divergence is caught.

Hermetic: grpcio channels are lazy, so a `connect("grpc://…")` opens no socket
until a verb runs. No server is contacted.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path

import grpc
import pytest

import jammi
import jammi_native


def _embed_method(verb: str):
    """Resolve `verb` against the COMPOSED embedded surface: the thin Python
    `jammi.EmbeddedBackend`'s explicit method if it declares one, else the
    `_NativeDatabase` low-level handle's method it delegates to via
    ``__getattr__``.

    The embedded `Database` is a thin wrapper over the compiled
    `_NativeDatabase`: the migrated training verbs are explicit Python methods on
    the wrapper (driving the shared request assembly), while every un-migrated
    verb lives on the native handle and is forwarded at runtime. A class-level
    introspection (which is what the conformance guard does) must look through the
    same composition — wrapper first, native handle behind it — to see the verb a
    caller actually invokes."""
    if verb in vars(jammi.EmbeddedBackend):
        return getattr(jammi.EmbeddedBackend, verb)
    return getattr(jammi_native._NativeDatabase, verb)


def _embed_has(verb: str) -> bool:
    """Whether the composed embedded surface carries `verb` — declared on the thin
    `Database` wrapper or on the `_NativeDatabase` handle behind it."""
    return verb in vars(jammi.EmbeddedBackend) or hasattr(
        jammi_native._NativeDatabase, verb
    )


def test_embed_remote_is_the_client_remote_database():
    """`jammi.connect(remote)` returns the client's `RemoteDatabase` — the
    remote arm is defined once, in `jammi-ai`, and reused by composition."""
    db = jammi.connect("grpc://127.0.0.1:8081")
    try:
        assert isinstance(db, jammi.RemoteDatabase)
    finally:
        db.close()


def test_connect_routes_local_to_the_compiled_engine(tmp_path):
    """A `file://` target resolves to the in-process engine — a
    `jammi.EmbeddedBackend`, never the remote client."""
    db = jammi.connect(f"file://{tmp_path}")
    assert isinstance(db, jammi.EmbeddedBackend)
    assert not isinstance(db, jammi.RemoteDatabase)


def test_base_client_resolves_the_engine_when_the_extra_is_present():
    """With the `[embedded]` extra installed (this lane carries `jammi_native`),
    the BASE client's `file://` front door resolves to the in-process engine — the
    base discovers the native backend on its own.

    The truthful no-engine ERROR (the extra ABSENT) is pinned where it is real, in
    the client-only lane `clients/python/tests/test_target.py` — that lane installs
    `jammi-ai` without the extra, so its `file://` raises; here, with the
    engine present, the same call resolves it instead of raising."""
    import tempfile

    db = jammi.connect(f"file://{tempfile.mkdtemp()}")
    assert isinstance(db, jammi.EmbeddedBackend)
    assert isinstance(db, jammi.Session)


# The remote verb vocabulary both wheels speak. These are the Stage-1 embedding
# verbs plus the session/handshake trio — the transport-agnostic surface.
_REMOTE_VERBS = {
    "add_source",
    "generate_embeddings",
    "encode_query",
    "search",
    "sql",
    "list_sources",
    "describe_source",
    "set_tenant",
    "tenant_scope",
    "tenant",
    "get_server_info",
}


# The training + predict verbs. Unlike the conformal numerics, these DO hit the
# wire: training is offloaded to the remote GPU server (`JobService.SubmitJob`)
# and the predict verb runs the trained predictor remotely
# (`InferenceService.Predict`). The embedded `Database` submits/serves in the
# compiled engine; the client's `RemoteDatabase` submits/serves over gRPC. The
# call surface must agree so a caller swaps transports without changing the
# call — pinned here against the embed `jammi.EmbeddedBackend`.
_TRAINING_VERBS = {
    "fine_tune",
    "fine_tune_graph",
    "train_context_predictor",
    "predict_with_context_predictor",
    # The job verbs: a submitted job is reachable by id from a session that
    # never submitted it, the tenant's jobs are listable, cancellable, and
    # prunable, and the worker fleet is listable. Both arms carry all five —
    # none is a `Capability`, because nothing about a transport makes "look up
    # a job I already have the id of" (or list/cancel/prune it, or list the
    # fleet) unavailable. Generic across every `jobs` row (training or
    # compute), per `JobService`.
    "job",
    "list_jobs",
    "cancel_job",
    "list_workers",
    "prune_jobs",
}


# The bulk inference verb. It DOES hit the wire: the model and the registered
# source both live in the engine, so the compute runs where the data is
# (`InferenceService.Infer`) and only the output rows cross the wire. The
# embedded `Database` runs it in the compiled engine; the client's
# `RemoteDatabase` drives it over gRPC. The call surface must agree so a caller
# swaps transports without changing the call — pinned here against the embed
# `jammi.EmbeddedBackend`.
_INFERENCE_VERBS = {
    "infer",
}


# The engine-state pipeline verbs. Like the training/predict verbs, these DO hit
# the wire: they build durable graph/embedding artifacts or assemble a target's
# conditioning context against the remote engine's state (`PipelineService`).
# The embedded `Database` runs them in the compiled engine; the client's
# `RemoteDatabase` drives them over gRPC. The call surface must agree so a
# caller swaps transports without changing the call — pinned here against the
# embed `jammi.EmbeddedBackend`.
_PIPELINE_VERBS = {
    "build_neighbor_graph",
    "propagate_embeddings",
    "asof_join",
    "assemble_context",
    "recompute",
    "verify_materialization",
    "staleness",
    "derives_from",
}


# The evaluation verbs. They DO hit the wire: the model and the golden data
# both live in the engine, so the compute runs where the data is
# (`EvalService`) and only the typed report crosses the wire. The embedded
# `Database` runs them in the compiled engine; the client's `RemoteDatabase`
# drives them over gRPC. The call surface must agree so a caller swaps
# transports without changing the call — pinned here against the embed
# `jammi.EmbeddedBackend`.
_EVAL_VERBS = {
    "eval_embeddings",
    "eval_per_query",
    "eval_inference",
    "eval_compare",
    "eval_calibration",
}


# The evidence-channel registry verbs. These DO hit the wire: the channel
# catalog lives in the engine, so register/append/list run against the remote
# engine's state (`CatalogService`). The embedded `Database` mutates/reads the
# compiled engine's catalog; the client's `RemoteDatabase` drives them over
# gRPC. The catalog is tenant-scoped, so both honour the session's bound tenant.
# The call surface must agree so a caller swaps transports without changing the
# call — pinned here against the embed `jammi.EmbeddedBackend`.
_CHANNEL_VERBS = {
    "register_channel",
    "add_channel_columns",
    "list_channels",
}


# The stateless conformal / RRF numerics. These are NOT on the gRPC wire: their
# inputs are caller-supplied arrays the engine never holds, so a wire hop would
# only ship data the caller already has. The embedded `Database` computes them
# in the compiled engine; the client's `RemoteDatabase` computes them locally in
# pure Python from the SAME algorithm — so the verb surface agrees on both
# transports without a server round-trip. Pinned here so the two stay in lockstep.
_NUMERIC_VERBS = {
    "conformalize",
    "conformalize_interval",
    "conformalize_cqr",
    "rrf_fuse",
}


# The mutable-companion-table lifecycle and the trigger topic verbs. These DO hit
# the wire: the create/drop/list-mutable-table and register/drop/list-topic verbs
# are control-plane (`CatalogService`); `publish_topic` / `subscribe_collect` are
# data-plane (`TriggerService`). The embedded `Database` registers/publishes in
# the compiled engine; the client's `RemoteDatabase` drives them over gRPC. The
# call surface must agree so a caller swaps transports without changing the call —
# pinned here against the embed `jammi.EmbeddedBackend`. `subscribe_collect` mirrors
# the embedded replay+live-tail collect (bounded by `max_batches`), not a
# replay-only drain.
_MUTABLE_TOPIC_VERBS = {
    "create_mutable_table",
    "drop_mutable_table",
    "list_mutable_tables",
    "register_topic",
    "drop_topic",
    "list_topics",
    "publish_topic",
    "subscribe_collect",
}


# The model-lifecycle verbs. These DO hit the wire: the model catalog lives in
# the engine, so list/describe/delete run against the remote engine's state
# (`CatalogService`). The embedded `Database` mutates/reads the compiled engine's
# catalog; the client's `RemoteDatabase` drives them over gRPC. The catalog is
# tenant-scoped, so both honour the session's bound tenant. The call surface must
# agree so a caller swaps transports without changing the call — pinned here
# against the embed `jammi.EmbeddedBackend`.
_LIFECYCLE_VERBS = {
    "list_models",
    "describe_model",
    "delete_model",
}


# The index-segment listing. A result table's ANN index is a SET of immutable
# segments (one `index_segments` catalog row each); this is the reader for that
# set. It DOES hit the wire on the remote arm (`CatalogService.
# ListIndexSegments`) because the catalog lives in the engine. Both arms carry
# it — it is a verb, never a `Capability`: nothing about a transport makes a
# segment listing unavailable, so a caller never has to ask whether it exists.
_SEGMENT_VERBS = {
    "list_index_segments",
}


# The catalog/object-store cross-check. It DOES hit the wire on the remote arm
# (`CatalogService.Reconcile`) because the catalog and the object store both
# live in the engine. The embedded `Database` runs it in the compiled engine
# (`all=True` calling `ResultStore::reconcile_all` directly, no authorizer
# gate — the in-process caller is trusted); the client's `RemoteDatabase`
# drives it over gRPC, where the server gates `all=True` behind a
# deployment-supplied admin authorizer. The call surface must still agree so a
# caller swaps transports without changing the call — pinned here against the
# embed `jammi.EmbeddedBackend`.
_RECONCILE_VERBS = {
    "reconcile",
}


def test_remote_surface_has_every_verb():
    """The client's `RemoteDatabase` exposes the full transport-agnostic verb
    set — the same vocabulary the embedded `Database` carries."""
    for verb in (
        _REMOTE_VERBS
        | _NUMERIC_VERBS
        | _TRAINING_VERBS
        | _INFERENCE_VERBS
        | _PIPELINE_VERBS
        | _EVAL_VERBS
        | _CHANNEL_VERBS
        | _MUTABLE_TOPIC_VERBS
        | _LIFECYCLE_VERBS
        | _SEGMENT_VERBS
        | _RECONCILE_VERBS
    ):
        assert callable(getattr(jammi.RemoteDatabase, verb)), verb


def test_lifecycle_verbs_have_identical_signatures_across_wheels():
    """The model-lifecycle verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _LIFECYCLE_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_segment_verbs_have_identical_signatures_across_wheels():
    """The index-segment listing carries the SAME call surface on both
    transports — one positional `table_name`, no transport-shaped extra."""
    for verb in _SEGMENT_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_reconcile_verb_has_identical_signature_across_wheels():
    """`reconcile` carries the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend` —
    `apply` / `grace_secs` / `all`, same names, same kinds, same defaults —
    even though `all=True` means something different on each transport (a
    trusted in-process caller embedded, an authorizer-gated admin pass
    remote)."""
    for verb in _RECONCILE_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


# The client-facing segment projection: exactly the keys a `list_index_segments`
# entry carries on BOTH transports. The embed wheel builds the dict at its FFI
# boundary from the engine's `IndexSegment`; the remote builds it from the wire
# `IndexSegment` — so the two agree key-for-key, and neither carries the row's
# `tenant_id` / `created_at` bookkeeping.
_INDEX_SEGMENT_DICT_KEYS = {"segment_id", "index_path", "row_count", "version"}


def test_index_segment_projection_is_the_whole_row_and_nothing_more():
    """The wire `IndexSegment` message — the single source of the client-facing
    segment shape — carries exactly the four projected fields. Pinned against
    the proto descriptor, so adding a catalog-internal column (`tenant_id`,
    `created_at`) to the projection reds here.

    Hermetic: reads the generated proto descriptor, never dialing a server."""
    from jammi._generated.jammi.v1 import catalog_pb2

    proto_fields = {f.name for f in catalog_pb2.IndexSegment.DESCRIPTOR.fields}
    assert proto_fields == _INDEX_SEGMENT_DICT_KEYS, (
        f"wire IndexSegment fields {proto_fields} != the client projection "
        f"{_INDEX_SEGMENT_DICT_KEYS} — a catalog-internal field leaked"
    )


def test_embed_list_index_segments_returns_the_projection_shape(tmp_path):
    """The embedded `list_index_segments` returns a list (never `None`, never an
    error) for a table that does not exist — the unknown-table arm of the
    four-way empty contract — and every entry it ever yields carries exactly the
    projected keys.

    Hermetic: opens a local engine (`file://`), contacts no server."""
    db = jammi.connect(f"file://{tmp_path}")
    try:
        segments = db.list_index_segments("no_such_table")
        assert segments == []
    finally:
        db.close()


# The client-facing model projection: exactly the keys a `list_models` /
# `describe_model` entry carries on BOTH transports. The embed wheel projects its
# catalog record through `ModelDescriptor` and the remote builds the same dict
# from the wire `Model`, so the two agree key-for-key — and, critically, neither
# exposes the record's version/lineage/path bookkeeping.
_MODEL_DICT_KEYS = {"model_id", "backend", "task", "status"}


def test_model_projection_is_minimal_and_leaks_no_internal_fields():
    """The wire `Model` message — the single source of the client-facing model
    shape — carries exactly the minimal projection and no server-internal
    bookkeeping. Pinned against the proto descriptor so adding an internal field
    to the projection (the leak the `ModelDescriptor` split prevents) fails here.

    Hermetic: reads the generated proto descriptor, never dialing a server."""
    from jammi._generated.jammi.v1 import catalog_pb2

    proto_fields = {f.name for f in catalog_pb2.Model.DESCRIPTOR.fields}
    assert proto_fields == _MODEL_DICT_KEYS, (
        f"wire Model fields {proto_fields} != the minimal client projection "
        f"{_MODEL_DICT_KEYS} — a server-internal field leaked into the projection"
    )


def test_embed_list_models_returns_the_projection_shape(tmp_path):
    """The embedded `Database.list_models` returns the `ModelDescriptor`
    projection — a list of dicts whose keys are exactly the client-facing set. An
    empty engine lists nothing, so this pins the list-shape and (when populated)
    the key contract via the shared assertion.

    Hermetic: opens a local engine (`file://`), contacts no server."""
    db = jammi.connect(f"file://{tmp_path}")
    models = db.list_models()
    assert isinstance(models, list)
    for m in models:
        assert set(m) == _MODEL_DICT_KEYS, (
            f"embed list_models entry keys {set(m)} != {_MODEL_DICT_KEYS}"
        )


_RECONCILE_REPORT_DICT_KEYS = {
    "scope",
    "applied",
    "rows_failed",
    "rows_failed_count",
    "orphans",
    "orphan_count",
    "pending",
    "pending_count",
    "unattributed",
    "unattributed_count",
    "damaged",
    "damaged_count",
    "referenced",
    "referenced_count",
    "truncated",
    "bytes_reclaimed",
}


def test_reconcile_report_projection_is_the_whole_row_and_nothing_more():
    """The wire `ReconcileReport` message — the single source of the
    client-facing report shape — carries exactly the projected fields, AND
    the remote client's own projection function actually produces that shape
    (not merely the proto descriptor, which a stale/unused projection could
    still pass vacuously against).

    Pinned two ways:
    1. The proto descriptor's field set matches `_RECONCILE_REPORT_DICT_KEYS`
       — a field added to the engine's `ReconcileReport` without a matching
       wire update fails here.
    2. `jammi._database._reconcile_report_to_dict` is actually CALLED on a
       fully-populated `ReconcileReport` (every list non-empty, `truncated`
       True, a `damaged` entry present, every `*_count` deliberately NOT
       equal to its list's length so a projection that derived a count from
       `len(list)` instead of projecting the wire field would be caught) and
       its result's key set matches — byte-for-byte — the embedded PyO3 arm's
       key set (`_RECONCILE_REPORT_DICT_KEYS`, independently pinned against
       `crates/jammi-python/src/database.rs`'s `reconcile` doc comment by
       `test_embed_reconcile_both_arms_return_the_report_shape`), with every
       value round-tripping unchanged.

    Hermetic: reads the generated proto descriptor and calls a pure-Python
    projection function, never dialing a server."""
    from jammi._database import _reconcile_report_to_dict
    from jammi._generated.jammi.v1 import catalog_pb2

    proto_fields = {f.name for f in catalog_pb2.ReconcileReport.DESCRIPTOR.fields}
    assert proto_fields == _RECONCILE_REPORT_DICT_KEYS, (
        f"wire ReconcileReport fields {proto_fields} != the client projection "
        f"{_RECONCILE_REPORT_DICT_KEYS}"
    )

    report = catalog_pb2.ReconcileReport(
        scope="tenant:11111111-1111-4111-8111-111111111111",
        applied=True,
        rows_failed=["t1", "t2"],
        rows_failed_count=99,
        orphans=["o1"],
        orphan_count=50,
        pending=["p1", "p2", "p3"],
        pending_count=77,
        unattributed=["u1"],
        unattributed_count=12,
        damaged=["d1"],
        damaged_count=33,
        referenced=["r1", "r2"],
        referenced_count=44,
        truncated=True,
        bytes_reclaimed=123456,
    )
    projected = _reconcile_report_to_dict(report)

    assert set(projected) == _RECONCILE_REPORT_DICT_KEYS, (
        f"_reconcile_report_to_dict returned {set(projected)} != "
        f"{_RECONCILE_REPORT_DICT_KEYS} (the whole row, and nothing more)"
    )
    assert projected == {
        "scope": "tenant:11111111-1111-4111-8111-111111111111",
        "applied": True,
        "rows_failed": ["t1", "t2"],
        "rows_failed_count": 99,
        "orphans": ["o1"],
        "orphan_count": 50,
        "pending": ["p1", "p2", "p3"],
        "pending_count": 77,
        "unattributed": ["u1"],
        "unattributed_count": 12,
        "damaged": ["d1"],
        "damaged_count": 33,
        "referenced": ["r1", "r2"],
        "referenced_count": 44,
        "truncated": True,
        "bytes_reclaimed": 123456,
    }, f"every field must round-trip unchanged: {projected}"


def test_embed_reconcile_both_arms_return_the_report_shape(tmp_path):
    """The embedded `Database.reconcile` returns the same report dict shape on
    BOTH the tenant-scoped pass (`all=False`, `ResultStore::reconcile`) and the
    cross-tenant admin pass (`all=True`, `ResultStore::reconcile_all`) — the
    embedded engine trusts its own caller, so `all=True` needs no wired
    authorizer to exercise here, unlike the remote arm (gated server-side).
    `apply=False` on an empty, freshly-opened engine reports (and mutates)
    nothing on either arm.

    Hermetic: opens a local engine (`file://`), contacts no server."""
    db = jammi.connect(f"file://{tmp_path}")
    try:
        for all_tenants, expected_scope in ((False, "_global"), (True, "all")):
            report = db.reconcile(apply=False, grace_secs=0, all=all_tenants)
            assert set(report) == _RECONCILE_REPORT_DICT_KEYS, (
                f"embed reconcile(all={all_tenants}) keys {set(report)} != "
                f"{_RECONCILE_REPORT_DICT_KEYS}"
            )
            assert report["applied"] is False
            assert report["scope"] == expected_scope
            for key in (
                "rows_failed",
                "orphans",
                "pending",
                "unattributed",
                "damaged",
                "referenced",
            ):
                assert report[key] == [], f"a freshly-opened engine reports nothing: {report}"
            for key in (
                "rows_failed_count",
                "orphan_count",
                "pending_count",
                "unattributed_count",
                "damaged_count",
                "referenced_count",
            ):
                assert report[key] == 0, f"a freshly-opened engine reports nothing: {report}"
            assert report["truncated"] is False
    finally:
        db.close()


def test_embed_reconcile_referenced_list_is_populated_and_matches_the_remote_key_set(
    tmp_path,
):
    """`test_embed_reconcile_both_arms_return_the_report_shape` above only ever
    exercises the embedded `Database.reconcile` on an EMPTY, freshly-opened
    engine, so the `referenced` / `referenced_count` fields (the reap-site
    consult's own output — see `ResultStore::reconcile`'s doc and
    `crates/jammi-db/tests/it/reconcile.rs::a_stray_file_under_a_referenced_attempt_level_prefix_survives_via_the_reap_site_consult`)
    are asserted structurally (always `[]` / `0`) there, never EXECUTED on a
    non-empty case through this binding.

    This test reproduces that exact it-test's scenario through the embedded
    engine's own on-disk layout instead: a `models` row is registered naming
    an attempt-level artifact prefix (`models/_global/{job}/worker-1/0`, the
    `[job_id, worker_id, attempt]` shape `worker.rs` registers in production),
    a valid bundle (`adapter.safetensors` + `manifest.json`) is published
    under it, and a STRAY file the manifest does not name is written directly
    alongside it. `reconcile(apply=True)` must find that stray file, consult
    `prefix_is_referenced` on its own key, and report it under `referenced` —
    never reclaim it — the same live-through-containment case the Rust
    it-test proves at the engine layer, now proven not to drop across this
    binding's serde projection.

    No embedded verb exists to register a model or publish an artifact
    bundle directly, so both are constructed the same way the engine itself
    would lay them out on disk: a raw `sqlite3` INSERT mirroring
    `Catalog::register_model`'s own statement (the same close-before-inject
    discipline `test_remote_and_embedded_job_metrics_agree_on_all_three_states`
    already uses against the `jobs` table), and `manifest.json` written by
    hand in the exact shape `ArtifactStore::put_artifact` produces. The
    `reconcile` CALL ITSELF — the artifact under test — is the real,
    compiled engine's, not a stand-in.

    **Close-before-inject is not optional here, it is load-bearing**: the
    SQLite catalog's own module doc
    (`crates/jammi-db/src/catalog/backend_sqlite.rs`, "Residual, deliberately
    not closed") names exactly this shape — a foreign SQLite library
    instance (CPython's own `sqlite3`, linked against the platform
    `libsqlite3`, as opposed to the engine's bundled amalgamation) writing to
    `catalog.db` while an engine connection is ALSO open is an
    out-of-contract topology: the engine's own `unix-excl` VFS keeps its
    wal-index on the HEAP (never re-reading the on-disk `-wal`), so a raw
    write landing while an engine `Database` is live is silently invisible to
    it, never a loud failure. Every raw `sqlite3` write below therefore runs
    with NO embedded engine connection open at all — verified upstream by
    `test_remote_and_embedded_job_metrics_agree_on_all_three_states`'s own
    docstring — and only the FRESH `db` opened after is ever used to read it.

    Hermetic: opens a local engine (`file://`), contacts no server.
    """
    import hashlib
    import json
    import sqlite3
    import uuid

    from jammi._database import _reconcile_report_to_dict
    from jammi._generated.jammi.v1 import catalog_pb2

    # The remote projection's own shape, computed against a POPULATED
    # `ReconcileReport` — every list field non-empty, `referenced_count`
    # equal to `len(referenced)`, `truncated` False — never an EMPTY
    # message. A proto's field set is fixed by its descriptor regardless of
    # which fields carry values, so comparing key sets against an empty
    # message is structurally true for any message and would keep passing
    # even if `_reconcile_report_to_dict` mis-typed or miscounted a
    # populated field; only a populated fixture exercises that. Pinned here
    # directly (not via `_RECONCILE_REPORT_DICT_KEYS`, which this test's
    # non-empty case must agree with independently) so this test alone
    # still catches a dropped `referenced` field even if the module-level
    # constant above were wrong.
    remote_report = catalog_pb2.ReconcileReport(
        scope="tenant:22222222-2222-4222-8222-222222222222",
        applied=True,
        rows_failed=["rf1", "rf2"],
        rows_failed_count=2,
        orphans=["o1"],
        orphan_count=1,
        pending=["p1", "p2"],
        pending_count=2,
        unattributed=["u1"],
        unattributed_count=1,
        damaged=["d1"],
        damaged_count=1,
        referenced=["r1", "r2", "r3"],
        referenced_count=3,
        truncated=False,
        bytes_reclaimed=42,
    )
    remote_projected = _reconcile_report_to_dict(remote_report)
    remote_keys = set(remote_projected)
    assert remote_projected["referenced_count"] == len(remote_projected["referenced"]), (
        "the populated fixture itself must be internally consistent before it "
        f"is used as the comparison oracle: {remote_projected}"
    )

    # Bootstrap: open + immediately close, so `catalog.db` and the
    # `jammi_db/` root exist (migrations applied) with NO engine connection
    # left attached before the raw-sqlite3 injection below.
    bootstrap_db = jammi.connect(f"file://{tmp_path}")
    bootstrap_db.close()
    del bootstrap_db

    job_id = str(uuid.uuid4())  # attribution requires a canonical v4 UUID
    prefix_dir = tmp_path / "jammi_db" / "models" / "_global" / job_id / "worker-1" / "0"
    prefix_dir.mkdir(parents=True)

    weights = b"weights"
    (prefix_dir / "adapter.safetensors").write_bytes(weights)
    manifest = {
        "files": [
            {
                "name": "adapter.safetensors",
                "sha256": hashlib.sha256(weights).hexdigest(),
            }
        ]
    }
    # Manifest LAST, mirroring `ArtifactStore::put_artifact`'s own write
    # order — its presence is what marks the bundle complete.
    (prefix_dir / "manifest.json").write_text(json.dumps(manifest))

    # A stray object the manifest does not name, directly under the SAME
    # attempt-level directory the model row's `artifact_path` will EQUAL
    # exactly — a strict descendant of it, never reclaimable through the
    # ordinary age-gated orphan arm regardless of age; only the reap-site's
    # own `prefix_is_referenced` consult on this exact key protects it.
    (prefix_dir / "debug_dump.tmp").write_bytes(b"leftover")

    artifact_path = f"file://{prefix_dir}"
    catalog_db = tmp_path / "catalog.db"
    conn = sqlite3.connect(str(catalog_db))
    try:
        # `models.{created_at, updated_at}` are canonical-stamp columns
        # (migration 039): the schema edge refuses any other shape, including
        # the legacy `CURRENT_TIMESTAMP` DEFAULT a raw INSERT would otherwise
        # fall back on — so this fixture stamps explicitly, exactly as every
        # catalog writer does (`%Y-%m-%dT%H:%M:%S%.6fZ`).
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn.execute(
            "INSERT INTO models "
            "(model_id, name, model_type, task, backend, version, status, "
            " metadata, artifact_path, tenant_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'registered', ?, ?, NULL, ?, ?)",
            (
                "attempt-level-model::1",  # untenanted `model_pk(None, name, version)`
                "attempt-level-model",
                "lora",
                "text_embedding",
                "candle",
                1,
                json.dumps({"base_model_id": None, "config_json": None}),
                artifact_path,
                stamp,
                stamp,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Fresh db, opened only AFTER the raw-sqlite3 injection above is fully
    # committed and its own connection fully closed.
    db = jammi.connect(f"file://{tmp_path}")
    try:
        # `grace_secs` need only clear the deployment's configured lease
        # duration (default 30s; `apply=True` refuses a shorter grace) — the
        # reap-site's `referenced` consult itself runs before any age gate,
        # so a fresh object still lands in `referenced`, never `orphans`.
        report = db.reconcile(apply=True, grace_secs=3600, all=False)

        assert set(report) == remote_keys, (
            f"embed reconcile(apply=True) keys {set(report)} != the remote "
            f"projection's {remote_keys}"
        )
        for key in remote_keys:
            assert type(report[key]) is type(remote_projected[key]), (
                f"{key}: embedded value {report[key]!r} ({type(report[key])}) != "
                f"remote-projection value type {type(remote_projected[key])}"
            )
        assert any(r.endswith("debug_dump.tmp") for r in report["referenced"]), (
            f"the reap-site consult must name the stray file referenced: {report}"
        )
        assert report["referenced_count"] == len(report["referenced"]), (
            f"referenced_count must be the true total: {report}"
        )
        assert report["truncated"] is False
        assert all(not o.endswith("debug_dump.tmp") for o in report["orphans"]), (
            f"a referenced stray file must never be reclaimed: {report}"
        )
        assert (prefix_dir / "debug_dump.tmp").exists()
    finally:
        db.close()


def test_mutable_topic_verbs_have_identical_signatures_across_wheels():
    """The mutable-table + topic + pub/sub verbs carry the SAME call surface on the
    client's `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`. Both
    drive over the same verb vocabulary (the client over gRPC, the embed
    in-process), so a caller swaps transports without changing the call — pinned
    name-for-name, kind-for-kind, and default-for-default so a divergence in either
    is caught."""
    for verb in _MUTABLE_TOPIC_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_inference_verbs_have_identical_signatures_across_wheels():
    """The bulk inference verb carries the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _INFERENCE_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_pipeline_verbs_have_identical_signatures_across_wheels():
    """The engine-state pipeline verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _PIPELINE_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_eval_verbs_have_identical_signatures_across_wheels():
    """The evaluation verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _EVAL_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


# The vector-search verb. It DOES hit the wire: the source's embedding tables
# live in the engine, so the ANN search runs where the vectors are
# (`EmbeddingService.Search`) and only the hits cross the wire. The embedded
# `Database` searches in the compiled engine; the client's `RemoteDatabase`
# drives it over gRPC. The call surface — including the `embedding_table=`
# selector that names WHICH of a source's embedding tables to search — must
# agree so a caller swaps transports without changing the call. Pinned against
# the embed `jammi.EmbeddedBackend`.
_SEARCH_VERBS = {
    "search",
}


def test_search_verb_has_identical_signature_across_wheels():
    """`search` carries the SAME call surface on the client's `RemoteDatabase`
    as on the embedded engine's `jammi.EmbeddedBackend` — including the
    `embedding_table=` table selector. Both drive over the same verb vocabulary
    (the client over gRPC, the embed in-process), so a caller swaps transports
    without changing the call — pinned name-for-name, kind-for-kind, and
    default-for-default so a divergence in either is caught."""
    for verb in _SEARCH_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"
        names = {p[0] for p in embed}
        assert "embedding_table" in names, f"{verb} must expose embedding_table"


def test_channel_verbs_have_identical_signatures_across_wheels():
    """The evidence-channel registry verbs carry the SAME call surface on the
    client's `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`.
    Both drive over the same verb vocabulary (the client over gRPC, the embed
    in-process), so a caller swaps transports without changing the call — pinned
    name-for-name, kind-for-kind, and default-for-default so a divergence in
    either is caught."""
    for verb in _CHANNEL_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_training_verbs_have_identical_signatures_across_wheels():
    """The training + predict verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`. Both submit
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _TRAINING_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_remote_job_matches_the_local_handle_shape():
    """The client's `RemoteJob` carries the SAME handle surface as the
    embedded engine's `Job`: the `job_id` / `kind` / `output_model_id`
    properties, the `status()` / `wait()` / `progress()` / `cancel()`
    methods, `metrics()`, and `acceleration_report()`. A remote `wait()`
    polls `JobStatus` and raises on a
    failed job with the wire error, mirroring the local handle, so a caller
    treats the two interchangeably."""
    local = jammi_native.Job
    remote = jammi.RemoteJob
    for member in (
        "job_id",
        "kind",
        "output_model_id",
        "status",
        "wait",
        "progress",
        "cancel",
        "metrics",
        "acceleration_report",
    ):
        assert hasattr(remote, member), member
        assert hasattr(local, member), member
    # `job_id` / `kind` / `output_model_id` are read-only attributes on both
    # handles (a property on the pure client, a getter on the native handle);
    # `status` / `wait` / `progress` / `cancel` are callable methods on both.
    for method in ("status", "wait", "progress", "cancel"):
        assert callable(getattr(remote, method))
        assert callable(getattr(local, method))


def _call_surface(fn) -> list:
    """The (name, kind, default) of each parameter — the call surface, ignoring
    type annotations (a native PyO3 method and a typed Python method differ
    there by construction, but must agree on what a caller passes and how). A
    leading `self` is dropped so an unbound pure-Python method (which lists it)
    compares against the native method descriptor (which may not)."""
    params = [
        (p.name, p.kind, p.default)
        for p in inspect.signature(fn).parameters.values()
    ]
    if params and params[0][0] == "self":
        params = params[1:]
    return params


def test_numeric_verbs_have_identical_signatures_across_wheels():
    """The conformal / RRF numerics carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi.EmbeddedBackend`. They are
    computed locally on both wheels (no wire hop), so the verb surface must agree
    name-for-name, kind-for-kind, and default-for-default — pinned here so a
    divergence in either implementation is caught."""
    for verb in _NUMERIC_VERBS:
        client = _call_surface(getattr(jammi.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_numeric_verbs_compute_identically_across_wheels(tmp_path):
    """The client's pure-Python conformal / RRF numerics produce output EQUAL to
    the embedded engine's on shared fixtures. Both are computed locally (no
    server), so this asserts the two implementations reproduce the same
    finite-sample quantile, score families, interval construction, and fusion
    order — the byte-identical agreement the compute-to-data split requires.

    Hermetic: the embedded side opens a local engine (`file://`); the client side
    runs entirely in process. No server is contacted by either."""
    local = jammi.connect(f"file://{tmp_path}")
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        assert type(remote) is jammi.RemoteDatabase

        # Classification: one row per family, shared calibration / test fixtures.
        calibration = [
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
        ]
        true_labels = [0, 1, 2, 0, 1]
        test = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.34, 0.33, 0.33]]
        for score, raps_params in (
            ("lac", None),
            ("aps", None),
            ("raps", (0.5, 1)),
        ):
            assert local.conformalize(
                calibration, true_labels, test,
                alpha=0.2, score=score, raps_params=raps_params,
            ) == remote.conformalize(
                calibration, true_labels, test,
                alpha=0.2, score=score, raps_params=raps_params,
            ), score

        # Absolute-residual regression interval.
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.2, 1.7, 3.4, 3.6, 5.5]
        test_predictions = [2.5, 6.0]
        assert local.conformalize_interval(
            predictions, observed, test_predictions, alpha=0.25
        ) == remote.conformalize_interval(
            predictions, observed, test_predictions, alpha=0.25
        )

        # CQR regression interval.
        lower = [-1.0, -2.0, -1.5, -1.0, -0.5]
        upper = [1.0, 2.0, 1.5, 1.0, 0.5]
        cqr_observed = [0.5, -2.5, 1.0, -1.5, 0.0]
        test_lower = [-1.0, -3.0]
        test_upper = [1.0, 3.0]
        assert local.conformalize_cqr(
            lower, upper, cqr_observed, test_lower, test_upper, alpha=0.25
        ) == remote.conformalize_cqr(
            lower, upper, cqr_observed, test_lower, test_upper, alpha=0.25
        )

        # Reciprocal-rank fusion, default and explicit k_rrf.
        ranked_lists = [["a", "b", "c"], ["c", "a", "d"], ["b", "d", "a"]]
        assert local.rrf_fuse(ranked_lists) == remote.rrf_fuse(ranked_lists)
        assert local.rrf_fuse(ranked_lists, k_rrf=40) == remote.rrf_fuse(
            ranked_lists, k_rrf=40
        )
    finally:
        # Only the remote client's gRPC channel needs an explicit close in a
        # `finally`. The embedded arm carries a real `close()` too (the catalog
        # release), but this test opens a fresh per-test directory no successor
        # ever reopens, so letting the handle drop is enough here.
        remote.close()


def test_embed_remote_and_client_share_identical_signatures():
    """Because the embed wheel's remote arm IS `jammi.RemoteDatabase`,
    every verb's signature is identical by construction. Asserting it here pins
    the invariant so any future hand-rolled remote class in the embed wheel
    (which would re-introduce the very drift the composition removes) fails
    this test."""
    embed_remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        assert type(embed_remote) is jammi.RemoteDatabase
        for verb in _REMOTE_VERBS:
            client_sig = inspect.signature(getattr(jammi.RemoteDatabase, verb))
            embed_sig = inspect.signature(getattr(type(embed_remote), verb))
            assert client_sig == embed_sig, f"{verb}: {embed_sig} != {client_sig}"
    finally:
        embed_remote.close()


def test_embedded_database_shares_the_unified_modality_verbs():
    """The embedded `Database` carries the unified `encode_query` /
    `generate_embeddings` (the `modality=` form), matching the client — and the
    per-modality names are gone (the deferred Stage-1 unification)."""
    for verb in ("encode_query", "generate_embeddings", "get_server_info"):
        assert _embed_has(verb), verb
    for gone in (
        "encode_text_query",
        "encode_image_query",
        "encode_audio_query",
        "generate_text_embeddings",
        "generate_image_embeddings",
        "generate_audio_embeddings",
        "server_info",
    ):
        assert not _embed_has(gone), f"{gone} should be hard-cut"


def test_tenant_surface_agrees_across_wheels():
    """The tenant verbs carry the SAME names on the embedded `Database` and the
    client's `RemoteDatabase`: the sticky `set_tenant` setter, the block-scoped
    `tenant_scope` context manager, and the `tenant` getter. The old
    `with_tenant` (which mutated in place yet read like a builder, returning
    ``None``) is gone from BOTH surfaces — a caller swaps transports without
    changing the call, and neither surface carries the footgun."""
    for verb in ("set_tenant", "tenant_scope", "tenant"):
        assert callable(_embed_method(verb)), verb
        assert callable(getattr(jammi.RemoteDatabase, verb)), verb
    assert not _embed_has("with_tenant"), "with_tenant must be hard-cut"
    assert not hasattr(
        jammi.RemoteDatabase, "with_tenant"
    ), "with_tenant must be hard-cut"
    # `tenant_scope(tenant_id)` takes the same caller-visible parameter on both —
    # the embedded native method and the client context manager agree name-for-name.
    embed = _call_surface(_embed_method("tenant_scope"))
    client = _call_surface(jammi.RemoteDatabase.tenant_scope)
    assert embed == client, f"tenant_scope: {embed} != {client}"


def test_get_server_info_shape_agrees_across_transports(tmp_path):
    """`get_server_info` returns the SAME key set whether it crossed the gRPC
    wire (client `RemoteDatabase`) or came from the in-process engine (embed
    `Database`). The embedded dict is the whole `jammi_db::ServerInfo` struct;
    the client projects the `jammi.v1.ServerInfo` message field-by-field. Both
    are pinned to the proto's field set here so a forgotten projection (a field
    present in the proto but dropped by one transport) fails — the very
    embedded-vs-remote drift the composition removes.

    Hermetic: the embedded side opens a real local engine; the remote side reads
    the generated proto descriptor, never dialing a server.
    """
    from jammi._generated.jammi.v1 import catalog_pb2

    proto_fields = {f.name for f in catalog_pb2.ServerInfo.DESCRIPTOR.fields}

    embedded = jammi.connect(f"file://{tmp_path}").get_server_info()
    assert set(embedded) == proto_fields, (
        f"embedded get_server_info keys {set(embedded)} != proto ServerInfo "
        f"fields {proto_fields}"
    )

    # The client builds its dict from exactly these keys; assert it maps every
    # proto field so the remote dict matches the embedded one key-for-key.
    client_keys = _client_server_info_keys()
    assert client_keys == proto_fields, (
        f"client get_server_info keys {client_keys} != proto ServerInfo "
        f"fields {proto_fields} — a field is unmapped"
    )


def _client_server_info_keys() -> set:
    """The key set `jammi.RemoteDatabase.get_server_info` returns, read
    off a stub response so the assertion sees what the method actually builds —
    no server contact."""

    class _StubServerInfo:
        version = "0.0.0"
        features = []
        storage_backends = []
        services = []
        broker = "in_memory"

    class _StubCatalog:
        def GetServerInfo(self, *_a, **_k):
            return _StubServerInfo()

    db = jammi.connect("grpc://127.0.0.1:8081")
    try:
        db._catalog = _StubCatalog()
        return set(db.get_server_info())
    finally:
        db.close()


# ---------------------------------------------------------------------------
# The unified surface: one Session protocol, one JammiError taxonomy, one
# capability contract — both transports map onto them.
#
# Raised-type conformance is TWO tiers, honest about hermeticity:
#   * Tier A (end-to-end): connect/target/validation errors are driven on BOTH
#     transports and asserted to raise the SAME taxonomy class — no server, no
#     compute.
#   * Tier B (converter-level): a failed training `wait()` needs the worker to
#     actually run and fail, which is not hermetic (it would pull a base model /
#     run compute), so the EMBEDDED arm cannot be driven end-to-end here. The
#     remote raise-site is driven directly; the embedded raise-site is pinned at
#     the shared class it is wired to. Neither is silently capped — the seam is
#     named explicitly in the test.
# ---------------------------------------------------------------------------


def test_both_backends_satisfy_the_session_protocol(tmp_path):
    """The embedded `Database` and the remote `RemoteDatabase` both structurally
    satisfy `jammi.Session` — the transport-agnostic surface `connect`
    returns is one protocol, not two parallel classes that might drift."""
    embed = jammi.connect(f"file://{tmp_path}")
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        assert isinstance(embed, jammi.Session)
        assert isinstance(remote, jammi.Session)
    finally:
        remote.close()


def test_connect_file_with_credentials_raises_no_embedded_engine_on_both_front_doors(
    tmp_path,
):
    """Tier A — a `file://` target opened WITH credentials raises
    `NoEmbeddedEngineError` on both front doors, even in this native-present lane:
    credentials are meaningless for an in-process engine (it has no channel to
    authenticate), so the target-vs-credential mismatch is a caller error rejected
    BEFORE an engine opens. Hermetic: no engine opened, no server dialed. The error
    is a `NotSupportedOnBackend`, itself a `JammiError`.

    (The extra-ABSENT `file://` error — no credentials — is a build-time condition
    pinned in the client-only lane, `clients/python/tests/test_target.py`.)"""
    assert issubclass(
        jammi.NoEmbeddedEngineError, jammi.NotSupportedOnBackend
    )
    assert issubclass(jammi.NotSupportedOnBackend, jammi.JammiError)

    bearer = jammi.BearerCredentials(token="tok")
    with pytest.raises(jammi.NoEmbeddedEngineError):
        jammi.connect(f"file://{tmp_path}", credentials=bearer)
    with pytest.raises(jammi.NoEmbeddedEngineError):
        jammi.connect(f"file://{tmp_path}", credentials=bearer)


def test_bad_format_add_source_raises_invalid_argument_on_both_backends(tmp_path):
    """Tier A — an unknown source `format` is an `InvalidArgument` on BOTH
    transports: the remote client rejects it client-side, the embedded engine at
    its `parse_file_format` seam. Hermetic: the format is rejected before any I/O.
    `InvalidArgument` refines `ValueError`, so an `except ValueError` still fires."""
    assert issubclass(jammi.InvalidArgument, jammi.JammiError)
    assert issubclass(jammi.InvalidArgument, ValueError)

    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        with pytest.raises(jammi.InvalidArgument):
            remote.add_source("s", url="/tmp/x.parquet", format="bogus")
    finally:
        remote.close()

    embed = jammi.connect(f"file://{tmp_path}")
    with pytest.raises(jammi.InvalidArgument):
        embed.add_source("s", url="/tmp/x.parquet", format="bogus")


def test_jsonl_and_ndjson_add_source_are_accepted_on_both_backends(tmp_path):
    """Cross-surface parity: `"jsonl"`/`"ndjson"` must be accepted on
    BOTH transports, not just one.

    The embedded arm calls the engine's `FileFormat::from_str` directly
    (`jammi.EmbeddedBackend.add_source` → `_native.add_source`), unmediated by
    any Python dict — so a REAL 2-row jsonl fixture registers end-to-end here,
    proving the whole embedded path, not just that no exception was raised.

    The remote arm instead resolves `format=` through the hand-maintained
    `jammi._assembly._FILE_FORMAT` dict BEFORE any channel I/O runs — a dict
    that drifted out of sync with the engine's vocabulary is exactly how the
    reported symptom reproduced (jsonl worked embedded, failed remote). `_call`
    is mocked so the remote assertion is on the client-side validation seam
    alone (hermetic, no server), and the mocked request is inspected to confirm
    both spellings land on the SAME wire value, `FILE_FORMAT_JSONL`."""
    from unittest.mock import patch

    from jammi._generated.jammi.v1 import catalog_pb2

    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        for token in ("jsonl", "ndjson"):
            with patch.object(remote, "_call", return_value=None) as mock_call:
                remote.add_source("s", url="/tmp/x.jsonl", format=token)
            sent_request = mock_call.call_args[0][1]
            assert (
                sent_request.connection.format
                == catalog_pb2.FileFormat.FILE_FORMAT_JSONL
            )
    finally:
        remote.close()

    fixture = tmp_path / "events.jsonl"
    fixture.write_text('{"id": 1}\n{"id": 2}\n')
    embed = jammi.connect(f"file://{tmp_path}")
    for i, token in enumerate(("jsonl", "ndjson")):
        source_id = f"events_{i}"
        embed.add_source(source_id, url=f"file://{fixture}", format=token)
        table = embed.sql(f"SELECT id FROM {source_id}.public.events")
        assert table.num_rows == 2, (
            f"format={token!r} must register the fixture's 2 rows, not just "
            "not raise"
        )


def test_failed_job_wait_raises_training_error_on_both_raise_sites():
    """Tier B (converter-level) — a failed job's `wait()` maps to ONE class,
    `jammi.errors.TrainingError`, on both transports.

    Converter-level on the embedded arm BY NECESSITY: a failed embedded job needs
    the worker to run and fail, which is not hermetic (it would pull a
    base model / run compute), so driving one here would violate test discipline.
    We therefore pin the two raise-sites at the converter:
      * the REMOTE raise-site is driven directly — a stubbed `JobStatus` of
        `failed` makes `RemoteJob.wait()` raise, asserted to `TrainingError`;
      * the EMBEDDED raise-site (`job.rs` failed `wait()` → `to_pyerr`'s
        `JammiError::FineTune` → `TrainingError` variant-dispatch) is wired to the
        SAME class object: the native engine imports `jammi.errors` and
        raises `TrainingError` from there, so the class both produce is one and the
        same. The seam is named here, not silently skipped."""
    from jammi._generated.jammi.v1 import job_pb2

    class _FailedJobStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(status="failed", error="boom")

    job = jammi.RemoteJob(
        _FailedJobStub(), (), job_id="job-1", kind="fine_tune", output_model_id="model-1"
    )
    with pytest.raises(jammi.TrainingError) as info:
        job.wait()
    assert type(info.value) is jammi.TrainingError
    assert "boom" in str(info.value)

    # Both raise-sites bind to THIS class: a `JammiError` refining `RuntimeError`,
    # so one `except JammiError` / `except RuntimeError` catches a failed job on
    # either transport. The embedded `Job.wait` raises it by importing
    # `jammi.errors.TrainingError` — the same object asserted above.
    assert issubclass(jammi.TrainingError, jammi.JammiError)
    assert issubclass(jammi.TrainingError, RuntimeError)
    from jammi import errors as client_errors

    assert client_errors.TrainingError is jammi.TrainingError


def test_cancelled_job_wait_raises_job_cancelled_not_training_error():
    """Tier B (converter-level) — a cancelled job's `wait()` raises
    `jammi.errors.JobCancelled` on both transports, distinct from a failure.

    Converter-level for the reason the failed-job test above gives. The REMOTE
    raise-site is driven through a stubbed `JobStatus` of `cancelled`; the
    EMBEDDED raise-site (`JobRecord::unsuccessful_error` →
    `JammiError::JobCancelled` → `to_pyerr`) raises the same class object,
    imported from `jammi.errors`."""
    from jammi._generated.jammi.v1 import job_pb2

    class _CancelledJobStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(status="cancelled", error="job 'job-1' cancelled")

    job = jammi.RemoteJob(
        _CancelledJobStub(), (), job_id="job-1", kind="fine_tune", output_model_id="model-1"
    )
    with pytest.raises(jammi.JobCancelled) as info:
        job.wait()
    assert type(info.value) is jammi.JobCancelled
    assert "job-1" in str(info.value)

    # The end a caller asked for is not a fault: `except TrainingError` must not
    # swallow it, while `except JammiError` / `except RuntimeError` still catch
    # any unsuccessful job.
    assert not issubclass(jammi.JobCancelled, jammi.TrainingError)
    assert issubclass(jammi.JobCancelled, jammi.JammiError)
    assert issubclass(jammi.JobCancelled, RuntimeError)
    from jammi import errors as client_errors

    assert client_errors.JobCancelled is jammi.JobCancelled


def test_empty_training_set_refusal_over_recompute_is_invalid_argument_on_both_transports():
    """Tier B (converter-level) — the refusal of an EMPTY training set, WHEN
    IT SURFACES OVER THE `Recompute` RPC (`grpc/pipeline.rs:139`,
    `jammi_ai::pipeline::recompute`'s `TrainingSet` replay arm), maps to ONE
    class, `jammi.errors.InvalidArgument`, on both transports.

    This is deliberately scoped to the `Recompute` surface, not "the"
    empty-training-set refusal generally: the SAME typed engine error
    (`JammiError::EmptyTrainingSet`) reaches the caller as a DIFFERENT class,
    `jammi.errors.TrainingError`, when it is instead raised on the fine-tune
    JOB path (`worker.rs::run_spec`'s `materialize_projection` call) — see
    `test_empty_training_set_refusal_on_the_job_path_is_training_error_on_both_transports`
    below. A test named "the" refusal would misstate that as one universal
    class; it is not — which surface carried the refusal determines the class.

    On the `Recompute` RPC path: the server sends the refusal as
    `INVALID_ARGUMENT` (`jammi_server::grpc::wire::map_engine_error`), so the
    remote client raises `InvalidArgument` — and the embedded converter must
    not classify the same engine error as a `BackendError`, or one
    `except InvalidArgument` would catch the caller's own degenerate input
    remotely and miss it in-process.

    Converter-level on the EMBEDDED arm BY NECESSITY, the same shape (and for the
    same reason) as the failed-job parity test above: driving a real refusal needs
    a fine-tune over a real source and base model, which is not hermetic. The two
    raise-sites are therefore pinned at their converters:
      * the REMOTE raise-site is driven directly — `_rpc_to_jammi` over a status
        carrying the engine's own `EmptyTrainingSet` message text;
      * the EMBEDDED raise-site (`error.rs::jammi_error_class`'s
        `JammiError::EmptyTrainingSet` arm → `client_error("InvalidArgument", …)`)
        is pinned in Rust by
        `empty_training_set_raises_the_class_the_remote_transport_raises`, which
        asserts that arm equals the class this crate maps `INVALID_ARGUMENT` to;
        the class name it looks up is asserted here to resolve to the very object
        the remote arm raised.
    """
    from jammi._database import _rpc_to_jammi

    message = (
        "training set over `SELECT text, label FROM reviews.public.rows` is "
        "empty: the projection yielded zero rows"
    )

    class _EmptyTrainingSetStatus(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.INVALID_ARGUMENT

        def details(self):
            return message

    mapped = _rpc_to_jammi(_EmptyTrainingSetStatus())
    assert type(mapped) is jammi.InvalidArgument
    assert mapped.code is grpc.StatusCode.INVALID_ARGUMENT
    assert message in str(mapped)

    # The embedded converter raises BY NAME out of `jammi.errors` — the same
    # module object, so the class it resolves is the one asserted above and a
    # caller's single `except` holds on either transport.
    from jammi import errors as client_errors

    assert client_errors.InvalidArgument is jammi.InvalidArgument
    assert issubclass(jammi.InvalidArgument, jammi.JammiError)
    assert issubclass(jammi.InvalidArgument, ValueError)


def test_empty_training_set_refusal_on_the_job_path_is_training_error_on_both_transports():
    """Tier B (converter-level) — the SAME typed engine error
    (`JammiError::EmptyTrainingSet`), WHEN IT SURFACES ON THE FINE-TUNE JOB
    PATH instead of the `Recompute` RPC, does NOT reach the caller as
    `InvalidArgument` on either transport: both `job.wait()` raise-sites
    collapse it (and every other job-failure cause) to
    `jammi.errors.TrainingError`. This is the class-equality check the
    `Recompute`-scoped test above deliberately does not claim to cover.

    On the JOB path, `run_spec` (`crates/jammi-ai/src/fine_tune/worker.rs`,
    the `materialize_projection` call `.map_err(WorkerJobError::from)?`)
    turns the typed `EmptyTrainingSet` into a plain string stored as the
    job's `error_message` — the variant does not survive past that point.
    Both raise-sites then rebuild the caller-facing error from that STORED
    STRING alone, with NO branch on its content:
      * the REMOTE raise-site, `RemoteJob.wait()`
        (`clients/python/jammi/_database.py`), raises
        `jammi.errors.TrainingError(resp.error)` for ANY `status == "failed"`
        — driven directly here with the refusal's message as the stubbed error;
      * the EMBEDDED raise-site, `wait_for_result`
        (`crates/jammi-python/src/job.rs:380-382`), raises
        `JammiError::FineTune(record.error)` for ANY `JobStatus::Failed` —
        the exact same unconditional wrap the generic
        `test_failed_job_wait_raises_training_error_on_both_raise_sites`
        above already pins with an unrelated message ("boom") — which
        `jammi_error_class` (`crates/jammi-python/src/error.rs:54`) maps to
        `TrainingError`. Neither raise-site inspects the failure's original
        cause, so the class the two transports agree on for THIS failure is
        the SAME class already pinned for every OTHER job failure — proven
        by the shared code path, not restated as a separate coincidence.

    Converter-level on the EMBEDDED arm for the same reason as the other
    Tier B tests: driving a real `EmptyTrainingSet` job failure end-to-end
    needs a real source and base model, which is not hermetic.
    """
    from jammi._generated.jammi.v1 import job_pb2

    message = (
        "training set over `SELECT text, label FROM reviews.public.rows` is "
        "empty: the projection yielded zero rows"
    )

    class _EmptyTrainingSetJobStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(status="failed", error=message)

    job = jammi.RemoteJob(
        _EmptyTrainingSetJobStub(),
        (),
        job_id="job-empty-training-set",
        kind="fine_tune",
        output_model_id="model-empty-training-set",
    )
    with pytest.raises(jammi.TrainingError) as info:
        job.wait()
    assert type(info.value) is jammi.TrainingError
    assert message in str(info.value)

    # This is NOT the class the Recompute-RPC-scoped test above pins for the
    # very same underlying refusal — the two surfaces genuinely disagree,
    # which is exactly why neither test names itself as covering "the"
    # empty-training-set refusal universally.
    assert type(info.value) is not jammi.InvalidArgument

    # The embedded raise-site binds to THIS class the same way the generic
    # failed-job test binds it: by importing `jammi.errors.TrainingError`,
    # the same module object asserted above, so a caller's single `except`
    # holds on either transport for this failure too.
    from jammi import errors as client_errors

    assert client_errors.TrainingError is jammi.TrainingError
    assert issubclass(jammi.TrainingError, jammi.JammiError)
    assert issubclass(jammi.TrainingError, RuntimeError)


def test_job_handle_protocol_is_satisfied_by_both_handles():
    """Both `jammi_native.Job` (native) and `jammi.RemoteJob`
    satisfy the `JobHandle` protocol — a caller treats the two
    interchangeably. The remote handle is checked by `isinstance` on a
    stub-constructed instance; the native handle, which needs a live engine to
    instantiate, is checked structurally at the class level for the same members
    (no engine is opened here)."""
    from jammi._generated.jammi.v1 import job_pb2

    class _CompletedStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(status="completed")

    remote_job = jammi.RemoteJob(
        _CompletedStub(), (), job_id="job-1", kind="fine_tune", output_model_id="model-1"
    )
    assert isinstance(remote_job, jammi.JobHandle)
    for member in (
        "job_id",
        "kind",
        "output_model_id",
        "status",
        "wait",
        "progress",
        "cancel",
        "metrics",
        "acceleration_report",
    ):
        assert hasattr(jammi_native.Job, member), member


# crates/jammi-python/tests/this_file -> repo root is three parents up. Shared
# with test_embedded_training.py: the embedded arm of the metrics-parity test
# below needs a real `Job` handle bound to a genuine catalog row, which
# needs a real base model + source.
_METRICS_TEST_ROOT = Path(__file__).resolve().parents[3]
_METRICS_TEST_TINY_BERT = _METRICS_TEST_ROOT / "cookbook" / "fixtures" / "tiny_bert"
_METRICS_TEST_TRAINING_PAIRS = (
    _METRICS_TEST_ROOT / "tests" / "fixtures" / "training_pairs.csv"
)


@pytest.mark.skipif(
    not _METRICS_TEST_TINY_BERT.is_dir() or not _METRICS_TEST_TRAINING_PAIRS.is_file(),
    reason="local tiny_bert / training_pairs fixtures not present",
)
def test_remote_and_embedded_job_metrics_agree_on_all_three_states(tmp_path):
    """`RemoteJob.metrics()` and the embedded `Job.metrics()` agree on the
    SAME three states the catalog's `jobs.result` payload's nested `metrics`
    field can carry —
    proven against a REAL embedded engine + catalog on one arm,
    not a stub of both:

      * absent (`jobs.result` NULL, or `metrics` unset within it) -> `{}` on
        both.
      * present + valid JSON -> the SAME parsed dict on both.
      * present + unparseable JSON -> `jammi.errors.BackendError` on both,
        NEVER silently folded into the absent `{}` case — a malformed-but-
        present blob is a catalog data-integrity fault, not "not recorded
        yet", and must fail loudly on both transports identically.

    Drives the compiled `jammi_native._NativeDatabase` primitives DIRECTLY —
    never the `jammi.EmbeddedBackend`/`jammi.connect` wrapper (`close()` /
    `job()` are native-only today; see the asymmetry note at the
    bottom of this docstring) — so this proves the built artifact's `close()`
    and `job()` (attach-by-id), not a mock of either.

    A raw `sqlite3` seed write must never coexist with a live engine
    connection on the same catalog file: two independent SQLite library
    instances (Python's stdlib `sqlite3` vs. Rust's vendored `sqlx-sqlite`)
    committing to the same `-wal`/`-shm` files at the same instant is not
    always safe, and can crash the interpreter (`Fatal Python error: Bus
    error`, SIGBUS inside SQLite's own WAL commit path) — the engine crashes
    rather than refusing typed on that unsupported topology, and that
    behavior lives in `jammi-db`, out of this crate's reach. The shape below
    structurally excludes this test's exposure to it, rather than merely
    making it rare.

    `Database.close()` (deterministic worker stop + session release) +
    `Database.job(job_id)` (attach-by-id, the embedded peer of
    `RemoteJob`'s always-attach-by-id shape) let one job serve all
    three states with NO window where a raw `sqlite3` write and a live
    engine connection coexist:

      1. ONE real job is submitted and driven to completion (`wait()`),
         then `close()`d — which blocks until the embedded worker this
         connection owned has actually stopped and this connection's own
         session handle is released (see `PyDatabase.close`'s doc); the
         Python references are `del`eted too, so no persistent handle this
         test still holds (there are none by this point) could keep the
         session alive past `close()`. No connection this test opened is
         attached to the catalog file from here on, until step 3 opens a
         new one.
      2. Each state injects its own seed value through a raw `sqlite3`
         connection that is itself fully closed (commit + close) before
         the next step runs — never overlapping a live engine connection.
      3. A FRESH `Database` attaches to the job by id and reads
         `metrics()` — the new pool's first (and only) read of that row,
         sidestepping the separate, pre-existing stale-pooled-connection
         hazard (a second read on an already-warm pool not always
         observing a write that landed through a separate handle) the same
         way the prior version of this test did — then is itself `close()`d
         before the next state's injection.

    The "absent" state's `jobs.result = NULL` is INJECTED, not the job's
    natural just-submitted state: `wait()` blocks until the job reaches
    `completed` (required before `close()` can run), and by then the worker
    has already stamped a real tagged `result` (with a real `metrics`
    string) onto the row. The state under test is the READ arm on the
    payload's three possible SHAPES (absent / valid JSON / malformed JSON)
    — reusing one job's completed row and overwriting its `result` column
    directly exercises the identical read path the naturally-absent case
    would (both are `None` at the `Option<String>` boundary `metrics()`
    matches on), for one real training run instead of three.

    Asymmetry, named rather than papered over: `RemoteDatabase` already has
    `close()` (used throughout this module); `close()` is therefore symmetric
    across transports. `job(job_id)` is NOT — `RemoteDatabase` has
    no equivalent convenience method, though a `RemoteJob` can always
    be constructed directly from a `job_id` (every one of its verbs re-fetches
    state over the wire per call, so it needs no server-side "attach" step).
    A `RemoteDatabase.job(...)` convenience method would be
    `clients/python` surface, not this crate's.
    """
    import json
    import sqlite3

    import jammi_native
    from jammi._assembly import build_fine_tune_request
    from jammi._generated.jammi.v1 import job_pb2
    from jammi.errors import BackendError

    valid_payload = (
        '{"final_loss": 0.42, "total_steps": 100, '
        '"train_loss_curve": [{"epoch": 0, "loss": 1.0}, {"epoch": 1, "loss": 0.5}], '
        '"val_loss_curve": [{"epoch": 0, "loss": 1.1}, {"epoch": 1, "loss": 0.6}]}'
    )
    valid_expected = {
        "final_loss": 0.42,
        "total_steps": 100,
        "train_loss_curve": [{"epoch": 0, "loss": 1.0}, {"epoch": 1, "loss": 0.5}],
        "val_loss_curve": [{"epoch": 0, "loss": 1.1}, {"epoch": 1, "loss": 0.6}],
    }
    malformed_payload = '{"final_loss": 0.42, "total_steps"'  # truncated, invalid JSON

    # --- Remote arm: three stubbed `JobStatusResponse`s. -------------------
    class _AbsentStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(status="running")

    class _ValidStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(
                status="completed",
                model=job_pb2.ModelResult(
                    model_id="model-2", artifact_path="a2", metrics_json=valid_payload
                ),
            )

    class _MalformedStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(
                status="completed",
                model=job_pb2.ModelResult(
                    model_id="model-3",
                    artifact_path="a3",
                    metrics_json=malformed_payload,
                ),
            )

    remote_absent = jammi.RemoteJob(
        _AbsentStub(), (), job_id="remote-1", kind="fine_tune", output_model_id="model-1"
    )
    remote_valid = jammi.RemoteJob(
        _ValidStub(), (), job_id="remote-2", kind="fine_tune", output_model_id="model-2"
    )
    remote_malformed = jammi.RemoteJob(
        _MalformedStub(), (), job_id="remote-3", kind="fine_tune", output_model_id="model-3"
    )

    assert remote_absent.metrics() == {}
    assert remote_valid.metrics() == valid_expected
    with pytest.raises(BackendError, match=r"metrics blob failed to parse as JSON"):
        remote_malformed.metrics()

    # --- Embedded arm: one real job, three close-before-inject cycles. ----
    catalog_db = tmp_path / "catalog.db"

    def _set_result(job_id: str, metrics_value) -> None:
        """Overwrite `jobs.result` with a tagged `model` result envelope
        carrying `metrics_value` as its nested `metrics` field, or SQL NULL
        when `metrics_value` is `None` (the natural pre-completion /
        never-recorded state)."""
        result_json = (
            None
            if metrics_value is None
            else json.dumps(
                {
                    "kind": "model",
                    "model_id": "irrelevant-for-this-test",
                    "artifact_path": "irrelevant-for-this-test",
                    "metrics": metrics_value,
                    "cache_outcome": "computed",
                }
            )
        )
        conn = sqlite3.connect(str(catalog_db))
        try:
            conn.execute(
                "UPDATE jobs SET result = ? WHERE job_id = ?",
                (result_json, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    # Submit + run ONE real fine-tune job directly against the compiled
    # `_NativeDatabase` — the same request assembly `EmbeddedBackend.fine_tune`
    # drives, minus the wrapper, so the artifact under test is the native
    # `close()` / `job()` primitives themselves.
    submit_db = jammi_native.open_local(artifact_dir=str(tmp_path))
    submit_db.add_source(
        "metrics_states", url=str(_METRICS_TEST_TRAINING_PAIRS), format="csv"
    )
    request = build_fine_tune_request(
        source="metrics_states",
        base_model=f"local:{_METRICS_TEST_TINY_BERT}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        epochs=2,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )
    submit_job = submit_db._start_training_proto(request.SerializeToString())
    submit_job.wait()
    assert submit_job.status() == "completed"
    job_id = submit_job.job_id

    # Deterministic teardown BEFORE any raw-sqlite3 injection (see the
    # docstring): no engine connection this test opened is attached to the
    # catalog file past this point.
    submit_db.close()
    del submit_job, submit_db

    def _seed_and_attach(seed):
        """Inject `seed` through a raw connection that is fully closed
        before this returns, then attach a FRESH `Database` to the shared
        job by id — that pool's first read. Returns the still-open
        `(db, job)` pair so the caller can assert before tearing this
        state down (`db.close()`) ahead of the next state's injection."""
        _set_result(job_id, seed)
        db = jammi_native.open_local(artifact_dir=str(tmp_path))
        return db, db.job(job_id)

    # State 1: absent (`jobs.result` NULL) — injected explicitly; see the
    # docstring on why the job's OWN natural post-submit state cannot be
    # reused here.
    absent_db, absent_job = _seed_and_attach(None)
    assert absent_job.metrics() == {}
    absent_db.close()
    del absent_job, absent_db

    # State 2: present + valid — the SAME payload string fed to the remote
    # stub above, so a genuine equal-dicts comparison, not just "parses to
    # something".
    valid_db, valid_job = _seed_and_attach(valid_payload)
    assert valid_job.metrics() == valid_expected
    valid_db.close()
    del valid_job, valid_db

    # State 3: present + malformed — the SAME truncated payload string.
    malformed_db, malformed_job = _seed_and_attach(malformed_payload)
    with pytest.raises(
        BackendError, match=r"metrics blob failed to parse as JSON"
    ) as embedded_info:
        malformed_job.metrics()
    assert job_id in str(embedded_info.value)
    malformed_db.close()
    del malformed_job, malformed_db


@pytest.mark.skipif(
    not _METRICS_TEST_TINY_BERT.is_dir() or not _METRICS_TEST_TRAINING_PAIRS.is_file(),
    reason="local tiny_bert / training_pairs fixtures not present",
)
def test_remote_and_embedded_job_acceleration_report_agree_on_all_three_states(
    tmp_path,
):
    """`RemoteJob.acceleration_report()` and the embedded
    `Job.acceleration_report()` agree on the SAME three states the
    catalog's `jobs.acceleration_report` column can carry — VALUE parity,
    not merely
    `test_remote_job_matches_the_local_handle_shape`'s method-
    existence check above:

      * `NULL` (column absent / wire field unset) -> `None` on BOTH — an
        honest absence, never coerced to `{}` or any acceleration-state claim
        on either transport. This is the tri-state's load-bearing case: it is
        NOT `metrics()`'s "absent means `{}`" shape (that column's `NULL` and
        "not recorded yet" are one state); `acceleration_report`'s `NULL` and
        the submission-time `"pending"` marker are deliberately two different,
        distinguishable states, and a transport that collapsed `NULL` to `{}`
        would erase that distinction.
      * present + `{"state": "pending"}` -> the SAME parsed dict on both.
      * present + `{"state": "determined", ...}` -> the SAME parsed dict on
        both, using the byte-identical JSON text a REAL run's acceleration probe
        produced on the embedded side (never a hand-built stand-in) fed
        verbatim to the remote stub — so this proves the two `JSON-decode`
        implementations agree on the real producer's actual output shape, not
        just on a convenient literal.

    Same close-before-inject / one-real-job discipline as
    `test_remote_and_embedded_job_metrics_agree_on_all_three_states` —
    see that test's docstring for why.
    """
    import json
    import sqlite3

    import jammi_native
    from jammi._assembly import build_fine_tune_request
    from jammi._generated.jammi.v1 import job_pb2

    pending_payload = '{"state":"pending"}'
    pending_expected = {"state": "pending"}

    # --- Embedded arm: one real job, three close-before-inject cycles. ----
    catalog_db = tmp_path / "catalog.db"

    def _set_acceleration_report(job_id: str, value) -> None:
        conn = sqlite3.connect(str(catalog_db))
        try:
            conn.execute(
                "UPDATE jobs SET acceleration_report = ? WHERE job_id = ?",
                (value, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    submit_db = jammi_native.open_local(artifact_dir=str(tmp_path))
    submit_db.add_source(
        "acceleration_states", url=str(_METRICS_TEST_TRAINING_PAIRS), format="csv"
    )
    request = build_fine_tune_request(
        source="acceleration_states",
        base_model=f"local:{_METRICS_TEST_TINY_BERT}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        epochs=2,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )
    submit_job = submit_db._start_training_proto(request.SerializeToString())
    submit_job.wait()
    assert submit_job.status() == "completed"
    job_id = submit_job.job_id

    # State "determined": the job's OWN natural post-completion payload — a
    # REAL acceleration-probe result, not a hand-built stand-in.
    embedded_determined = submit_job.acceleration_report()
    assert isinstance(embedded_determined, dict)
    assert embedded_determined["state"] == "determined"
    # The byte-identical raw JSON text the catalog itself holds, fed to the
    # remote stub below so the parity assertion covers the real producer's
    # actual output shape, not a convenient literal.
    determined_payload = json.dumps(embedded_determined)

    submit_db.close()
    del submit_job, submit_db

    def _seed_and_attach(seed):
        _set_acceleration_report(job_id, seed)
        db = jammi_native.open_local(artifact_dir=str(tmp_path))
        return db, db.job(job_id)

    absent_db, absent_job = _seed_and_attach(None)
    embedded_absent = absent_job.acceleration_report()
    absent_db.close()
    del absent_job, absent_db

    pending_db, pending_job = _seed_and_attach(pending_payload)
    embedded_pending = pending_job.acceleration_report()
    pending_db.close()
    del pending_job, pending_db

    # --- Remote arm: three stubbed `JobStatusResponse`s. -------------------
    class _AbsentStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(status="running")

    class _PendingStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(
                status="running", acceleration_report_json=pending_payload
            )

    class _DeterminedStub:
        def JobStatus(self, *_args, **_kwargs):
            return job_pb2.JobStatusResponse(
                status="completed", acceleration_report_json=determined_payload
            )

    remote_absent = jammi.RemoteJob(
        _AbsentStub(), (), job_id="remote-accel-1", kind="fine_tune", output_model_id="model-1"
    )
    remote_pending = jammi.RemoteJob(
        _PendingStub(), (), job_id="remote-accel-2", kind="fine_tune", output_model_id="model-2"
    )
    remote_determined = jammi.RemoteJob(
        _DeterminedStub(), (), job_id="remote-accel-3", kind="fine_tune", output_model_id="model-3"
    )

    # --- Value parity, both directions of the tri-state. -------------------
    assert embedded_absent is None
    assert remote_absent.acceleration_report() is None

    assert embedded_pending == pending_expected
    assert remote_pending.acceleration_report() == pending_expected
    assert embedded_pending == remote_pending.acceleration_report()

    assert remote_determined.acceleration_report() == embedded_determined


class _StubTenant:
    id = ""


class _StubTenantResp:
    tenant = _StubTenant()


class _StubTenantCatalog:
    """A catalog stub for the remote tenant trio so `tenant_scope` enters without
    a server: `GetTenant` reports the unscoped state, `SetTenant`/`ClearTenant`
    are no-ops."""

    def GetTenant(self, *_args, **_kwargs):
        return _StubTenantResp()

    def SetTenant(self, *_args, **_kwargs):
        return None

    def ClearTenant(self, *_args, **_kwargs):
        return None


def test_tenant_scope_yields_a_session_surface_on_both_backends(tmp_path):
    """`with s.tenant_scope(t) as x` yields a Session-surface object — NOT an inert
    scope token — on BOTH backends, so `x.search(...)` / `x.sql(...)` work inside
    the block. This is the block-fix: the embedded wrapper yields itself (the
    tenant-scoped `Database`), matching the remote generator, which yields the
    `RemoteDatabase`. Embedded is driven on a real local engine; remote enters
    over a stubbed catalog so no server is dialed."""
    session_verbs = ("search", "sql", "add_source", "supports", "list_sources")
    valid_tenant = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"

    embed = jammi.connect(f"file://{tmp_path}")
    with embed.tenant_scope(valid_tenant) as scoped:
        assert scoped is embed
        assert isinstance(scoped, jammi.Session)
        for verb in session_verbs:
            assert callable(getattr(scoped, verb)), verb

    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        remote._catalog = _StubTenantCatalog()
        with remote.tenant_scope("tenant-x") as scoped:
            assert scoped is remote
            assert isinstance(scoped, jammi.Session)
            for verb in session_verbs:
                assert callable(getattr(scoped, verb)), verb
    finally:
        remote.close()


def test_supports_and_not_supported_on_backend_contract(tmp_path):
    """`supports(capability)` answers the one-sided divergence on both backends,
    and invoking a capability the backend lacks raises the typed
    `NotSupportedOnBackend` — never a bare `AttributeError`. The embedded engine
    carries the in-process primitives (audit / ephemeral / preload); the remote
    carries the per-connection scoping key (session_id); each raises for the
    other's.

    `close` is NOT among them, on either side: it is a SHARED verb both
    backends implement (a channel teardown remote, the catalog-file release
    embedded), so it is asserted as an ordinary member of the Session surface
    here rather than as a capability."""
    from jammi import Capability
    from jammi.errors import NotSupportedOnBackend

    embed = jammi.connect(f"file://{tmp_path}")
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        assert embed.supports(Capability.AUDIT) is True
        assert embed.supports(Capability.EPHEMERAL_SESSION) is True
        assert embed.supports(Capability.PRELOAD_MODEL) is True
        assert embed.supports(Capability.SESSION_ID) is False

        assert remote.supports(Capability.SESSION_ID) is True
        assert remote.supports(Capability.AUDIT) is False
        assert remote.supports(Capability.EPHEMERAL_SESSION) is False
        assert remote.supports(Capability.PRELOAD_MODEL) is False

        # `close` is a shared verb, not a capability: both backends carry it,
        # and neither raises for it.
        assert callable(embed.close)
        assert callable(remote.close)
        assert not hasattr(Capability, "CLOSE")

        # A one-sided op on the wrong backend raises the typed error.
        with pytest.raises(NotSupportedOnBackend):
            embed.session_id
        with pytest.raises(NotSupportedOnBackend):
            remote.audit
        with pytest.raises(NotSupportedOnBackend):
            remote.ephemeral_session(timeout_seconds=60)
        with pytest.raises(NotSupportedOnBackend):
            remote.preload_model("some-model")
    finally:
        embed.close()
        remote.close()


def test_capability_enum_is_the_closed_four():
    """The capability set is CLOSED — exactly the four one-sided features that
    diverge between the transports, no more. Pinned so a fifth is a deliberate
    decision, not a silent addition.

    `close` is not one of them: both backends implement it (the embedded arm
    releases the catalog file), and the enum's charter — "exactly the features
    that genuinely diverge between the two transports" — excludes it: a
    `Capability` every backend supports is a predicate that never
    discriminates, and listing it would say the embedded engine lacks a
    primitive it has. Pinned as an absence, not merely by the set below, in
    `test_supports_and_not_supported_on_backend_contract`."""
    from jammi import Capability

    assert {c.value for c in Capability} == {
        "audit",
        "ephemeral_session",
        "preload_model",
        "session_id",
    }


def test_close_is_idempotent_and_use_after_close_raises_the_same_error_on_both(
    tmp_path,
):
    """Value-parity on the closed-session contract: BOTH backends make `close()`
    idempotent and BOTH raise the same typed `BackendError` from every verb
    afterwards.

    This is the parity that `Capability.CLOSE`'s removal asserts. "Both have a
    `close()`" would be a shallow claim if the two disagreed on what a closed
    session then does — and they did: the embedded arm raised the typed
    `BackendError` from its FFI-boundary guard while the remote arm let grpcio's
    bare `ValueError("Cannot invoke RPC on closed channel!")` escape the
    `JammiError` taxonomy entirely. A caller writing one program against
    :class:`jammi.Session` must be able to catch one exception type.

    The CLASS is pinned identical (that is the contract a caller writes
    `except`ing); the message is pinned only to say "closed", because each arm
    truthfully names the resource it released — the embedded arm the catalog,
    the remote arm the channel — and forcing one sentence would make one of them
    lie.

    The verbs cover both remote transports: the typed gRPC lane (`list_sources`,
    `tenant`, `get_server_info`) and the separate Flight SQL lane (`sql`), so a
    guard installed on only one of them fails here.

    Hermetic on the remote side — the guard fires before any wire hop, so the
    unreachable endpoint is never dialed."""
    from jammi.errors import BackendError, JammiError

    embed = jammi.connect(f"file://{tmp_path}")
    remote = jammi.connect("grpc://127.0.0.1:8081")

    raised: dict[tuple[str, str], type] = {}
    for name, session in (("embedded", embed), ("remote", remote)):
        session.close()
        # Idempotent: a second and third close are no-ops, not errors.
        session.close()
        session.close()

        for verb, call in (
            ("list_sources", lambda s=session: s.list_sources()),
            ("tenant", lambda s=session: s.tenant()),
            ("get_server_info", lambda s=session: s.get_server_info()),
            ("sql", lambda s=session: s.sql("SELECT 1")),
        ):
            with pytest.raises(BackendError) as excinfo:
                call()
            assert issubclass(excinfo.type, JammiError), (name, verb)
            assert "closed" in str(excinfo.value), (name, verb, str(excinfo.value))
            raised[(name, verb)] = excinfo.type

    # One class across both transports and every verb — the whole point.
    assert set(raised.values()) == {BackendError}


class _FakeRpcError(grpc.RpcError):
    """A `grpc.RpcError` carrying a chosen status code — lets a test drive the
    remote error-mapping choke point without a live server."""

    def __init__(self, code, details):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _RaisingStub:
    """A gRPC service stub whose every method raises a fixed error — swapped onto
    a `RemoteDatabase`'s `_catalog` so a verb's wire hop raises hermetically."""

    def __init__(self, err):
        self._err = err

    def __getattr__(self, _name):
        def _raise(*_args, **_kwargs):
            raise self._err

        return _raise


def test_remote_rpc_status_errors_map_onto_the_taxonomy():
    """A SERVER-detected fault the remote stub raises as a ``grpc.RpcError`` is
    remapped onto the taxonomy by status code — so the whole remote surface, not
    only client-constructed errors, descends from ``JammiError``:

    * ``INVALID_ARGUMENT`` → :class:`InvalidArgument` — the SAME class the
      embedded engine raises for a server-detected bad argument (``status_to_pyerr``),
      the two-sided parity the client-side format pre-rejection could NOT prove;
    * ``RESOURCE_EXHAUSTED`` (the 64 MiB receive-cap edge) / ``UNAVAILABLE`` /
      ``DEADLINE_EXCEEDED`` / ``INTERNAL`` → :class:`BackendError`;
    * ``UNIMPLEMENTED`` → :class:`NotSupportedOnBackend`.

    Hermetic: the stub raises, so no server is dialed; the verb still reaches the
    wire arm (``_call``), unlike a client-side pre-rejection."""
    from jammi.errors import (
        BackendError,
        InvalidArgument,
        NotSupportedOnBackend,
    )

    cases = [
        (grpc.StatusCode.INVALID_ARGUMENT, InvalidArgument),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, BackendError),
        (grpc.StatusCode.UNAVAILABLE, BackendError),
        (grpc.StatusCode.DEADLINE_EXCEEDED, BackendError),
        (grpc.StatusCode.INTERNAL, BackendError),
        (grpc.StatusCode.UNIMPLEMENTED, NotSupportedOnBackend),
    ]
    for code, expected in cases:
        remote = jammi.connect("grpc://127.0.0.1:8081")
        try:
            remote._catalog = _RaisingStub(_FakeRpcError(code, f"server said {code}"))
            with pytest.raises(expected) as info:
                remote.list_sources()  # a unary verb → routes through `_call`
            assert type(info.value) is expected, f"{code}: {type(info.value)}"
            assert isinstance(info.value, jammi.JammiError)
            assert getattr(info.value, "code", None) == code
        finally:
            remote.close()


def test_remote_not_found_semantics_survive_the_taxonomy_mapping():
    """The mapping must not regress the ``NOT_FOUND`` special-cases: ``describe_*``
    returns ``None`` and an ``if_exists`` drop is a no-op even though the raw
    ``grpc.RpcError`` is now remapped to a ``JammiError`` first — the call-sites
    branch on the status code carried on the mapped exception's ``.code``."""
    not_found = _FakeRpcError(grpc.StatusCode.NOT_FOUND, "absent")
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        remote._catalog = _RaisingStub(not_found)
        assert remote.describe_source("nope") is None
        assert remote.describe_model("nope") is None
        remote.drop_mutable_table("nope", if_exists=True)  # no raise
        # Without if_exists, the NOT_FOUND surfaces as a typed JammiError.
        with pytest.raises(jammi.JammiError):
            remote.drop_mutable_table("nope", if_exists=False)
    finally:
        remote.close()


# ---------------------------------------------------------------------------
# The base client discovers the native engine as an in-process backend.
#
# `jammi` is the BASE: `jammi.connect("file://…")` resolves
# the local target to an `EmbeddedBackend` (direct FFI) when `jammi_native` is
# importable — discovering the native engine directly. These tests pin that base
# front door and the lazy-native discipline (`import jammi` stays native-free;
# the engine loads only when a `file://` target is opened), the positive
# complement of the CI negative import-direction guard.
# ---------------------------------------------------------------------------


def test_client_connect_file_returns_embedded_session_via_the_base_front_door(tmp_path):
    """`jammi.connect("file://…")` — the BASE front door — returns an
    `EmbeddedBackend` that satisfies the `Session` protocol and runs a verb
    in-process (direct FFI): the base client discovers `jammi_native` as an
    in-process backend on its own.

    Hermetic: opens a local engine (`file://`), contacts no server."""
    db = jammi.connect(f"file://{tmp_path}")
    assert isinstance(db, jammi.EmbeddedBackend)
    assert type(db).__module__ == "jammi._embedded"
    assert isinstance(db, jammi.Session)
    assert type(db) is jammi.EmbeddedBackend
    # A verb runs in-process against the compiled engine (empty catalog → []).
    assert db.list_models() == []


def test_import_jammi_is_native_free_then_lazily_loads_the_engine(tmp_path):
    """Positive import-direction guard (the complement of the CI negative guard):
    in a FRESH interpreter, `import jammi` loads NO `jammi_native` (the base
    package is native-free — the engine is imported lazily), and opening a
    `file://` target THEN loads `jammi_native` (the base discovers the engine
    directly, on demand).

    Run in a subprocess because THIS test process already imported `jammi` /
    `jammi_native` at module load, so the clean import direction can only be
    observed in an interpreter that has not."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys
        import jammi
        assert "jammi_native" not in sys.modules, (
            "import jammi eagerly pulled jammi_native: "
            + str([m for m in sys.modules if m == "jammi_native"])
        )
        db = jammi.connect("file://" + {str(tmp_path)!r})
        assert "jammi_native" in sys.modules, "file:// did not load the engine"
        assert type(db).__name__ == "EmbeddedBackend"
        print("POSITIVE_IMPORT_GUARD_OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "POSITIVE_IMPORT_GUARD_OK" in proc.stdout


# ---------------------------------------------------------------------------
# The projection shape pin, ANCHORED TO THE PROTO SCHEMA.
#
# The 20 `_*_to_dict` projections in `jammi._database` decode a wire proto
# into the client-facing dict. The embedded engine returns the SAME dict shapes
# by serialising the mirror serde structs (`serializable_to_pydict`). The proto
# message is the single source of truth for that shape, so this pin asserts each
# REMOTE projection's key-structure reconciles with its proto message's LIVE
# descriptor (identity, plus a small DECLARED set of renames / oneof-tags / dropped
# fields) — NOT against a hand-authored golden. A proto field added or removed
# then forces the projection (or the declared delta) to change, so the two
# projection paths, both anchored to the proto, cannot drift from each other.
#
# This is the honest FLOOR: true per-VALUE equality on real data is
# covered by the Rust it-suite (`grpc_remote_session.rs` / `grpc_remote_compute.rs`,
# the embedded/remote parity anchor) and — for the three eval reports — by the
# Rust-serde-GENERATED golden the client's `tests/test_eval_projection.py` locks,
# not hermetically constructible in Python (the native links no tonic/proto, so
# there is no in-process Python value-parity oracle to stand up here).
# ---------------------------------------------------------------------------


def _ipc_schema_bytes():
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, pa.schema([("id", pa.int64()), ("v", pa.float32())])):
        pass
    return sink.getvalue().to_pybytes()


def test_struct_projection_shapes_are_proto_anchored():
    """Each non-oneof projection's output key-set == its proto message's field
    set, minus a DECLARED drop-set, with DECLARED renames — read off the live
    descriptor, so a proto change breaks this until the projection tracks it."""
    from jammi import _database as D
    from jammi._generated.jammi.v1 import catalog_pb2, embedding_pb2, eval_pb2

    def _model():
        return catalog_pb2.Model(model_id="m", backend="b", task=1, status="ready")

    def _mutable():
        return catalog_pb2.MutableTableDefinition(id="t", schema=_ipc_schema_bytes())

    def _inference_report():
        # `aggregate` is a oneof — a variant must be set or the projection raises.
        r = eval_pb2.InferenceEvalReport()
        r.aggregate.classification.SetInParent()
        return r

    # (projection, proto message, {output_key: proto_field} renames, {dropped fields}, fixture)
    specs = [
        (D._result_table_to_dict, embedding_pb2.ResultTable, {}, {"task", "cache_outcome"}, embedding_pb2.ResultTable),
        (D._source_descriptor_to_dict, catalog_pb2.SourceDescriptor, {"source_type": "kind"}, set(), catalog_pb2.SourceDescriptor),
        (D._model_to_dict, catalog_pb2.Model, {}, set(), _model),
        (D._mutable_table_definition_to_dict, catalog_pb2.MutableTableDefinition, {}, {"user_metadata"}, _mutable),
        (D._aggregate_metrics_to_dict, eval_pb2.AggregateMetrics, {}, set(), eval_pb2.AggregateMetrics),
        (D._per_query_record_to_dict, eval_pb2.PerQueryRecord, {}, set(), eval_pb2.PerQueryRecord),
        (D._embedding_report_to_dict, eval_pb2.EmbeddingEvalReport, {}, set(), eval_pb2.EmbeddingEvalReport),
        (D._entity_to_dict, eval_pb2.Entity, {}, set(), eval_pb2.Entity),
        (D._inference_report_to_dict, eval_pb2.InferenceEvalReport, {}, set(), _inference_report),
        (D._metric_significance_to_dict, eval_pb2.MetricSignificance, {}, set(), eval_pb2.MetricSignificance),
        (D._aggregate_delta_to_dict, eval_pb2.AggregateDelta, {}, set(), eval_pb2.AggregateDelta),
        (D._compare_report_to_dict, eval_pb2.CompareEvalReport, {}, set(), eval_pb2.CompareEvalReport),
        (D._calibration_report_to_dict, eval_pb2.CalibrationEvalReport, {}, set(), eval_pb2.CalibrationEvalReport),
        (D._channel_spec_to_dict, catalog_pb2.Channel, {}, set(), catalog_pb2.Channel),
    ]
    for project, msg, renames, dropped, fixture in specs:
        got = set(project(fixture()))
        proto_fields = {f.name for f in msg.DESCRIPTOR.fields}
        expected = set(proto_fields) - dropped
        for out_key, proto_field in renames.items():
            assert proto_field in proto_fields, f"{msg.DESCRIPTOR.name}: rename source {proto_field!r} not a proto field"
            expected.discard(proto_field)
            expected.add(out_key)
        assert got == expected, (
            f"{project.__name__} vs proto {msg.DESCRIPTOR.name}: "
            f"got {sorted(got)} != proto-anchored {sorted(expected)} "
            f"(proto fields {sorted(proto_fields)}, dropped {sorted(dropped)}, renames {renames})"
        )

    # The list projection: each edge dict reconciles with the `DerivesFromEdge`
    # message's fields, anchored the same way.
    resp = catalog_pb2.DerivesFromResponse()
    edge = resp.edges.add()
    edge.kind = catalog_pb2.ANCHOR_KIND_RESULT_DIGEST
    got = set(D._derives_from_edges_to_list(resp)[0])
    expected = {f.name for f in catalog_pb2.DerivesFromEdge.DESCRIPTOR.fields}
    assert got == expected, f"_derives_from_edges_to_list edge: {sorted(got)} != {sorted(expected)}"


def test_oneof_projection_shapes_are_proto_anchored():
    """Each oneof projection flattens the SELECTED variant into `{tag} + variant
    fields`. For every variant of every oneof message, the projection's output
    key-set == the tag key plus the variant sub-message's LIVE descriptor fields
    (read off the oneof, so a new variant field breaks this until projected)."""
    from jammi import _database as D
    from jammi._generated.jammi.v1 import catalog_pb2, eval_pb2

    # (projection, proto message, oneof name, tag key, [variant field names])
    specs = [
        (D._inference_aggregate_to_dict, eval_pb2.InferenceAggregate, "aggregate", "task", ["classification", "ner"]),
        (D._per_record_prediction_to_dict, eval_pb2.PerRecordPrediction, "prediction", "task", ["classification", "ner"]),
        (D._verify_verdict_to_dict, catalog_pb2.VerifyMaterializationResponse, "verdict", "verdict", ["match", "mismatch", "match_with_unpinned_inputs", "missing_manifest"]),
        (D._stale_reason_to_dict, catalog_pb2.StaleReason, "reason", "reason", ["definition_changed", "input_advanced", "input_vanished"]),
        (D._staleness_verdict_to_dict, catalog_pb2.StalenessResponse, "staleness", "staleness", ["fresh", "stale", "undecidable", "missing_manifest"]),
    ]
    for project, msg, oneof_name, tag_key, variant_names in specs:
        oneof = {o.name: o for o in msg.DESCRIPTOR.oneofs}[oneof_name]
        variant_fields = {f.name: f for f in oneof.fields}
        assert set(variant_fields) == set(variant_names), (
            f"{msg.DESCRIPTOR.name}.{oneof_name}: proto variants {sorted(variant_fields)} "
            f"!= pinned {sorted(variant_names)} — a variant was added or removed"
        )
        for variant in variant_names:
            m = msg()
            getattr(m, variant).SetInParent()
            got = set(project(m))
            sub = variant_fields[variant].message_type
            sub_fields = {f.name for f in sub.fields} if sub is not None else set()
            expected = {tag_key} | sub_fields
            assert got == expected, (
                f"{project.__name__}.{variant} vs proto {sub.name if sub else variant}: "
                f"got {sorted(got)} != proto-anchored {sorted(expected)}"
            )
