"""Conformance: the embed wheel's remote arm IS `jammi-ai`, by construction.

The whole point of the composition (M2 §2) is that the remote surface is
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
    """U2: with the `[embedded]` extra installed (this lane carries `jammi_native`),
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
# wire: training is offloaded to the remote GPU server (`TrainingService`) and the
# predict verb runs the trained predictor remotely (`InferenceService.Predict`).
# The embedded `Database` submits/serves in the compiled engine; the client's
# `RemoteDatabase` submits/serves over gRPC. The call surface must agree so a
# caller swaps transports without changing the call — pinned here against the
# embed `jammi.EmbeddedBackend`.
_TRAINING_VERBS = {
    "fine_tune",
    "fine_tune_graph",
    "train_context_predictor",
    "predict_with_context_predictor",
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


def test_remote_training_job_matches_the_local_handle_shape():
    """The client's `RemoteTrainingJob` carries the SAME handle surface as the
    embedded engine's `TrainingJob`: the `job_id` / `model_id` properties, the
    `status()` / `wait()` methods, and `metrics()` (issue #441). A remote
    `wait()` polls `TrainingStatus` and raises on a failed job with the wire
    error, mirroring the local handle, so a caller treats the two
    interchangeably."""
    local = jammi_native.TrainingJob
    remote = jammi.RemoteTrainingJob
    for member in ("job_id", "model_id", "status", "wait", "metrics"):
        assert hasattr(remote, member), member
        assert hasattr(local, member), member
    # `job_id` / `model_id` are read-only attributes on both handles (a property
    # on the pure client, a getter on the native handle); `status` / `wait` are
    # callable methods on both.
    assert callable(remote.status) and callable(remote.wait)
    assert callable(local.status) and callable(local.wait)


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
        # The embedded engine releases its resources on drop; only the remote
        # client holds a gRPC channel that needs an explicit close.
        remote.close()


def test_embed_remote_and_client_share_identical_signatures():
    """Because the embed wheel's remote arm IS `jammi.RemoteDatabase`,
    every verb's signature is identical by construction. Asserting it here pins
    the invariant so any future hand-rolled remote class in the embed wheel
    (which would re-introduce the very drift M2 removes) fails this test."""
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
    embedded-vs-remote drift M2 §2 removes.

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
    """Cross-surface parity for #346: `"jsonl"`/`"ndjson"` must be accepted on
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


def test_failed_training_wait_raises_training_error_on_both_raise_sites():
    """Tier B (converter-level) — a failed training `wait()` maps to ONE class,
    `jammi.errors.TrainingError`, on both transports.

    Converter-level on the embedded arm BY NECESSITY: a failed embedded job needs
    the training worker to run and fail, which is not hermetic (it would pull a
    base model / run compute), so driving one here would violate test discipline.
    We therefore pin the two raise-sites at the converter:
      * the REMOTE raise-site is driven directly — a stubbed `TrainingStatus` of
        `failed` makes `RemoteTrainingJob.wait()` raise, asserted to `TrainingError`;
      * the EMBEDDED raise-site (`job.rs` failed `wait()` → `to_pyerr`'s
        `JammiError::FineTune` → `TrainingError` variant-dispatch) is wired to the
        SAME class object: the native engine imports `jammi.errors` and
        raises `TrainingError` from there, so the class both produce is one and the
        same. The seam is named here, not silently skipped."""
    from jammi._generated.jammi.v1 import training_pb2

    class _FailedTrainingStub:
        def TrainingStatus(self, *_args, **_kwargs):
            return training_pb2.TrainingStatusResponse(status="failed", error="boom")

    job = jammi.RemoteTrainingJob(
        _FailedTrainingStub(), (), job_id="job-1", model_id="model-1"
    )
    with pytest.raises(jammi.TrainingError) as info:
        job.wait()
    assert type(info.value) is jammi.TrainingError
    assert "boom" in str(info.value)

    # Both raise-sites bind to THIS class: a `JammiError` refining `RuntimeError`,
    # so one `except JammiError` / `except RuntimeError` catches a failed job on
    # either transport. The embedded `TrainingJob.wait` raises it by importing
    # `jammi.errors.TrainingError` — the same object asserted above.
    assert issubclass(jammi.TrainingError, jammi.JammiError)
    assert issubclass(jammi.TrainingError, RuntimeError)
    from jammi import errors as client_errors

    assert client_errors.TrainingError is jammi.TrainingError


def test_training_job_handle_protocol_is_satisfied_by_both_handles():
    """Both `jammi_native.TrainingJob` (native) and `jammi.RemoteTrainingJob`
    satisfy the `TrainingJobHandle` protocol — a caller treats the two
    interchangeably. The remote handle is checked by `isinstance` on a
    stub-constructed instance; the native handle, which needs a live engine to
    instantiate, is checked structurally at the class level for the same members
    (no engine is opened here)."""
    from jammi._generated.jammi.v1 import training_pb2

    class _CompletedStub:
        def TrainingStatus(self, *_args, **_kwargs):
            return training_pb2.TrainingStatusResponse(status="completed")

    remote_job = jammi.RemoteTrainingJob(
        _CompletedStub(), (), job_id="job-1", model_id="model-1"
    )
    assert isinstance(remote_job, jammi.TrainingJobHandle)
    for member in ("job_id", "model_id", "status", "wait"):
        assert hasattr(jammi_native.TrainingJob, member), member


# crates/jammi-python/tests/this_file -> repo root is three parents up. Shared
# with test_embedded_training.py: the embedded arm of the metrics-parity test
# below needs a real `TrainingJob` handle bound to a genuine catalog row, which
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
def test_remote_and_embedded_training_job_metrics_agree_on_all_three_states(tmp_path):
    """`RemoteTrainingJob.metrics()` and the embedded `TrainingJob.metrics()`
    agree on the SAME three states the catalog's `training_jobs.metrics`
    column can carry (issue #441 / adversarial-audit r3 BLOCK 4) — proven
    against a REAL embedded engine + catalog on one arm, not a stub of both:

      * absent (column NULL) -> `{}` on both.
      * present + valid JSON -> the SAME parsed dict on both.
      * present + unparseable JSON -> `jammi.errors.BackendError` on both,
        NEVER silently folded into the absent `{}` case — a malformed-but-
        present blob is a catalog data-integrity fault, not "not recorded
        yet", and must fail loudly on both transports identically.

    Drives the compiled `jammi_native._NativeDatabase` primitives DIRECTLY —
    never the `jammi.EmbeddedBackend`/`jammi.connect` wrapper (`close()` /
    `training_job()` are native-only today; see the asymmetry note at the
    bottom of this docstring) — so this proves the built artifact's `close()`
    and `training_job()` (attach-by-id), not a mock of either.

    History this shape fixes (`esc-073`): `jammi_python::PyTrainingJob` used
    to expose no way to obtain a handle for a job a session did not itself
    submit (unlike `RemoteTrainingJob`, built here straight from a stub gRPC
    channel), so an earlier version of this test opened a FRESH
    `Database`/live worker PER STATE and seeded its value through a raw
    `sqlite3` connection while that state's OWN worker was still live on the
    same catalog file. That reproduced a hard interpreter crash (`Fatal
    Python error: Bus error`, SIGBUS inside SQLite's own WAL commit path)
    roughly every other full-suite run: the raw `sqlite3` seed write and the
    live worker's pooled `sqlx` connection committed to the same
    `-wal`/`-shm` files at the same instant — two independent SQLite library
    instances (Python's stdlib `sqlite3` vs. Rust's vendored `sqlx-sqlite`)
    are not always safe to coexist on one WAL file. `esc-073` tracks the
    underlying engine-side behavior (a crash instead of a typed refusal for
    an unsupported topology) as a `jammi-db`-scope defect, out of this
    crate's reach; THIS test's exposure to it is what the shape below
    structurally excludes, not merely makes rare.

    `Database.close()` (deterministic worker stop + session release) +
    `Database.training_job(job_id)` (attach-by-id, the embedded peer of
    `RemoteTrainingJob`'s always-attach-by-id shape) let one job serve all
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

    The "absent" state's `NULL` is INJECTED, not the job's natural
    just-submitted state: `wait()` blocks until the job reaches `completed`
    (required before `close()` can run), and by then the worker has already
    stamped a real run summary into `metrics`. The state under test is the
    READ arm on the column's three possible SHAPES (`NULL` / valid JSON /
    malformed JSON) — reusing one job's completed row and overwriting its
    `metrics` column directly exercises the identical read path the
    naturally-absent case would (both are `None` at the `Option<String>`
    boundary `metrics()` matches on), for one real training run instead of
    three.

    Asymmetry, named rather than papered over: `RemoteDatabase` already has
    `close()` (used throughout this module); `close()` is therefore symmetric
    across transports. `training_job(job_id)` is NOT — `RemoteDatabase` has
    no equivalent convenience method, though a `RemoteTrainingJob` can always
    be constructed directly from a `job_id` (every one of its verbs re-fetches
    state over the wire per call, so it needs no server-side "attach" step).
    Adding a `RemoteDatabase.training_job(...)` convenience method is
    `clients/python` surface, out of this contract's scope.
    """
    import sqlite3

    import jammi_native
    from jammi._assembly import build_fine_tune_request
    from jammi._generated.jammi.v1 import training_pb2
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

    # --- Remote arm: three stubbed `TrainingStatusResponse`s. -------------
    class _AbsentStub:
        def TrainingStatus(self, *_args, **_kwargs):
            return training_pb2.TrainingStatusResponse(status="running")

    class _ValidStub:
        def TrainingStatus(self, *_args, **_kwargs):
            return training_pb2.TrainingStatusResponse(
                status="completed", metrics_json=valid_payload
            )

    class _MalformedStub:
        def TrainingStatus(self, *_args, **_kwargs):
            return training_pb2.TrainingStatusResponse(
                status="completed", metrics_json=malformed_payload
            )

    remote_absent = jammi.RemoteTrainingJob(
        _AbsentStub(), (), job_id="remote-1", model_id="model-1"
    )
    remote_valid = jammi.RemoteTrainingJob(
        _ValidStub(), (), job_id="remote-2", model_id="model-2"
    )
    remote_malformed = jammi.RemoteTrainingJob(
        _MalformedStub(), (), job_id="remote-3", model_id="model-3"
    )

    assert remote_absent.metrics() == {}
    assert remote_valid.metrics() == valid_expected
    with pytest.raises(BackendError, match=r"metrics blob failed to parse as JSON"):
        remote_malformed.metrics()

    # --- Embedded arm: one real job, three close-before-inject cycles. ----
    catalog_db = tmp_path / "catalog.db"

    def _set_metrics(job_id: str, value) -> None:
        conn = sqlite3.connect(str(catalog_db))
        try:
            conn.execute(
                "UPDATE training_jobs SET metrics = ? WHERE job_id = ?",
                (value, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    # Submit + run ONE real fine-tune job directly against the compiled
    # `_NativeDatabase` — the same request assembly `EmbeddedBackend.fine_tune`
    # drives, minus the wrapper, so the artifact under test is the native
    # `close()` / `training_job()` primitives themselves.
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
        _set_metrics(job_id, seed)
        db = jammi_native.open_local(artifact_dir=str(tmp_path))
        return db, db.training_job(job_id)

    # State 1: absent (`NULL`) — injected explicitly; see the docstring on
    # why the job's OWN natural post-submit state cannot be reused here.
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
    carries the transport lifecycle (close / session_id); each raises for the
    other's."""
    from jammi import Capability
    from jammi.errors import NotSupportedOnBackend

    embed = jammi.connect(f"file://{tmp_path}")
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        assert embed.supports(Capability.AUDIT) is True
        assert embed.supports(Capability.EPHEMERAL_SESSION) is True
        assert embed.supports(Capability.PRELOAD_MODEL) is True
        assert embed.supports(Capability.CLOSE) is False
        assert embed.supports(Capability.SESSION_ID) is False

        assert remote.supports(Capability.CLOSE) is True
        assert remote.supports(Capability.SESSION_ID) is True
        assert remote.supports(Capability.AUDIT) is False
        assert remote.supports(Capability.EPHEMERAL_SESSION) is False
        assert remote.supports(Capability.PRELOAD_MODEL) is False

        # A one-sided op on the wrong backend raises the typed error.
        with pytest.raises(NotSupportedOnBackend):
            embed.session_id
        with pytest.raises(NotSupportedOnBackend):
            embed.close()
        with pytest.raises(NotSupportedOnBackend):
            remote.audit
        with pytest.raises(NotSupportedOnBackend):
            remote.ephemeral_session(timeout_seconds=60)
        with pytest.raises(NotSupportedOnBackend):
            remote.preload_model("some-model")
    finally:
        remote.close()


def test_capability_enum_is_the_closed_five():
    """The capability set is CLOSED — exactly the five one-sided features that
    diverge between the transports, no more. Pinned so a sixth is a deliberate
    decision, not a silent addition."""
    from jammi import Capability

    assert {c.value for c in Capability} == {
        "audit",
        "ephemeral_session",
        "preload_model",
        "close",
        "session_id",
    }


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
# `jammi` is the BASE (U2): `jammi.connect("file://…")` resolves
# the local target to an `EmbeddedBackend` (direct FFI) when `jammi_native` is
# importable — discovering the native engine directly. These tests pin that base
# front door and the lazy-native discipline (`import jammi` stays native-free;
# the engine loads only when a `file://` target is opened), the positive
# complement of the CI negative import-direction guard.
# ---------------------------------------------------------------------------


def test_client_connect_file_returns_embedded_session_via_the_base_front_door(tmp_path):
    """`jammi.connect("file://…")` — the BASE front door — returns an
    `EmbeddedBackend` that satisfies the `Session` protocol and runs a verb
    in-process (direct FFI, B4). This is the U2 feature: the base client
    discovers `jammi_native` as an in-process backend on its own.

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
# §5.5 — the projection shape pin, ANCHORED TO THE PROTO SCHEMA.
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
# This is the honest FLOOR (§5.5 / K4): true per-VALUE equality on real data is
# covered by the Rust it-suite (`grpc_remote_session.rs` / `grpc_remote_compute.rs`,
# the CONSTITUTION K4 anchor) and — for the three eval reports — by the
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
