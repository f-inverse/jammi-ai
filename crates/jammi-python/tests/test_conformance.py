"""Conformance: the embed wheel's remote arm IS `jammi-client`, by construction.

The whole point of the composition (M2 §2) is that the remote surface is
DEFINED ONCE — in `jammi-client` — and `jammi-ai`'s remote target delegates to
it. So these tests assert the construction holds rather than re-listing a
parallel surface that could drift:

  1. `jammi_ai.connect(remote)` returns a `jammi_client.RemoteDatabase` — the
     embed wheel's remote-capable surface IS the client's, not a copy.
  2. `connect(target)` routes by scheme: `file://` → the compiled local engine
     (`jammi_ai.Database`); `https://`/`grpc://` → the client's
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

import jammi_ai
import jammi_client


def _embed_method(verb: str):
    """Resolve `verb` against the COMPOSED embedded surface: the thin Python
    `jammi_ai.Database`'s explicit method if it declares one, else the
    `_NativeDatabase` low-level handle's method it delegates to via
    ``__getattr__``.

    The embedded `Database` is a thin wrapper over the compiled
    `_NativeDatabase`: the migrated training verbs are explicit Python methods on
    the wrapper (driving the shared request assembly), while every un-migrated
    verb lives on the native handle and is forwarded at runtime. A class-level
    introspection (which is what the conformance guard does) must look through the
    same composition — wrapper first, native handle behind it — to see the verb a
    caller actually invokes."""
    if verb in vars(jammi_ai.Database):
        return getattr(jammi_ai.Database, verb)
    return getattr(jammi_ai._native._NativeDatabase, verb)


def _embed_has(verb: str) -> bool:
    """Whether the composed embedded surface carries `verb` — declared on the thin
    `Database` wrapper or on the `_NativeDatabase` handle behind it."""
    return verb in vars(jammi_ai.Database) or hasattr(
        jammi_ai._native._NativeDatabase, verb
    )


def test_embed_remote_is_the_client_remote_database():
    """`jammi_ai.connect(remote)` returns the client's `RemoteDatabase` — the
    remote arm is defined once, in `jammi-client`, and reused by composition."""
    db = jammi_ai.connect("grpc://127.0.0.1:8081")
    try:
        assert isinstance(db, jammi_client.RemoteDatabase)
    finally:
        db.close()


def test_connect_routes_local_to_the_compiled_engine(tmp_path):
    """A `file://` target resolves to the in-process engine — a
    `jammi_ai.Database`, never the remote client."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    assert isinstance(db, jammi_ai.Database)
    assert not isinstance(db, jammi_client.RemoteDatabase)


def test_pure_client_local_target_is_a_truthful_error():
    """The pure client carries no engine; `file://` raises the no-engine error
    pointing at the embed wheel, never a silent default to remote."""
    import pytest

    with pytest.raises(jammi_client.NoEmbeddedEngineError):
        jammi_client.connect("file:///tmp/x")


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
# embed `jammi_ai.Database`.
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
# `jammi_ai.Database`.
_INFERENCE_VERBS = {
    "infer",
}


# The engine-state pipeline verbs. Like the training/predict verbs, these DO hit
# the wire: they build durable graph/embedding artifacts or assemble a target's
# conditioning context against the remote engine's state (`PipelineService`).
# The embedded `Database` runs them in the compiled engine; the client's
# `RemoteDatabase` drives them over gRPC. The call surface must agree so a
# caller swaps transports without changing the call — pinned here against the
# embed `jammi_ai.Database`.
_PIPELINE_VERBS = {
    "build_neighbor_graph",
    "propagate_embeddings",
    "assemble_context",
}


# The evaluation verbs. They DO hit the wire: the model and the golden data
# both live in the engine, so the compute runs where the data is
# (`EvalService`) and only the typed report crosses the wire. The embedded
# `Database` runs them in the compiled engine; the client's `RemoteDatabase`
# drives them over gRPC. The call surface must agree so a caller swaps
# transports without changing the call — pinned here against the embed
# `jammi_ai.Database`.
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
# call — pinned here against the embed `jammi_ai.Database`.
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
# pinned here against the embed `jammi_ai.Database`. `subscribe_collect` mirrors
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
# against the embed `jammi_ai.Database`.
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
        assert callable(getattr(jammi_client.RemoteDatabase, verb)), verb


def test_lifecycle_verbs_have_identical_signatures_across_wheels():
    """The model-lifecycle verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _LIFECYCLE_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
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
    from jammi_client._generated.jammi.v1 import catalog_pb2

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
    db = jammi_ai.connect(f"file://{tmp_path}")
    models = db.list_models()
    assert isinstance(models, list)
    for m in models:
        assert set(m) == _MODEL_DICT_KEYS, (
            f"embed list_models entry keys {set(m)} != {_MODEL_DICT_KEYS}"
        )


def test_mutable_topic_verbs_have_identical_signatures_across_wheels():
    """The mutable-table + topic + pub/sub verbs carry the SAME call surface on the
    client's `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`. Both
    drive over the same verb vocabulary (the client over gRPC, the embed
    in-process), so a caller swaps transports without changing the call — pinned
    name-for-name, kind-for-kind, and default-for-default so a divergence in either
    is caught."""
    for verb in _MUTABLE_TOPIC_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_inference_verbs_have_identical_signatures_across_wheels():
    """The bulk inference verb carries the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _INFERENCE_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_pipeline_verbs_have_identical_signatures_across_wheels():
    """The engine-state pipeline verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _PIPELINE_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_eval_verbs_have_identical_signatures_across_wheels():
    """The evaluation verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`. Both drive
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _EVAL_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


# The vector-search verb. It DOES hit the wire: the source's embedding tables
# live in the engine, so the ANN search runs where the vectors are
# (`EmbeddingService.Search`) and only the hits cross the wire. The embedded
# `Database` searches in the compiled engine; the client's `RemoteDatabase`
# drives it over gRPC. The call surface — including the `embedding_table=`
# selector that names WHICH of a source's embedding tables to search — must
# agree so a caller swaps transports without changing the call. Pinned against
# the embed `jammi_ai.Database`.
_SEARCH_VERBS = {
    "search",
}


def test_search_verb_has_identical_signature_across_wheels():
    """`search` carries the SAME call surface on the client's `RemoteDatabase`
    as on the embedded engine's `jammi_ai.Database` — including the
    `embedding_table=` table selector. Both drive over the same verb vocabulary
    (the client over gRPC, the embed in-process), so a caller swaps transports
    without changing the call — pinned name-for-name, kind-for-kind, and
    default-for-default so a divergence in either is caught."""
    for verb in _SEARCH_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"
        names = {p[0] for p in embed}
        assert "embedding_table" in names, f"{verb} must expose embedding_table"


def test_channel_verbs_have_identical_signatures_across_wheels():
    """The evidence-channel registry verbs carry the SAME call surface on the
    client's `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`.
    Both drive over the same verb vocabulary (the client over gRPC, the embed
    in-process), so a caller swaps transports without changing the call — pinned
    name-for-name, kind-for-kind, and default-for-default so a divergence in
    either is caught."""
    for verb in _CHANNEL_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_training_verbs_have_identical_signatures_across_wheels():
    """The training + predict verbs carry the SAME call surface on the client's
    `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`. Both submit
    over the same verb vocabulary (the client over gRPC, the embed in-process), so
    a caller swaps transports without changing the call — pinned name-for-name,
    kind-for-kind, and default-for-default so a divergence in either is caught."""
    for verb in _TRAINING_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
        embed = _call_surface(_embed_method(verb))
        assert client == embed, f"{verb}: {embed} != {client}"


def test_remote_training_job_matches_the_local_handle_shape():
    """The client's `RemoteTrainingJob` carries the SAME handle surface as the
    embedded engine's `TrainingJob`: the `job_id` / `model_id` properties and the
    `status()` / `wait()` methods. A remote `wait()` polls `TrainingStatus` and
    raises on a failed job with the wire error, mirroring the local handle, so a
    caller treats the two interchangeably."""
    local = jammi_ai.TrainingJob
    remote = jammi_client.RemoteTrainingJob
    for member in ("job_id", "model_id", "status", "wait"):
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
    `RemoteDatabase` as on the embedded engine's `jammi_ai.Database`. They are
    computed locally on both wheels (no wire hop), so the verb surface must agree
    name-for-name, kind-for-kind, and default-for-default — pinned here so a
    divergence in either implementation is caught."""
    for verb in _NUMERIC_VERBS:
        client = _call_surface(getattr(jammi_client.RemoteDatabase, verb))
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
    local = jammi_ai.connect(f"file://{tmp_path}")
    remote = jammi_ai.connect("grpc://127.0.0.1:8081")
    try:
        assert type(remote) is jammi_client.RemoteDatabase

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
    """Because the embed wheel's remote arm IS `jammi_client.RemoteDatabase`,
    every verb's signature is identical by construction. Asserting it here pins
    the invariant so any future hand-rolled remote class in the embed wheel
    (which would re-introduce the very drift M2 removes) fails this test."""
    embed_remote = jammi_ai.connect("grpc://127.0.0.1:8081")
    try:
        assert type(embed_remote) is jammi_client.RemoteDatabase
        for verb in _REMOTE_VERBS:
            client_sig = inspect.signature(getattr(jammi_client.RemoteDatabase, verb))
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
        assert callable(getattr(jammi_client.RemoteDatabase, verb)), verb
    assert not _embed_has("with_tenant"), "with_tenant must be hard-cut"
    assert not hasattr(
        jammi_client.RemoteDatabase, "with_tenant"
    ), "with_tenant must be hard-cut"
    # `tenant_scope(tenant_id)` takes the same caller-visible parameter on both —
    # the embedded native method and the client context manager agree name-for-name.
    embed = _call_surface(_embed_method("tenant_scope"))
    client = _call_surface(jammi_client.RemoteDatabase.tenant_scope)
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
    from jammi_client._generated.jammi.v1 import catalog_pb2

    proto_fields = {f.name for f in catalog_pb2.ServerInfo.DESCRIPTOR.fields}

    embedded = jammi_ai.connect(f"file://{tmp_path}").get_server_info()
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
    """The key set `jammi_client.RemoteDatabase.get_server_info` returns, read
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

    db = jammi_client.connect("grpc://127.0.0.1:8081")
    try:
        db._catalog = _StubCatalog()
        return set(db.get_server_info())
    finally:
        db.close()
