"""Cache-backed checks for the unified-client surface (U1/U2) — CPU, hermetic.

These re-derive the chapter's load-bearing surface facts LIVE (no server socket,
no GPU, no upstream recompute) and assert they equal the committed
`unified_client.json` record. The surface IS the subject here, so the live
re-derivation is cheap and self-contained — there is no heavy upstream cache to
recompute, only the client contract to hold to its frozen shape:

* **the install surface** — `jammi-ai` carries the engine as the opt-in
  `[embedded]` extra (the `jammi-ai-native` dist), and `connect("file://…")`
  resolves the local target to a direct-FFI `EmbeddedBackend` while
  `connect("grpc://…")` resolves to a `RemoteDatabase` — one base package,
  two arms;
* **embedded↔remote parity** — every member the shared `Session` Protocol names
  is present on BOTH concrete backends, and both satisfy `Session` structurally;
* **the capability contract** — `supports(Capability.X)` is the committed boolean
  per backend (the two CLOSED-five sets are exactly complementary), and a
  one-sided feature on the wrong backend raises `NotSupportedOnBackend`;
* **the two-sided taxonomy** — one `except JammiError` catches a failure on BOTH
  transports, and a bad argument is the SAME `InvalidArgument` class on both;
* **the honest edge** — a real capped gRPC channel raises `RESOURCE_EXHAUSTED`
  mapped onto `BackendError` for a payload above the cap, while the embedded
  engine returns the same-scale payload in-process.

The embedded backend under test is the REAL base-client `EmbeddedBackend` that
`jammi.connect("file://…")` returns (the `[embedded]` extra's direct-FFI
engine), not a convenience-bundle alias. If the emitted cache is absent the
checks skip; the committed record, once present, is always asserted equal to the
live surface.
"""

from __future__ import annotations

import concurrent.futures
import importlib.metadata as importlib_metadata
import tempfile

import grpc
import jammi
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jammi import Capability, Session
from jammi._database import MAX_RECEIVE_MESSAGE_LENGTH, _rpc_to_jammi
from jammi.errors import (
    BackendError,
    InvalidArgument,
    JammiError,
    NotSupportedOnBackend,
)

from jammi_cookbook import contracts

_UC = contracts._dataset_dir("unified_client")
_HAVE_CACHE = (_UC / "unified_client.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="unified-client cache not emitted")


def _record() -> dict:
    return contracts.load_artifact("unified_client.record")


def _session_protocol_verbs() -> list[str]:
    """The public members the runtime-checkable `Session` Protocol names — read
    off the live Protocol, the same way the emit script does."""
    attrs = getattr(Session, "__protocol_attrs__", None)
    if attrs is None:  # pragma: no cover - <3.12 fallback
        from typing import _get_protocol_attrs  # type: ignore[attr-defined]

        attrs = _get_protocol_attrs(Session)
    return sorted(a for a in attrs if not a.startswith("_"))


# --------------------------------------------------------------------------- #
# 0. install surface — the [embedded] extra + the two connect arms
# --------------------------------------------------------------------------- #


@_needs_cache
def test_embedded_extra_pins_the_native_engine():
    """`jammi-ai` declares the opt-in `[embedded]` extra, pinning the compiled
    `jammi-ai-native` dist — read off the live dist metadata, matching the golden."""
    rec = _record()["install"]
    md = importlib_metadata.metadata("jammi-ai")
    extras = sorted(md.get_all("Provides-Extra") or [])
    reqs = md.get_all("Requires-Dist") or []
    pins = sorted(
        r.split(";", 1)[0].strip()
        for r in reqs
        if "extra ==" in r and "'embedded'" in r
    )
    assert "embedded" in extras
    assert pins == rec["embedded_extra_pins"]
    assert any(p.startswith("jammi-ai-native==") for p in pins), "the extra pins the engine"
    assert extras == rec["extras"]


@_needs_cache
def test_file_uri_returns_embedded_backend_grpc_returns_remote():
    """The ONE front door, two arms: `connect("file://…")` returns the in-process
    `EmbeddedBackend` (direct FFI, no server), `connect("grpc://…")` a
    `RemoteDatabase` — both off the base `jammi`, no convenience bundle."""
    rec = _record()["install"]
    with tempfile.TemporaryDirectory() as d:
        embedded = jammi.connect(f"file://{d}")
        remote = jammi.connect("grpc://127.0.0.1:8081")
        try:
            assert type(embedded).__name__ == rec["file_uri_backend"] == "EmbeddedBackend"
            assert type(embedded).__module__ == rec["file_uri_backend_module"]
            assert type(remote).__name__ == rec["grpc_uri_backend"] == "RemoteDatabase"
        finally:
            remote.close()


# --------------------------------------------------------------------------- #
# 0b. embedded↔remote parity — the shared Session Protocol vocabulary
# --------------------------------------------------------------------------- #


@_needs_cache
def test_session_protocol_vocabulary_present_on_both_backends():
    """Every member the shared `Session` Protocol names is present on BOTH concrete
    backends, and both satisfy `Session` structurally — the parity the one-front-
    door thesis rests on, measured member-for-member against the golden."""
    rec = _record()["parity"]
    verbs = _session_protocol_verbs()
    assert verbs == rec["session_protocol_verbs"]
    assert len(verbs) == rec["verb_count"]
    with tempfile.TemporaryDirectory() as d:
        embedded = jammi.connect(f"file://{d}")
        remote = jammi.connect("grpc://127.0.0.1:8081")
        try:
            emb_missing = sorted(v for v in verbs if not hasattr(type(embedded), v))
            rem_missing = sorted(v for v in verbs if not hasattr(type(remote), v))
            assert isinstance(embedded, Session) is rec["embedded_is_session"] is True
            assert isinstance(remote, Session) is rec["remote_is_session"] is True
        finally:
            remote.close()
    assert emb_missing == rec["embedded_missing"] == []
    assert rem_missing == rec["remote_missing"] == []
    assert rec["both_complete"] is True


# --------------------------------------------------------------------------- #
# 1. capability contract
# --------------------------------------------------------------------------- #


@_needs_cache
def test_supports_booleans_match_and_are_complementary():
    """`supports(Capability.X)` is the committed boolean per backend, and the two
    CLOSED-five capability sets are exactly complementary."""
    rec = _record()["capability"]
    with tempfile.TemporaryDirectory() as d:
        embedded = jammi.connect(f"file://{d}")
        remote = jammi.connect("grpc://127.0.0.1:8081")
        try:
            emb = {c.value: embedded.supports(c) for c in Capability}
            rem = {c.value: remote.supports(c) for c in Capability}
        finally:
            remote.close()
    assert emb == rec["embedded_supports"]
    assert rem == rec["remote_supports"]
    assert all(emb[k] != rem[k] for k in emb), "the two capability sets must be complementary"
    assert rec["complementary"] is True


@_needs_cache
def test_wrong_side_capability_raises_not_supported():
    """A one-sided feature invoked on the backend that lacks it raises the typed
    `NotSupportedOnBackend` (never a bare `AttributeError`), naming the capability."""
    rec = _record()["capability"]
    with tempfile.TemporaryDirectory() as d:
        embedded = jammi.connect(f"file://{d}")
        remote = jammi.connect("grpc://127.0.0.1:8081")
        try:
            with pytest.raises(NotSupportedOnBackend) as emb_info:
                _ = embedded.session_id  # remote-only, on embedded
            with pytest.raises(NotSupportedOnBackend) as rem_info:
                _ = remote.audit  # embedded-only, on remote
        finally:
            remote.close()
    assert str(emb_info.value.capability) == rec["embedded_wrong_side"]["capability"]
    assert str(rem_info.value.capability) == rec["remote_wrong_side"]["capability"]
    assert rec["embedded_wrong_side"]["raised"] == "NotSupportedOnBackend"
    assert rec["remote_wrong_side"]["raised"] == "NotSupportedOnBackend"


@_needs_cache
def test_capability_enum_is_the_closed_five():
    """The capability enum is the committed CLOSED five — no silent sixth."""
    rec = _record()["capability"]
    assert [c.value for c in Capability] == rec["enum_values"]
    assert set(rec["enum_values"]) == {
        "audit",
        "ephemeral_session",
        "preload_model",
        "close",
        "session_id",
    }


# --------------------------------------------------------------------------- #
# 2. JammiError taxonomy — two-sided
# --------------------------------------------------------------------------- #


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code, self._details = code, details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class _RaisingStub:
    def __init__(self, err: grpc.RpcError) -> None:
        self._err = err

    def __getattr__(self, _name: str):
        def _raise(*_a: object, **_k: object):
            raise self._err

        return _raise


@_needs_cache
def test_one_except_jammi_error_catches_both_transports():
    """One `except JammiError` catches a bad-argument failure on BOTH the embedded
    engine (rejected in-process) and the remote client (a server status), and both
    are the SAME `InvalidArgument` class — the two-sided parity."""
    rec = _record()["taxonomy"]
    caught: list[JammiError] = []

    # Embedded side — the in-process engine rejects a malformed tenant id.
    with tempfile.TemporaryDirectory() as d:
        embedded = jammi.connect(f"file://{d}")
        try:
            embedded.set_tenant("not-a-uuid")
        except JammiError as exc:
            caught.append(exc)

    # Remote side — a server INVALID_ARGUMENT status arrives over the (stubbed)
    # wire and the client maps it. The SAME `except JammiError` clause catches it.
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        remote._catalog = _RaisingStub(
            _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "server said bad arg")
        )
        try:
            remote.list_sources()
        except JammiError as exc:
            caught.append(exc)
    finally:
        remote.close()

    assert len(caught) == 2, "both transports must raise a JammiError caught by one clause"
    embedded_exc, remote_exc = caught
    assert type(embedded_exc) is InvalidArgument
    assert type(remote_exc) is InvalidArgument
    assert isinstance(embedded_exc, JammiError) and isinstance(remote_exc, JammiError)
    assert isinstance(embedded_exc, ValueError) and isinstance(remote_exc, ValueError)
    assert rec["two_sided_same_class"] is True
    assert rec["embedded_bad_arg"]["raised"] == "InvalidArgument"
    assert rec["remote_bad_arg"]["raised"] == "InvalidArgument"


@_needs_cache
def test_taxonomy_hierarchy_refines_honest_builtins():
    """Each leaf refines the closest honest built-in — the committed hierarchy holds
    on the live classes (an `except ValueError` still fires for a bad argument)."""
    rec = _record()["taxonomy"]["hierarchy"]
    assert "ValueError" in [b.__name__ for b in InvalidArgument.__bases__]
    assert "RuntimeError" in [b.__name__ for b in BackendError.__bases__]
    assert issubclass(InvalidArgument, JammiError)
    assert issubclass(NotSupportedOnBackend, JammiError)
    assert issubclass(BackendError, JammiError)
    assert "ValueError" in rec["InvalidArgument"]
    assert "RuntimeError" in rec["BackendError"]


# --------------------------------------------------------------------------- #
# 3. remote honest edge
# --------------------------------------------------------------------------- #

_ECHO_METHOD = "/jammi.probe.Echo/Get"


def _echo_server(payload: bytes):
    def _handler(_request: bytes, _context) -> bytes:
        return payload

    rpc = grpc.unary_unary_rpc_method_handler(
        _handler,
        request_deserializer=lambda b: b,
        response_serializer=lambda b: b,
    )
    generic = grpc.method_handlers_generic_handler("jammi.probe.Echo", {"Get": rpc})
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=2), handlers=[generic]
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def _receive_at_cap(port: int, cap: int):
    channel = grpc.insecure_channel(
        f"127.0.0.1:{port}", options=[("grpc.max_receive_message_length", cap)]
    )
    stub = channel.unary_unary(
        _ECHO_METHOD,
        request_serializer=lambda b: b,
        response_deserializer=lambda b: b,
    )
    try:
        return len(stub(b"go")), None
    except grpc.RpcError as exc:
        return None, exc
    finally:
        channel.close()


@_needs_cache
def test_finite_receive_cap_raises_backend_error_remote_returns_embedded():
    """A real capped gRPC channel raises `RESOURCE_EXHAUSTED` for a payload above
    the cap — mapped by the real client onto `BackendError` — while a channel at
    the production cap returns it, mirroring the embedded engine's unbounded
    in-process return. The production default (64 MiB) is exercised as the
    generous arm; the reject arm is shown at a scaled-down configured cap."""
    rec = _record()["honest_edge"]
    assert MAX_RECEIVE_MESSAGE_LENGTH == rec["production_cap_bytes"] == 64 * 1024 * 1024

    demo_cap = rec["demo_cap_bytes"]
    payload = b"x" * rec["payload_bytes"]
    server, port = _echo_server(payload)
    try:
        generous_bytes, generous_err = _receive_at_cap(port, MAX_RECEIVE_MESSAGE_LENGTH)
        capped_bytes, capped_err = _receive_at_cap(port, demo_cap)
    finally:
        server.stop(0)

    assert generous_err is None and generous_bytes == len(payload)
    assert capped_bytes is None and capped_err is not None
    assert capped_err.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
    mapped = _rpc_to_jammi(capped_err)
    assert type(mapped) is BackendError and isinstance(mapped, JammiError)

    assert rec["capped_channel_grpc_code"] == "RESOURCE_EXHAUSTED"
    assert rec["capped_channel_mapped"] == "BackendError"
    assert rec["generous_channel_returned_bytes"] == len(payload)


@_needs_cache
def test_embedded_returns_the_same_scale_payload_unbounded():
    """The embedded engine returns the same-scale payload in-process, exceeding the
    demo cap the remote channel rejected — the unbounded in-process counterpart."""
    rec = _record()["honest_edge"]
    with tempfile.TemporaryDirectory() as d:
        embedded = jammi.connect(f"file://{d}")
        table = pa.table({"id": pa.array(range(rec["embedded_returned_rows"]), type=pa.int32())})
        path = f"{d}/big.parquet"
        pq.write_table(table, path)
        embedded.add_source("big", url=path, format="parquet")
        got = embedded.sql("SELECT id FROM big.public.big")
    assert got.num_rows == rec["embedded_returned_rows"]
    assert got.nbytes == rec["embedded_returned_bytes"]
    assert got.nbytes > rec["demo_cap_bytes"], "the embedded return exceeds the remote cap"
    assert rec["embedded_exceeds_demo_cap"] is True
