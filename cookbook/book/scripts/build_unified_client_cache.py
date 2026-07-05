#!/usr/bin/env python3
"""Emit the unified-client-surface cache — CPU, dual-transport, hermetic.

The engine<->cookbook validator for the one-front-door client contract: one
``connect(target)`` returns a transport-agnostic ``Session`` whose capability
divergence, error taxonomy, and finite-cap honest edge are the SAME shape on the
embedded ``jammi_client.EmbeddedBackend`` (the base client's in-process,
direct-FFI backend — the ``[embedded]`` extra) and the remote
``jammi_client.RemoteDatabase``. This script measures five families of surface
facts and freezes them:

* **the install surface** — ``jammi-client`` carries the engine as the opt-in
  ``[embedded]`` extra (the compiled ``jammi-ai-native`` dist, importable as
  ``jammi_native``); with it, ``jammi_client.connect("file://…")`` resolves the
  local target to an in-process ``EmbeddedBackend`` (direct FFI, no server),
  ``connect("grpc://…")`` to a ``RemoteDatabase`` — one base package, two arms;
* **embedded↔remote parity** — the shared ``Session`` Protocol enumerates the
  verb vocabulary both transports carry; every one of its members is present on
  BOTH concrete backends, and both concrete backends satisfy ``Session``
  structurally (``isinstance`` runtime-checkable), so a program written against
  the Protocol runs local or remote by target alone;
* **the capability contract** — ``supports(Capability.X)`` is a hard boolean per
  backend (the two sets are exactly complementary across the CLOSED five), and
  invoking a one-sided feature on the backend that lacks it raises the typed
  ``NotSupportedOnBackend`` rather than a bare ``AttributeError``;
* **the JammiError taxonomy, two-sided** — one ``except JammiError`` catches a
  failure on BOTH transports, and a bad argument surfaces as the SAME
  ``InvalidArgument`` class whether the embedded engine rejected it in-process or
  a server rejected it over the wire;
* **the remote honest edge** — the remote channel raises its per-message receive
  cap to a finite 64 MiB (the embedded engine returns a result in-process,
  unbounded); a result exceeding the cap surfaces as ``BackendError`` mapped from
  gRPC ``RESOURCE_EXHAUSTED``, exercised for real at a scaled-down configured cap,
  beside the embedded engine returning the same-scale payload whole in-process.

Every fact is measured against the LIVE surface — no fact is transcribed. The
remote server faults (a wire ``INVALID_ARGUMENT`` / ``RESOURCE_EXHAUSTED``) are
injected at the transport boundary exactly as a live server's status arrives —
the same hermetic technique the engine's own U1 conformance suite uses
(`crates/jammi-python/tests/test_conformance.py`) — so no server socket is
dialed. This is an emit-only script; PR CI reads the committed cache
(`test_unified_client_cache.py` re-derives the same facts live and asserts
equality) and never re-drives a server. Names no consumer — every capability and
error class is a generic engine feature.

Usage::

    python scripts/build_unified_client_cache.py
"""

from __future__ import annotations

import concurrent.futures
import importlib.metadata as importlib_metadata
import json
import tempfile
from importlib.util import find_spec
from pathlib import Path

import grpc
import jammi_client
import pyarrow as pa
import pyarrow.parquet as pq
from jammi_client import Capability, Session
from jammi_client._database import MAX_RECEIVE_MESSAGE_LENGTH, _rpc_to_jammi
from jammi_client.errors import (
    BackendError,
    InvalidArgument,
    JammiError,
    NotSupportedOnBackend,
)

import jammi_cookbook  # noqa: F401  # applies the determinism env on import

_OUT = Path(__file__).resolve().parent.parent / "artifacts" / "unified_client"


# --------------------------------------------------------------------------- #
# The hermetic remote-fault injection — one real RemoteDatabase, a transport
# stub that raises a chosen gRPC status. Mirrors the engine's conformance suite:
# the verb still routes through the real client `_call` -> `_rpc_to_jammi`
# mapping, only the wire hop is stubbed, so no server socket is opened.
# --------------------------------------------------------------------------- #


class _FakeRpcError(grpc.RpcError):
    """A ``grpc.RpcError`` carrying a chosen status code, as a server would set."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class _RaisingStub:
    """A gRPC service stub whose every method raises a fixed error — swapped onto
    a ``RemoteDatabase._catalog`` so a verb's wire hop raises hermetically."""

    def __init__(self, err: grpc.RpcError) -> None:
        self._err = err

    def __getattr__(self, _name: str):
        def _raise(*_a: object, **_k: object):
            raise self._err

        return _raise


def _remote_verb_status(code: grpc.StatusCode) -> JammiError:
    """Drive a real ``RemoteDatabase`` unary verb whose wire hop raises ``code``;
    return the mapped taxonomy exception (through the real ``_call`` path)."""
    remote = jammi_client.connect("grpc://127.0.0.1:8081")  # lazy; opens no socket
    try:
        remote._catalog = _RaisingStub(_FakeRpcError(code, f"server said {code}"))
        try:
            remote.list_sources()
        except JammiError as exc:
            return exc
        raise AssertionError(f"{code} did not raise")
    finally:
        remote.close()


# --------------------------------------------------------------------------- #
# 0. install surface — the [embedded] extra + the two connect arms
# --------------------------------------------------------------------------- #


def _install_record(embedded, remote) -> dict:
    """The base client's install surface: the ``[embedded]`` extra (read off the
    live dist metadata — never transcribed) and the concrete backend each URI
    scheme resolves to."""
    md = importlib_metadata.metadata("jammi-client")
    extras = sorted(md.get_all("Provides-Extra") or [])
    reqs = md.get_all("Requires-Dist") or []
    embedded_pins = sorted(
        r.split(";", 1)[0].strip()
        for r in reqs
        if "extra ==" in r and "'embedded'" in r
    )
    return {
        "extras": extras,
        "embedded_extra_name": "embedded",
        "embedded_extra_pins": embedded_pins,
        "native_dist": "jammi-ai-native",
        "native_module": "jammi_native",
        "native_importable": find_spec("jammi_native") is not None,
        # The two arms of the ONE front door, by URI scheme.
        "file_uri_backend": type(embedded).__name__,
        "file_uri_backend_module": type(embedded).__module__,
        "grpc_uri_backend": type(remote).__name__,
    }


# --------------------------------------------------------------------------- #
# 0b. embedded↔remote parity — the shared Session Protocol vocabulary
# --------------------------------------------------------------------------- #


def _session_protocol_verbs() -> list[str]:
    """The public members the runtime-checkable ``Session`` Protocol names — the
    shared verb vocabulary both transports MUST carry. Read off the Protocol
    itself (``__protocol_attrs__`` on 3.12+, the ``typing`` helper below it), so
    the list is the live contract, never a transcription."""
    attrs = getattr(Session, "__protocol_attrs__", None)
    if attrs is None:  # pragma: no cover - <3.12 fallback
        from typing import _get_protocol_attrs  # type: ignore[attr-defined]

        attrs = _get_protocol_attrs(Session)
    return sorted(a for a in attrs if not a.startswith("_"))


def _parity_record(embedded, remote) -> dict:
    """Every member the ``Session`` Protocol names is present on BOTH concrete
    backends, and both backends satisfy ``Session`` structurally — the parity the
    one-front-door thesis rests on, measured member-for-member."""
    verbs = _session_protocol_verbs()
    emb_missing = sorted(v for v in verbs if not hasattr(type(embedded), v))
    rem_missing = sorted(v for v in verbs if not hasattr(type(remote), v))
    return {
        "session_protocol_verbs": verbs,
        "verb_count": len(verbs),
        "embedded_missing": emb_missing,
        "remote_missing": rem_missing,
        "both_complete": not emb_missing and not rem_missing,
        "embedded_is_session": isinstance(embedded, Session),
        "remote_is_session": isinstance(remote, Session),
        "embedded_backend": type(embedded).__name__,
        "remote_backend": type(remote).__name__,
    }


# --------------------------------------------------------------------------- #
# 1. capability contract
# --------------------------------------------------------------------------- #


def _capability_record(embedded, remote) -> dict:
    order = list(Capability)
    emb = {c.value: embedded.supports(c) for c in order}
    rem = {c.value: remote.supports(c) for c in order}

    # A one-sided feature invoked on the backend that lacks it raises the typed
    # NotSupportedOnBackend, carrying the capability it named.
    def _wrong_side(fn) -> dict:
        try:
            fn()
        except NotSupportedOnBackend as exc:
            return {"raised": type(exc).__name__, "capability": str(exc.capability)}
        raise AssertionError("expected NotSupportedOnBackend")

    embedded_wrong = _wrong_side(lambda: embedded.session_id)  # remote-only on embedded
    remote_wrong = _wrong_side(lambda: remote.audit)  # embedded-only on remote

    return {
        "enum_values": [c.value for c in order],
        "embedded_supports": emb,
        "remote_supports": rem,
        "complementary": all(emb[k] != rem[k] for k in emb),
        "embedded_wrong_side": embedded_wrong,
        "remote_wrong_side": remote_wrong,
    }


# --------------------------------------------------------------------------- #
# 2. JammiError taxonomy — two-sided parity
# --------------------------------------------------------------------------- #


def _taxonomy_record(embedded) -> dict:
    # The hierarchy, read off the live classes (each leaf refines the closest
    # honest built-in).
    hierarchy = {
        "JammiError": [b.__name__ for b in JammiError.__bases__],
        "InvalidArgument": [b.__name__ for b in InvalidArgument.__bases__],
        "NotSupportedOnBackend": [b.__name__ for b in NotSupportedOnBackend.__bases__],
        "BackendError": [b.__name__ for b in BackendError.__bases__],
    }

    # Embedded bad argument — a malformed tenant id the in-process engine rejects.
    try:
        embedded.set_tenant("not-a-uuid")
    except JammiError as exc:
        embedded_bad = exc
    else:
        raise AssertionError("embedded bad tenant id did not raise")

    # Remote bad argument — a server INVALID_ARGUMENT status, mapped by the client.
    remote_bad = _remote_verb_status(grpc.StatusCode.INVALID_ARGUMENT)

    def _describe(exc: BaseException) -> dict:
        return {
            "raised": type(exc).__name__,
            "is_jammi_error": isinstance(exc, JammiError),
            "is_value_error": isinstance(exc, ValueError),
        }

    return {
        "hierarchy": hierarchy,
        "embedded_bad_arg": _describe(embedded_bad),
        "remote_bad_arg": _describe(remote_bad),
        "two_sided_same_class": type(embedded_bad) is type(remote_bad) is InvalidArgument,
        # One except JammiError catches both — proven by the fact both were caught
        # by the `except JammiError` above without escaping.
        "one_except_catches_both": True,
    }


# --------------------------------------------------------------------------- #
# 3. remote honest edge — the finite receive cap, exercised for real
# --------------------------------------------------------------------------- #

_ECHO_METHOD = "/jammi.probe.Echo/Get"


def _echo_server(payload: bytes):
    """A minimal in-process gRPC server: one unary method returns ``payload``.

    Stands in for a remote verb whose whole result rides back as one message. No
    proto is needed — bytes pass through an identity ser/de — so the demonstration
    is of the CHANNEL's receive cap, nothing engine-specific."""

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


def _receive_at_cap(port: int, cap: int) -> tuple[int | None, grpc.RpcError | None]:
    """Call the echo server over a channel whose receive cap is ``cap``; return
    ``(bytes_returned, None)`` on success or ``(None, rpc_error)`` on the cap."""
    channel = grpc.insecure_channel(
        f"127.0.0.1:{port}",
        options=[("grpc.max_receive_message_length", cap)],
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


def _honest_edge_record(embedded) -> dict:
    # A payload well above a scaled-down configured cap (and far below the 64 MiB
    # production default, so the harness stays light). The demo cap stands in for
    # the production 64 MiB — the mechanism is the channel's receive cap, whatever
    # its value.
    demo_cap = 64 * 1024  # 64 KiB
    payload = b"x" * (256 * 1024)  # 256 KiB — exceeds the demo cap

    server, port = _echo_server(payload)
    try:
        # A channel at the PRODUCTION cap returns the payload (mirrors embedded).
        generous_bytes, generous_err = _receive_at_cap(port, MAX_RECEIVE_MESSAGE_LENGTH)
        assert generous_err is None and generous_bytes == len(payload)
        # A channel at the demo cap rejects it — a real gRPC RESOURCE_EXHAUSTED,
        # mapped by the real client mapping onto BackendError.
        capped_bytes, capped_err = _receive_at_cap(port, demo_cap)
        assert capped_bytes is None and capped_err is not None
        grpc_code = capped_err.code().name
        mapped = _rpc_to_jammi(capped_err)
    finally:
        server.stop(0)

    # The embedded contrast: the SAME-scale payload returned in-process, with no
    # message cap at all. Build a ~payload-scale table and return it via sql.
    with tempfile.TemporaryDirectory() as d:
        n_rows = 40_000
        table = pa.table({"id": pa.array(range(n_rows), type=pa.int32())})
        path = f"{d}/big.parquet"
        pq.write_table(table, path)
        embedded.add_source("big", url=path, format="parquet")
        got = embedded.sql("SELECT id FROM big.public.big")
        embedded_bytes = got.nbytes
        embedded_rows = got.num_rows

    return {
        "production_cap_bytes": MAX_RECEIVE_MESSAGE_LENGTH,
        "production_cap_mib": MAX_RECEIVE_MESSAGE_LENGTH // (1024 * 1024),
        "demo_cap_bytes": demo_cap,
        "payload_bytes": len(payload),
        "generous_channel_returned_bytes": generous_bytes,
        "capped_channel_grpc_code": grpc_code,
        "capped_channel_mapped": type(mapped).__name__,
        "capped_channel_is_jammi_error": isinstance(mapped, JammiError),
        "capped_channel_is_backend_error": isinstance(mapped, BackendError),
        "embedded_returned_rows": embedded_rows,
        "embedded_returned_bytes": embedded_bytes,
        "embedded_exceeds_demo_cap": embedded_bytes > demo_cap,
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as artifact_dir:
        # The base client discovers the in-process engine on demand: a `file://`
        # target resolves to the direct-FFI EmbeddedBackend when the `[embedded]`
        # extra is installed. This is the REAL new surface — no convenience bundle.
        embedded = jammi_client.connect(f"file://{artifact_dir}")
        remote = jammi_client.connect("grpc://127.0.0.1:8081")  # lazy; supports() is local
        try:
            record = {
                "install": _install_record(embedded, remote),
                "parity": _parity_record(embedded, remote),
                "capability": _capability_record(embedded, remote),
                "taxonomy": _taxonomy_record(embedded),
                "honest_edge": _honest_edge_record(embedded),
            }
        finally:
            remote.close()

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "unified_client.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    # No numeric tolerances here — every fact is an exact boolean / class name /
    # byte count, asserted by equality — so golden_metrics.json is an empty map
    # (the loader contract still resolves the dataset dir).
    (_OUT / "golden_metrics.json").write_text("{}\n")
    print(f"unified-client cache emitted to {_OUT}")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
