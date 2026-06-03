"""Shape C — the remote-transport Python binding (`connect_remote`).

A Python consumer that wants to drive a *remote* jammi engine over the
`jammi.v1` gRPC wire calls `jammi_ai.connect_remote(endpoint=...)` and gets a
`RemoteDatabase` whose verbs (encode-query, generate-embeddings, search, infer,
the tenant trio) reach the server through the one Rust `RemoteSession` gRPC
client — never a second Python-side client.

These tests cover what only the Python boundary can: the constructor is
exposed, validates its endpoint, and maps transport / engine errors to Python
exceptions (errors-as-values, never a panic across the FFI). The full
over-the-wire parity proof — a query encoded through the remote Python session
equals the byte-identical vector a `LocalSession` produces against the same
engine — lives in the Rust integration test `crates/jammi-python/tests/it.rs`,
which can stand up an in-process gRPC server hermetically (bundled `tiny_bert`
encoder + `patents.parquet` corpus, no live network).

These tests run via the `test-python` CI job, which builds the wheel with
`maturin develop --release` before invoking `pytest`.
"""

import pytest

import jammi_ai


def test_connect_remote_is_exposed():
    """The remote front door is on the module surface alongside `connect`."""
    assert hasattr(jammi_ai._native, "connect_remote")
    assert hasattr(jammi_ai._native, "RemoteDatabase")


def test_connect_remote_rejects_invalid_endpoint():
    """A malformed endpoint URI raises `ValueError` at the binding edge
    (PyO3 mapping of the `Endpoint::from_shared` parse failure) — not a
    panic across the FFI."""
    with pytest.raises((ValueError, RuntimeError)) as info:
        jammi_ai._native.connect_remote(endpoint="not a url")
    msg = str(info.value).lower()
    assert "endpoint" in msg or "invalid" in msg or "uri" in msg, (
        f"expected the message to name the endpoint problem, got: {info.value}"
    )


def test_connect_remote_surfaces_connection_failure():
    """Connecting to an address with nothing listening surfaces a Python
    exception (the `RemoteSession::connect` error mapped through `to_pyerr`),
    not a hang or a panic. Port 1 is in the reserved low range and never has a
    jammi server behind it, so the dial fails fast and hermetically."""
    with pytest.raises((RuntimeError, ValueError)):
        jammi_ai._native.connect_remote(endpoint="http://127.0.0.1:1")
