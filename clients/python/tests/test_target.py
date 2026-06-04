"""Unit tests for `connect(target)` scheme dispatch and the `Target` sum.

Hermetic: no channel is dialed (grpcio channels are lazy), so these assert the
classification + the truthful local-on-pure-client error without a server.
"""

from __future__ import annotations

import pytest

import jammi_client
from jammi_client import (
    LocalTarget,
    NoEmbeddedEngineError,
    RemoteDatabase,
    RemoteTarget,
    parse_target,
)


def test_file_scheme_parses_to_local_target():
    t = parse_target("file:///var/lib/jammi")
    assert isinstance(t, LocalTarget)
    assert t.artifact_dir == "/var/lib/jammi"


def test_relative_file_target_keeps_path():
    t = parse_target("file://.jammi")
    assert isinstance(t, LocalTarget)
    assert ".jammi" in t.artifact_dir


@pytest.mark.parametrize(
    "uri,tls",
    [
        ("https://engine.example.com", True),
        ("grpcs://engine.example.com:8081", True),
        ("http://127.0.0.1:8081", False),
        ("grpc://127.0.0.1:8081", False),
    ],
)
def test_remote_schemes_parse_to_remote_target(uri, tls):
    t = parse_target(uri)
    assert isinstance(t, RemoteTarget)
    assert t.tls is tls
    assert t.endpoint  # host[:port] authority, non-empty


def test_structured_target_passes_through():
    given = RemoteTarget(endpoint="host:9", tls=False)
    assert parse_target(given) is given


def test_unknown_scheme_raises_without_silent_default():
    with pytest.raises(ValueError) as info:
        parse_target("ftp://nope")
    assert "scheme" in str(info.value).lower()


def test_connect_local_raises_truthful_no_engine_error():
    """The pure client carries no embedded engine; `file://` is a truthful
    error pointing at the embed wheel — the runtime echo of the Rust
    `#[cfg(feature = "local")]` gate, never a silent failure."""
    with pytest.raises(NoEmbeddedEngineError) as info:
        jammi_client.connect("file:///tmp/data")
    msg = str(info.value)
    assert "pip install jammi-ai" in msg
    assert "/tmp/data" in msg


def test_connect_remote_returns_remote_database_without_dialing():
    """A remote target yields a `RemoteDatabase` over a lazy channel; no server
    contact happens until a verb is called, so this stays hermetic."""
    db = jammi_client.connect("grpc://127.0.0.1:8081")
    try:
        assert isinstance(db, RemoteDatabase)
        assert len(db.session_id) == 36  # a v4 UUID
    finally:
        db.close()
