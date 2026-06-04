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
    "list_sources",
    "describe_source",
    "with_tenant",
    "tenant",
    "get_server_info",
}


def test_remote_surface_has_every_verb():
    """The client's `RemoteDatabase` exposes the full transport-agnostic verb
    set — the same vocabulary the embedded `Database` carries."""
    for verb in _REMOTE_VERBS:
        assert callable(getattr(jammi_client.RemoteDatabase, verb)), verb


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
        assert hasattr(jammi_ai.Database, verb), verb
    for gone in (
        "encode_text_query",
        "encode_image_query",
        "encode_audio_query",
        "generate_text_embeddings",
        "generate_image_embeddings",
        "generate_audio_embeddings",
        "server_info",
    ):
        assert not hasattr(jammi_ai.Database, gone), f"{gone} should be hard-cut"
