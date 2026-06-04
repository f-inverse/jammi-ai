"""Hermetic tests for the channel-credentials layer.

Each test stands up a real in-process `SessionService` that records the inbound
metadata of every call, drives a real :class:`RemoteDatabase` through
`open_remote`, and asserts on what reached the server. No network, no
certificate file, nothing that expires:

* the plaintext tests bind an insecure loopback port;
* the secure test binds a loopback port with `grpc.local_server_credentials`
  (`LOCAL_TCP`) — an encrypted, certificate-free transport — and the client
  trusts the matching `grpc.local_channel_credentials`.
"""

from __future__ import annotations

from concurrent import futures
from contextlib import contextmanager

import grpc
import pytest

from jammi_client import BearerCredentials
from jammi_client._credentials import AUTHORIZATION_HEADER
from jammi_client._database import SESSION_HEADER, open_remote
from jammi_client._generated.jammi.v1 import session_pb2, session_pb2_grpc


class _RecordingSessionService(session_pb2_grpc.SessionServiceServicer):
    """Records each call's inbound metadata and answers `GetServerInfo`."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def GetServerInfo(self, request, context):
        self.calls.append(dict(context.invocation_metadata()))
        return session_pb2.ServerInfo(
            version="0.20.0",
            features=["postgres"],
            storage_backends=["file", "memory"],
            services=["core"],
        )


@contextmanager
def _server(server_credentials=None):
    """Run a recording `SessionService` on an ephemeral loopback port.

    Yields ``(endpoint, recorder)``. With `server_credentials` the port is
    secure; without, it is plaintext insecure.
    """
    recorder = _RecordingSessionService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    session_pb2_grpc.add_SessionServiceServicer_to_server(recorder, server)
    if server_credentials is None:
        port = server.add_insecure_port("127.0.0.1:0")
    else:
        port = server.add_secure_port("127.0.0.1:0", server_credentials)
    server.start()
    try:
        yield f"127.0.0.1:{port}", recorder
    finally:
        server.stop(grace=None).wait()


def _local_server_credentials() -> grpc.ServerCredentials:
    return grpc.local_server_credentials(grpc.LocalConnectionType.LOCAL_TCP)


def _local_channel_credentials() -> grpc.ChannelCredentials:
    return grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)


def test_anonymous_sends_session_header_and_no_authorization():
    with _server() as (endpoint, recorder):
        db = open_remote(endpoint, tls=False, credentials=None)
        try:
            db.get_server_info()
        finally:
            db.close()
    assert len(recorder.calls) == 1
    metadata = recorder.calls[0]
    assert AUTHORIZATION_HEADER not in metadata
    assert metadata[SESSION_HEADER] == db.session_id


def test_plaintext_bearer_rides_every_call_alongside_session_header():
    with _server() as (endpoint, recorder):
        db = open_remote(
            endpoint, tls=False, credentials=BearerCredentials("tok-plain")
        )
        try:
            db.get_server_info()
        finally:
            db.close()
    assert len(recorder.calls) == 1
    metadata = recorder.calls[0]
    assert metadata[AUTHORIZATION_HEADER] == "Bearer tok-plain"
    assert metadata[SESSION_HEADER] == db.session_id


def test_secure_bearer_rides_composite_call_credentials_alongside_session():
    with _server(_local_server_credentials()) as (endpoint, recorder):
        db = open_remote(
            endpoint,
            tls=True,
            credentials=BearerCredentials(
                "tok-tls", secure_credentials=_local_channel_credentials
            ),
        )
        try:
            db.get_server_info()
        finally:
            db.close()
    assert len(recorder.calls) == 1
    metadata = recorder.calls[0]
    assert metadata[AUTHORIZATION_HEADER] == "Bearer tok-tls"
    assert metadata[SESSION_HEADER] == db.session_id


@pytest.mark.parametrize("token", ["", "   "])
def test_empty_token_is_rejected_at_construction(token):
    with pytest.raises(ValueError):
        BearerCredentials(token)
