"""Hermetic proof that the Flight SQL lane carries the channel's bearer.

The typed gRPC verbs attach `authorization: Bearer <token>` via the channel
credentials; `RemoteDatabase.sql` is a separate `pyarrow.flight` transport that
does not share those credentials, so the bearer is threaded into the Flight call
options from the credential at `open_remote`. These tests stand up a minimal
`pyarrow.flight` stub server whose `ServerMiddleware` records the inbound
`authorization` header — the same place a server-side BYO-auth seam would read
it — and drive a real :class:`RemoteDatabase` (built by `open_remote`, so the
production token-threading runs) through `db.sql` against the stub.

No jammi-server build is involved. Server-side enforcement of the BYO-auth seam
over Flight is a separate concern tracked at
https://github.com/f-inverse/jammi-ai/issues/220; what is proven here is the
client scope: the client *sends* the bearer on the Flight lane (and sends none
when anonymous), over both the plaintext (`grpc+tcp`) and TLS (`grpc+tls`)
schemes — pyarrow does not gate metadata on transport security, mirroring the
typed path's plaintext interceptor.
"""

from __future__ import annotations

import datetime
import socket

import pyarrow as pa
import pyarrow.flight as flight
import pytest

from jammi_client import BearerCredentials
from jammi_client._credentials import AnonymousCredentials
from jammi_client._database import open_remote

_TOKEN = "good-token"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RecordingAuthFactory(flight.ServerMiddlewareFactory):
    """Records the inbound `authorization` header of every Flight call.

    pyarrow lowercases header keys and presents each value as a list; the
    factory stores the first value (or ``None`` when the header is absent) so a
    test can assert exactly what the client put on the wire.
    """

    def __init__(self) -> None:
        self.authorizations: list[str | None] = []

    def start_call(self, info, headers):
        auth = headers.get("authorization")
        self.authorizations.append(auth[0] if auth else None)
        return None


class _StubFlightServer(flight.FlightServerBase):
    """A Flight server that answers any query with a one-row table.

    It exists only to terminate a real `get_flight_info` + `do_get` round-trip
    so the client's `sql` path runs end to end; the assertions are on what the
    recording middleware saw, not the returned data.
    """

    def __init__(self, location, factory: _RecordingAuthFactory, **kwargs) -> None:
        super().__init__(location, middleware={"auth": factory}, **kwargs)
        self._schema = pa.schema([("ok", pa.int64())])

    def get_flight_info(self, context, descriptor):
        endpoint = flight.FlightEndpoint(b"ticket", [])
        return flight.FlightInfo(self._schema, descriptor, [endpoint], -1, -1)

    def do_get(self, context, ticket):
        return flight.RecordBatchStream(pa.table({"ok": [1]}))


def _self_signed_cert() -> tuple[bytes, bytes]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), False)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


@pytest.fixture(params=["grpc+tcp", "grpc+tls"])
def stub(request):
    """A running stub keyed by transport scheme.

    Yields ``(scheme, endpoint, factory, tls_root_certs)``. The TLS leg binds a
    self-signed loopback cert; the plaintext leg binds none. ``tls_root_certs``
    is the PEM a Flight client must trust to complete the TLS handshake (the
    production `_flight_client` builds a bare client, so the test substitutes a
    cert-trusting client for the same `RemoteDatabase` — the header assembly it
    exercises is unchanged).
    """
    scheme = request.param
    factory = _RecordingAuthFactory()
    if scheme == "grpc+tls":
        cert_pem, key_pem = _self_signed_cert()
        server = _StubFlightServer(
            flight.Location.for_grpc_tls("127.0.0.1", _free_port()),
            factory,
            tls_certificates=[flight.CertKeyPair(cert_pem, key_pem)],
        )
        tls_root_certs: bytes | None = cert_pem
    else:
        server = _StubFlightServer(
            flight.Location.for_grpc_tcp("127.0.0.1", _free_port()), factory
        )
        tls_root_certs = None
    endpoint = f"localhost:{server.port}"
    try:
        yield scheme, endpoint, factory, tls_root_certs
    finally:
        server.shutdown()


def _query(db, scheme: str, tls_root_certs: bytes | None) -> pa.Table:
    """Run `db.sql` against the stub, trusting the test cert on the TLS leg.

    The production `_flight_client` builds a bare `FlightClient` that would not
    trust a self-signed cert, so the test seeds `db._flight` with a client that
    trusts the test cert. `db.sql` still calls the real `_flight_options`, which
    is the code under test — the header assembly that decides what rides the
    wire — so the bearer threading is exercised, not bypassed.
    """
    location = f"{scheme}://{db._endpoint}"
    kwargs = {"tls_root_certs": tls_root_certs} if tls_root_certs is not None else {}
    db._flight = flight.FlightClient(location, **kwargs)
    return db.sql("SELECT 1")


def test_bearer_credentials_send_authorization_on_flight(stub):
    scheme, endpoint, factory, tls_root_certs = stub
    db = open_remote(
        endpoint, tls=(scheme == "grpc+tls"), credentials=BearerCredentials(_TOKEN)
    )
    try:
        table = _query(db, scheme, tls_root_certs)
    finally:
        db.close()

    assert table.num_rows == 1
    # The stub saw the bearer on every Flight call (info + get), proving the
    # client put `authorization: Bearer <token>` on the lane over this scheme.
    assert factory.authorizations
    assert all(a == f"Bearer {_TOKEN}" for a in factory.authorizations)


def test_anonymous_credentials_send_no_bearer_on_flight(stub):
    scheme, endpoint, factory, tls_root_certs = stub
    db = open_remote(
        endpoint, tls=(scheme == "grpc+tls"), credentials=AnonymousCredentials()
    )
    try:
        table = _query(db, scheme, tls_root_certs)
    finally:
        db.close()

    assert table.num_rows == 1
    # An anonymous connection carries no bearer — the stub saw the lane's calls
    # but no `authorization` header on any of them.
    assert factory.authorizations
    assert all(a is None for a in factory.authorizations)


def test_no_credentials_send_no_bearer_on_flight(stub):
    scheme, endpoint, factory, tls_root_certs = stub
    # `credentials=None` is the `connect()` default; it resolves to anonymous
    # and so must put no bearer on the Flight lane either.
    db = open_remote(endpoint, tls=(scheme == "grpc+tls"), credentials=None)
    try:
        _query(db, scheme, tls_root_certs)
    finally:
        db.close()

    assert factory.authorizations
    assert all(a is None for a in factory.authorizations)


def test_forged_bearer_is_distinguishable_from_the_real_one(stub):
    scheme, endpoint, factory, tls_root_certs = stub
    db = open_remote(
        endpoint,
        tls=(scheme == "grpc+tls"),
        credentials=BearerCredentials("forged-token"),
    )
    try:
        _query(db, scheme, tls_root_certs)
    finally:
        db.close()

    # The header the stub observes is the exact token the client holds, so a
    # forged credential is distinguishable from the genuine one server-side: it
    # is neither absent nor equal to the real bearer.
    assert factory.authorizations
    assert all(a == "Bearer forged-token" for a in factory.authorizations)
    assert all(a != f"Bearer {_TOKEN}" for a in factory.authorizations)
