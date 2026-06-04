"""Channel credentials — turning an endpoint into a gRPC channel.

A :class:`ChannelCredentials` is the boundary between `connect` / `open_remote`
and the gRPC transport: given an endpoint and whether the transport is secure,
it returns a ready channel. The credential decides what (if anything) rides
that channel — an anonymous channel carries no caller identity, a
:class:`BearerCredentials` attaches `authorization: Bearer <token>` to every
outbound call.

The bearer rides the channel, not each call. On a secure channel that is
composite call-credentials (`ssl + metadata_call_credentials`); on a plaintext
channel — where gRPC refuses call-credentials in cleartext — it is a client
interceptor that appends the same header. Either way a caller never threads the
token through individual RPC invocations, and any per-call metadata the caller
does send (such as the session header) composes with the bearer on the same
call.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, List, Optional

import grpc

# The header carrying the bearer credential. Single source so the value never
# drifts between the secure (metadata plugin) and plaintext (interceptor) paths.
AUTHORIZATION_HEADER = "authorization"


def _bearer_metadata(token: str) -> tuple:
    """The `(authorization, Bearer <token>)` pair attached to every call."""
    return (AUTHORIZATION_HEADER, f"Bearer {token}")


# The transport-security root a secure channel trusts when no other is chosen.
# Both credential types default their `secure_credentials` factory to this so
# the secure-base construction is written once; a test injects a different
# factory (e.g. a loopback transport) without touching the production default.
_default_secure_credentials: Callable[[], grpc.ChannelCredentials] = (
    grpc.ssl_channel_credentials
)


class ChannelCredentials(abc.ABC):
    """Opens an authenticated (or anonymous) gRPC channel to an endpoint.

    Implementations decide what identity rides the channel; `open_channel`
    builds a secure channel when `tls` is true and a plaintext one otherwise.
    """

    @abc.abstractmethod
    def open_channel(self, endpoint: str, *, tls: bool) -> grpc.Channel:
        """Open a channel to `endpoint`, secure when `tls` else plaintext."""


@dataclass(frozen=True)
class AnonymousCredentials(ChannelCredentials):
    """A channel that carries no caller identity.

    The default when `connect` is called without credentials: a bare secure
    channel for `tls=True`, a bare insecure channel otherwise.
    """

    secure_credentials: Callable[[], grpc.ChannelCredentials] = (
        _default_secure_credentials
    )

    def open_channel(self, endpoint: str, *, tls: bool) -> grpc.Channel:
        if tls:
            return grpc.secure_channel(endpoint, self.secure_credentials())
        return grpc.insecure_channel(endpoint)


@dataclass(frozen=True)
class BearerCredentials(ChannelCredentials):
    """A channel that attaches `authorization: Bearer <token>` to every call.

    `token` is stripped of surrounding whitespace and must be non-empty.
    `secure_credentials` selects which transport-security root to trust on the
    secure path, defaulting to the system TLS roots; the secure path composes
    those roots with the bearer as call-credentials, while the plaintext path
    appends the bearer through a client interceptor (gRPC declines to send
    call-credentials over an insecure channel).
    """

    token: str
    secure_credentials: Callable[[], grpc.ChannelCredentials] = (
        _default_secure_credentials
    )

    def __post_init__(self) -> None:
        stripped = self.token.strip()
        if not stripped:
            raise ValueError("BearerCredentials requires a non-empty token")
        object.__setattr__(self, "token", stripped)

    def open_channel(self, endpoint: str, *, tls: bool) -> grpc.Channel:
        if tls:
            call_creds = grpc.metadata_call_credentials(
                _BearerMetadataPlugin(self.token)
            )
            composite = grpc.composite_channel_credentials(
                self.secure_credentials(), call_creds
            )
            return grpc.secure_channel(endpoint, composite)
        return grpc.intercept_channel(
            grpc.insecure_channel(endpoint), _BearerCallInterceptor(self.token)
        )


class _BearerMetadataPlugin(grpc.AuthMetadataPlugin):
    """Supplies the bearer header as call-credentials on a secure channel.

    gRPC invokes the plugin per call and merges the returned metadata into the
    outbound request, so the bearer rides the channel rather than each RPC's
    explicit metadata argument.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def __call__(
        self,
        context: grpc.AuthMetadataContext,
        callback: grpc.AuthMetadataPluginCallback,
    ) -> None:
        callback((_bearer_metadata(self._token),), None)


class _BearerCallInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """Appends the bearer header on every call over a plaintext channel.

    gRPC refuses to send call-credentials over an insecure channel, so the
    plaintext path carries the bearer as ordinary per-call metadata appended by
    this interceptor. Existing metadata (such as the session header) is kept and
    the bearer is added alongside it.
    """

    def __init__(self, token: str) -> None:
        self._meta = _bearer_metadata(token)

    def _augment(
        self, call_details: grpc.ClientCallDetails
    ) -> grpc.ClientCallDetails:
        metadata: List[tuple] = list(call_details.metadata or ())
        metadata.append(self._meta)
        return _ClientCallDetails(
            method=call_details.method,
            timeout=call_details.timeout,
            metadata=metadata,
            credentials=call_details.credentials,
            wait_for_ready=call_details.wait_for_ready,
            compression=call_details.compression,
        )

    def intercept_unary_unary(self, continuation, call_details, request):
        return continuation(self._augment(call_details), request)

    def intercept_unary_stream(self, continuation, call_details, request):
        return continuation(self._augment(call_details), request)

    def intercept_stream_unary(self, continuation, call_details, request_iterator):
        return continuation(self._augment(call_details), request_iterator)

    def intercept_stream_stream(self, continuation, call_details, request_iterator):
        return continuation(self._augment(call_details), request_iterator)


class _ClientCallDetails(grpc.ClientCallDetails):
    """A `ClientCallDetails` carrying the interceptor's augmented metadata.

    `grpc.ClientCallDetails` is a structural protocol with no public
    constructor; the interceptor builds one with every field set explicitly so a
    missing piece is a construction error rather than a silently dropped option.
    """

    def __init__(
        self,
        *,
        method: str,
        timeout: Optional[float],
        metadata: Optional[List[tuple]],
        credentials: Optional[grpc.CallCredentials],
        wait_for_ready: Optional[bool],
        compression: Optional[grpc.Compression],
    ) -> None:
        self.method = method
        self.timeout = timeout
        self.metadata = metadata
        self.credentials = credentials
        self.wait_for_ready = wait_for_ready
        self.compression = compression
