"""`Target` — where a session should run, parsed once from a URI or struct.

This is the Python mirror of the Rust `jammi_ai::jammi::Target` sum
(`Local(JammiConfig) | Remote(Endpoint)`): one closed choice the `connect`
front door dispatches on, rather than two `connect_local` / `connect_remote`
constructors. Transport is chosen here, at open time, and never leaks into the
call site of any verb.

Parsing lives in `jammi-client` — the always-present dependency of the embed
wheel — so BOTH `jammi_client.connect` and `jammi_ai.connect` classify a target
through the *same* code. The scheme→transport mapping is defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
from urllib.parse import urlparse


@dataclass(frozen=True)
class LocalTarget:
    """An embedded, in-process engine rooted at `artifact_dir`.

    Resolved by the compiled engine in the `jammi-ai` wheel; the pure-Python
    `jammi-client` cannot honour it and raises a truthful no-engine error — the
    runtime echo of the Rust build's `#[cfg(feature = "local")]` gate.
    """

    artifact_dir: str


@dataclass(frozen=True)
class RemoteTarget:
    """A remote engine reached over the `jammi.v1` gRPC wire at `endpoint`.

    `endpoint` is normalised to the `host:port` (or `host`) authority a grpcio
    channel takes; `tls` records whether the original scheme implied transport
    security (`https://` / `grpcs://`).
    """

    endpoint: str
    tls: bool


Target = Union[LocalTarget, RemoteTarget]


# Schemes that name an embedded, in-process engine.
_LOCAL_SCHEMES = frozenset({"file"})
# Schemes that name a remote engine over gRPC, paired with whether the scheme
# implies TLS. `https`/`grpcs` are secure; `http`/`grpc` are plaintext.
_REMOTE_SCHEMES = {
    "https": True,
    "grpcs": True,
    "http": False,
    "grpc": False,
}


def parse_target(target: Union[str, Target]) -> Target:
    """Classify `target` into the one `Local | Remote` sum.

    Accepts either an already-structured :class:`Target` (returned as-is) or a
    URI string whose scheme selects the transport:

    * ``file:///data`` → :class:`LocalTarget` (embedded engine)
    * ``https://host`` / ``grpcs://host:8081`` → secure :class:`RemoteTarget`
    * ``http://host`` / ``grpc://host:8081`` → plaintext :class:`RemoteTarget`

    A URI with no recognised scheme raises ``ValueError`` naming the offending
    scheme — never a silent default to one transport.
    """
    if isinstance(target, (LocalTarget, RemoteTarget)):
        return target
    if not isinstance(target, str):
        raise TypeError(
            f"target must be a URI string or a Target, got {type(target).__name__}"
        )

    parsed = urlparse(target)
    scheme = parsed.scheme.lower()

    if scheme in _LOCAL_SCHEMES:
        # `file:///data` → path `/data`; `file://./rel` keeps the netloc as part
        # of the path so a relative artifact dir survives.
        artifact_dir = parsed.path or parsed.netloc
        if parsed.netloc and parsed.path:
            artifact_dir = parsed.netloc + parsed.path
        if not artifact_dir:
            raise ValueError(f"file:// target carries no artifact directory: {target!r}")
        return LocalTarget(artifact_dir=artifact_dir)

    if scheme in _REMOTE_SCHEMES:
        if not parsed.netloc:
            raise ValueError(f"{scheme}:// target carries no host: {target!r}")
        return RemoteTarget(endpoint=parsed.netloc, tls=_REMOTE_SCHEMES[scheme])

    raise ValueError(
        f"unrecognised target scheme {scheme!r} in {target!r}: "
        "use file:// (local), or http(s):// / grpc(s):// (remote)"
    )
