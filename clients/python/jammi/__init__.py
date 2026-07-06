"""jammi — the base Python client for a Jammi engine, local or remote.

The single front door: one `connect(target)`, the Python mirror of the Rust
`Jammi::open(Target)`, where transport is config, not a code path.

* ``https://`` / ``grpc://`` → a remote :class:`RemoteDatabase` over the
  `jammi.v1` gRPC wire — the lean, candle-free surface, always available.
* ``file://`` → the compiled in-process engine (an :class:`EmbeddedBackend`,
  direct FFI) when the ``jammi-ai[embedded]`` extra is installed (it pulls
  the `jammi-ai-native` engine, importable as `jammi_native`); otherwise a
  truthful :class:`NoEmbeddedEngineError` pointing at that extra — the runtime
  echo of the Rust `#[cfg]` gate.

`jammi-ai` (import `jammi`) is the BASE: it discovers the native engine as an
in-process backend without carrying it as a hard dependency. `import jammi` stays
native-free — the engine is imported LAZILY, only when a `file://` target is
opened (`jammi._embedded._open_embedded`) — so the lean deploy build and
the embed build are one package, differing only by whether the extra is present.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Optional, Union

from . import errors
from ._backend import Backend, Session, TrainingJobHandle
from ._capability import Capability
from ._credentials import BearerCredentials, ChannelCredentials
from ._database import RemoteDatabase, RemoteTrainingJob
# `_embedded` is native-free at import (it imports `jammi_native` lazily, inside
# `_open_embedded`), so naming `EmbeddedBackend` here keeps `import jammi`
# native-free — the client-import guard the positive conformance test pins.
from ._embedded import EmbeddedBackend
from ._target import LocalTarget, RemoteTarget, Target, parse_target
from .errors import (
    BackendError,
    InvalidArgument,
    JammiError,
    NoEmbeddedEngineError,
    NotSupportedOnBackend,
    TrainingError,
)

__version__ = version("jammi-ai")

__all__ = [
    "connect",
    # Protocols — the transport-agnostic surface a caller writes against.
    "Session",
    "Backend",
    "TrainingJobHandle",
    "Capability",
    # Concrete remote transport.
    "RemoteDatabase",
    "RemoteTrainingJob",
    # Concrete embedded transport (the compiled in-process engine). Named here,
    # but its module (`_embedded`) imports `jammi_native` lazily, so referencing
    # the name does not pull the engine — only opening a `file://` target does.
    "EmbeddedBackend",
    # The JammiError taxonomy (also `jammi.errors`).
    "errors",
    "JammiError",
    "InvalidArgument",
    "NotSupportedOnBackend",
    "NoEmbeddedEngineError",
    "TrainingError",
    "BackendError",
    # Credentials + targets.
    "ChannelCredentials",
    "BearerCredentials",
    "LocalTarget",
    "RemoteTarget",
    "Target",
    "parse_target",
    "__version__",
]


def connect(
    target: Union[str, Target],
    *,
    credentials: Optional[ChannelCredentials] = None,
) -> Session:
    """Open a session against `target`, selecting its transport once.

    `target` is a URI string or a structured :class:`Target`:

    * ``https://host`` / ``grpcs://host:8081`` → a TLS :class:`RemoteDatabase`.
    * ``http://host`` / ``grpc://host:8081`` → a plaintext :class:`RemoteDatabase`.
    * ``file:///data`` → the compiled in-process engine (an
      :class:`EmbeddedBackend`, direct FFI) when the ``jammi-ai[embedded]``
      extra is installed (it makes `jammi_native` importable); otherwise a
      :class:`NoEmbeddedEngineError` pointing at that extra — the runtime echo of
      the Rust `#[cfg]` gate.

    Both arms return a :class:`Session` — the transport-agnostic surface — so a
    caller writes one program and flips local↔remote by target alone.

    `credentials` decides what identity rides a REMOTE channel. ``None`` opens an
    anonymous channel; a :class:`BearerCredentials` attaches
    `authorization: Bearer <token>` to every call on both TLS and plaintext
    transports — the bearer rides the channel, not each verb. The bearer covers
    both transports: the typed gRPC verbs on the channel credentials, and
    :meth:`RemoteDatabase.sql` (the Flight SQL lane) per call. Server-side
    enforcement of the BYO-auth seam over Flight is tracked at
    https://github.com/f-inverse/jammi-ai/issues/220 (§5.8 — external; the
    EMBEDDED `sql` runs in-process over the native DataFusion engine and carries
    no channel to authenticate). Credentials are meaningless on a `file://`
    target — an in-process engine has no channel — so a local target opened
    *with* credentials is a caller error, rejected as a
    :class:`NoEmbeddedEngineError` before an engine is opened.

    Scaling local→remote is an env flip (``connect(os.environ["JAMMI_TARGET"])``)
    with no code change; the embed build and this lean build are one package,
    differing only by whether the ``[embedded]`` extra is present.
    """
    parsed = parse_target(target)
    if isinstance(parsed, LocalTarget):
        if credentials is not None:
            # A local target has no channel to authenticate — the target/credential
            # mismatch is a caller error, caught before an engine is opened.
            raise NoEmbeddedEngineError(parsed.artifact_dir)
        # The base client discovers the engine on demand: probe the extra WITHOUT
        # importing it into this module's namespace (keeps `import jammi`
        # native-free). If it is absent, this build carries no engine — raise the
        # truthful error hinting at the extra.
        from importlib.util import find_spec

        if find_spec("jammi_native") is None:
            raise NoEmbeddedEngineError(parsed.artifact_dir)
        # Known-importable — the factory does the lazy `import jammi_native`.
        from ._embedded import _open_embedded

        return _open_embedded(parsed.artifact_dir)
    # RemoteTarget — the gRPC transport.
    from ._database import open_remote

    return open_remote(parsed.endpoint, tls=parsed.tls, credentials=credentials)
