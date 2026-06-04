"""jammi-client — pure-Python gRPC client for a remote Jammi engine.

The deploy half of the develop→deploy journey: `jammi-ai` runs an embedded
engine and bundles this client for its remote targets; `jammi-client` is the
lean, candle-free, `py3-none-any` variant for production where the engine runs
behind a server.

One `connect(target)` is the single front door, the Python mirror of the Rust
`Jammi::open(Target)`: transport is config, not a code path. A `https://` /
`grpc://` target opens a remote :class:`RemoteDatabase`; a `file://` target is a
local engine this build does not carry, so it raises the truthful
:class:`NoEmbeddedEngineError` (the runtime echo of the Rust `#[cfg]` gate).
"""

from __future__ import annotations

from typing import Union

from ._database import RemoteDatabase
from ._errors import NoEmbeddedEngineError
from ._target import LocalTarget, RemoteTarget, Target, parse_target

__version__ = "0.19.0"

__all__ = [
    "connect",
    "RemoteDatabase",
    "NoEmbeddedEngineError",
    "LocalTarget",
    "RemoteTarget",
    "Target",
    "parse_target",
    "__version__",
]


def connect(target: Union[str, Target]) -> RemoteDatabase:
    """Open a session against `target`, selecting its transport once.

    `target` is a URI string or a structured :class:`Target`:

    * ``https://host`` / ``grpcs://host:8081`` → a TLS :class:`RemoteDatabase`.
    * ``http://host`` / ``grpc://host:8081`` → a plaintext :class:`RemoteDatabase`.
    * ``file:///data`` → a local engine. `jammi-client` carries no compiled
      engine, so this raises :class:`NoEmbeddedEngineError` pointing at
      `pip install jammi-ai`.

    Scaling local→remote is an env flip (``connect(os.environ["JAMMI_TARGET"])``)
    with no code change; productionising from the embed wheel to this lean client
    is a one-line import swap (`import jammi_ai` → `import jammi_client`),
    `connect` unchanged.
    """
    parsed = parse_target(target)
    if isinstance(parsed, LocalTarget):
        raise NoEmbeddedEngineError(parsed.artifact_dir)
    # RemoteTarget — the only transport this build carries.
    from ._database import open_remote

    return open_remote(parsed.endpoint, tls=parsed.tls)
