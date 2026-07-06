"""Jammi AI — embeddable inference engine for structured data.

The **develop** half of the develop→deploy journey: a compiled, in-process
engine (the `jammi_native` extension) plus the bundled pure-Python `jammi-client` for
remote targets. One `connect(target)` is the single front door — the Python
mirror of the Rust `Jammi::open(Target)` — where transport is config, not a code
path:

* ``connect("file:///data")`` → the compiled local engine (`jammi_native`).
* ``connect("https://host")`` / ``"grpc://host:8081"`` → the bundled
  `jammi-client`, by composition. The remote surface is DEFINED ONCE, in
  `jammi-client`; `jammi-ai`'s remote IS `jammi-client`'s, so the two agree by
  construction rather than by a parallel reimplementation.

Scaling local→remote is an env flip (``connect(os.environ["JAMMI_TARGET"])``)
with no code change; productionising to the lean client is a one-line import
swap (``import jammi_ai`` → ``import jammi_client``), `connect` unchanged.
"""

from __future__ import annotations

# The one front door — DEFINED ONCE in `jammi-client`, re-exported here. `jammi_ai`
# is a convenience bundle, not a parallel implementation: `jammi_ai.connect` IS
# `jammi_client.connect` (same function object), which dispatches `file://` to the
# in-process engine and remote targets to the gRPC client. The only thing this
# bundle adds is that it PINS the `[embedded]` extra (see this dist's
# `pyproject.toml`), so `jammi_native` is always present and a `file://` target
# always resolves — the "batteries-included" develop half of develop→deploy.
from jammi_client import (
    NoEmbeddedEngineError,
    RemoteDatabase,
    Session,
    connect,
)

# The embedded backend — the local `Session` a `file://` target returns. Defined
# once in `jammi_client._embedded`; re-exposed here under its historical name
# `jammi_ai.Database` for the convenience surface (retired in U3's rename).
from jammi_client import EmbeddedBackend as Database

# The compiled engine's handle + primitive types, re-exported so a caller reaches
# them off `jammi_ai` without naming `jammi_native`. `jammi_ai` carries the engine
# by construction (it pins the extra), so this eager native import is honest here
# — unlike `jammi_client`, whose native import is lazy.
from jammi_native import (
    AuditHandle,
    EphemeralSession,
    ModelTask,
    PerQueryAudit,
    TrainingJob,
    open_local,
)

__all__ = [
    "connect",
    "Session",
    "Database",
    "RemoteDatabase",
    "NoEmbeddedEngineError",
    "TrainingJob",
    "ModelTask",
    "PerQueryAudit",
    "AuditHandle",
    "EphemeralSession",
    "open_local",
]
