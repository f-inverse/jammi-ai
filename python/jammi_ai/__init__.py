"""Jammi AI — embeddable inference engine for structured data.

The **develop** half of the develop→deploy journey: a compiled, in-process
engine (the `_native` extension) plus the bundled pure-Python `jammi-client` for
remote targets. One `connect(target)` is the single front door — the Python
mirror of the Rust `Jammi::open(Target)` — where transport is config, not a code
path:

* ``connect("file:///data")`` → the compiled local engine (`_native`).
* ``connect("https://host")`` / ``"grpc://host:8081"`` → the bundled
  `jammi-client`, by composition. The remote surface is DEFINED ONCE, in
  `jammi-client`; `jammi-ai`'s remote IS `jammi-client`'s, so the two agree by
  construction rather than by a parallel reimplementation.

Scaling local→remote is an env flip (``connect(os.environ["JAMMI_TARGET"])``)
with no code change; productionising to the lean client is a one-line import
swap (``import jammi_ai`` → ``import jammi_client``), `connect` unchanged.
"""

from __future__ import annotations

from typing import Union

import jammi_client
from jammi_client import LocalTarget, RemoteDatabase, Target, parse_target

from jammi_ai._native import (
    open_local,
    Database,
    FineTuneJob,
    ModelTask,
    PerQueryAudit,
    AuditHandle,
    EphemeralSession,
)

__all__ = [
    "connect",
    "Database",
    "RemoteDatabase",
    "FineTuneJob",
    "ModelTask",
    "PerQueryAudit",
    "AuditHandle",
    "EphemeralSession",
]


def connect(target: Union[str, Target]) -> Union[Database, RemoteDatabase]:
    """Open a session against `target`, selecting its transport once.

    `target` is a URI string or a structured :class:`~jammi_client.Target`:

    * ``file:///data`` → the compiled, in-process engine (a :class:`Database`),
      rooted at the target's path.
    * ``https://host`` / ``grpcs://host:8081`` / ``http://host`` /
      ``grpc://host:8081`` → a remote engine over the `jammi.v1` gRPC wire (a
      :class:`~jammi_client.RemoteDatabase`), via the bundled `jammi-client`.

    The remote arm delegates to `jammi_client.connect`, so the remote verb
    surface here is exactly `jammi-client`'s — defined once, agreeing by
    construction.
    """
    parsed = parse_target(target)
    if isinstance(parsed, LocalTarget):
        return open_local(artifact_dir=parsed.artifact_dir)
    # Remote — hand the original target to the bundled client; it re-parses to
    # the same RemoteTarget and opens the channel. One remote definition.
    return jammi_client.connect(parsed)
